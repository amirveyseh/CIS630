"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.tree import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils


class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """

    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()

    def forward(self, inputs):
        outputs, pooling_output = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, pooling_output


class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb, self.ner_emb)
        self.init_embeddings()

        # gcn layer
        self.gcn = GCN(opt, embeddings, opt['hidden_dim'], opt['num_layers'])

        # output mlp layers
        in_dim = opt['hidden_dim'] * 2
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers'] - 1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: \
                                              torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def forward(self, inputs):
        words, masks, pos, deprel, head, subj_pos, doc_pos, doc_tokens, doc_masks, doc_triggers, doc_events, bert = inputs  # unpack
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)

        def inputs_to_tree_reps(head, words, l):
            trees = [head_to_tree(head[i], words[i], l[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in
                   trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)

        adj = inputs_to_tree_reps(head.data, words.data, l)
        h, pool_mask = self.gcn(adj, inputs)

        # pooling
        subj_mask = subj_pos.eq(0).eq(0).unsqueeze(2)
        pool_type = self.opt['pooling']
        h_out = pool(h, pool_mask, type=pool_type)
        subj_out = pool(h, subj_mask, type=pool_type)
        outputs = torch.cat([h_out, subj_out], dim=1)
        outputs = self.out_mlp(outputs)
        return outputs, h_out


class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """

    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim
        self.in_dim = opt['emb_dim'] + opt['pos_dim']

        self.emb, self.pos_emb, self.ner_emb = embeddings

        # rnn layer
        if self.opt.get('rnn', False):
            input_size = self.in_dim
            self.rnn = nn.LSTM(input_size + opt['rnn_hidden'] * 2 + 768, opt['rnn_hidden'], opt['rnn_layers'],
                               batch_first=True, \
                               dropout=opt['rnn_dropout'], bidirectional=True)
            self.rnn_word = nn.LSTM(input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                                    dropout=opt['rnn_dropout'], bidirectional=True)
            self.rnn_sent = nn.LSTM(opt['rnn_hidden'] * 2, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                                    dropout=opt['rnn_dropout'], bidirectional=True)
            self.in_dim = opt['rnn_hidden'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout'])  # use on last layer output

        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.cw = nn.Parameter(torch.rand(opt['hidden_dim']))
        self.fc_W = nn.Sequential(nn.Linear(2 * opt['rnn_hidden'], opt['hidden_dim']), nn.Tanh())
        self.cs = nn.Parameter(torch.rand(opt['hidden_dim']))
        self.fc_S = nn.Sequential(nn.Linear(2 * opt['rnn_hidden'], opt['hidden_dim']), nn.Tanh())

        self.mu = nn.Parameter(torch.rand(1))
        self.lambd = nn.Parameter(torch.rand(1))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size, rnn):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
        words, masks, pos, deprel, head, subj_pos, doc_pos, doc_tokens, doc_masks, doc_triggers, doc_events, bert = inputs  # unpack
        word_embs = self.emb(words)
        doc_word_embeds = self.emb(doc_tokens)

        embs = [word_embs, bert]
        doc_embs = [doc_word_embeds]
        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
            doc_embs += [self.pos_emb(doc_pos)]
        # doc_embs += [doc_bert]
        embs = torch.cat(embs, dim=2)
        doc_embs = torch.cat(doc_embs, dim=3)
        embs = self.in_drop(embs)
        doc_embs = self.in_drop(doc_embs)

        the_mask = doc_masks.view(doc_embs.shape[0] * doc_embs.shape[1], -1)
        the_words = doc_embs.view(doc_embs.shape[0] * doc_embs.shape[1], doc_embs.shape[2], -1)
        bs = doc_embs.shape[0] * doc_embs.shape[1]
        seq_lens = [a[1] for a in
                    sorted(list(zip(list(the_mask.data.eq(constant.PAD_ID).long().sum(1).squeeze()), range(bs))),
                           key=lambda a: -a[0].item())]
        the_words = the_words[seq_lens]
        the_mask = the_mask[seq_lens]
        seq_lens = [a[1] for a in sorted(list(zip(seq_lens, range(len(seq_lens)))), key=lambda a: a[0])]
        lstm_W = self.rnn_drop(
            self.encode_with_rnn(the_words, the_mask, bs, self.rnn_word))[seq_lens].view(doc_embs.shape[0],
                                                                                         doc_embs.shape[1],
                                                                                         doc_embs.shape[2], -1)
        lstm_W = lstm_W.sum(2).squeeze()
        lstm_S = self.rnn_drop(
            self.encode_with_rnn(lstm_W, torch.zeros(lstm_W.shape[0], lstm_W.shape[1]), lstm_W.shape[0], self.rnn_sent))
        lstm_S = lstm_S.sum(1).squeeze().repeat(1,embs.shape[1]).view(embs.shape[0],embs.shape[1],-1)
        embs = torch.cat([embs,lstm_S], dim=2)
        # exit(1)

        # new_emb = []
        # for i in range(len(doc_pos)):
        #     lens = torch.stack(list(doc_masks[i].data.eq(constant.PAD_ID).long().sum(1).squeeze()), dim=0)
        #     lens[lens > 0] = 1
        #     bs = lens.sum()
        #     lstm_W = self.rnn_drop(self.encode_with_rnn(doc_embs[i][:bs], doc_masks[i][:bs], bs, self.rnn_word))
        #     fc_w = (lstm_W).masked_fill(doc_masks[i][:bs, :lstm_W.shape[1]].unsqueeze(2), 0).sum(1).unsqueeze(0)
        #     lstm_S = \
        #         self.rnn_drop(
        #             self.encode_with_rnn(fc_w.repeat(2, 1, 1), torch.zeros(2, fc_w.shape[1]), 2, self.rnn_sent))[
        #             0]
        #     fc_s = (lstm_S).sum(0).repeat(words.shape[1]).view(words.shape[1], -1)
        #     new_emb += [torch.cat([embs[i], fc_s], dim=1)]
        # embs2 = torch.stack(new_emb, dim=0)

        # rnn layer
        if self.opt.get('rnn', False):
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, words.size()[0], self.rnn))
        else:
            gcn_inputs = embs

        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        # zero out adj for ablation
        if self.opt.get('no_adj', False):
            adj = torch.zeros_like(adj)

        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs)  # self loop
            AxW = AxW / denom

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return gcn_inputs, mask


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0
