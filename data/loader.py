"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np

from utils import constant, helper, vocab

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False, low=0, high=-1):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID
        self.low = low
        self.high = high

        with open(filename) as infile:
            data = json.load(infile)[low:high]
        self.raw_data = data
        data = self.preprocess(data, vocab, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])
        self.labels = [d[-1] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            tokens = list(d['token'])
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            # doc_pos = [map_to_ids(d['doc'][i]['pos'], constant.POS_TO_ID) if i < len(d['doc']) else map_to_ids(['DT'], constant.POS_TO_ID) for i in range(20)]
            doc_pos = [map_to_ids(d['doc'][i]['pos'], constant.POS_TO_ID) if i < len(d['doc']) else map_to_ids(['DT'], constant.POS_TO_ID) for i in range(5)]
            doc_token = [map_to_ids(d['doc'][i]['token'], vocab.word2id) if i < len(d['doc']) else map_to_ids(['the'], vocab.word2id) for i in range(5)]
            doc_triggers = [d['doc'][i]['triggers'] if i < len(d['doc']) else [0] for i in range(5)]
            doc_event = [d['doc'][i]['event'] if i < len(d['doc']) else 0 for i in range(5)]
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            head = [int(x) for x in d['stanford_head']]
            assert any([x == 0 for x in head])
            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            relation = d['relation']
            lens = sorted(list(zip([len(doc) for doc in doc_token],range(len(doc_token)))), key= lambda p: -p[0])
            doc_pos = [doc_pos[p[1]] for p in lens]
            doc_token = [doc_token[p[1]] for p in lens]
            doc_triggers = [doc_triggers[p[1]] for p in lens]
            doc_event = [doc_event[p[1]] for p in lens]
            try:
                bert = d['bert']
            except:
                bert = []
                for _ in range(len(d['token'])):
                    bert += [[0]*768]
            # doc_berts = []
            # for i in range(5):
            #     try:
            #         doc_bert = d['doc'][i]['bert']
            #     except:
            #         if i < len(d['doc']):
            #             l = len(d['doc'][i]['token'])
            #         else:
            #             l = 1
            #         doc_bert = []
            #         for _ in range(l):
            #             doc_bert += [[0]*768]
            #     doc_berts += [doc_bert]
            processed += [(tokens, pos, deprel, head, subj_positions, doc_pos, doc_token, doc_triggers, doc_event, bert, relation)]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 11

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        deprel = get_long_tensor(batch[2], batch_size)
        head = get_long_tensor(batch[3], batch_size)
        subj_positions = get_long_tensor(batch[4], batch_size)

        doc_pos = get_doc_tensor(batch[5], batch_size)
        doc_tokens = get_doc_tensor(batch[6], batch_size)
        doc_masks = torch.eq(doc_tokens, 0)
        doc_triggers = get_doc_tensor(batch[7], batch_size)
        doc_events = get_long_tensor(batch[8], batch_size)

        bert = get_bert_tensor(batch[9], batch_size)
        # doc_berts = get_doc_bert_tensor(batch[10], batch_size)

        rels = torch.LongTensor(batch[10])

        return (words, masks, pos, deprel, head, subj_positions, doc_pos, doc_tokens, doc_masks, doc_triggers, doc_events, bert, rels, orig_idx)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def get_doc_tensor(tokens_list, batch_size):
    doc_len = max(len(x) for x in tokens_list)
    sent_len = 0
    for x in tokens_list:
        for s in x:
            if len(s) > sent_len:
                sent_len = len(s)
    tokens = torch.LongTensor(batch_size, doc_len, sent_len).fill_(constant.PAD_ID)
    for i, d in enumerate(tokens_list):
        for j, s in enumerate(d):
            tokens[i, j, :len(s)] = torch.LongTensor(s)
    return tokens

def get_bert_tensor(tokens_list, batch_size):
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len, 768).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s), :] = torch.FloatTensor(s)
    return tokens

def get_doc_bert_tensor(tokens_list, batch_size):
    doc_len = max(len(x) for x in tokens_list)
    sent_len = 0
    for x in tokens_list:
        for s in x:
            if len(s) > sent_len:
                sent_len = len(s)
    tokens = torch.FloatTensor(batch_size, doc_len, sent_len, 768).fill_(constant.PAD_ID)
    for i, d in enumerate(tokens_list):
        for j, s in enumerate(d):
            for k, t in enumerate(s):
                tokens[i, j, k, :] = torch.FloatTensor(t)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

