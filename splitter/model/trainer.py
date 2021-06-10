"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.gcn import GCNClassifier
from utils import constant, torch_utils
from model.sinkhorn import SinkhornDistance


class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda):
    if cuda:
        inputs = [Variable(b.cuda()) for b in batch[:12]]
        labels = Variable(batch[12].cuda())
    else:
        inputs = [Variable(b) for b in batch[:10]]
        labels = Variable(batch[10])
    tokens = batch[0]
    head = batch[5]
    lens = batch[1].eq(0).long().sum(1).squeeze()
    return inputs, labels, tokens, head, lens

class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(opt, emb_matrix=emb_matrix)
        self.sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean')
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

        print('number of parameters: ', sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def update(self, batch):
        inputs, labels, tokens, head, lens = unpack_batch(batch, self.opt['cuda'])

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, words = self.model(inputs)

        groups = []
        dirty = []
        sel_ind = {}
        for j in range(4):
            selected = []
            values = []
            for i in range(len(logits)):
                if i not in dirty:
                    if len(selected) < self.opt['batch_size'] // 4:
                        selected.append((words[i],logits[i,j], i))
                        values.append(logits[i,j].item())
                        dirty.append(i)
                        sel_ind[i] = j
                    else:
                        if min(values) < logits[i,j].item():
                            ind = values.index(min(values))
                            dirty.remove(selected[ind][2])
                            del sel_ind[selected[ind][2]]
                            selected[ind] = (words[i],logits[i,j],i)
                            values[ind] = logits[i,j].item()
                            dirty.append(i)
                            sel_ind[i] = j
            groups.append(selected)
        loss = 0
        for i in range(4):
            for j in range(i+1,4):
                loss += self.sinkhorn(torch.stack([a[0].cpu() for a in groups[i]], dim=0).cpu(),torch.stack([a[0].cpu() for a in groups[j]],dim=0).cpu(),torch.stack([a[1].cpu() for a in groups[i]],dim=0).cpu(),torch.stack([a[1].cpu() for a in groups[j]],dim=0).cpu(),cuda=False)[0]
        loss_val = loss.item()
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss_val, sel_ind
