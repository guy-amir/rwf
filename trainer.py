from model_conf import *
from typing import Iterable
from callbacks import *
import torch
from model_conf import Tree

class Runner():
    def __init__(self, cbs=None, cb_funcs=None,conf=None):
        self.conf = conf
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop,self.cbs = False,[TrainEvalCallback()]+cbs

    @property
    def opt(self):       return self.learner.opt
    @property
    def model(self):     return self.learner.model
    @property
    def loss_func(self): return self.learner.loss_func
    @property
    def data(self):      return self.learner.data

    def one_batch(self, xb, yb):
        try:
            self.xb,self.yb = xb,yb
            self('begin_batch')

            #construct routing probability tree:
            self.mu = self.model(self.xb)

            #find the nodes that are leaves:
            mu_midpoint = int(self.mu.size(1)/2)
            self.mu_leaves = self.mu[:,mu_midpoint:]

            #create a normalizing factor for leaves:
            N = self.mu.sum(0)

            if self.in_train:
                self.y_hat = self.yb @ self.mu/N
                self.y_hat_leaves = self.y_hat[mu_midpoint:]

            self.pred = self.mu_leaves @ self.y_hat_leaves
            self('after_pred')
            
            self.loss = self.loss_func(self.pred, self.yb)
            self('after_loss')
            if not self.in_train: return
            self.loss.backward()
            self('after_backward')
            self.opt.step()
            self('after_step')
            self.opt.zero_grad()
        except CancelBatchException: self('after_cancel_batch')
        finally: self('after_batch')

    def all_batches(self, dl):
        self.iters = len(dl)
        try:
            for xb,yb in dl: self.one_batch(xb, yb)
        except CancelEpochException: self('after_cancel_epoch')

    def fit(self, epochs, learner):
        self.epochs,self.learner,self.loss = epochs,learner,torch.tensor(0.)
        self.in_train = self.learner.model.training

        try:
            for cb in self.cbs: cb.set_runner(self)
            self('begin_fit')
            for epoch in range(epochs):
                self.epoch = epoch
                if not self('begin_epoch'): 
                    self.all_batches(self.data.train_dl)

                with torch.no_grad(): 
                    if not self('begin_validate'): 
                        self.all_batches(self.data.valid_dl)
                self('after_epoch')
            
        except CancelTrainException: self('after_cancel_train')
        finally:
            self('after_fit')
            self.learner = None

    def __call__(self, cb_name):
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name) or res
        return res