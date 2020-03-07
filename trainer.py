#This is where the training loop takes place
import torch
from model_conf import entire_network
from typing import Iterable
from callbacks import *

def accuracy(out, yb): return (torch.argmax(out, dim=1)==yb).float().mean()

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

def print_progress(epoch,batch_number):
    # if batch_number % 300 == 0:
    #     print(f"epoch {epoch}, batch {batch_number}")
    if batch_number % 100 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(\
            epoch, batch_idx * len(data), num_train,\
            100. * batch_idx / len(train_loader), loss.data.item()))  

def fit(conf, learner): #model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(conf.epochs):

        learner.model.train()

        for i,(xb,yb) in enumerate(learner.data.train_dl):
            learner.model.every_batch(yb)
            loss = learner.loss_func(torch.log(learner.model(xb)), yb)
            loss.backward()
            learner.opt.step()
            learner.opt.zero_grad()

            print_progress(epoch,i)

        

        learner.model.eval()

        with torch.no_grad():
            tot_loss,tot_acc = 0.,0.
            for xb,yb in learner.data.valid_dl:
                pred = learner.model(xb)
                ##! may be neccessary in future:
                pred = pred.clamp(min=1e-6, max=1) # resolve some numerical issue
                tot_loss += learner.loss_func(torch.log(pred), yb)
                tot_acc  += accuracy(pred,yb)
        nv = len(learner.data.valid_dl)
        print(f"epoch: {epoch}, loss: {tot_loss/nv}, accuracy: {tot_acc/nv}")
        # learner.model.forest.trees[0].mu_cache = []

        learner.model.every_epoch()
    return tot_loss/nv, tot_acc/nv

##! go over fast.ai #10 and deep-dive the hellouta these classes
######################################################################################################################
class Runner():
    def __init__(self, cbs=None, cb_funcs=None):
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
            ##! add to seperate callback later:
            #HT seperate_callback
            # self.model.every_batch(yb)
            self('begin_batch')
            self.pred = self.model(self.xb)
            self('after_pred')
            ##! make sure torch.log is part of seperate callback
            #HT seperate_callback
            #self.pred = torch.log(self.pred)
            self.loss = self.loss_func(torch.log(self.pred), self.yb)
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