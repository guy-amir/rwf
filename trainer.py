#This is where the training loop takes place
import torch
from model_conf import entire_network

def accuracy(out, yb): return (torch.argmax(out, dim=1)==yb).float().mean()
def print_progress(epoch,batch_number):
    if batch_number % 300 == 0:
        print(f"epoch {epoch}, batch {batch_number}")

def fit(conf, learner): #model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(conf.epochs):

        learner.model.train()

        for i,(xb,yb) in enumerate(learner.data.train_dl):
            learner.model.every_batch(yb)
            loss = learner.loss_func(learner.model(xb), yb)
            loss.backward()
            learner.opt.step()
            learner.opt.zero_grad()

            print_progress(epoch,i)

        learner.model.every_epoch()

        learner.model.eval()

        with torch.no_grad():
            tot_loss,tot_acc = 0.,0.
            for xb,yb in learner.data.valid_dl:
                pred = learner.model(xb)
                ##! may be neccessary in future:
                # pred = pred.clamp(min=1e-6, max=1) # resolve some numerical issue
                tot_loss += learner.loss_func(pred, yb)
                tot_acc  += accuracy(pred,yb)
        nv = len(learner.data.valid_dl)
        print(epoch, tot_loss/nv, tot_acc/nv)
        # learner.model.forest.trees[0].mu_cache = []
    return tot_loss/nv, tot_acc/nv

##! go over fast.ai #10 and deep-dive the hellouta these classes
######################################################################################################################
class Callback():
    _order = 0
    def set_runner(self, run): self.run = run
    def __getattr__(self, k): return getattr(self.run, k)

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f(): return True #? why?
        return False

class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.run.n_epochs=0.
        self.run.n_iter=0
    
    def after_batch(self):
        if not self.in_train: return
        self.run.n_epochs += 1./self.iters
        self.run.n_iter   += 1
        
    def begin_epoch(self):
        self.run.n_epochs=self.epoch
        self.model.train()
        self.run.in_train=True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train=False

class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass


class Runner():
    def __init__(self, cbs=None, cb_funcs=None):
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop,self.cbs = False,[TrainEvalCallback()]+cbs

    @property
    def opt(self):       return self.learn.opt
    @property
    def model(self):     return self.learn.model
    @property
    def loss_func(self): return self.learn.loss_func
    @property
    def data(self):      return self.learn.data

    def one_batch(self, xb, yb):
        try:
            self.xb,self.yb = xb,yb
            self('begin_batch')
            self.pred = self.model(self.xb)
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

    def fit(self, epochs, learn):
        self.epochs,self.learn,self.loss = epochs,learn,tensor(0.)

        try:
            for cb in self.cbs: cb.set_runner(self)
            self('begin_fit')
            for epoch in range(epochs):
                self.epoch = epoch
                if not self('begin_epoch'): self.all_batches(self.data.train_dl)

                with torch.no_grad(): 
                    if not self('begin_validate'): self.all_batches(self.data.valid_dl)
                self('after_epoch')
            
        except CancelTrainException: self('after_cancel_train')
        finally:
            self('after_fit')
            self.learn = None

    def __call__(self, cb_name):
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name) or res
        return res