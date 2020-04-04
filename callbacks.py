import re
import torch
from typing import Iterable
import matplotlib.pyplot as plt
from wavelets import psi

device = torch.device('cuda',0)
torch.cuda.set_device(device)

class Callback():
    _order=0
    def set_runner(self, run): self.run=run
    def __getattr__(self, k): return getattr(self.run, k)
    
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')
    
    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f(): return True
        return False

class wavelets(Callback):
    _order=1
    # def begin_batch(self):
    #     self.model.target_batches.append(self.model.target_indicator[self.run.yb])
        
    # def after_pred(self):
    #     self.pred = torch.log(self.pred)

    
    def prune(self,even_cutoff):

        self.wavelet_acc = []
        self.wavelet_loss = []
        self.nu_list = []
        self.y_hat_list = []
        self.wavelet_pred_list = []

        self.train_stats,self.valid_stats = AvgStats(accuracy,True),AvgStats(accuracy,False)
        

        with torch.no_grad(): 
            self.all_batches(self.data.valid_dl)
            # for tree in self.model.forest.trees:         
            # mean_mu = torch.mean(torch.stack(tree.mu_cache[:-1]),0)
            ##! this should be running over each mu in mu_cache or each yb in loader
            self.psi = psi(self)
            self.psi.cutoff_value_list = []
            for i in range(1,self.intervals):#range(2*(2**(opt.tree_depth))-1): 
                self.tot_loss = 0
                self.tot_acc = 0
                self.tot_samples = 0
                # ########################avg stats callback
                # self.train_stats.reset()
                # self.valid_stats.reset()
                # ########################end avg stats callback
                if even_cutoff == True:
                    self.psi.cutoff_value=int(self.mu.size(1)*i/self.intervals)
                else:
                    if i*5<self.model.mu.size(1):
                        self.psi.cutoff_value=i*5
                    else:
                        self.psi.cutoff_value=self.model.mu.size(1)
                self.psi.cutoff_value_list.append(self.psi.cutoff_value)
                
                self.all_wavelet_batches(self.data.valid_dl,self)
                
                # ########################avg stats callback
                print(f"\n cuttoff value is: {self.psi.cutoff_value}")
                print(f"loss is {self.tot_loss}")
                print(f"acc is {self.tot_acc}")
                # print(self.train_stats)
                # print(self.valid_stats)
                # ########################end avg stats callback


    def all_wavelet_batches(self, dl,runner):

        self.iters = len(dl)
        for i,[_,yb] in enumerate(dl): self.one_wavelet_batch(i,yb,runner)
        
        self.tot_acc = self.tot_acc/self.tot_samples
        self.wavelet_acc.append(self.tot_acc.cpu().numpy())
        self.tot_loss = self.tot_loss/self.tot_samples
        self.wavelet_loss.append(self.tot_loss)

    def one_wavelet_batch(self, i, yb, runner):
        yb = yb.cuda()
        leaf_list = runner.psi.cutoff(runner.psi.cutoff_value)
        self.nu  = self.model.mu[:,leaf_list]
        N = self.nu.sum(0)
        self.pred = self.nu @ self.model.y_hat[leaf_list]
        self.loss = float(self.loss_func(self.pred, yb))
        self.tot_loss += self.loss*yb.size(0)
        self.acc = accuracy(self.pred,yb)
        self.tot_acc += self.acc* yb.size(0)
        self.tot_samples += yb.size(0)

        self.nu_list.append(self.nu)
        self.y_hat_list.append(self.model.y_hat[leaf_list])
        self.wavelet_pred_list.append(self.pred)
    
# class Softmax(Callback):
#     continue

class Recorder(Callback):
    def begin_fit(self):
        self.pred_switch = False
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []

    def after_batch(self):
        if not self.in_train: return
        for pg,lr in zip(self.opt.param_groups,self.lrs): lr.append(pg['lr'])
        self.losses.append(self.loss.detach().cpu())        

    def plot_lr  (self, pgid=-1): plt.plot(self.lrs[pgid])
    def plot_loss(self, skip_last=0): plt.plot(self.losses[:len(self.losses)-skip_last])

    def after_fit(self):
        with torch.no_grad(): 
            self.in_train = True
            self.one_batch(self.data.valid_dl.ds.x, self.data.valid_dl.ds.y)
        self.record_pred()

    def record_pred(self):
        self.tot_pred = self.pred

    def plot(self, skip_last=0, pgid=-1):
        losses = [o.item() for o in self.losses]
        lrs    = self.lrs[pgid]
        n = len(losses)-skip_last
        plt.xscale('log')
        plt.plot(lrs[:n], losses[:n])

class CudaCallback(Callback):
    def __init__(self,device): self.device = device
    def begin_fit(self): self.model.cuda()
    def begin_batch(self): 
        self.run.xb, self.run.yb = self.xb.cuda(), self.yb.cuda()

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

class AvgStats():
    def __init__(self, metrics, in_train): self.metrics,self.in_train = listify(metrics),in_train
    
    def reset(self):
        self.tot_loss,self.count = 0.,0
        self.tot_mets = [0.] * len(self.metrics)
        
    @property
    def all_stats(self): return [self.tot_loss.item()] + self.tot_mets
    @property
    def avg_stats(self): return [o/self.count for o in self.all_stats]
    
    def __repr__(self):
        if not self.count: return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):
        bn = run.yb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i,m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn

class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)
        
    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        
    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)
    
    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)

class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass

def accuracy(pred, yb): return (pred-yb).pow(2).mean()

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]