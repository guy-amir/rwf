import torch
import re
from typing import Iterable
from wavelets import wavelet

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

class DeepNeuralForest(Callback):
    _order=1
    def begin_batch(self):
        self.model.target_batches.append(self.model.target_indicator[self.run.yb])
        
    # def after_pred(self):
    #     self.pred = torch.log(self.pred)

    def after_epoch(self):
        for tree in self.model.forest.trees:
            tree.update_pi(self.model.target_batches)
            del tree.mu_cache
            tree.mu_cache = []
            self.model.target_batches = []

class DeepNeuralWavelet(DeepNeuralForest):
    _order=1

    # def begin_fit(self):
    #     self.model.conf.wavelets = False

    def after_fit(self):
        # self.model.conf.wavelets = True
        with torch.no_grad(): 
            self.all_batches(self.data.valid_dl)
            for tree in self.model.forest.trees:
                tree.wavelet = wavelet(tree)
                mean_mu = torch.mean(torch.stack(tree.mu_cache[:-1]),0)
                ##! this should be running over each mu in mu_cache or each yb in loader
                ##! mod pi remains constant for each cutoff of course
                for i in range(1,15):#range(2*(2**(opt.tree_depth))-1): 
                    j=5*i
                    leaf_list = tree.psi.cutoff(j)

                    nu  = tree.psi.mod_mu(mean_mu,leaf_list)
                    pu  = tree.psi.mod_pi(tree.pi,leaf_list)
                    p = nu @ pu
                    self.loss = self.loss_func(torch.log(p), self.yb)
                    print("hi!!!")

    #     # loss function                
    #     test_loss += F.nll_loss(torch.log(p), TARGET, reduction='sum').data.item() # sum up batch loss ##GG turned output to output[0] to avoid tuple error
    #     # get class prediction
    #     ##GG old: pred = output.data.max(1, keepdim = True)[1] # get the index of the max log-probability
    #     ##GG mod:
    #     pred = p.max(1, keepdim = True)[1]
    #     # count correct prediction
    #     correct += pred.eq(TARGET.data.view_as(pred)).cpu().sum()
    #     # averaging
    #     test_loss /= len(test_loader.dataset)
    #     test_acc = int(correct) / len(dataset)
    #     record = {'loss':test_loss, 'acc':test_acc, 'corr':correct, 'cutoff':j}
    #     cutoff_record.append(record)
    #     J.append(j)
    #     LOSS.append(test_loss)
    # return [J,LOSS]

    def after_epoch(self):
        for tree in self.model.forest.trees:
            # if not self.model.conf.wavelets:
                tree.update_pi(self.model.target_batches)
                del tree.mu_cache
                tree.mu_cache = []
                self.model.target_batches = []
            # else:
            #     tree.w = wavelet(tree)

            


    
# class Softmax(Callback):
#     continue

class Recorder(Callback):
    def begin_fit(self):
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []

    def after_batch(self):
        if not self.in_train: return
        for pg,lr in zip(self.opt.param_groups,self.lrs): lr.append(pg['lr'])
        self.losses.append(self.loss.detach().cpu())        

    def plot_lr  (self, pgid=-1): plt.plot(self.lrs[pgid])
    def plot_loss(self, skip_last=0): plt.plot(self.losses[:len(self.losses)-skip_last])
        
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
        bn = run.xb.shape[0]
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

def accuracy(out, yb): return (torch.argmax(out, dim=1)==yb).float().mean()

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