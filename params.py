from callbacks import *
from functools import partial

class parameters():
    def __init__(self):

        #Computational parameters:
        self.set_CUDA = True

        #Dataset parameters:
        # self.dataset = 'mnist'

        #NN parameters:
        # self.batchnorm = True

        #Forest parameters:
        self.use_tree = True
        self.use_prenet = True

        self.n_trees = 1

        #Tree parameters:
        self.tree_depth = 10
        self.cascading = True
        # self.single_sigmoid = False
        # self.softmax_normalization = True ##! replace softmax_normalization in tree_conf

        #Training parameters:
        self.epochs = 200
        self.batch_size = 64
        self.one_batch = True
        self.learning_rate = 0.1
        self.weight_decay=1e-5

        #Wavelet parameters:
        self.wavelets = True
        self.intervals = 200
   

        #Callback parameters
        self.cbfs = [Recorder, partial(AvgStatsCallback,accuracy),partial(CudaCallback,device)]
        if self.wavelets:
            self.cbfs.append(wavelets)
