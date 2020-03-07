#This file contains all the configuration aparameters
from callbacks import *
from functools import partial

class parameters():
    def __init__(self,use_tree=True):

        #Computational parameters:
        self.set_CUDA = True

        #Dataset parameters:
        self.dataset = 'mnist'

        #NN parameters:
        self.batchnorm = True

        #Forest parameters:
        self.use_tree = use_tree
        self.n_trees = 1

        #Tree parameters:
        self.tree_depth = 6
        self.single_sigmoid = False
        self.softmax_normalization = True ##! replace softmax_normalization in tree_conf

        #Training parameters:
        self.epochs = 3
        self.batch_size = 128
        self.learning_rate = 0.001
        self.weight_decay=1e-5

        #Wavelet parameters:
        self.wavelets = True
        self.cutoff = 30
   

        #Callback parameters
        self.cbfs = [Recorder, partial(AvgStatsCallback,accuracy),partial(CudaCallback,device)]
        if self.use_tree:
            self.cbfs.append(DeepNeuralWavelet)
