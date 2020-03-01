#This file contains all the configuration aparameters

class parameters():
    def __init__(self):

        #Computational parameters:
        self.set_CUDA = True

        #Dataset parameters:
        self.dataset = 'mnist'

        #NN parameters:

        #Forest parameters:
        self.n_trees = 2

        #Tree parameters:
        self.tree_depth = 6

        #Training parameters:
        self.epochs = 30
        self.batch_size = 64
        self.learning_rate = 0.001
        self.weight_decay=1e-6
