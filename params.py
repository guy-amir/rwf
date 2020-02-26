#This file contains all the configuration aparameters

class parameters():
    def __init__(self):

        #Computational parameters:
        self.set_CUDA = True

        #Dataset parameters:
        self.dataset = 'mnist'

        #NN parameters:

        #Forest parameters:
        self.n_trees = 1

        #Tree parameters:
        self.tree_depth = 3

        #Training parameters:
        self.epochs = 10
        self.batch_size = 64
        self.learning_rate = 0.01
