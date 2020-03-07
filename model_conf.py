#this is where the nn is defined
import torch.nn as nn
import torch
from forest_conf import Forest

class Learner():
    def __init__(self, model, opt, loss_func, data):
        self.model,self.opt,self.loss_func,self.data = model,opt,loss_func,data

def get_model(conf,data):
    
    model = entire_network(conf,data)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate , weight_decay=conf.weight_decay)

    return model, optimizer

class entire_network(nn.Module):
    
    def __init__(self,conf,data):
        super(entire_network, self).__init__()
        if conf.use_tree:  
            self.forest = Forest(conf, data)
        self.target_indicator = nn.Parameter(torch.eye(data.c),requires_grad=False)
        self.target_batches = []
        self.data = data
        self.conf = conf
        
        self.nn_model = self.get_cnn_model(data)
        # self.layers = len(self.nn_model)
        # self.act_means = [[] for _ in self.nn_model]
        # self.act_stds  = [[] for _ in self.nn_model]
        

    def forward(self,x):
        nn_output_data = self.nn_model(x)
        # for i,l in enumerate(self.nn_model):
        #     x = l(x)
        #     self.act_means[i].append(x.data.mean())
        #     self.act_stds[i].append(x.data.std())
        # nn_output_data = x
        if self.conf.use_tree:    
            return self.forest(nn_output_data)
        else:
            return nn.functional.softmax(nn_output_data, dim=1)

    def every_batch(self,yb):
        self.target_batches.append(self.target_indicator[yb])


    def every_epoch(self):
        for tree in self.forest.trees:
            tree.update_pi(self.target_batches)
            del tree.mu_cache
            tree.mu_cache = []
            self.target_batches = []

    def get_cnn_model(self,data):

            self.conv_layers = nn.Sequential()
            self.conv_layers.add_module('square', Lambda(square_data))
            self.conv_layers.add_module('conv1', nn.Conv2d(1, 32, kernel_size=3, padding=1))
            if self.conf.batchnorm: self.conv_layers.add_module('bn1', nn.BatchNorm2d(32))
            self.conv_layers.add_module('relu1', nn.ReLU())
            self.conv_layers.add_module('pool1', nn.MaxPool2d(kernel_size=2))
            #self.add_module('drop1', nn.Dropout(dropout_rate))
            self.conv_layers.add_module('conv2', nn.Conv2d(32, 64, kernel_size=3, padding=1))
            if self.conf.batchnorm: self.conv_layers.add_module('bn2', nn.BatchNorm2d(64))
            self.conv_layers.add_module('relu2', nn.ReLU())
            self.conv_layers.add_module('pool2', nn.MaxPool2d(kernel_size=2))
            #self.add_module('drop2', nn.Dropout(dropout_rate))
            self.conv_layers.add_module('conv3', nn.Conv2d(64, 128, kernel_size=3, padding=1))
            if self.conf.batchnorm: self.conv_layers.add_module('bn3', nn.BatchNorm2d(128))
            self.conv_layers.add_module('relu3', nn.ReLU())
            self.conv_layers.add_module('pool3', nn.MaxPool2d(kernel_size=2))
            self.conv_layers.add_module('flatten', Lambda(flatten_data))
            self.conv_layers.add_module('linear', nn.Linear(1152,data.features4tree))
            if self.conf.batchnorm: self.conv_layers.add_module('bn4', nn.BatchNorm1d(data.features4tree))

            if self.conf.single_sigmoid:
                self.conv_layers.add_module('linear2', nn.Linear(data.features4tree,1))
            return self.conv_layers

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)

def flatten_data(x): return x.view(x.shape[0], -1)
def square_data(x): return x.view(-1,1,28,28)

##! in the future compose feature that changes the data according to the nn input
##! that feature will include:

# def flatten_feature_shape(feature_shape):
#     m = 1
#     for i in feature_shape:
#         m *= i
#     return m.item()