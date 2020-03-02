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
        self.nn_model = self.get_cnn_model(data)
        self.forest = Forest(conf, data)
        self.target_batches = []
        self.data = data
        self.target_indicator = nn.Parameter(torch.eye(data.c))
        

    def forward(self,x):
        nn_output_data = self.nn_model(x)
        predictions = self.forest(nn_output_data)

        return predictions

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
            self.conv_layers.add_module('bn1', nn.BatchNorm2d(32))
            self.conv_layers.add_module('relu1', nn.ReLU())
            self.conv_layers.add_module('pool1', nn.MaxPool2d(kernel_size=2))
            #self.add_module('drop1', nn.Dropout(dropout_rate))
            self.conv_layers.add_module('conv2', nn.Conv2d(32, 64, kernel_size=3, padding=1))
            self.conv_layers.add_module('bn2', nn.BatchNorm2d(64))
            self.conv_layers.add_module('relu2', nn.ReLU())
            self.conv_layers.add_module('pool2', nn.MaxPool2d(kernel_size=2))
            #self.add_module('drop2', nn.Dropout(dropout_rate))
            self.conv_layers.add_module('conv3', nn.Conv2d(64, 128, kernel_size=3, padding=1))
            self.conv_layers.add_module('bn3', nn.BatchNorm2d(128))
            self.conv_layers.add_module('relu3', nn.ReLU())
            self.conv_layers.add_module('pool3', nn.MaxPool2d(kernel_size=2))
            self.conv_layers.add_module('flatten', Lambda(flatten_data))
            self.conv_layers.add_module('linear', nn.Linear(1152,data.features4tree))
            return self.conv_layers
    # return nn.Sequential(
    #     Lambda(square_data),
    #     nn.Conv2d( 1, 8, 5, padding=2,stride=2), nn.ReLU(), #14
    #     nn.Conv2d( 8,16, 3, padding=1,stride=2), nn.ReLU(), # 7
    #     nn.Conv2d(16,32, 3, padding=1,stride=2), nn.ReLU(), # 4
    #     nn.Conv2d(32,32, 3, padding=1,stride=2), nn.ReLU(), # 2
    #     nn.AdaptiveAvgPool2d(1),
    #     Lambda(flatten_data),
    #     nn.Linear(32,data.features4tree)
    # )


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)

def flatten_data(x):      return x.view(x.shape[0], -1)
def square_data(x): return x.view(-1,1,28,28)

##! in the future compose feature that changes the data according to the nn input
##! that feature will include:

# def flatten_feature_shape(feature_shape):
#     m = 1
#     for i in feature_shape:
#         m *= i
#     return m.item()

#GG old get_model
# def get_model(conf,data):
    
#     nh = 100 #hidden layer size
#     m = flatten_feature_shape(feature_shape)
#     # model_a = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,m))
#     model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,n_classes))
#     # model = nn.Sequential(nn_model(m,n_classes),forest(...))
    
#     optimizer = torch.optim.SGD(model.parameters(), lr=conf.learning_rate)

#     return model, optimizer