#this is where the nn is defined
import torch.nn as nn
import torch
from forest_conf import Forest

class Learner():
    def __init__(self, model, opt, loss_func, data):
        self.model,self.opt,self.loss_func,self.data = model,opt,loss_func,data

def get_model(conf,data):
    
    model = entire_network(conf,data)
    optimizer = torch.optim.SGD(model.parameters(), lr=conf.learning_rate)

    return model, optimizer

class entire_network(nn.Module):
    
    def __init__(self,conf,data):
        super(entire_network, self).__init__()
        self.nn_model = get_cnn_model(data)
        self.forest = Forest(conf, data)
        self.target_batches = []
        self.target_indicator = torch.eye(data.c)

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

        


def get_cnn_model(data):
    return nn.Sequential(
        Lambda(square_data),
        nn.Conv2d( 1, 8, 5, padding=2,stride=2), nn.ReLU(), #14
        nn.Conv2d( 8,16, 3, padding=1,stride=2), nn.ReLU(), # 7
        nn.Conv2d(16,32, 3, padding=1,stride=2), nn.ReLU(), # 4
        nn.Conv2d(32,32, 3, padding=1,stride=2), nn.ReLU(), # 2
        nn.AdaptiveAvgPool2d(1),
        Lambda(flatten_data),
        nn.Linear(32,data.features4tree)
    )

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