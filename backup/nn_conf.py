#this is where the nn is defined
import torch.nn as nn
import torch

class Learner():
    def __init__(self, model, opt, loss_func, data):
        self.model,self.opt,self.loss_func,self.data = model,opt,loss_func,data
        
def flatten_feature_shape(feature_shape):
    m = 1
    for i in feature_shape:
        m *= i
    return m.item()

def get_model(conf,feature_shape,n_classes):
    nh = 100 #hidden layer size
    m = flatten_feature_shape(feature_shape)

    model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,n_classes))
    return model, torch.optim.SGD(model.parameters(), lr=conf.learning_rate)