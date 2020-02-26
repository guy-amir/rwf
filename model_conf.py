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
    # model_a = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,m))
    model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,n_classes))
    # model = nn.Sequential(model_a,model_b)
    return model, torch.optim.SGD(model.parameters(), lr=conf.learning_rate)

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)

def flatten(x):      return x.view(x.shape[0], -1)
def anti_flatten(x): return x.view(-1,1,28,28)

def get_cnn_model(data):
    return nn.Sequential(
        Lambda(mnist_resize),
        nn.Conv2d( 1, 8, 5, padding=2,stride=2), nn.ReLU(), #14
        nn.Conv2d( 8,16, 3, padding=1,stride=2), nn.ReLU(), # 7
        nn.Conv2d(16,32, 3, padding=1,stride=2), nn.ReLU(), # 4
        nn.Conv2d(32,32, 3, padding=1,stride=2), nn.ReLU(), # 2
        nn.AdaptiveAvgPool2d(1),
        Lambda(flatten),
        nn.Linear(32,data.c)
    )