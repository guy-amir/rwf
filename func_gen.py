import numpy as np
from sklearn.model_selection import train_test_split
import torch
from dataset_loader import Dataset, DataLoader

def step_gen(range = (0,10),step=0.1):
    midpoint = (range[1] - range[0])/2+range[0]
    x = np.arange(range[0],range[1], step)
    y = np.zeros_like(x)
    y[x>midpoint] = 2
    y = y-1+np.sin(0.1*x)
    x = x-np.mean(x)
    # x = x/np.max(x)
    return x,y

def get_fun(x):
    # y = np.zeros_like(x)
    # y[x>midpoint] = 2
    # y = y-1
    y = np.sin(0.1*x)
    return y

def split(x,y,percent=0.33):
    return train_test_split(x, y, test_size=percent, random_state=197)

# def create_dl(x,y,batch_size=64):
    
#     permutation = torch.randperm(x.size()[0])
#     X = []
#     Y = []

#     for i in range(0,x.size()[0], batch_size):
#         indices = permutation[i:i+batch_size]
#         X.append(x[indices])
#         Y.append(y[indices])
    
#     return zip(X,Y)

def dl_maker(x,y, conf):
    batch_size=conf.batch_size
    xt, xv, yt, yv = split(x,y)
    xt, xv, yt, yv =  map(torch.tensor, (xt, xv, yt, yv))

    train_ds,valid_ds = Dataset(xt.unsqueeze(1).float(), yt.float()),Dataset(xv.unsqueeze(1).float(), yv.float())

    if conf.one_batch:
        train_dl = DataLoader(train_ds, len(train_ds))
        valid_dl = DataLoader(valid_ds, len(valid_ds))

    else:
        train_dl = DataLoader(train_ds, batch_size)
        valid_dl = DataLoader(valid_ds, batch_size)

    return train_dl,valid_dl



    


