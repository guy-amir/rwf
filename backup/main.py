#This the main script file it is intended to be minimal and to run all other parts of the code
import torch
##!fix the class cropper
##!fix the data flattening 
##!replace cross entropy with NLL later on

#load configuration
from params import parameters
conf = parameters()

#load dataset
import dataset_loader

# c = class_counter(conf,datasets=None, n_classes=2)
n,m,c = dataset_loader.data_shape(conf, datasets=None, n_classes=2)
data_loaders = dataset_loader.get_dls(conf,n_classes = c)


#initialize network
import model_conf

loss_func = torch.nn.CrossEntropyLoss() ##!replace with NLL later on
# model,optimizer = get_model(conf,m,c)
learner = model_conf.Learner(*model_conf.get_model(conf,m,c), loss_func, data_loaders)

#train
from trainer import fit
fit(conf,learner)

#plot
print("hi!")