#This the main script file it is intended to be minimal and to run all other parts of the code

##!fix the class cropper
##!fix the data flattening 
##!replace cross entropy with NLL later on

#in code from "visualizing NDF":
##! check out save flag
##! check out logger
##! check out cache
##! check out getting parametrs form command line
##! perhaps randomize target batches

import torch

#load configuration
from params import parameters
conf = parameters()

#load dataset
import dataset_loader

# c = class_counter(conf,datasets=None, n_classes=2)
n,m,c = dataset_loader.data_shape(conf, datasets=None, n_classes=2)
data = dataset_loader.DataBunch(*dataset_loader.get_dls(conf,n_classes = c),c,features4tree=10)

#initialize network
import model_conf

loss_func = torch.nn.CrossEntropyLoss() ##!replace with NLL later on
learner = model_conf.Learner(*model_conf.get_model(conf,data), loss_func, data)

#train
from trainer import fit
fit(conf,learner)
# fit(conf, model, loss_func, opt, train_dl=data_loaders['train'], valid_dl=data_loaders['val'])

#plot
print("hi!")

# if __name__ == '__main__':
#     main()

