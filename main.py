#This the main script file it is intended to be minimal and to run all other parts of the code

##! UPLOAD EVERTHING TO CUDA!!!
##!fix the class cropper
##!fix the data flattening 
##!replace cross entropy with NLL later on

#in code from "visualizing NDF":
##! check out save flag
##! check out logger
##! check out cache
##! check out getting parametrs form command line
##! perhaps randomize target batches
##! replace all .cuda() with torch.new
##! add grad=False to all not learn

import torch
from functools import partial

device = torch.device('cuda',0)
torch.cuda.set_device(device)

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

loss_func = torch.nn.functional.nll_loss ##!replace with NLL later on
learn = model_conf.Learner(*model_conf.get_model(conf,data), loss_func, data)

#train
import trainer
from callbacks import *

# from trainer import fit
# fit(conf,learner)

#add callbacks functionallity:
# cbfs = [Recorder, partial(AvgStatsCallback,accuracy),partial(CudaCallback,device)]
cbfs = [partial(AvgStatsCallback,accuracy),partial(CudaCallback,device)]
run = trainer.Runner(cb_funcs=cbfs)
run.fit(30, learn)

#plot
print("hi!")

# if __name__ == '__main__':
#     main()

