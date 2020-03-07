#This the main script file it is intended to be minimal and to run all other parts of the code

##! UPLOAD EVERTHING TO CUDA!!!
##!fix the class cropper
##!fix the data flattening 
##!replace cross entropy with NLL later on
##!having both wavelet and psi is redundant
##!add layer for every additional tree
##!average multiple tree data in wavelets like in regular training

#in code from "visualizing NDF":
##! check out save flag
##! check out logger
##! check out cache
##! check out getting parametrs form command line
##! perhaps ra5ndomize target batches
##! replace all .cuda() with torch.new
##! add grad=False to all not learn
##! add batchnorm
##! fix issue with softmax_initialization:
#   TypeError: cannot assign 'torch.cuda.FloatTensor' as parameter 'pi' (torch.nn.Parameter or None expected)


import torch


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

#add callbacks functionallity:

run = trainer.Runner(cb_funcs=conf.cbfs)
run.fit(conf.epochs, learn)

#plot
print("hi!")

# if __name__ == '__main__':
#     main()