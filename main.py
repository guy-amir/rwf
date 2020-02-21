#This the main script file it is intended to be minimal and to run all other parts of the code

#load configuration
from params import opts
conf = opts()

#load dataset
from dataset_loader import get_dls
data_loaders = get_dls(conf)

print(data_loaders)

#initialize network

#train

#plot

print("hi! :)")