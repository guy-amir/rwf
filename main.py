from func_gen import *
import sklearn
import torch
from params import parameters
import model_conf
import dataset_loader
import trainer

conf = parameters()

x,y = step_gen(range = (0,1000),step=0.1)
tdl,vdl = dl_maker(x,y,conf.batch_size)
data = dataset_loader.DataBunch(tdl,vdl,c=1,features4tree=1)

loss_func =  torch.nn.MSELoss()

learn = model_conf.Learner(*model_conf.get_model(conf,data), loss_func, data)

device = torch.device('cuda',0)
torch.cuda.set_device(device)

run = trainer.Runner(cb_funcs=conf.cbfs)
run.fit(conf.epochs, learn)

run.recorder.plot_lr()

# from training import trainer
# train = trainer(xt,yt)

