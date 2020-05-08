##! work about multiple trees

from func_gen import *
import sklearn
import torch
from params import parameters
import model_conf
import dataset_loader
import trainer
import pandas as pd

conf = parameters()

x,y = step_gen(range = (0,60),step=0.1)
tdl,vdl = dl_maker(x,y,conf)
data = dataset_loader.DataBunch(tdl,vdl,c=1,features4tree=1)

learn = model_conf.Learner(*model_conf.get_model(conf,data), data)

device = torch.device('cuda',0)
torch.cuda.set_device(device)

run = trainer.Runner(cb_funcs=conf.cbfs,conf=conf)
run.fit(conf.epochs, learn)

# run.wavelets.prune(even_cutoff=False)

# xx = learn.data.valid_dl.ds.x.numpy()
# yy = learn.data.valid_dl.ds.y.numpy()
# zz = run.recorder.tot_pred.cpu().numpy()
# real_pred = pd.DataFrame([xx,yy,zz]) 

# pred_list = [i.tolist() for i in run.wavelets.wavelet_pred_list]
# pred_df = pd.DataFrame(pred_list)
# loss_df = pd.DataFrame(run.wavelets.wavelet_loss)
# cutoff_df = pd.DataFrame(run.wavelets.psi.cutoff_value_list)

# pred_df.to_csv('wav_pred.csv',index=False)
# loss_df.to_csv('wav_loss.csv',index=False)
# cutoff_df.to_csv('wav_cutoff.csv',index=False)