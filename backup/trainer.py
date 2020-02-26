#This is where the training loop takes place
import torch

def accuracy(out, yb): return (torch.argmax(out, dim=1)==yb).float().mean()

def fit(opts, learner): #model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(opts.epochs):

        learner.model.train()

        for xb,yb in learner.data['train']:
            xb = xb.view(xb.shape[0],28*28)
            loss = learner.loss_func(learner.model(xb), yb)
            loss.backward()
            learner.opt.step()
            learner.opt.zero_grad()

        learner.model.eval()

        with torch.no_grad():
            tot_loss,tot_acc = 0.,0.
            for xb,yb in learner.data['val']:
                xb = xb.view(xb.shape[0],28*28)
                pred = learner.model(xb)
                tot_loss += learner.loss_func(pred, yb)
                tot_acc  += accuracy(pred,yb)
        nv = len(learner.data['val'])
        print(epoch, tot_loss/nv, tot_acc/nv)
    return tot_loss/nv, tot_acc/nv