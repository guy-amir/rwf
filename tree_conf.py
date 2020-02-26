import numpy as np
import torch
import torch.nn as nn
class Tree(nn.Module):
    def __init__(self, conf, data):
        super(Tree, self).__init__()
        self.depth = conf.tree_depth
        self.n_leaf = 2 ** conf.tree_depth
        self.n_features = data.features4tree
        self.n_classes = data.c

        self.feature_idx = np.random.choice(self.n_features, self.n_leaf, replace=False)
        #GG^ create a random vector of length n_leaf composed of numbers out of feature_length
        #GG actually choosing the features for each leaf 
        self.feature_mask = torch.eye(self.n_features)[self.feature_idx].t()
        #GG^ feature mask is a tensor of size (n_features X feature_idx) with 1 at the location of each of the noted random variables

        self.pi = torch.ones((self.n_leaf, self.n_classes))/self.n_classes
        self.mu_cache = []
        self.decision = torch.nn.Sigmoid()


    def forward(self, x, save_flag = False):

        feats = torch.mm(x, self.feature_mask) # ->[batch_size,n_leaf]
        #GG^ x[batch size, feature_length] mm with feature_mask[feature_length,n_leaf]
        #GG^ what this does is extract only the relevant features chosen from the feature mask
        #  out of all the features in x
        self.d = self.decision(feats)
         # passed sigmoid->[batch_size,n_leaf]
        self.d = torch.unsqueeze(self.d,dim=2) # ->[batch_size,n_leaf,1]
        decision = torch.cat((self.d,1-self.d),dim=2) # -> [batch_size,n_leaf,2]

        # for debug
        #decision.register_hook(debug_hook)

        # compute route probability
        # note: we do not use decision[:,0]
        # save some intermediate results for analysis
        ##! add save flag
     
        batch_size = x.size()[0]
        
        mu = torch.ones(batch_size,1,1)
        ##! may need to update CUDA
        
        begin_idx = 1
        end_idx = 2
        for n_layer in range(0, self.depth):
            # mu stores the probability a sample is routed at certain node
            # repeat it to be multiplied for left and right routing
            mu = mu.repeat(1, 1, 2)
            # the routing probability at n_layer
            _decision = decision[:, begin_idx:end_idx, :] # -> [batch_size,2**n_layer,2]
            #GG^ original decision tensor is [feature length, leaf_number,decision&compliment]
            mu = mu*_decision # -> [batch_size,2**n_layer,2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer+1)
            # merge left and right nodes to the same layer
            mu = mu.view(batch_size, -1, 1)
            #GG print(f'begin_idx: {begin_idx}, end_idx {end_idx}, delta {-begin_idx+end_idx}')
        mu = mu.view(batch_size, -1)
        
        return mu
    
    def update_pi(self,target_batches):
        """
        compute new mean vector based on a simple update rule inspired from traditional regression tree 
        Args:
            param feat_batch (Tensor): feature batch of size [batch_size, feature_length]
            param target_batch (Tensor): target batch of size [batch_size, vector_length]
        """  

        with torch.no_grad():
            new_pi = torch.zeros((self.n_leaf, self.n_classes)) ##GG 1/self.vector_length) # Tensor [n_leaf,n_class] 
                
            for mu, target in zip(self.mu_cache, target_batches):
                prob = torch.mm(mu, self.pi)  # [batch_size,n_class]
                _target = target.unsqueeze(1) # [batch_size,1,n_class]
                _pi = self.pi.unsqueeze(0) # [1,n_leaf,n_class]
                _mu = mu.unsqueeze(2) # [batch_size,n_leaf,1]
                _prob = torch.clamp(prob.unsqueeze(1),min=1e-6,max=1.) # [batch_size,1,n_class] #this is perhaps the normalization
                ##! check later if clamping is neccessary
                _new_pi = torch.mul(torch.mul(_target,_pi),_mu)/_prob # [batch_size,n_leaf,n_class]
                new_pi += torch.sum(_new_pi,dim=0)

        Z = new_pi.sum(1) # Z is a normalising factor ensuring the sum of probabilities from each leaf to each class is 1
        self.pi = (new_pi.t()/Z).t() ##! add a smarter way to broadcast
        self.mu_cache = []
        return