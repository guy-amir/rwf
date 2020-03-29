import torch
import torch.nn as nn
from tree_conf import Tree

class Forest(nn.Module):
    def __init__(self, conf, data):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        # self.n_tree  = conf.n_trees #remove
        self.conf = conf

        for _ in range(self.conf.n_trees):
            tree = Tree(conf, data)
            self.trees.append(tree)

    def forward(self, x, save_flag = False):
        predictions = []
        cache = []
        for tree in self.trees: 
                mu = tree(x)
                # if self.training or self.conf.wavelets:
                #     tree.mu_cache.append(mu) #find a way to add a test/train switch for mu_cache
                # p = torch.mm(mu,tree.pi)

                #GG::
                # if wavelets:
                #     ww = w.wavelet(tree)
                #     leaf_list = ww.cutoff(50)
                #     nu  = w.mod_mu(mu,leaf_list)
                #     pu  = w.mod_pi(tree.pi,leaf_list)
                #     p = tree.cal_prob(nu, pu)

                predictions.append(p.unsqueeze(2))
        prediction = torch.cat(predictions, dim=2)
        prediction = torch.sum(prediction, dim=2)/self.conf.n_trees
        return prediction