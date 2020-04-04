import torch
import torch.nn as nn
from tree_conf import Tree

class Forest(nn.Module):
    def __init__(self, conf, data):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.conf = conf

        for _ in range(self.conf.n_trees):
            tree = Tree(conf, data)
            self.trees.append(tree)

    def forward(self, x, save_flag = False):
        predictions = []
        cache = []
        for tree in self.trees: 
                mu = tree(x)

                predictions.append(p.unsqueeze(2))
        prediction = torch.cat(predictions, dim=2)
        prediction = torch.sum(prediction, dim=2)/self.conf.n_trees
        return prediction