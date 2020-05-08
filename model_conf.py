import torch
import torch.nn as nn
import torch.nn.functional as F

class Learner():

    # def weight_init(self,m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.xavier_uniform_(m.weight.data)
    #         m.bias.data.zero_()

    def __init__(self, model, opt, loss_func, data):
        self.model,self.opt,self.loss_func,self.data = model,opt,loss_func,data
        # self.model.apply(self.weight_init)

def get_model(conf,data):
    
    model = Forest(conf,data)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate) # , weight_decay=conf.weight_decay)
    loss_func =  torch.nn.MSELoss()

    return model, optimizer, loss_func

class Forest(nn.Module):
    def __init__(self, conf, data):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.conf = conf
        self.n_features = data.features4tree

        #The neural network that feeds into the trees:
        self.prenet = nn.Sequential(nn.Linear(self.n_features, 16), nn.LeakyReLU(),nn.BatchNorm1d(num_features=16), nn.Linear(16, self.n_features), nn.LeakyReLU(),nn.BatchNorm1d(num_features=self.n_features))

        
        for _ in range(self.conf.n_trees):
            tree = Tree(conf, data)
            self.trees.append(tree)

    def forward(self, xb,yb=None,layer=None):
        # if layer==None:

        self.predictions = []
        if self.training:
            self.y_hat_avg= []
        self.mu = []

        if (self.conf.use_prenet == True):
            xb = self.prenet(xb)

        if (self.conf.use_tree == False):
            return xb

        for tree in self.trees: 
            
            #construct routing probability tree:
            mu = tree(xb)

            #find the nodes that are leaves:
            mu_midpoint = int(mu.size(1)/2)
            mu_leaves = mu[:,mu_midpoint:]

            #create a normalizing factor for leaves:
            N = mu.sum(0)

            if self.training:
                self.y_hat = yb @ mu/N
                y_hat_leaves = self.y_hat[mu_midpoint:]
                self.y_hat_avg.append(self.y_hat.unsqueeze(1))
            else:
                y_hat_leaves = self.y_hat_avg[mu_midpoint:]
            pred = mu_leaves @ y_hat_leaves

            self.predictions.append(pred.unsqueeze(1))
            self.mu.append(mu.unsqueeze(1))
            
        self.mu = torch.cat(self.mu, dim=1)
        self.mu = torch.sum(self.mu, dim=1)/self.conf.n_trees

        if self.training:
            self.y_hat_avg = torch.cat(self.y_hat_avg, dim=1)
            self.y_hat_avg = torch.sum(self.y_hat_avg, dim=1)/self.conf.n_trees

        self.prediction = torch.cat(self.predictions, dim=1)
        self.prediction = torch.sum(self.prediction, dim=1)/self.conf.n_trees
        return self.prediction

class Tree(nn.Module):
    def __init__(self, conf, data):
        super(Tree, self).__init__()
        self.depth = conf.tree_depth
        self.n_leaf = 2 ** conf.tree_depth
        self.n_nodes = self.n_leaf#-1
        self.n_features = data.features4tree
        self.mu_cache = []
        self.conf = conf

        ##! attend to number of features!
        #prenet shouldn't be in tree
        
        # self.fc1 = nn.ModuleList([nn.Linear(self.n_features, self.n_features).float() for i in range(self.n_nodes)])
        # self.bn1 = nn.ModuleList([nn.BatchNorm1d(num_features=self.n_features).float() for i in range(self.n_nodes)])
        self.fc = nn.ModuleList([nn.Linear(self.n_features, self.n_features).float() for i in range(self.n_nodes)])
        self.decision = torch.nn.Sigmoid()


    def forward(self, x, save_flag = False):
        self.d = []

        for node in self.fc:
            n = node(x)
            self.d.append(self.decision(n))
        self.d = torch.stack(self.d).permute(1,0,2) #[batch_size,n_leafs,1]
        # self.d=torch.unsqueeze(self.d,dim=2)# ->[batch_size,n_leaf,1]
            #GG^ x[batch size, feature_length] mm with feature_mask[feature_length,n_leaf]
            #GG^ what this does is extract only the relevant features chosen from the feature mask
            #  out of all the features in x
         # passed sigmoid->[batch_size,n_leaf]
        decision = torch.cat((self.d,1-self.d),dim=2) # -> [batch_size,n_leaf,2]

        # with torch.no_grad():
        batch_size = x.size()[0]
        
        mu = torch.ones(batch_size,2,1).cuda()
        ##! may need to update CUDA
        
        for tree_level in range(0, self.depth):
            [begin_idx, end_idx] = level2node_delta(tree_level)
            # mu stores the probability a sample is routed at certain node
            # repeat it to be multiplied for left and right routing
            nu = mu[:, begin_idx:, :].repeat(1, 1, 2)
            # the routing probability at tree_level
            _decision = decision[:, begin_idx:end_idx, :] # -> [batch_size,2**tree_level,2]
            #GG^ original decision tensor is [batch size, leaf_number,decision&compliment]
            nu = nu*_decision # -> [batch_size,2**tree_level,2]

            # merge left and right nodes to the same layer
            nu = nu.view(batch_size, -1, 1)
            #GG print(f'begin_idx: {begin_idx}, end_idx {end_idx}, delta {-begin_idx+end_idx}')
            mu = torch.cat((mu,nu),1)

        mu = mu.squeeze(-1)
        mu[:,0] = 1 #define the first value in mu (the non-existing zeroth tree) as 1, don't worry, we don't use it anyway.
        
        return mu


def level2nodes(tree_level):
    return 2**(tree_level+1)

def level2node_delta(tree_level):
    start = level2nodes(tree_level-1)
    end = level2nodes(tree_level)
    return [start,end]
