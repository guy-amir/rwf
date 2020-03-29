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
    
    model = Tree(conf,data)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.learning_rate) # , weight_decay=conf.weight_decay)

    return model, optimizer

# class tree_layers(nn.Module):
#     def __init__(self,input_dim):
#         super(tree_layers, self).__init__()
#         self.n_nodes = 15
#         self.linears = nn.ModuleList([nn.Linear(input_dim, input_dim) for i in range(self.n_nodes)])
#         print(self.linears)

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
        self.prenet = nn.Sequential(nn.Linear(self.n_features, 16), nn.LeakyReLU(),nn.BatchNorm1d(num_features=16), nn.Linear(16, self.n_features), nn.LeakyReLU(),nn.BatchNorm1d(num_features=self.n_features))
        # self.fc1 = nn.ModuleList([nn.Linear(self.n_features, self.n_features).float() for i in range(self.n_nodes)])
        # self.bn1 = nn.ModuleList([nn.BatchNorm1d(num_features=self.n_features).float() for i in range(self.n_nodes)])
        self.fc = nn.ModuleList([nn.Linear(self.n_features, self.n_features).float() for i in range(self.n_nodes)])
        self.decision = torch.nn.Sigmoid()


    def forward(self, x, save_flag = False):
        self.d = []
        if (self.conf.use_prenet == True):
            x = self.prenet(x)

        if (self.conf.use_tree == False):
            return x

        for layer in self.fc:
            l = layer(x)
            # b = bn1(l)
            # r = torch.sigmoid(x)
            self.d.append(self.decision(l))
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
        
        begin_idx = 1
        end_idx = 2
        for n_layer in range(0, self.depth):
            # mu stores the probability a sample is routed at certain node
            # repeat it to be multiplied for left and right routing
            nu = mu[:, begin_idx:, :].repeat(1, 1, 2)
            # the routing probability at n_layer
            _decision = decision[:, begin_idx:end_idx, :] # -> [batch_size,2**n_layer,2]
            #GG^ original decision tensor is [batch size, leaf_number,decision&compliment]
            nu = nu*_decision # -> [batch_size,2**n_layer,2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer+1)
            # merge left and right nodes to the same layer
            nu = nu.view(batch_size, -1, 1)
            #GG print(f'begin_idx: {begin_idx}, end_idx {end_idx}, delta {-begin_idx+end_idx}')
            mu = torch.cat((mu,nu),1)

        mu = mu.squeeze(-1)
        mu[:,0] = 0
        
        return mu

