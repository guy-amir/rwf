import torch

class psi():
    def __init__(self,wavelet):
        self.Vectors = wavelet.mu_nodes*(wavelet.pi_nodes-self.find_parent(wavelet.pi_nodes))
        self.Norm = self.Vectors.norm(p=2,dim=1)
        self.Norm_value,self.Norm_order = self.Norm.sort(descending=True)

    def find_parent(self,pi):
        parent_tree = pi.new(pi.size())
        parent_tree[0,:] = torch.tensor(float('nan'))
        n_parent_nodes = int(pi.size(0)/2)
        for ii in range(0,n_parent_nodes):
            parent_tree[2*ii+1:2*ii+3,:] = pi[ii,:]

        return parent_tree

    def find_child(self,pi):
        child_tree = pi.new(pi.size()[0],2)
        n_parent_nodes = int(pi.size(0)/2)
        for ii in range(pi.size()[0]//2):
            child_tree[ii,0]=2*ii+1
            child_tree[ii,1]=2*ii+2

        return child_tree

    
    def cutoff(self,N):

        #cutoff after N+1 (or N) values: 
        node_list = self.Norm_order[:N]
        #find the nodes that are leaves:
        leaf_list = self.find_leaves(node_list)
        #add parents of all such nodes:
        node_list = torch.unique(torch.cat([self.find_parents(N),node_list],0))
        # print(f"naive node no.: {N} \n effective nodes: {node_list}")
        # print(f"leaves: {leaf_list}")
        return leaf_list

    def find_parents(self,N):
        parent_list = []
        current_parent = (N-1)//2
        while(current_parent is not 0):
            parent_list.append(current_parent)
            current_parent = (current_parent-1)//2
        return torch.LongTensor(parent_list).cuda()



    def find_leaves(self,node_list):
        leaf_list = []
        for node in node_list:
            #check if no left-node children:
            if (not (node_list == 2*node+1).sum().item()) & (not (node_list == 2*node+2).sum().item()):
                if (node%2).item():
                    if (node_list == node+1).sum().item():
                        leaf_list.append(node)
                    else:
                        leaf_list.append((node-1)//2)
                else: 
                    if (node_list == node-1).sum().item():
                        leaf_list.append(node)
                    else:
                        leaf_list.append((node-1)//2)
        return torch.unique(torch.FloatTensor(leaf_list))

    def mod_mu(self,mu,leaf_list=None):
        nu = mu.clone()
        leaves = mu.size(1)
        while (leaves >= 1):
            leaves = int(leaves/2)
            mu_temp = mu.new(mu.size(0),leaves)
            for row in range(leaves):
                place = 2*row
                mu_temp[:,row] = nu[:,place:place+2].sum(1)
            nu = torch.cat((mu_temp,nu),1) 

        makeshift_nu = []
        for i in leaf_list:
            i=int(i.item())
            makeshift_nu.append(nu[:,i].unsqueeze(1))

        nu = torch.cat(makeshift_nu,1)
        return nu

    def mod_pi(self,pi,leaf_list=None):
        #there is a major problem with the probability summation heare
        pu = pi.clone()
        leaves = pi.size(0)
        while (leaves >= 1):
            leaves = int(leaves/2)
            pi_temp = pi.new(leaves,pi.size(1))
            for row in range(leaves):
                place = 2*row
                pi_temp[row,:] = pu[place:place+2,:].sum(0)/2 #we devide by 2 in order to normalize the probability
            pu = torch.cat((pi_temp,pu),0) 

        makeshift_pu = []
        for i in leaf_list:
            i=int(i.item())
            makeshift_pu.append(pu[i,:].unsqueeze(0))

        pu = torch.cat(makeshift_pu,0)

        return pu

class wavelet():
    def __init__(self,tree):
        self.tree = tree
        self.construct_wavelet()
   
    def construct_wavelet(self):
        
        #Basically, what goes down here is this:
        #1. We collect the mu_cache tree when it's done training and NOT! average it

        #2. mu_chace is averaged and a new enlarged data structure is created to contain proper number of nodes:
        self.construct_mu_nodes()

        # print('mu_nodes: ',self.mu_nodes.size())

        #3. a new pi clone is created, big enough to contain entire tree and leaf nodes are summed and averaged in order to fill it up.
        self.construct_pi_nodes()

        #4. wavelets are calculated based on new mu and pi
        self.tree.psi = self.get_psi()

        # self.find_child(self.pi_nodes)
        return 

    def get_psi(self):
        return psi(self)
        # self.psi = self.mu_nodes*(self.pi_nodes-self.find_parent(self.pi_nodes))
        # self.psi.Norm = self.psi.norm(p=2,dim=1)
        # self.psi.Norm_value,self.psi.Norm_order = self.psi.Norm.sort(descending=True)
        # print(f"self.psi: {self.psi}")
        
    # def cutoff(self, psi)
    # to be concluded

    # def find_parent(self,pi):
    #     parent_tree = pi.new(pi.size())
    #     parent_tree[0,:] = torch.tensor(float('nan'))
    #     n_parent_nodes = int(pi.size(0)/2)
    #     for ii in range(0,n_parent_nodes):
    #         parent_tree[2*ii+1:2*ii+3,:] = pi[ii,:]

    #     return parent_tree

    # def find_child(self,pi):
    #     child_tree = pi.new(pi.size()[0],2)
    #     n_parent_nodes = int(pi.size(0)/2)
    #     for ii in range(pi.size()[0]//2):
    #         child_tree[ii,0]=2*ii+1
    #         child_tree[ii,1]=2*ii+2
    #         # print(f'ii={ii}')
    #         # print(f'child_tree[ii,0]={child_tree[ii,0]}')
    #         # print(f'child_tree[ii,1]={child_tree[ii,1]}')

        # return child_tree

    # def mu_tree(self,mu)

    #     return mu_tree

    # def pi_tree(self,mu)

    #     return pi_tree

    def construct_mu_nodes(self):        
        mean_mu = torch.mean(torch.stack(self.tree.mu_cache[:-1]),0) ##! add last tensor in mu_cache list later
        mean_mu = mean_mu.mean(0).unsqueeze(1)
        mu_level = mean_mu
        self.mu_nodes = mean_mu
        place_holder = 0
        nodes_per_level = 0
        for level in range(0,self.tree.depth):
            mu_temp = mean_mu.new(int(mu_level.size(0)/2),1)
            for row in range(mu_temp.size(0)):
                place = 2*row
                mu_temp[row] = mu_level[place:place+2].sum()
            mu_level = mu_temp
            self.mu_nodes = torch.cat((mu_level,self.mu_nodes),0)

    def construct_pi_nodes(self):
        self.pi_nodes = torch.tensor(self.tree.pi)
        pi_level = self.pi_nodes
        for level in range(0,self.tree.depth):
            pi_temp = pi_level.new(int(pi_level.size(0)/2),pi_level.size(1))
            for row in range(pi_temp.size(0)):
                place = 2*row
                pi_temp[row] = pi_level[place:place+2,:].sum(0)/2
            pi_level = pi_temp
            self.pi_nodes = torch.cat((pi_level,self.pi_nodes),0)
            pass

    