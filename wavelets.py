import torch

class psi():
    def __init__(self,runner):
        model = runner.model
        self.Vectors = model.mu*(model.y_hat-self.find_parent(model.y_hat))
        self.Norm = self.Vectors.norm(p=2,dim=0)
        self.Norm_value,self.Norm_order = self.Norm.sort(descending=True)
        self.intervals = model.conf.intervals

    def find_parent(self,y_hat):
        y_parent = y_hat.new(y_hat.size())
        y_parent[0] = torch.tensor(0)
        n_parent_nodes = int(y_hat.size(0)/2)
        for ii in range(0,n_parent_nodes):
            y_parent[2*ii+1:2*ii+3] = y_hat[ii]

        return y_parent

    
    def cutoff(self,N):

        #cutoff after N+1 (or N) values: 
        node_list = self.Norm_order[:N]
        #find the nodes that are leaves:
        leaf_list = self.find_leaves(node_list)
        leaf_list = [int(leaf.item()) for leaf in leaf_list]
        #add parents of all such nodes:
        # node_list = torch.unique(torch.cat([self.find_parents(N),node_list],0)) ##?
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