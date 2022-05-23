import torch 
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree
x = torch.tensor([[2,1],[5,6],[3,7],[12,0]],dtype=torch.float)
y = torch.tensor([0,1,0,1],dtype=torch.long)
edge_index = torch.tensor([[0,1,1,2,2,3,1,3],[1,0,2,1,3,2,3,1]],dtype=torch.long)
data = Data(x=x,y=y,edge_index=edge_index)

adjacency_matrix, temp = add_self_loops(edge_index, num_nodes=x.size(0)
