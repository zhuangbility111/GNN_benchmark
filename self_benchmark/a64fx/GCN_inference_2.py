import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

import time

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]
data.adj_t = gcn_norm(data.adj_t)  # Pre-process GCN normalization.


class Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super().__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, dataset.num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5,
            shared_weights=True, dropout=0.6).to(device)
data = data.to(device)
model.load_state_dict(torch.load('GCNNet_v1.pt'))
model.eval()

@torch.no_grad()
def test():
    model.eval()
    pred, accs = model(data.x, data.adj_t).argmax(dim=-1), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

# Warm-up
for _ in range(5):
    start = time.time()
    outputs = model(data.x, data.adj_t).argmax(dim=-1)
    end = time.time()
    print('Time:{}ms'.format((end-start)*1000))

with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=False, profile_memory=False) as prof:
    for _ in range(100):
       outputs = model(data.x, data.adj_t).argmax(dim=-1)
print(prof.key_averages().table(sort_by="self_cpu_time_total"))

# ------------------------------------
# import torchvision.models as models
# from torch.profiler import profile, record_function, ProfilerActivity

# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#     with record_function("model_inference"):
#         for _ in range(100):
#             outputs = model(data.x, data.adj_t).argmax(dim=-1)

# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))