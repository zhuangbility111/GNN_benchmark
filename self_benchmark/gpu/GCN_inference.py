from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import time
import pdb


dataset = Planetoid(root='../data/Cora', name='Cora')
print(dataset.data)
print('Number of classes:', dataset.num_classes)
print('Number of edges features:', dataset.num_edge_features)
print('Number of edges index:', dataset.data.edge_index.shape[1] / 2)
print('Number of nodes features：', dataset.num_node_features)
print('Number of nodes:', dataset.data.x.shape[0])
# print(':', dataset.data.x)

print(dataset.data.edge_index)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
model.load_state_dict(torch.load('GCNNet_v0.pt'))
data = dataset[0].to(device)
model.eval()

# Warm-up
for _ in range(5):
    start = time.time()
    pdb.set_trace()
    outputs = model(data)
    end = time.time()
    print('Time:{}ms'.format((end-start)*1000))

with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=False, profile_memory=False, with_stack=True) as prof:
    for _ in range(100):
        outputs = model(data)
print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total"))

# ------------------------------------
# import torchvision.models as models
# from torch.profiler import profile, record_function, ProfilerActivity

# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#     with record_function("model_inference"):
#         for _ in range(100):
#             outputs = model(data)

# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
