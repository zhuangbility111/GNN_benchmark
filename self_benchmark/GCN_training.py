from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# dataset = Planetoid(root='/tmp/Cora', name='Cora')
dataset = Planetoid(root='~/data/PubMed', name='PubMed')
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
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
train_loss_all = []
val_loss_all = []

with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=False, profile_memory=False, with_stack=True) as prof:
    model.train()
    for epoch in range(200):
        print('=' * 100)
        print('epoch:', epoch)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_loss_all.append(loss.data.numpy())

        loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
        val_loss_all.append(loss.data.numpy())
        print('Epoch:{},train_loss:{},val_loss:{}'.format(epoch, train_loss_all[-1], val_loss_all[-1]))
print(prof.key_averages(group_by_stack_n=2).table(sort_by="self_cpu_time_total"))
# print(prof.key_averages().table(sort_by="self_cpu_time_total"))

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')

torch.save(model.state_dict(), 'GCNNet_v0.pt')
