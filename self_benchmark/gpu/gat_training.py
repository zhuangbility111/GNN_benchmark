import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

import time

dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid("../../data/Cora", dataset, transform=T.NormalizeFeatures())
data = dataset[0]
torch.cuda.set_device(5)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)

def train(data, model, optimizer):
    total_forward_dur = 0
    total_backward_dur = 0
    total_update_weight_dur = 0
    model.train()
    for epoch in range(200):
        print('=' * 100)
        print('epoch:', epoch)
        optimizer.zero_grad()
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        out = model(data.x, data.edge_index)
        torch.cuda.synchronize()
        backward_start = time.perf_counter()
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.cuda.synchronize()
        update_weight_start = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        update_weight_end = time.perf_counter()
        total_forward_dur += backward_start - forward_start
        total_backward_dur += update_weight_start - backward_start
        total_update_weight_dur += update_weight_end - update_weight_start
    return total_forward_dur, total_backward_dur, total_update_weight_dur


@torch.no_grad()
def test(data):
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
        accs.append(acc)
    return accs


def run_model_with_profiler(data, model, optimizer):
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False, with_stack=False) as prof:
        train(data, model, optimizer)
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    torch.save(model.state_dict(), 'GATNet.pt')


def run_model_without_profiler(data, model, optimizer):
    total_forward_dur, total_backward_dur, total_update_weight_dur = train(data, model, optimizer)
    torch.save(model.state_dict(), 'GATNet.pt')
    print("forward_time(ms), {}".format(total_forward_dur * 1000))
    print("backward_time(ms), {}".format(total_backward_dur * 1000))
    print("update_weight_time(ms), {}".format(total_update_weight_dur * 1000))

    # train_acc, val_acc, test_acc = test(data)
    # print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
    #     f'Test: {test_acc:.4f}')
# print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total"))

start = time.perf_counter()
device = torch.device('cuda')

load_start = time.perf_counter()
model = Net(dataset.num_features, dataset.num_classes).to(device)
data = data.to(device)
# torch.cuda.synchronize()
total_load_data_and_model_dur = time.perf_counter() - load_start

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# run_model_with_profiler(data, model, optimizer)
run_model_without_profiler(data, model, optimizer)
torch.cuda.synchronize()
end = time.perf_counter()
print("load_data_and_model_time(ms), {}".format(total_load_data_and_model_dur * 1000))
print("total_training_time(ms), {}".format((end - start) * 1000))
