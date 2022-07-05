import os.path as osp
import argparse

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor

import time

parser = argparse.ArgumentParser()
parser.add_argument('--tensor_type', type=str, default='tensor', help='The type of Tensor: \'tensor\' or \'sparse_tensor\'')
parser.add_argument('--use_profiler', type=str, default='false', help='Use profiler: \'false\' or \'true\'')
args = parser.parse_args()
tensor_type = args.tensor_type
use_profiler = args.use_profiler

if torch.cuda.is_available():
    print("GPU version")
    print("CUDA version:" + torch.version.cuda)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
else:
    print("CPU version")

if tensor_type == 'tensor':
    print('tensor_type: Tensor')
    dataset = Planetoid(root='../../data/Cora', name='Cora', transform=T.NormalizeFeatures())
elif tensor_type == 'sparse_tensor':
    print('tensor_type: SparseTensor')
    dataset = Planetoid(root='../../data/Cora', name='Cora', transform=T.ToSparseTensor())

# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
# dataset = Planetoid("../../data/Cora", dataset, transform=T.NormalizeFeatures())
# data = dataset[0]
# torch.cuda.set_device(5)
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

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

def train_with_instrumentation(data, model, optimizer):
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
        # out = model(data.x, data.edge_index)
        if tensor_type == 'tensor':
            out = model(data.x, data.edge_index)
        elif tensor_type == 'sparse_tensor':
            out = model(data.x, data.adj_t)
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

def train(data, model, optimizer):
    model.train()
    for epoch in range(200):
        print('=' * 100)
        print('epoch:', epoch)
        optimizer.zero_grad()
        # out = model(data.x, data.edge_index)
        if tensor_type == 'tensor':
            out = model(data.x, data.edge_index)
        elif tensor_type == 'sparse_tensor':
            out = model(data.x, data.adj_t)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

@torch.no_grad()
def test(model, data):
    model.eval()
    # out, accs = model(data.x, data.edge_index), []
    if tensor_type == 'tensor':
        out, accs = model(data.x, data.edge_index), []
    elif tensor_type == 'sparse_tensor':
        out, accs = model(data.x, data.adj_t), []

    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
        accs.append(acc)
    return accs


def run_model_with_profiler():
    # device = torch.device('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load_start = time.perf_counter()
    model = Net(dataset.num_features, dataset.num_classes).to(device)
    data = dataset[0].to(device)
    # total_load_data_and_model_dur += time.perf_counter() - load_start
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False, with_stack=False) as prof:
        train(data, model, optimizer)
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))

    train_acc, val_acc, test_acc = test(model, data)
    print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    torch.save(model.state_dict(), 'GATNet.pt')


def run_model_without_profiler():
    total_forward_dur = 0
    total_backward_dur = 0
    total_update_weight_dur = 0
    total_load_data_and_model_dur = 0
    dur = 0

    # warmup
    # device = torch.device('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset.num_features, dataset.num_classes).to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    train_with_instrumentation(data, model, optimizer)

    repeat_round = 10
    for i in range(repeat_round):
        start = time.perf_counter()
        # device = torch.device('cuda')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        load_start = time.perf_counter()
        model = Net(dataset.num_features, dataset.num_classes).to(device)
        data = dataset[0].to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_data_dur = time.perf_counter() - load_start
        total_load_data_and_model_dur += load_data_dur
        # print("=======repeat round " + str(i) + " ========")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        forward_dur, backward_dur, update_weight_dur = train_with_instrumentation(data, model, optimizer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        total_forward_dur += forward_dur
        total_backward_dur += backward_dur
        total_update_weight_dur += update_weight_dur
        dur += (end - start)

    print("load_data_and_model_time(ms): {}".format(total_load_data_and_model_dur * 1000.0 / repeat_round))
    print("forward_time(ms): {}".format(total_forward_dur * 1000.0 / repeat_round))
    print("backward_time(ms): {}".format(total_backward_dur * 1000.0 / repeat_round))
    print("update_weight_time(ms): {}".format(total_update_weight_dur * 1000.0 / repeat_round))
    print("total_training_time(ms): {}".format(dur * 1000.0 / repeat_round))

    train_acc, val_acc, test_acc = test(model, data)
    print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    torch.save(model.state_dict(), 'GATNet.pt')

    # train_acc, val_acc, test_acc = test(data)
    # print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
    #     f'Test: {test_acc:.4f}')
# print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total"))

if use_profiler == 'true':
    run_model_with_profiler()
else:
    run_model_without_profiler()
