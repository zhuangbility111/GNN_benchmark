from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
import time
import argparse
# import pdb

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
    dataset = Planetoid(root='../../data/Cora', name='Cora')
elif tensor_type == 'sparse_tensor':
    print('tensor_type: SparseTensor')
    dataset = Planetoid(root='../../data/Cora', name='Cora', transform=T.ToSparseTensor())

# dataset = Planetoid(root='~/data/PubMed', name='PubMed')
print(dataset.data)
print('Number of classes:', dataset.num_classes)
print('Number of edges features:', dataset.num_edge_features)
print('Number of edges index:', dataset.data.edge_index.shape[1] / 2)
print('Number of nodes features：', dataset.num_node_features)
print('Number of nodes:', dataset.data.x.shape[0])
# print(':', dataset.data.x)
# print(dataset.data.edge_index)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = GCNConv(dataset.num_node_features, 16)
        # self.conv2 = GCNConv(16, dataset.num_classes)
        self.conv1 = GCNConv(dataset.num_node_features, dataset.num_node_features)
        self.conv2 = GCNConv(dataset.num_node_features, dataset.num_classes)

    # forward() for tensor
    def forward(self, x, edge):
        # x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge)
        return F.log_softmax(x, dim=1)

def train(data, model, optimizer):
    model.train()
    for epoch in range(200):
        print('=' * 100)
        print('epoch:', epoch)
        optimizer.zero_grad()
        if tensor_type == 'tensor':
            out = model(data.x, data.edge_index)
        elif tensor_type == 'sparse_tensor':
            out = model(data.x, data.adj_t)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

def train_with_instrumentation(data, model, optimizer):
    model.train()
    total_forward_dur = 0
    total_backward_dur = 0
    total_update_weight_dur = 0
    for epoch in range(200):
        print('=' * 100)
        print('epoch:', epoch)
        optimizer.zero_grad()
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        # out = model(data)
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
	

def run_model_without_profiler():
    total_forward_dur = 0
    total_backward_dur = 0
    total_update_weight_dur = 0
    total_load_data_and_model_dur = 0
    dur = 0
    
    # warm up
    # device = torch.device('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train_loss_all = []
    val_loss_all = []
    train_with_instrumentation(data, model, optimizer)

    print("warmup over.")
    
    repeat_round = 10
    for _ in range(repeat_round):
        start = time.perf_counter()
        # device = torch.device('cuda')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        load_start = time.perf_counter()
        model = GCN().to(device)
        data = dataset[0].to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_load_data_and_model_dur += time.perf_counter() - load_start
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        train_loss_all = []
        val_loss_all = []
        forward_dur, backward_dur, update_weight_dur = train_with_instrumentation(data, model, optimizer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        total_forward_dur += forward_dur
        total_backward_dur += backward_dur
        total_update_weight_dur += update_weight_dur
        dur += end - start
    print("load_data_and_model_time(ms): {}".format(total_load_data_and_model_dur * 1000 / repeat_round))
    print("forward_time(ms): {}".format(total_forward_dur * 1000 / repeat_round))
    print("backward_time(ms): {}".format(total_backward_dur * 1000 / repeat_round))
    print("update_weight_time(ms): {}".format(total_update_weight_dur * 1000 / repeat_round))
    print("total_training_time(ms): {}".format(dur * 1000 / repeat_round))

    model.eval()
    if tensor_type == 'tensor':
        pred = model(data.x, data.edge_index).argmax(dim=1)
    elif tensor_type == 'sparse_tensor':
        pred = model(data.x, data.adj_t).argmax(dim=1)
    # pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')
        
    torch.save(model.state_dict(), 'GCNNet_v0.pt')

def run_model_with_profiler():
    # warm up
    # device = torch.device('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train_loss_all = []
    val_loss_all = []
    train(data, model, optimizer)

    print("warmup over.")

    # device = torch.device('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train_loss_all = []
    val_loss_all = []
    
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False, with_stack=False) as prof:
        start = time.perf_counter()
        train(data, model, optimizer)
        torch.cuda.synchronize()
        end = time.perf_counter()
        dur = end - start
        print("training_time(s):{}".format(dur))
            # train_loss_all.append(loss.data.numpy())
        # print("training_time(s): {}, forward_time(s): {}, backward_time(s): {}, update_weight_time(s): {}".format((end - start), total_forward_dur, total_backward_dur, total_update_weight_dur))
    
        # loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
        # torch.cuda.synchronize()
        # end = time.perf_counter()
        # backward_dur = end - backward_start
        # total_dur = end - start
            # val_loss_all.append(loss.data.numpy())
            # print('Epoch:{},train_loss:{},val_loss:{}'.format(epoch, train_loss_all[-1], val_loss_all[-1]))
            # print('Epoch:{}'.format(epoch))
            # print("forward_time(ms): {}, backward_time(ms): {}, total_time(ms): {}".format(forward_dur*10e3, backward_dur*10e3, total_dur*10e3))
    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total"))
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    
    model.eval()
    if tensor_type == 'tensor':
        pred = model(data.x, data.edge_index).argmax(dim=1)
    elif tensor_type == 'sparse_tensor':
        pred = model(data.x, data.adj_t).argmax(dim=1)
    # pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')
    
    torch.save(model.state_dict(), 'GCNNet_v0.pt')

if use_profiler == 'true':
    run_model_with_profiler()
else:
    run_model_without_profiler()
