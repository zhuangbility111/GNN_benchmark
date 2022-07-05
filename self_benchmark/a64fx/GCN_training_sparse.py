import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
import time 
from torch_geometric.datasets import Planetoid
# import pdb

dataset = Planetoid(root='../../data/Cora', name='Cora', transform=T.ToSparseTensor())
# dataset = Planetoid(root='~/data/PubMed', name='PubMed')
print(dataset.data)
print('Number of classes:', dataset.num_classes)
print('Number of edges features:', dataset.num_edge_features)
print('Number of edges index:', dataset.data.edge_index.shape[1] / 2)
print('Number of nodes features:', dataset.num_node_features)
print('Number of nodes:', dataset.data.x.shape[0])
# print(':', dataset.data.x)

print(dataset.data.edge_index)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = GCNConv(dataset.num_node_features, 16, cached=True)
        # self.conv2 = GCNConv(16, dataset.num_classes, cached=True)

        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, adj_t):

        x = self.conv1(x, adj_t)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj_t)

        return F.log_softmax(x, dim=1)

def run_model_without_profiler(dataset):
    start = time.perf_counter()

    total_forward_dur = 0
    total_backward_dur = 0
    total_update_weight_dur = 0
    total_load_data_and_model_dur = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_start = time.perf_counter()
    model = GCN().to(device)
    data = dataset[0].to(device)
    total_load_data_and_model_dur = time.perf_counter() - load_start
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train_loss_all = []
    val_loss_all = []

    # with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=False, profile_memory=False, with_stack=True) as prof:
    model.train()
    for epoch in range(200):
        print('=' * 100)
        print('epoch:', epoch)
        optimizer.zero_grad()
        forward_start = time.perf_counter()
        out = model(data.x, data.adj_t)
        backward_start = time.perf_counter()
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        update_weight_start = time.perf_counter()
        optimizer.step()
        update_weight_end = time.perf_counter()
        # train_loss_all.append(loss.data.numpy())
        total_forward_dur += backward_start - forward_start
        total_backward_dur += update_weight_start - backward_start
        total_update_weight_dur += update_weight_end - update_weight_start
        loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
        # val_loss_all.append(loss.data.numpy())
        # print('Epoch:{},train_loss:{},val_loss:{}'.format(epoch, train_loss_all[-1], val_loss_all[-1]))
        print('Epoch:{},train_loss:{},val_loss:{}'.format(epoch, loss.data.numpy(), loss.data.numpy()))
    # print(prof.key_averages(group_by_stack_n=2).table(sort_by="self_cpu_time_total"))
    end = time.perf_counter()
    print("load_data_and_model_time(ms), {}".format(total_load_data_and_model_dur * 1000))
    print("forward_time(ms), {}".format(total_forward_dur * 1000))
    print("backward_time(ms), {}".format(total_backward_dur * 1000))
    print("update_weight_time(ms), {}".format(total_update_weight_dur * 1000))
    print("total_training_time(ms), {}:".format((end - start) * 1000))
    
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')

    torch.save(model.state_dict(), 'GCNNet_v0.pt')

def run_model_with_profiler(dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train_loss_all = []
    val_loss_all = []
    total_forward_dur = 0
    total_backward_dur = 0
    total_update_weight_dur = 0

    with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=False, profile_memory=False, with_stack=True) as prof:
        start = time.perf_counter()
        model.train()
        for epoch in range(200):
            print('=' * 100)
            print('epoch:', epoch)
            optimizer.zero_grad()
            # forward_start = time.perf_counter()
            out = model(data.x, data.adj_t)
            # backward_start = time.perf_counter()
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            # update_weight_start = time.perf_counter()
            optimizer.step()
            # update_weight_end = time.perf_counter()
            # train_loss_all.append(loss.data.numpy())
            # total_forward_dur += backward_start - forward_start
            # total_backward_dur += update_weight_start - backward_start
            # total_update_weight_dur += update_weight_end - update_weight_start
            loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
            # val_loss_all.append(loss.data.numpy())
            # print('Epoch:{},train_loss:{},val_loss:{}'.format(epoch, train_loss_all[-1], val_loss_all[-1]))
            print('Epoch:{},train_loss:{},val_loss:{}'.format(epoch, loss.data.numpy(), loss.data.numpy()))
        # print(prof.key_averages(group_by_stack_n=2).table(sort_by="self_cpu_time_total"))
        end = time.perf_counter()
        # print("training_time(s): {}, forward_time(s): {}, backward_time(s): {}, update_weight_time(s): {}".format((end - start), total_forward_dur, total_backward_dur, total_update_weight_dur))
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')

    torch.save(model.state_dict(), 'GCNNet_v0.pt')

run_model_without_profiler(dataset)
