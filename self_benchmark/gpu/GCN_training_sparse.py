from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
import time
# import pdb

torch.cuda.set_device(4)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

dataset = Planetoid(root='../data/Cora', name='Cora', transform=T.ToSparseTensor())
# dataset = Planetoid(root='../data/Cora', name='Cora')
# dataset = Planetoid(root='~/data/PubMed', name='PubMed')
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
        # self.conv1 = GCNConv(dataset.num_node_features, 16)
        # self.conv2 = GCNConv(16, dataset.num_classes)
        self.conv1 = GCNConv(dataset.num_node_features, dataset.num_node_features)
        self.conv2 = GCNConv(dataset.num_node_features, dataset.num_classes)

    def forward(self, x, adj_t):
        x = self.conv1(x, adj_t)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adj_t)
        return F.log_softmax(x, dim=1)

def run_model_without_profiler(dataset):
    total_forward_dur = 0
    total_backward_dur = 0
    total_update_weight_dur = 0
    total_load_data_and_model_dur = 0
    dur = 0
    repeat_round = 1
    for _ in range(repeat_round):
        print("num of gpu:{}".format(torch.cuda.device_count()))
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        start = time.perf_counter()
        device = torch.device('cuda')
        load_start = time.perf_counter()
        model = GCN().to(device)
        data = dataset[0].to(device)
        total_load_data_and_model_dur += time.perf_counter() - load_start
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        train_loss_all = []
        val_loss_all = []
        torch.cuda.synchronize()
        model.train()

        for epoch in range(200):
            print('=' * 100)
            print('epoch:', epoch)
            optimizer.zero_grad()
            torch.cuda.synchronize()
            forward_start = time.perf_counter()
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
        torch.cuda.synchronize()
        end = time.perf_counter()
        dur += end - start

    print("total_training_time(ms): {}:".format(dur * 1000 / repeat_round))
    print("load_data_and_model_time(ms): {}".format(total_load_data_and_model_dur * 1000 / repeat_round))
    print("forward_time(ms): {}".format(total_forward_dur * 1000 / repeat_round))
    print("backward_time(ms): {}".format(total_backward_dur * 1000  / repeat_round))
    print("update_weight_time(ms): {}".format(total_update_weight_dur * 1000 / repeat_round))

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
        # print(prof.key_averages().table(sort_by="self_cuda_time_total"))
'''
        model.eval()
        pred = model(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        print(f'Accuracy: {acc:.4f}')

        torch.save(model.state_dict(), 'GCNNet_v0.pt')
'''

def run_model_with_profiler(dataset):
    torch.cuda.set_device(2)
    print("num of gpu:{}".format(torch.cuda.device_count()))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')
    print(device)
    model = GCN().to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train_loss_all = []
    val_loss_all = []
    total_forward_dur = 0
    total_backward_dur = 0
    total_update_weight_dur = 0

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False, with_stack=False) as prof:
        start = time.perf_counter()
        # data = dataset[0].to(device)
        torch.cuda.synchronize()
        model.train()
        for epoch in range(200):
            print('=' * 100)
            print('epoch:', epoch)
            optimizer.zero_grad()
            # torch.cuda.synchronize()
            # forward_start = time.perf_counter()
            out = model(data.x, data.adj_t)
            # torch.cuda.synchronize()
            # backward_start = time.perf_counter()
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            # torch.cuda.synchronize()
            # update_weight_start = time.perf_counter()
            optimizer.step()
            # torch.cuda.synchronize()
            # update_weight_end = time.perf_counter()
            # total_forward_dur += backward_start - forward_start
            # total_backward_dur += update_weight_start - backward_start
            # total_update_weight_dur += update_weight_end - update_weight_start
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
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')

    torch.save(model.state_dict(), 'GCNNet_v0.pt')

run_model_without_profiler(dataset)
