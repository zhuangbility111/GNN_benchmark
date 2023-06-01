import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn import DistGCNConv
from torch_geometric.nn import DistGCNConvGrad
from torch_geometric.nn import DistSAGEConvGrad
import numpy as np
import pandas as pd
import time
import argparse
import os
import random

import psutil

try:
    import torch_ccl
except ImportError as e:
    print(e)

def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

class DistGCNGrad(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 local_nodes_required_by_other,
                 num_local_nodes_required_by_other,
                 remote_nodes_list,
                 remote_nodes_num_from_each_subgraph,
                 range_of_remote_nodes_on_local_graph,
                 num_local_nodes,
                 rank,
                 num_part,
                 cached):
        super().__init__()
        num_layers = 3
        dropout = 0.5
        cached = True

        max_feat_len = max(in_channels, hidden_channels, out_channels)
        num_send_nodes = local_nodes_required_by_other.size(0)
        send_nodes_feat_buf = torch.zeros((num_send_nodes, max_feat_len), dtype=torch.float32)

        num_recv_nodes = remote_nodes_list.size(0)
        recv_nodes_feat_buf = torch.zeros((num_recv_nodes, max_feat_len), dtype=torch.float32)

        self.convs = torch.nn.ModuleList()
        self.convs.append(DistGCNConvGrad(in_channels, hidden_channels,
                                 local_nodes_required_by_other,
                                 num_local_nodes_required_by_other,
                                 remote_nodes_list,
                                 remote_nodes_num_from_each_subgraph,
                                 range_of_remote_nodes_on_local_graph,
                                 rank,
                                 num_part,
                                 send_nodes_feat_buf,
                                 recv_nodes_feat_buf,
                                 cached=cached))
        # self.bns = torch.nn.ModuleList()
        # self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                DistGCNConvGrad(hidden_channels, hidden_channels,
                                 local_nodes_required_by_other,
                                 num_local_nodes_required_by_other,
                                 remote_nodes_list,
                                 remote_nodes_num_from_each_subgraph,
                                 range_of_remote_nodes_on_local_graph,
                                 rank,
                                 num_part,
                                 send_nodes_feat_buf,
                                 recv_nodes_feat_buf,
                                 cached=cached))
            # self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(DistGCNConvGrad(hidden_channels, out_channels,
                                 local_nodes_required_by_other,
                                 num_local_nodes_required_by_other,
                                 remote_nodes_list,
                                 remote_nodes_num_from_each_subgraph,
                                 range_of_remote_nodes_on_local_graph,
                                 rank,
                                 num_part,
                                 send_nodes_feat_buf,
                                 recv_nodes_feat_buf,
                                 cached=cached))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # for bn in self.bns:
        #     bn.reset_parameters()

    def forward(self, x, local_edges_list, remote_edges_list):
        total_conv_time = 0.0
        total_relu_time = 0.0
        total_dropout_time = 0.0
        for i, conv in enumerate(self.convs[:-1]):
            conv_begin = time.perf_counter()
            x = conv(x, local_edges_list, remote_edges_list)
            # x = self.bns[i](x)
            relu_begin = time.perf_counter()
            x = F.relu(x)
            dropout_begin = time.perf_counter()
            x = F.dropout(x, p=self.dropout, training=self.training)
            dropout_end = time.perf_counter()
            # total_conv_time += relu_begin - conv_begin
            total_conv_time = relu_begin - conv_begin
            # total_relu_time += dropout_begin - relu_begin
            total_relu_time = dropout_begin - relu_begin
            # total_dropout_time += dropout_end - dropout_begin
            total_dropout_time = dropout_end - dropout_begin
            print("----------------------------------------")
            print("Time of conv(ms): {:.4f}".format(total_conv_time * 1000.0))
            print("Time of relu(ms): {:.4f}".format(total_relu_time * 1000.0))
            print("Time of dropout(ms): {:.4f}".format(total_dropout_time * 1000.0))
            print("----------------------------------------")

        conv_begin = time.perf_counter()
        x = self.convs[-1](x, local_edges_list, remote_edges_list)
        # total_conv_time += time.perf_counter() - conv_begin
        return F.log_softmax(x, dim=1)

class DistSAGEGrad(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 hidden_channels,
                 out_channels,
                 local_nodes_required_by_other,
                 num_local_nodes_required_by_other,
                 remote_nodes_list,
                 remote_nodes_num_from_each_subgraph,
                 range_of_remote_nodes_on_local_graph,
                 rank,
                 num_part):
        super().__init__()
        num_layers = 3
        dropout = 0.5

        max_feat_len = max(in_channels, hidden_channels, out_channels)
        num_send_nodes = local_nodes_required_by_other.size(0)
        send_nodes_feat_buf = torch.zeros((num_send_nodes, max_feat_len), dtype=torch.float32)

        num_recv_nodes = remote_nodes_list.size(0)
        recv_nodes_feat_buf = torch.zeros((num_recv_nodes, max_feat_len), dtype=torch.float32)

        self.convs = torch.nn.ModuleList()
        self.convs.append(DistSAGEConvGrad(in_channels, hidden_channels,
                                 local_nodes_required_by_other,
                                 num_local_nodes_required_by_other,
                                 remote_nodes_list,
                                 remote_nodes_num_from_each_subgraph,
                                 range_of_remote_nodes_on_local_graph,
                                 rank,
                                 num_part,
                                 send_nodes_feat_buf,
                                 recv_nodes_feat_buf))
        # self.bns = torch.nn.ModuleList()
        # self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                DistSAGEConvGrad(hidden_channels, hidden_channels,
                                 local_nodes_required_by_other,
                                 num_local_nodes_required_by_other,
                                 remote_nodes_list,
                                 remote_nodes_num_from_each_subgraph,
                                 range_of_remote_nodes_on_local_graph,
                                 rank,
                                 num_part,
                                 send_nodes_feat_buf,
                                 recv_nodes_feat_buf))
            # self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(DistSAGEConvGrad(hidden_channels, out_channels,
                                 local_nodes_required_by_other,
                                 num_local_nodes_required_by_other,
                                 remote_nodes_list,
                                 remote_nodes_num_from_each_subgraph,
                                 range_of_remote_nodes_on_local_graph,
                                 rank,
                                 num_part,
                                 send_nodes_feat_buf,
                                 recv_nodes_feat_buf))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # for bn in self.bns:
        #     bn.reset_parameters()

    def forward(self, x, local_edges_list, remote_edges_list):
        total_conv_time = 0.0
        total_relu_time = 0.0
        total_dropout_time = 0.0
        for i, conv in enumerate(self.convs[:-1]):
            conv_begin = time.perf_counter()
            x = conv(x, local_edges_list, remote_edges_list)
            # x = self.bns[i](x)
            relu_begin = time.perf_counter()
            x = F.relu(x)
            dropout_begin = time.perf_counter()
            x = F.dropout(x, p=self.dropout, training=self.training)
            dropout_end = time.perf_counter()
            # total_conv_time += relu_begin - conv_begin
            total_conv_time = relu_begin - conv_begin
            # total_relu_time += dropout_begin - relu_begin
            total_relu_time = dropout_begin - relu_begin
            # total_dropout_time += dropout_end - dropout_begin
            total_dropout_time = dropout_end - dropout_begin
            print("----------------------------------------")
            print("Time of conv(ms): {:.4f}".format(total_conv_time * 1000.0))
            print("Time of relu(ms): {:.4f}".format(total_relu_time * 1000.0))
            print("Time of dropout(ms): {:.4f}".format(total_dropout_time * 1000.0))
            print("----------------------------------------")

        conv_begin = time.perf_counter()
        x = self.convs[-1](x, local_edges_list, remote_edges_list)
        # total_conv_time += time.perf_counter() - conv_begin
        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self, cached):
        super().__init__()
        num_node_features = 500
        num_classes = 3
        num_hidden_channels = 16
        self.conv1 = GCNConv(num_node_features, num_hidden_channels, cached=cached)
        self.conv2 = GCNConv(num_hidden_channels, num_classes, cached=cached)

    # forward() for tensor
    def forward(self, x, edge):
        # x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge)
        return F.log_softmax(x, dim=1)

# compare two array and remap the elem in train_idx according to the mapping in nodes_id_list
# global id is mapped to local id in train_idx
def compare_array(train_idx, nodes_id_list, node_idx_begin):
    local_train_idx = []
    train_idx.sort()
    idx_in_mask = 0
    idx_in_node = 0
    len_mask = train_idx.shape[0]
    len_node_list = nodes_id_list.shape[0]
    while idx_in_mask < len_mask and idx_in_node < len_node_list:
        if train_idx[idx_in_mask] < nodes_id_list[idx_in_node][1]:
            idx_in_mask += 1
        elif train_idx[idx_in_mask] > nodes_id_list[idx_in_node][1]:
            idx_in_node += 1
        else:
            local_train_idx.append(nodes_id_list[idx_in_node][0].item() - node_idx_begin)
            idx_in_mask += 1
            idx_in_node += 1
    
    local_train_idx = torch.Tensor(local_train_idx).long()
    return local_train_idx

# To remap the training mask, test mask and validated mask according to new node id assignment
def remap_dataset_mask(dataset_mask, nodes_id_list, rank):
    remap_start = time.perf_counter()
    train_idx, valid_idx, test_idx = dataset_mask["train"], dataset_mask["valid"], dataset_mask["test"]
    node_idx_begin = nodes_id_list[0][0]
    # sort the original nodes_list according to the order of global id
    # this one could be eliminated by graph partitioning
    nodes_id_list = nodes_id_list[nodes_id_list[:,1].argsort()]

    # remap training mask
    local_train_idx = compare_array(train_idx, nodes_id_list, node_idx_begin)

    # remap validated mask
    local_valid_idx = compare_array(valid_idx, nodes_id_list, node_idx_begin)

    # remap test mask
    local_test_idx = compare_array(test_idx, nodes_id_list, node_idx_begin)
    remap_end = time.perf_counter()
    if rank == 0:
        print("elapsed time of remapping dataset mask(ms) = {}".format((remap_end - remap_start) * 1000))

    return local_train_idx, local_valid_idx, local_test_idx

# To load the training mask, test mask and validated mask from file 
def load_dataset_mask(dir_path, graph_name, rank, world_size):
    start = time.perf_counter()
    # train_idx = pd.read_csv(os.path.join(dir_path, "p{:0>3d}-{}_nodes_train_idx.txt".format(rank, graph_name)), sep=" ", header=None).values
    train_idx = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes_train_idx.npy".format(rank, graph_name)))
    print(train_idx.dtype)
    # valid_idx = pd.read_csv(os.path.join(dir_path, "p{:0>3d}-{}_nodes_valid_idx.txt".format(rank, graph_name)), sep=" ", header=None).values
    valid_idx = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes_valid_idx.npy".format(rank, graph_name)))
    # test_idx = pd.read_csv(os.path.join(dir_path, "p{:0>3d}-{}_nodes_test_idx.txt".format(rank, graph_name)), sep=" ", header=None).values
    test_idx = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes_test_idx.npy".format(rank, graph_name)))
    end = time.perf_counter()
    print("elapsed time of loading dataset mask(ms) = {}".format((end - start) * 1000))

    return torch.from_numpy(train_idx), torch.from_numpy(valid_idx), torch.from_numpy(test_idx)

def divide_edges_into_local_and_remote(edges_list, node_idx_begin, node_idx_end):
    local_edges_list = [[], []]
    remote_edges_list = [[], []]
    for edge in edges_list:
        # only the src node of each edges can be the remote node
        if edge[0] >= node_idx_begin and edge[0] <= node_idx_end:
            edge[0] -= node_idx_begin
            edge[1] -= node_idx_begin
            local_edges_list[0].append(edge[0])
            local_edges_list[1].append(edge[1])
        else:
            edge[1] -= node_idx_begin
            remote_edges_list[0].append(edge[0])
            remote_edges_list[1].append(edge[1])
    
    # convert list to numpy.array
    local_edges_list = np.array(local_edges_list, dtype= np.int64)
    remote_edges_list = np.array(remote_edges_list, dtype=np.int64)
    return local_edges_list, remote_edges_list

def sort_remote_edges_list_based_on_remote_nodes(remote_edges_list):
    remote_edges_row, remote_edges_col = remote_edges_list[0], remote_edges_list[1]
    sort_index = np.argsort(remote_edges_row)
    remote_edges_list[0] = remote_edges_row[sort_index]
    remote_edges_list[1] = remote_edges_col[sort_index]
    return remote_edges_list

def obtain_remote_nodes_list(remote_edges_list, num_local_nodes, num_nodes_on_each_subgraph, world_size):
    remote_nodes_list = []
    range_of_remote_nodes_on_local_graph = torch.zeros(world_size+1, dtype=torch.int64)
    remote_nodes_num_from_each_subgraph = torch.zeros(world_size, dtype=torch.int64)
    remote_edges_row = remote_edges_list[0]

    part_idx = 0
    local_node_idx = num_local_nodes - 1
    prev_node = -1
    tmp_len = len(remote_edges_row)
    for i in range(0, tmp_len):
        cur_node = remote_edges_row[i]
        if cur_node != prev_node:
            remote_nodes_list.append(cur_node)
            local_node_idx += 1
            while cur_node >= num_nodes_on_each_subgraph[part_idx+1]:
                part_idx += 1
                range_of_remote_nodes_on_local_graph[part_idx+1] = range_of_remote_nodes_on_local_graph[part_idx]
            range_of_remote_nodes_on_local_graph[part_idx+1] += 1
            remote_nodes_num_from_each_subgraph[part_idx] += 1
        prev_node = cur_node
        remote_edges_row[i] = local_node_idx

    for i in range(part_idx+1, world_size):
        range_of_remote_nodes_on_local_graph[i+1] = range_of_remote_nodes_on_local_graph[i]

    remote_nodes_list = np.array(remote_nodes_list, dtype=np.int64)
    print("local remote_nodes_num_from_each_subgraph:")
    print(remote_nodes_num_from_each_subgraph)

    '''
    # collect communication pattern
    global_list = [torch.zeros(world_size, dtype=torch.int64) for _ in range(world_size)]
    dist.gather(remote_nodes_num_from_each_subgraph, global_list if dist.get_rank() == 0 else None, 0)
    if dist.get_rank() == 0:
        global_comm_tesosr = torch.cat((global_list))

        global_comm_array = global_comm_tesosr.reshape(world_size, world_size).numpy()
        print(global_comm_array)
        np.save('./move_communication_pattern/global_comm_{}.npy'.format(world_size), global_comm_array)
    '''

    return remote_nodes_list, range_of_remote_nodes_on_local_graph, remote_nodes_num_from_each_subgraph

def load_graph_data(dir_path, graph_name, rank, world_size):
    # load vertices on subgraph
    load_nodes_start = time.perf_counter()
    # local_nodes_list = pd.read_csv(os.path.join(dir_path, "p{:0>3d}-{}_nodes.txt".format(rank, graph_name)), sep=" ", header=None, usecols=[0, 3]).values
    local_nodes_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes.npy".format(rank, graph_name)))
    print(local_nodes_list.dtype)
    node_idx_begin = local_nodes_list[0][0]
    node_idx_end = local_nodes_list[local_nodes_list.shape[0]-1][0]
    print("nodes_id_range: {} - {}".format(node_idx_begin, node_idx_end))
    num_local_nodes = node_idx_end - node_idx_begin + 1
    load_nodes_end = time.perf_counter()
    time_load_nodes = load_nodes_end - load_nodes_start

    # load features of vertices on subgraph
    # nodes_feat_list = pd.read_csv(os.path.join(dir_path, "p{:0>3d}-{}_nodes_feat.txt".format(rank, graph_name)), sep=" ", header=None, dtype=np.float32).values
    nodes_feat_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes_feat.npy".format(rank, graph_name)))
    print("nodes_feat_list.shape:")
    print(nodes_feat_list.shape)
    print(nodes_feat_list.dtype)
    load_nodes_feats_end = time.perf_counter()
    time_load_nodes_feats = load_nodes_feats_end - load_nodes_end

    # load labels of vertices on subgraph
    # nodes_label_list = pd.read_csv(os.path.join(dir_path, "p{:0>3d}-{}_nodes_label.txt".format(rank, graph_name)), sep=" ", header=None).values
    nodes_label_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes_label.npy".format(rank, graph_name)))
    print(nodes_label_list.dtype)
    load_nodes_labels_end = time.perf_counter()
    time_load_nodes_labels = load_nodes_labels_end - load_nodes_feats_end

    # load edges on subgraph
    # edges_list = pd.read_csv(os.path.join(dir_path, "p{:0>3d}-{}_edges.txt".format(rank, graph_name)), sep=" ", header=None).values
    # edges_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_edges.npy".format(rank, graph_name)))
    load_edges_list_end = time.perf_counter()
    time_load_edges_list = load_edges_list_end - load_nodes_labels_end

    # load number of nodes on each subgraph
    num_nodes_on_each_subgraph = np.loadtxt(os.path.join(dir_path, "begin_node_on_each_partition.txt"), dtype='int64', delimiter=' ')
    load_number_nodes_end = time.perf_counter()
    time_load_number_nodes = load_number_nodes_end - load_edges_list_end

    # divide the global edges list into the local edges list and the remote edges list
    # local_edges_list, remote_edges_list = divide_edges_into_local_and_remote(edges_list, node_idx_begin, node_idx_end)
    local_edges_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_local_edges.npy".format(rank, graph_name)))
    remote_edges_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_remote_edges.npy".format(rank, graph_name)))
    '''
    print(local_edges_list)
    print(local_edges_list.shape)
    print(local_edges_list.dtype)
    print(remote_edges_list)
    print(remote_edges_list.shape)
    print(remote_edges_list.dtype)
    '''
    print(local_edges_list)
    divide_edges_list_end = time.perf_counter()
    time_divide_edges_list = divide_edges_list_end - load_number_nodes_end

    # sort remote_edges_list based on the src(remote) nodes' global id
    sort_remote_edges_list_end = time.perf_counter()
    remote_edges_list = sort_remote_edges_list_based_on_remote_nodes(remote_edges_list)
    time_sort_remote_edges_list = sort_remote_edges_list_end - divide_edges_list_end

    # remove duplicated nodes
    # obtain remote nodes list and remap the global id of remote nodes to local id based on their rank
    remote_nodes_list, range_of_remote_nodes_on_local_graph, remote_nodes_num_from_each_subgraph = \
                                obtain_remote_nodes_list(remote_edges_list, num_local_nodes, num_nodes_on_each_subgraph, world_size)
    obtain_remote_nodes_list_end = time.perf_counter()
    time_obtain_remote_nodes_list = obtain_remote_nodes_list_end - sort_remote_edges_list_end

    time_load_and_preprocessing_graph = obtain_remote_nodes_list_end - load_nodes_start

    '''
    print("elapsed time of loading nodes(ms) = {}".format(time_load_nodes * 1000))
    print("elapsed time of loading nodes feats(ms) = {}".format(time_load_nodes_feats * 1000))
    print("elapsed time of loading nodes labels(ms) = {}".format(time_load_nodes_labels * 1000))
    print("elapsed time of loading edges(ms) = {}".format(time_load_edges_list * 1000))
    print("elapsed time of loading number of nodes(ms) = {}".format(time_load_number_nodes * 1000))
    print("elapsed time of dividing edges(ms) = {}".format(time_divide_edges_list * 1000))
    print("elapsed time of sorting edges(ms) = {}".format(time_sort_remote_edges_list * 1000))
    print("elapsed time of obtaining remote nodes(ms) = {}".format(time_obtain_remote_nodes_list * 1000))
    print("elapsed time of whole process of loading graph(ms) = {}".format(time_load_and_preprocessing_graph * 1000))
    print("number of remote nodes = {}".format(remote_nodes_list.shape[0]))
    '''

    return torch.from_numpy(local_nodes_list), torch.from_numpy(nodes_feat_list), \
           torch.from_numpy(nodes_label_list), torch.from_numpy(remote_nodes_list), \
           range_of_remote_nodes_on_local_graph, remote_nodes_num_from_each_subgraph, \
           torch.from_numpy(local_edges_list), torch.from_numpy(remote_edges_list)

def init_dist_group():
    if dist.is_mpi_available():
        print("mpi in torch.distributed is available!")
        dist.init_process_group(backend="mpi")
    else:
        world_size = int(os.environ.get("PMI_SIZE", -1))
        rank = int(os.environ.get("PMI_RANK", -1))
        print("PMI_SIZE = {}".format(world_size))
        print("PMI_RANK = {}".format(rank))
        print("use ccl backend for torch.distributed package on x86 cpu.")
        dist_url = "env://"
        dist.init_process_group(backend="ccl", init_method="env://", 
                                world_size=world_size, rank=rank)
    assert torch.distributed.is_initialized()
    print(f"dist_info RANK: {dist.get_rank()}, SIZE: {dist.get_world_size()}")
    # number of process in this MPI group
    world_size = dist.get_world_size() 
    # mpi rank in this MPI group
    rank = dist.get_rank()
    return (rank, world_size)

def obtain_local_nodes_required_by_other(remote_nodes_list, range_of_remote_nodes_on_local_graph, \
                                         remote_nodes_num_from_each_subgraph, world_size):
    # send the number of remote nodes we need to obtain from other subgrpah
    obtain_number_remote_nodes_start = time.perf_counter()
    send_num_nodes = [torch.tensor([remote_nodes_num_from_each_subgraph[i]], dtype=torch.int64) for i in range(world_size)]
    recv_num_nodes = [torch.zeros(1, dtype=torch.int64) for i in range(world_size)]
    dist.all_to_all(recv_num_nodes, send_num_nodes)
    num_local_nodes_required_by_other = recv_num_nodes
    num_local_nodes_required_by_other = torch.cat(num_local_nodes_required_by_other, dim=0)
    obtain_number_remote_nodes_end = time.perf_counter()
    print("elapsed time of obtaining number of remote nodes(ms) = {}".format( \
            (obtain_number_remote_nodes_end - obtain_number_remote_nodes_start) * 1000))

    # then we need to send the nodes_list which include the id of remote nodes we want
    # and receive the nodes_list from other subgraphs
    obtain_remote_nodes_list_start = time.perf_counter()
    send_nodes_list = [remote_nodes_list[range_of_remote_nodes_on_local_graph[i]: \
                       range_of_remote_nodes_on_local_graph[i+1]] for i in range(world_size)]
    recv_nodes_list = [torch.zeros(num_local_nodes_required_by_other[i], dtype=torch.int64) for i in range(world_size)]
    dist.all_to_all(recv_nodes_list, send_nodes_list)
    local_node_idx_begin = local_nodes_list[0][0]
    local_nodes_required_by_other = [i - local_node_idx_begin for i in recv_nodes_list]
    local_nodes_required_by_other = torch.cat(local_nodes_required_by_other, dim=0)
    obtain_remote_nodes_list_end = time.perf_counter()
    print("elapsed time of obtaining list of remote nodes(ms) = {}".format( \
            (obtain_remote_nodes_list_end - obtain_remote_nodes_list_start) * 1000))
    return local_nodes_required_by_other, num_local_nodes_required_by_other
    
def transform_edge_index_to_sparse_tensor(local_edges_list, remote_edges_list, num_local_nodes, num_remote_nodes):
    # local_edges_list = SparseTensor(row=local_edges_list[1], col=local_edges_list[0], value=None, sparse_sizes=(num_local_nodes, num_local_nodes + num_remote_nodes)).to_symmetric()
    # local_edges_list = SparseTensor(row=local_edges_list[1], col=local_edges_list[0], value=None, sparse_sizes=(num_local_nodes, num_local_nodes + num_remote_nodes))
    # remote_edges_list = SparseTensor(row=remote_edges_list[1], col=remote_edges_list[0], value=None, sparse_sizes=(num_local_nodes, num_local_nodes + num_remote_nodes))
    local_edges_list = SparseTensor(row=local_edges_list[1], col=local_edges_list[0], value=torch.ones(local_edges_list[1].shape[0], dtype=torch.float32), sparse_sizes=(num_local_nodes, num_local_nodes))
    tmp_col = remote_edges_list[0] - num_local_nodes
    remote_edges_list = SparseTensor(row=remote_edges_list[1], col=tmp_col, value=torch.ones(remote_edges_list[1].shape[0], dtype=torch.float32), sparse_sizes=(num_local_nodes, num_remote_nodes))
    '''
    local_edges_list = SparseTensor(row=local_edges_list[1], col=local_edges_list[0], value=None, sparse_sizes=(num_local_nodes, num_local_nodes))
    tmp_col = remote_edges_list[0] - num_local_nodes
    remote_edges_list = SparseTensor(row=remote_edges_list[1], col=tmp_col, value=None, sparse_sizes=(num_local_nodes, num_remote_nodes))
    '''
    return local_edges_list, remote_edges_list

def train(model, optimizer, nodes_feat_list, nodes_label_list, 
          local_edges_list, remote_edges_list, local_train_mask, rank, world_size):
    # start training
    start = time.perf_counter()
    total_forward_dur = 0
    total_backward_dur = 0
    total_share_grad_dur = 0
    total_update_weight_dur = 0

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        forward_start = time.perf_counter()
        if tensor_type == 'tensor':
            out = model(nodes_feat_list, local_edges_list, remote_edges_list)
        elif tensor_type == 'sparse_tensor':
            out = model(nodes_feat_list, local_edges_list, remote_edges_list)
        backward_start = time.perf_counter()
        loss = F.nll_loss(out[local_train_mask], nodes_label_list[local_train_mask])
        loss.backward()

        share_grad_start = time.perf_counter()
        '''
        # communicate gradients
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= float(world_size)
        '''
        update_weight_start = time.perf_counter()
        optimizer.step()
        print("rank: {}, epoch: {}, loss: {}".format(rank, epoch, loss.item()))
        update_weight_end = time.perf_counter()
        total_forward_dur += backward_start - forward_start
        total_backward_dur += share_grad_start - backward_start
        total_share_grad_dur += update_weight_start - share_grad_start
        total_update_weight_dur += update_weight_end - update_weight_start
    end = time.perf_counter()
    total_training_dur = (end - start)
    
    total_forward_dur = torch.tensor([total_forward_dur])
    total_backward_dur = torch.tensor([total_backward_dur])
    total_share_grad_dur = torch.tensor([total_share_grad_dur])
    total_update_weight_dur = torch.tensor([total_update_weight_dur])
    total_training_dur = torch.tensor([total_training_dur])

    dist.reduce(total_forward_dur, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total_backward_dur, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total_share_grad_dur, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total_update_weight_dur, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total_training_dur, 0, op=dist.ReduceOp.SUM)

    print("training end.")
    print("forward_time(ms): {}".format(total_forward_dur[0] / float(world_size) * 1000))
    print("backward_time(ms): {}".format(total_backward_dur[0] / float(world_size) * 1000))
    print("share_grad_time(ms): {}".format(total_share_grad_dur[0] / float(world_size) * 1000))
    print("update_weight_time(ms): {}".format(total_update_weight_dur[0] / float(world_size) * 1000))
    print("total_training_time(ms): {}".format(total_training_dur[0] / float(world_size) * 1000))

def test(model, nodes_feat_list, nodes_label_list, \
         local_edges_list, remote_edges_list, local_train_mask, local_valid_mask, local_test_mask, rank, world_size):
    # check accuracy
    model.eval()
    predict_result = []
    if tensor_type == 'tensor':
        out, accs = model(nodes_feat_list, local_edges_list, remote_edges_list), []
    elif tensor_type == 'sparse_tensor':
        print("sparse_tensor test!")
        out, accs = model(nodes_feat_list, local_edges_list, remote_edges_list), []
    for mask in (local_train_mask, local_valid_mask, local_test_mask):
        num_correct_data = (out[mask].argmax(-1) == nodes_label_list[mask]).sum()
        num_data = mask.size(0)
        print("local num_correct_data = {}, local num_entire_dataset = {}".format(num_correct_data, num_data))
        predict_result.append(num_correct_data) 
        predict_result.append(num_data)
    predict_result = torch.tensor(predict_result)
    dist.reduce(predict_result, 0, op=dist.ReduceOp.SUM)

    if rank == 0:
        train_acc = float(predict_result[0] / predict_result[1])
        val_acc = float(predict_result[2] / predict_result[3])
        test_acc = float(predict_result[4] / predict_result[5])
        print("size of correct training sample = {}, size of correct valid sample = {}, size of correct test sample = {}".format( \
                predict_result[0], predict_result[2], predict_result[4]))
        print("size of all training sample = {}, size of all valid sample = {}, size of all test sample = {}".format( \
                predict_result[1], predict_result[3], predict_result[5]))
        print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cached', type=str, default='true')
    parser.add_argument('--graph_name', type=str, default='arxiv')
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--is_async', type=str, default='false')
    parser.add_argument('--input_dir', type=str)

    setup_seed(0)

    args = parser.parse_args()
    cached = False if args.cached == 'false' else True
    is_async = True if args.is_async == 'true' else False
    input_dir = args.input_dir

    '''
    if is_async == True:
        torch.set_num_threads(11)
    else:
        torch.set_num_threads(12)
    '''
        
    graph_name = args.graph_name
    if graph_name == 'products':
        in_channels = 100
        hidden_channels = 256
        out_channels = 47
    elif graph_name == 'papers100M':
        in_channels = 128
        hidden_channels = 256
        out_channels = 172
    elif graph_name == 'arxiv':
        in_channels = 128
        hidden_channels = 256
        out_channels = 40

    model_name = args.model
    tensor_type = 'sparse_tensor'

    print("graph_name = {}, model_name = {}, is_async = {}".format(graph_name, model_name, is_async))
    print("in_channels = {}, hidden_channels = {}, out_channels = {}".format(in_channels, hidden_channels, out_channels))
    print("input_dir = {}".format(input_dir))
        
    rank, world_size = init_dist_group()
    num_part = world_size
    print("Rank = {}, Number of threads = {}".format(rank, torch.get_num_threads()))

    # obtain graph information
    local_nodes_list, nodes_feat_list, nodes_label_list, remote_nodes_list, \
        range_of_remote_nodes_on_local_graph, remote_nodes_num_from_each_subgraph, \
        local_edges_list, remote_edges_list = load_graph_data(input_dir, graph_name, rank, world_size)

    # obtain training, validated, testing mask
    local_train_mask, local_valid_mask, local_test_mask = load_dataset_mask(input_dir, graph_name, rank, world_size)
    # local_train_mask, local_valid_mask, local_test_mask = load_dataset_mask("./test_level_partition/{}_graph_{}_part/".format(graph_name, num_part), graph_name, rank, world_size)

    # obtain the idx of local nodes required by other subgraph
    local_nodes_required_by_other, num_local_nodes_required_by_other = \
        obtain_local_nodes_required_by_other(remote_nodes_list, range_of_remote_nodes_on_local_graph, \
                                             remote_nodes_num_from_each_subgraph, world_size)

    # transform the local edges list and remote edges list(both are edge_index) to SparseTensor if it needs
    if tensor_type == 'sparse_tensor':
        local_edges_list, remote_edges_list = transform_edge_index_to_sparse_tensor(local_edges_list, remote_edges_list, local_nodes_list.size(0), remote_nodes_list.size(0))
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_name == 'gcn':
        model = DistGCNGrad(in_channels,
                            hidden_channels,
                            out_channels,
                            local_nodes_required_by_other,
                            num_local_nodes_required_by_other,
                            remote_nodes_list,
                            remote_nodes_num_from_each_subgraph,
                            range_of_remote_nodes_on_local_graph,
                            local_nodes_list.size(0),
                            rank,
                            world_size,
                            cached).to(device)
    elif model_name == 'sage':
        model = DistSAGEGrad(in_channels,
                             hidden_channels,
                             out_channels,
                             local_nodes_required_by_other,
                             num_local_nodes_required_by_other,
                             remote_nodes_list,
                             remote_nodes_num_from_each_subgraph,
                             range_of_remote_nodes_on_local_graph,
                             rank,
                             world_size).to(device)

    for name, parameters in model.named_parameters():
        print(name, parameters.size())

    # DDP should synchronize between GPUs when doing batchnorm
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    para_model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.Adam(para_model.parameters(), lr=0.01)

    train(para_model, optimizer, nodes_feat_list, nodes_label_list, \
          local_edges_list, remote_edges_list, local_train_mask, rank, world_size)
    test(para_model, nodes_feat_list, nodes_label_list, \
         local_edges_list, remote_edges_list, local_train_mask, local_valid_mask, local_test_mask, rank, world_size)

    '''
    with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=False, profile_memory=True, with_stack=False) as prof:
        for _ in range(1):
        # para_model = torch.nn.parallel.DistributedDataParallel(model)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # optimizer = torch.optim.Adam(para_model.parameters(), lr=0.01, weight_decay=5e-4)
            train(para_model, optimizer, nodes_feat_list, nodes_label_list, \
                local_edges_list, remote_edges_list, local_train_mask, rank, world_size)

    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    test(para_model, nodes_feat_list, nodes_label_list, \
        local_edges_list, remote_edges_list, local_train_mask, local_valid_mask, local_test_mask, rank, world_size)
    '''
