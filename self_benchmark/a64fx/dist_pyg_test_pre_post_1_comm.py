import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn import DistGCNConv
from torch_geometric.nn import DistGCNConvGrad
from torch_geometric.nn import DistSAGEConvGrad
from torch_geometric.nn import DistributedGraphPre
from torch_geometric.nn import DistSAGEConvGradWithPre
# from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
import pandas as pd
import time
import argparse
import os
import random
import gc
import sys

class DistSAGEGradWithPre(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        num_layers = 3
        dropout = 0.5
        # dropout = 0.3

        self.convs = torch.nn.ModuleList()
        self.convs.append(DistSAGEConvGradWithPre(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                DistSAGEConvGradWithPre(hidden_channels, hidden_channels))
        self.convs.append(DistSAGEConvGradWithPre(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, graph: DistributedGraphPre, x):
        total_conv_time = 0.0
        total_relu_time = 0.0
        total_dropout_time = 0.0
        for i, conv in enumerate(self.convs[:-1]):
            conv_begin = time.perf_counter()
            x = conv(graph, x)
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
        x = self.convs[-1](graph, x)
        # total_conv_time += time.perf_counter() - conv_begin
        return F.log_softmax(x, dim=1)

# To load the training mask, test mask and validated mask from file 
def load_dataset_mask(dir_path, graph_name, rank, world_size):
    start = time.perf_counter()
    train_idx = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes_train_idx.npy".format(rank, graph_name)))
    valid_idx = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes_valid_idx.npy".format(rank, graph_name)))
    test_idx = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes_test_idx.npy".format(rank, graph_name)))
    end = time.perf_counter()
    print("elapsed time of loading dataset mask(ms) = {}".format((end - start) * 1000))

    return torch.from_numpy(train_idx), torch.from_numpy(valid_idx), torch.from_numpy(test_idx)

# to remap the nodes id in remote_nodes_list to local nodes id (from 0)
# the remote nodes list must be ordered
def remap_remote_nodes_id(remote_nodes_list, begin_node_on_each_partition):
    local_node_idx = -1
    for rank in range(begin_node_on_each_partition.shape[0]-1):
        prev_node = -1
        num_nodes = begin_node_on_each_partition[rank+1] - begin_node_on_each_partition[rank]
        begin_idx = begin_node_on_each_partition[rank]
        for i in range(num_nodes):
            # Attention !!! remote_nodes_list[i] must be transformed to scalar !!!
            cur_node = remote_nodes_list[begin_idx+i].item()
            if cur_node != prev_node:
                local_node_idx += 1
            prev_node = cur_node
            remote_nodes_list[begin_idx+i] = local_node_idx
    return local_node_idx + 1

def load_graph_data(dir_path, graph_name, rank, world_size):
    # load vertices on subgraph
    load_nodes_start = time.perf_counter()
    local_nodes_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes.npy".format(rank, graph_name)))
    node_idx_begin = local_nodes_list[0][0]
    node_idx_end = local_nodes_list[local_nodes_list.shape[0]-1][0]
    print("nodes_id_range: {} - {}".format(node_idx_begin, node_idx_end))
    num_local_nodes = node_idx_end - node_idx_begin + 1
    load_nodes_end = time.perf_counter()
    time_load_nodes = load_nodes_end - load_nodes_start

    # ----------------------------------------------------------

    # load features of vertices on subgraph
    nodes_feat_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes_feat.npy".format(rank, graph_name)))
    # nodes_feat_list = np.array([0,1,2], dtype=np.int64)
    print("nodes_feat_list.shape:")
    print(nodes_feat_list.shape)
    print(nodes_feat_list.dtype)
    load_nodes_feats_end = time.perf_counter()
    time_load_nodes_feats = load_nodes_feats_end - load_nodes_end

    # ----------------------------------------------------------

    # load labels of vertices on subgraph
    nodes_label_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes_label.npy".format(rank, graph_name)))
    print(nodes_label_list.dtype)
    load_nodes_labels_end = time.perf_counter()
    time_load_nodes_labels = load_nodes_labels_end - load_nodes_feats_end

    # ----------------------------------------------------------

    # load number of nodes on each subgraph
    begin_node_on_each_subgraph = np.loadtxt(os.path.join(dir_path, "begin_node_on_each_partition.txt"), dtype=np.int64, delimiter=' ')
    load_number_nodes_end = time.perf_counter()
    time_load_number_nodes = load_number_nodes_end - load_nodes_labels_end

    # ----------------------------------------------------------

    # divide the global edges list into the local edges list and the remote edges list
    local_edges_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_local_edges.npy".format(rank, graph_name)))
    remote_edges_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_remote_edges.npy".format(rank, graph_name)))

    # local nodes in local_edges_list and remote_edges_list has been localized
    # in order to perform pre_aggregation, the id of local nodes in remote_edges_list must be recover to global id
    remote_edges_list[1] += node_idx_begin

    print(local_edges_list)
    print(local_edges_list.shape)
    print(remote_edges_list)
    print(remote_edges_list.shape)
    divide_edges_list_end = time.perf_counter()
    time_divide_edges_list = divide_edges_list_end - load_number_nodes_end

    # ----------------------------------------------------------

    # sort remote_edges_list based on the src(remote) nodes' global id
    # sort_remote_edges_list_end = time.perf_counter()
    # remote_edges_list = sort_remote_edges_list_based_on_remote_nodes(remote_edges_list)
    # time_sort_remote_edges_list = sort_remote_edges_list_end - divide_edges_list_end

    # ----------------------------------------------------------

    # remove duplicated nodes
    # obtain remote nodes list and remap the global id of remote nodes to local id based on their rank
    # remote_nodes_list, range_of_remote_nodes_on_local_graph, remote_nodes_num_from_each_subgraph = \
    #                             obtain_remote_nodes_list(remote_edges_list, num_local_nodes, begin_node_on_each_subgraph, world_size)
    obtain_remote_nodes_list_end = time.perf_counter()
    time_obtain_remote_nodes_list = obtain_remote_nodes_list_end - divide_edges_list_end

    # ----------------------------------------------------------

    time_load_and_preprocessing_graph = obtain_remote_nodes_list_end - load_nodes_start

    print("elapsed time of loading nodes(ms) = {}".format(time_load_nodes * 1000))
    print("elapsed time of loading nodes feats(ms) = {}".format(time_load_nodes_feats * 1000))
    print("elapsed time of loading nodes labels(ms) = {}".format(time_load_nodes_labels * 1000))
    print("elapsed time of loading number of nodes(ms) = {}".format(time_load_number_nodes * 1000))
    print("elapsed time of dividing edges(ms) = {}".format(time_divide_edges_list * 1000))
    # print("elapsed time of sorting edges(ms) = {}".format(time_sort_remote_edges_list * 1000))
    print("elapsed time of obtaining remote nodes(ms) = {}".format(time_obtain_remote_nodes_list * 1000))
    print("elapsed time of whole process of loading graph(ms) = {}".format(time_load_and_preprocessing_graph * 1000))
    # print("number of remote nodes = {}".format(remote_nodes_list.shape[0]))

    return torch.from_numpy(local_nodes_list), torch.from_numpy(nodes_feat_list), torch.from_numpy(nodes_label_list), \
            torch.from_numpy(local_edges_list), torch.from_numpy(remote_edges_list), \
            torch.from_numpy(begin_node_on_each_subgraph)

def init_dist_group():
    if dist.is_mpi_available():
        print("mpi in torch.distributed is available!")
        dist.init_process_group(backend="mpi")
    assert torch.distributed.is_initialized()
    print(f"dist_info RANK: {dist.get_rank()}, SIZE: {dist.get_world_size()}")
    # number of process in this MPI group
    world_size = dist.get_world_size() 
    # mpi rank in this MPI group
    rank = dist.get_rank()
    return (rank, world_size)

def train(model, optimizer, graph, nodes_feat_list, nodes_label_list, 
          local_train_mask, rank, world_size):
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
        out = model(graph, nodes_feat_list)
        backward_start = time.perf_counter()
        loss = F.nll_loss(out[local_train_mask], nodes_label_list[local_train_mask])
        loss.backward()

        share_grad_start = time.perf_counter()
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
    ave_total_training_dur = torch.tensor([total_training_dur])
    max_total_training_dur = torch.tensor([total_training_dur])

    dist.reduce(total_forward_dur, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total_backward_dur, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total_share_grad_dur, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total_update_weight_dur, 0, op=dist.ReduceOp.SUM)
    dist.reduce(ave_total_training_dur, 0, op=dist.ReduceOp.SUM)
    dist.reduce(max_total_training_dur, 0, op=dist.ReduceOp.MAX)

    print("training end.")
    print("forward_time(ms): {}".format(total_forward_dur[0] / float(world_size) * 1000))
    print("backward_time(ms): {}".format(total_backward_dur[0] / float(world_size) * 1000))
    print("share_grad_time(ms): {}".format(total_share_grad_dur[0] / float(world_size) * 1000))
    print("update_weight_time(ms): {}".format(total_update_weight_dur[0] / float(world_size) * 1000))
    print("total_training_time(average)(ms): {}".format(ave_total_training_dur[0] / float(world_size) * 1000))
    print("total_training_time(max)(ms): {}".format(max_total_training_dur[0] * 1000.0))

def test(model, graph, nodes_feat_list, nodes_label_list, \
         local_train_mask, local_valid_mask, local_test_mask, rank, world_size):
    # check accuracy
    model.eval()
    predict_result = []
    print("sparse_tensor test!")
    out, accs = model(graph, nodes_feat_list), []
    # out, accs = model(nodes_feat_list, local_edges_list), []
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

def process_remote_edges_pre_post_aggr_to(is_pre_post_aggr_to, remote_edges_pre_post_aggr_to, world_size):
    remote_edges_list_pre_post_aggr_to = [torch.empty((0), dtype=torch.int64), torch.empty((0), dtype=torch.int64)]
    begin_edge_on_each_partition_to = torch.zeros(world_size+1, dtype=torch.int64)
    pre_aggr_to_splits = []
    post_aggr_to_splits = []
    for part_id in range(world_size):
        # post-aggregate
        if is_pre_post_aggr_to[part_id][0].item() == 0:
            # collect the number of local nodes current MPI rank needs
            post_aggr_to_splits.append(is_pre_post_aggr_to[part_id][2].item())
            pre_aggr_to_splits.append(0)
            # collect the local node id required by other MPI ranks, group them to edges list in which they will point to themselves
            remote_edges_list_pre_post_aggr_to[0] = torch.cat((remote_edges_list_pre_post_aggr_to[0], \
                                                               remote_edges_pre_post_aggr_to[part_id]), \
                                                               dim=0)
            remote_edges_list_pre_post_aggr_to[1] = torch.cat((remote_edges_list_pre_post_aggr_to[1], \
                                                               remote_edges_pre_post_aggr_to[part_id]), \
                                                               dim=0)
            # collect it for remapping nodes id
            begin_edge_on_each_partition_to[part_id+1] = begin_edge_on_each_partition_to[part_id] + remote_edges_pre_post_aggr_to[part_id].shape[0]
        # pre_aggregate
        else:
            # collect the number of post remote nodes current MPI rank needs
            pre_aggr_to_splits.append(is_pre_post_aggr_to[part_id][2].item())
            post_aggr_to_splits.append(0)
            # collect the subgraph sent from other MPI ranks for pre-aggregation
            num_remote_edges = int(is_pre_post_aggr_to[part_id][1].item() / 2)
            src_in_remote_edges = remote_edges_pre_post_aggr_to[part_id][:num_remote_edges]
            dst_in_remote_edges = remote_edges_pre_post_aggr_to[part_id][num_remote_edges:]
            
            # sort the remote edges based on the remote nodes (dst nodes)
            sort_index = torch.argsort(dst_in_remote_edges)
            remote_edges_list_pre_post_aggr_to[0] = torch.cat((remote_edges_list_pre_post_aggr_to[0], \
                                                               src_in_remote_edges[sort_index]), \
                                                               dim=0)
            remote_edges_list_pre_post_aggr_to[1] = torch.cat((remote_edges_list_pre_post_aggr_to[1], \
                                                               dst_in_remote_edges[sort_index]), \
                                                               dim=0)
            # collect it for remapping nodes id
            begin_edge_on_each_partition_to[part_id+1] = begin_edge_on_each_partition_to[part_id] + dst_in_remote_edges.shape[0]

        begin_edge_on_each_partition_to[world_size] = remote_edges_list_pre_post_aggr_to[0].shape[0]

    return remote_edges_list_pre_post_aggr_to, begin_edge_on_each_partition_to, \
           post_aggr_to_splits, pre_aggr_to_splits

def divide_remote_edges_list(begin_node_on_each_subgraph, remote_edges_list, world_size):
    is_pre_post_aggr_from = [torch.zeros((3), dtype=torch.int64) for i in range(world_size)]
    remote_edges_pre_post_aggr_from = []
    # remote_edges_list_post_aggr_from = [[], []]
    # local_nodes_idx_pre_aggr_from = []
    remote_edges_list_pre_post_aggr_from = [torch.empty((0), dtype=torch.int64), torch.empty((0), dtype=torch.int64)]
    begin_edge_on_each_partition_from = torch.zeros(world_size+1, dtype=torch.int64)
    remote_nodes_num_from_each_subgraph = torch.zeros(world_size, dtype=torch.int64)
    pre_aggr_from_splits = []
    post_aggr_from_splits = []
    num_diff_nodes = 0
    for i in range(world_size):
        # set the begin node idx and end node idx on current rank i
        begin_idx = begin_node_on_each_subgraph[i]
        end_idx = begin_node_on_each_subgraph[i+1]
        print("begin_idx = {}, end_idx = {}".format(begin_idx, end_idx))
        
        src_in_remote_edges = remote_edges_list[0]
        dst_in_remote_edges = remote_edges_list[1]

        # get the remote edges which are from current rank i
        edge_idx = ((src_in_remote_edges >= begin_idx) & (src_in_remote_edges < end_idx))
        src_in_remote_edges = src_in_remote_edges[edge_idx]
        dst_in_remote_edges = dst_in_remote_edges[edge_idx]

        # to get the number of remote nodes and local nodes to determine this rank is pre_aggr or post_aggr
        ids_src_nodes = torch.unique(src_in_remote_edges, sorted=True)
        ids_dst_nodes = torch.unique(dst_in_remote_edges, sorted=True)

        num_src_nodes = ids_src_nodes.shape[0]
        num_dst_nodes = ids_dst_nodes.shape[0]

        # print("total number of remote src nodes = {}".format(num_src_nodes))
        # print("total number of remote dst nodes = {}".format(num_dst_nodes))

        # accumulate the differences of remote src nodes and local dst nodes
        num_diff_nodes += abs(num_src_nodes - num_dst_nodes)
        remote_nodes_num_from_each_subgraph[i] = min(num_src_nodes, num_dst_nodes)

        # when the number of remote src_nodes > the number of local dst_nodes
        # pre_aggr is necessary to decrease the volumn of communication 
        # so pre_aggr  --> pre_post_aggr_from = 1 --> send the remote edges to src mpi rank
        #    post_aggr --> pre_post_aggr_from = 0 --> send the idx of src nodes to src mpi rank
        if num_src_nodes > num_dst_nodes:
            # pre_aggr
            # collect graph structure and send them to other MPI ransk to perform pre-aggregation
            tmp = torch.cat((src_in_remote_edges, \
                             dst_in_remote_edges), \
                             dim=0)
            remote_edges_pre_post_aggr_from.append(tmp)
            is_pre_post_aggr_from[i][0] = 1
            # number of remote edges = is_pre_post_aggr_from[i][1] / 2
            is_pre_post_aggr_from[i][1] = tmp.shape[0]
            # push the number of remote nodes current MPI rank needs
            is_pre_post_aggr_from[i][2] = ids_dst_nodes.shape[0]
            # collect number of nodes sent from other subgraphs for all_to_all_single
            pre_aggr_from_splits.append(ids_dst_nodes.shape[0])
            post_aggr_from_splits.append(0)
            # collect local node id sent from other MPI ranks, group them to edges list in which they will point to themselves
            remote_edges_list_pre_post_aggr_from[0] = torch.cat((remote_edges_list_pre_post_aggr_from[0], \
                                                                 ids_dst_nodes), \
                                                                 dim=0)
            remote_edges_list_pre_post_aggr_from[1] = torch.cat((remote_edges_list_pre_post_aggr_from[1], \
                                                                 ids_dst_nodes), \
                                                                 dim=0)
            # collect it for remapping nodes id
            begin_edge_on_each_partition_from[i+1] = begin_edge_on_each_partition_from[i] + ids_dst_nodes.shape[0]
        else:
            # post_aggr
            is_pre_post_aggr_from[i][0] = 0
            is_pre_post_aggr_from[i][1] = num_src_nodes
            # push the number of remote nodes current MPI rank needs
            is_pre_post_aggr_from[i][2] = ids_src_nodes.shape[0]
            # collect remote node id sent from other MPI ranks to notify other MPI ranks
            # which nodes current MPI rank needs
            remote_edges_pre_post_aggr_from.append(ids_src_nodes)
            # collect number of nodes sent from other subgraphs for all_to_all_single
            post_aggr_from_splits.append(ids_src_nodes.shape[0])
            pre_aggr_from_splits.append(0)

            # sort remote edges based on the remote nodes (src nodes)
            sort_index = torch.argsort(src_in_remote_edges)

            # collect remote edges for aggregation with SPMM later
            remote_edges_list_pre_post_aggr_from[0] = torch.cat((remote_edges_list_pre_post_aggr_from[0], \
                                                                 src_in_remote_edges[sort_index]), \
                                                                 dim=0)
            remote_edges_list_pre_post_aggr_from[1] = torch.cat((remote_edges_list_pre_post_aggr_from[1], \
                                                                 dst_in_remote_edges[sort_index]), \
                                                                 dim=0)
            # collect it for remapping nodes id
            begin_edge_on_each_partition_from[i+1] = begin_edge_on_each_partition_from[i] + src_in_remote_edges.shape[0]

    begin_edge_on_each_partition_from[world_size] = remote_edges_list_pre_post_aggr_from[0].shape[0]
    print("num_diff_nodes = {}".format(num_diff_nodes))

    # communicate with other mpi ranks to get the status of pre_aggr or post_aggr 
    # and number of remote edges(pre_aggr) or remote src nodes(post_aggr)
    is_pre_post_aggr_to = [torch.zeros((3), dtype=torch.int64) for i in range(world_size)]
    dist.all_to_all(is_pre_post_aggr_to, is_pre_post_aggr_from)

    # communicate with other mpi ranks to get the remote edges(pre_aggr) 
    # or remote src nodes(post_aggr)
    remote_edges_pre_post_aggr_to = [torch.empty((indices[1]), dtype=torch.int64) for indices in is_pre_post_aggr_to]
    dist.all_to_all(remote_edges_pre_post_aggr_to, remote_edges_pre_post_aggr_from)

    '''
    print("is_pre_post_aggr, remote_edges_pre_post_aggr:")
    print(is_pre_post_aggr_from)
    print(is_pre_post_aggr_to)
    print(remote_edges_pre_post_aggr_from)
    print(remote_edges_pre_post_aggr_to)
    '''

    remote_edges_list_pre_post_aggr_to, begin_edge_on_each_partition_to, \
    post_aggr_to_splits, pre_aggr_to_splits = \
        process_remote_edges_pre_post_aggr_to(is_pre_post_aggr_to, remote_edges_pre_post_aggr_to, world_size)

    del is_pre_post_aggr_from
    del is_pre_post_aggr_to
    del remote_edges_pre_post_aggr_from
    del remote_edges_pre_post_aggr_to

    '''
    # collect communication pattern
    global_list = [torch.zeros(world_size, dtype=torch.int64) for _ in range(world_size)]
    dist.gather(remote_nodes_num_from_each_subgraph, global_list if dist.get_rank() == 0 else None, 0)
    if dist.get_rank() == 0:
        global_comm_tensor = torch.cat((global_list))
        global_comm_array = global_comm_tesosr.reshape(world_size, world_size).numpy()
        print(global_comm_array)
        np.save('./move_communication_pattern/global_comm_{}.npy'.format(world_size), global_comm_array)
    '''
    '''
    print("local_nodes_idx_pre_post, remote_edges_list_pre_post_aggr:")
    print(local_nodes_idx_pre_aggr_from)
    print(local_nodes_idx_post_aggr_to)
    print(remote_edges_list_post_aggr_from)
    print(remote_edges_list_pre_aggr_to)
    '''

    return remote_edges_list_pre_post_aggr_from, remote_edges_list_pre_post_aggr_to, \
        begin_edge_on_each_partition_from, begin_edge_on_each_partition_to, \
        pre_aggr_from_splits, post_aggr_from_splits, \
        post_aggr_to_splits, pre_aggr_to_splits

def transform_edge_index_to_sparse_tensor(local_edges_list, \
                                          remote_edges_list_pre_post_aggr_from, \
                                          remote_edges_list_pre_post_aggr_to, \
                                          begin_edge_on_each_partition_from, \
                                          begin_edge_on_each_partition_to, \
                                          num_local_nodes, \
                                          local_node_begin_idx):
    # construct local sparse tensor for local aggregation
    # localize nodes
    # local_edges_list[0] -= local_node_begin_idx
    # local_edges_list[1] -= local_node_begin_idx

    # local_edges_list has been localized
    local_adj_t = SparseTensor(row=local_edges_list[1], \
                               col=local_edges_list[0], \
                               value=None, \
                               sparse_sizes=(num_local_nodes, num_local_nodes))

    del local_edges_list
    gc.collect()

    # ----------------------------------------------------------

    print("-----before remote_edges_list_pre_post_aggr_from[0]:-----")
    print(remote_edges_list_pre_post_aggr_from[0])
    print("-----before remote_edges_list_pre_post_aggr_from[1]:-----")
    print(remote_edges_list_pre_post_aggr_from[1])
    # localize the dst nodes id (local nodes id)
    remote_edges_list_pre_post_aggr_from[1] -= local_node_begin_idx
    # remap (localize) the sorted src nodes id (remote nodes id) for construction of SparseTensor
    num_remote_nodes_from = remap_remote_nodes_id(remote_edges_list_pre_post_aggr_from[0], begin_edge_on_each_partition_from)

    print("-----after remote_edges_list_pre_post_aggr_from[0]:-----")
    print(remote_edges_list_pre_post_aggr_from[0])
    print("-----after remote_edges_list_pre_post_aggr_from[1]:-----")
    print(remote_edges_list_pre_post_aggr_from[1])

    adj_t_pre_post_aggr_from = SparseTensor(row=remote_edges_list_pre_post_aggr_from[1], \
                                            col=remote_edges_list_pre_post_aggr_from[0], \
                                            value=None, \
                                            sparse_sizes=(num_local_nodes, num_remote_nodes_from))
    
    del remote_edges_list_pre_post_aggr_from
    del begin_edge_on_each_partition_from
    gc.collect()

    # ----------------------------------------------------------

    print("-----before remote_edges_list_pre_post_aggr_to[0]:-----")
    print(remote_edges_list_pre_post_aggr_to[0])
    print("-----before remote_edges_list_pre_post_aggr_to[1]:-----")
    print(remote_edges_list_pre_post_aggr_to[1])
    # localize the src nodes id (local nodes id)
    remote_edges_list_pre_post_aggr_to[0] -= local_node_begin_idx
    # remap (localize) the sorted dst nodes id (remote nodes id) for construction of SparseTensor
    num_remote_nodes_to = remap_remote_nodes_id(remote_edges_list_pre_post_aggr_to[1], begin_edge_on_each_partition_to)

    print("-----after remote_edges_list_pre_aggr_to[0]:-----")
    print(remote_edges_list_pre_post_aggr_to[0])
    print("-----after remote_edges_list_pre_aggr_to[1]:-----")
    print(remote_edges_list_pre_post_aggr_to[1])

    adj_t_pre_post_aggr_to = SparseTensor(row=remote_edges_list_pre_post_aggr_to[1], \
                                          col=remote_edges_list_pre_post_aggr_to[0], \
                                          value=None, \
                                          sparse_sizes=(num_remote_nodes_to, num_local_nodes))
    del remote_edges_list_pre_post_aggr_to
    del begin_edge_on_each_partition_to
    gc.collect()
    # ----------------------------------------------------------

    return local_adj_t, adj_t_pre_post_aggr_from, adj_t_pre_post_aggr_to

def init_adj_t(graph: DistributedGraphPre):
    if isinstance(graph.local_adj_t, SparseTensor) and \
      not graph.local_adj_t.has_value():
        graph.local_adj_t.fill_value_(1.)

    if isinstance(graph.adj_t_pre_post_aggr_from, SparseTensor) and \
      not graph.adj_t_pre_post_aggr_from.has_value():
        graph.adj_t_pre_post_aggr_from.fill_value_(1.)

    if isinstance(graph.adj_t_pre_post_aggr_to, SparseTensor) and \
      not graph.adj_t_pre_post_aggr_to.has_value():
        graph.adj_t_pre_post_aggr_to.fill_value_(1.)

def get_in_degrees(local_edges_list, remote_edges_list, num_local_nodes, begin_idx_local_nodes):
    local_degs = torch.zeros((num_local_nodes), dtype=torch.int64)
    source = torch.ones((local_edges_list[1].shape[0]), dtype=torch.int64)
    # tmp_index = local_edges_list[1] - begin_idx_local_nodes
    # local_degs.index_add_(dim=0, index=tmp_index, source=source)
    local_degs.index_add_(dim=0, index=local_edges_list[1], source=source)
    source = torch.ones((remote_edges_list[1].shape[0]), dtype=torch.int64)
    tmp_index = remote_edges_list[1] - begin_idx_local_nodes
    local_degs.index_add_(dim=0, index=tmp_index, source=source)
    return local_degs.unsqueeze(-1)

def test_forward(test_model, graph, feats):
    test_model.train()
    for epoch in range(10):
        out = test_model(graph, feats)
        print(out)
        # nodes_label_list = torch.tensor([2, 3, 3], dtype=torch.int64)
        # loss = F.nll_loss(out, nodes_label_list)
        # loss.backward()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_type', type=str, default='tensor')
    parser.add_argument('--use_profiler', type=str, default='false')
    parser.add_argument('--cached', type=str, default='true')
    parser.add_argument('--graph_name', type=str, default='arxiv')
    args = parser.parse_args()
    tensor_type = args.tensor_type
    use_profiler = args.use_profiler
    if args.cached == 'false':
        cached = False
    elif args.cached == 'true':
        cached = True
    graph_name = args.graph_name
    # graph_name = 'products'
    graph_name = 'papers100M'
    # graph_name = 'arxiv'
    # graph_name = 'test'
    # graph_name = 'pubmed'

    rank, world_size = init_dist_group()
    num_part = world_size
    # torch.set_num_threads(12)
    torch.set_num_threads(11)
    print("Rank = {}, Number of threads = {}".format(rank, torch.get_num_threads()))

    # obtain graph information
    local_nodes_list, nodes_feat_list, nodes_label_list, \
    local_edges_list, remote_edges_list, begin_node_on_each_subgraph = \
        load_graph_data("./ogbn_{}_new/{}_graph_{}_part/".format(graph_name, graph_name, num_part),
                        graph_name, 
                        rank, 
                        world_size)

    num_local_nodes = local_nodes_list.shape[0]
    local_in_degrees = get_in_degrees(local_edges_list, remote_edges_list, \
                                      num_local_nodes, begin_node_on_each_subgraph[rank])

    divide_remote_edges_begin = time.perf_counter()
    remote_edges_list_pre_post_aggr_from, remote_edges_list_pre_post_aggr_to, \
    begin_edge_on_each_partition_from, begin_edge_on_each_partition_to, \
    pre_aggr_from_splits, post_aggr_from_splits, \
    post_aggr_to_splits, pre_aggr_to_splits = \
        divide_remote_edges_list(begin_node_on_each_subgraph, \
                                 remote_edges_list, \
                                 world_size)

    divide_remote_edges_end = time.perf_counter()

    print("elapsed time of dividing remote edges(ms) = {}".format( \
            (divide_remote_edges_end - divide_remote_edges_begin) * 1000))

    pre_post_aggr_from_splits = []
    pre_post_aggr_to_splits = []
    for i in range(world_size):
        pre_post_aggr_from_splits.append(pre_aggr_from_splits[i] + post_aggr_from_splits[i])
        pre_post_aggr_to_splits.append(pre_aggr_to_splits[i] + post_aggr_to_splits[i])

    '''
    print("remote_edges_list_pre_post_aggr_from:")
    print(remote_edges_list_pre_post_aggr_from)
    print("remote_edges_list_pre_post_aggr_to:")
    print(remote_edges_list_pre_post_aggr_to)
    print("begin_edge_on_each_partition_from:")
    print(begin_edge_on_each_partition_from)
    print("begin_edge_on_each_partition_to:")
    print(begin_edge_on_each_partition_to)
    print("pre_aggr_from_splits:")
    print(pre_aggr_from_splits)
    print("post_aggr_from_splits:")
    print(post_aggr_from_splits)
    print("post_aggr_to_splits:")
    print(post_aggr_to_splits)
    print("pre_aggr_to_splits:")
    print(pre_aggr_to_splits)
    print("pre_post_aggr_from_splits:")
    print(pre_post_aggr_from_splits)
    print("pre_post_aggr_to_splits")
    print(pre_post_aggr_to_splits)
    print("local_in_degrees:")
    print(local_in_degrees)
    '''

    transform_remote_edges_begin = time.perf_counter()
    local_adj_t, adj_t_pre_post_aggr_from, adj_t_pre_post_aggr_to = \
        transform_edge_index_to_sparse_tensor(local_edges_list, \
                                              remote_edges_list_pre_post_aggr_from, \
                                              remote_edges_list_pre_post_aggr_to, \
                                              begin_edge_on_each_partition_from, \
                                              begin_edge_on_each_partition_to, \
                                              num_local_nodes, \
                                              begin_node_on_each_subgraph[rank])
    transform_remote_edges_end = time.perf_counter()
    print("elapsed time of transforming remote edges(ms) = {}".format( \
            (transform_remote_edges_end - transform_remote_edges_begin) * 1000))

    del local_nodes_list
    del local_edges_list
    del remote_edges_list
    del remote_edges_list_pre_post_aggr_from
    del remote_edges_list_pre_post_aggr_to
    gc.collect()

    dir_path = "./ogbn_{}_new/{}_graph_{}_part/".format(graph_name, graph_name, num_part)
    nodes_feat_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes_feat.npy".format(rank, graph_name)))
    nodes_feat_list = torch.from_numpy(nodes_feat_list)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''
    # ogbn-products
    input_channels = 100
    hidden_channels = 256
    # hidden_channels = 128
    output_channels = 47
    '''                                                                         
    # ogbn-papers100M
    input_channels = 128
    hidden_channels = 256
    # hidden_channels = 128
    output_channels = 172
    '''
    input_channels = 2
    hidden_channels = 8
    output_channels = 5
    # ogbn-arxiv
    input_channels = 128
    hidden_channels = 256
    output_channels = 40
    '''

    max_size = max(input_channels, hidden_channels, output_channels)
    buf_pre_post_aggr_from = torch.zeros((sum(pre_post_aggr_from_splits), max_size), dtype=torch.float32)
    buf_pre_post_aggr_to = torch.zeros((sum(pre_post_aggr_to_splits), max_size), dtype=torch.float32)
    g = DistributedGraphPre(local_adj_t, \
                            adj_t_pre_post_aggr_from, \
                            adj_t_pre_post_aggr_to, \
                            buf_pre_post_aggr_from, \
                            buf_pre_post_aggr_to, \
                            pre_post_aggr_from_splits, \
                            pre_post_aggr_to_splits, \
                            local_in_degrees)

    init_adj_t(g)

    model = DistSAGEGradWithPre(input_channels, hidden_channels, output_channels)
    para_model = torch.nn.parallel.DistributedDataParallel(model)
    # test_forward(para_model, g, nodes_feat_list)

    # test_communication(g, nodes_feat_list)

    # obtain training, validated, testing mask
    local_train_mask, local_valid_mask, local_test_mask = load_dataset_mask("./ogbn_{}_new/{}_graph_{}_part/".format(graph_name, graph_name, num_part), graph_name, rank, world_size)
    optimizer = torch.optim.Adam(para_model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(para_model.parameters(), lr=0.01, weight_decay=5e-4)
    train(para_model, optimizer, g, nodes_feat_list, nodes_label_list, local_train_mask, rank, world_size)
    test(para_model, g, nodes_feat_list, nodes_label_list, \
         local_train_mask, local_valid_mask, local_test_mask, rank, world_size)
        
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DistGCNGrad(local_nodes_required_by_other,
                            remote_nodes_list,
                            remote_nodes_num_from_each_subgraph,
                            range_of_remote_nodes_on_local_graph,
                            local_nodes_list.size(0),
                            rank,
                            world_size,
                            cached).to(device)
    '''

    '''
    model = DistSAGEGrad(local_nodes_required_by_other,
                            remote_nodes_list,
                            remote_nodes_num_from_each_subgraph,
                            range_of_remote_nodes_on_local_graph,
                            rank,
                            world_size).to(device)
    '''

    '''
    for name, parameters in model.named_parameters():
        print(name, parameters.size())

    para_model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.Adam(para_model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(para_model.parameters(), lr=0.01, weight_decay=5e-4)

    train(para_model, optimizer, nodes_feat_list, nodes_label_list, \
          local_edges_list, remote_edges_list, local_train_mask, rank, world_size)
    test(para_model, nodes_feat_list, nodes_label_list, \
         local_edges_list, remote_edges_list, local_train_mask, local_valid_mask, local_test_mask, rank, world_size)
    '''

