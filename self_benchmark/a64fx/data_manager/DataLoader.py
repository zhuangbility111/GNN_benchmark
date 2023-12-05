import time
import numpy as np
import torch
import torch.distributed as dist
from .DistributedGraph import DistributedGraph, DistributedGraphForPre
from .DataProcessor import DataProcessor, DataProcessorForPreAggresive
from .CommBuffer import CommBuffer, CommBufferForQuantization
from .CommSplits import CommSplits
import os


def load_graph_structures(input_dir, graph_name, rank):
    # load vertices on subgraph
    local_nodes_list = np.load(os.path.join(input_dir, "p{:0>3d}-{}_nodes.npy".format(rank, graph_name)))
    node_idx_begin = local_nodes_list[0][0]
    node_idx_end = local_nodes_list[local_nodes_list.shape[0] - 1][0]
    num_local_nodes = node_idx_end - node_idx_begin + 1

    # ----------------------------------------------------------
    # divide the global edges list into the local edges list and the remote edges list
    local_edges_list = np.load(
        os.path.join(input_dir, "p{:0>3d}-{}_local_edges.npy".format(rank, graph_name))
    )
    remote_edges_list = np.load(
        os.path.join(input_dir, "p{:0>3d}-{}_remote_edges.npy".format(rank, graph_name))
    )

    # ----------------------------------------------------------
    # load number of nodes on each subgraph
    nodes_range_on_each_subgraph = np.loadtxt(
        os.path.join(input_dir, "begin_node_on_each_partition.txt"), dtype=np.int64, delimiter=" "
    )

    return (
        torch.from_numpy(local_nodes_list),
        torch.from_numpy(local_edges_list),
        torch.from_numpy(remote_edges_list),
        torch.from_numpy(nodes_range_on_each_subgraph),
        num_local_nodes,
    )


def load_nodes_labels(input_dir, graph_name, rank):
    # load labels of vertices on subgraph
    nodes_label_list = np.load(
        os.path.join(input_dir, "p{:0>3d}-{}_nodes_label.npy".format(rank, graph_name))
    )
    return torch.from_numpy(nodes_label_list)


def load_nodes_features(input_dir, graph_name, rank):
    # load features of vertices on subgraph
    nodes_feat_list = np.load(os.path.join(input_dir, "p{:0>3d}-{}_nodes_feat.npy".format(rank, graph_name)))
    return torch.from_numpy(nodes_feat_list)


def load_dataset_mask(input_dir, graph_name, rank):
    train_idx = np.load(os.path.join(input_dir, "p{:0>3d}-{}_nodes_train_idx.npy".format(rank, graph_name)))
    valid_idx = np.load(os.path.join(input_dir, "p{:0>3d}-{}_nodes_valid_idx.npy".format(rank, graph_name)))
    test_idx = np.load(os.path.join(input_dir, "p{:0>3d}-{}_nodes_test_idx.npy".format(rank, graph_name)))
    return torch.from_numpy(train_idx), torch.from_numpy(valid_idx), torch.from_numpy(test_idx)


def get_distributed_graph(
    local_edges_list,
    remote_edges_list,
    local_nodes_list,
    nodes_range_on_each_subgraph,
    num_local_nodes: int,
    max_feat_len: int,
    world_size: int,
    bits: int,
):
    in_degrees = DataProcessor.get_in_degrees(local_edges_list, remote_edges_list, num_local_nodes)

    # sort remote_edges_list based on the src(remote) nodes' global id
    remote_edges_list = DataProcessor.sort_remote_edges_list_based_on_remote_nodes(remote_edges_list)

    # obtain remote nodes list and remap the global id of remote nodes to local id based on their rank
    (
        remote_nodes_list,
        range_of_remote_nodes_on_local_graph,
        remote_nodes_num_from_each_subgraph,
    ) = DataProcessor.obtain_remote_nodes_list(
        remote_edges_list, num_local_nodes, nodes_range_on_each_subgraph, world_size
    )

    (
        local_nodes_required_by_other,
        num_local_nodes_required_by_other,
    ) = DataProcessor.obtain_local_nodes_required_by_other(
        local_nodes_list,
        remote_nodes_list,
        range_of_remote_nodes_on_local_graph,
        remote_nodes_num_from_each_subgraph,
        world_size,
    )

    local_adj_t, remote_adj_t = DataProcessor.transform_edge_index_to_sparse_tensor(
        local_edges_list, remote_edges_list, local_nodes_list.size(0), remote_nodes_list.size(0)
    )

    comm_splits = CommSplits(remote_nodes_num_from_each_subgraph.tolist(), num_local_nodes_required_by_other.tolist(), world_size, bits)
    comm_buf = CommBuffer(comm_splits, max_feat_len, bits)
    comm_buf_for_quantization = None

    if bits == 2:
        comm_buf_for_quantization = CommBufferForQuantization(comm_splits, max_feat_len, bits)

    distributed_graph = DistributedGraph(
        local_adj_t,
        remote_adj_t,
        local_nodes_required_by_other,
        num_local_nodes_required_by_other.tolist(),
        remote_nodes_num_from_each_subgraph.tolist(),
        in_degrees,
        comm_splits,
        comm_buf,
        comm_buf_for_quantization,
    )

    return distributed_graph


def get_distributed_graph_for_pre_aggressive(
    local_edges_list,
    remote_edges_list,
    nodes_range_on_each_subgraph,
    num_local_nodes,
    max_feat_len,
    rank,
    world_size,
    bits,
):
    in_degrees = DataProcessor.get_in_degrees(local_edges_list, remote_edges_list, num_local_nodes)

    # local nodes in local_edges_list and remote_edges_list has been localized
    # in order to perform pre_aggregation, the id of local nodes in remote_edges_list must be recover to global id
    remote_edges_list[1] += nodes_range_on_each_subgraph[rank]

    (
        remote_edges_list_pre_post_aggr_from,
        remote_edges_list_pre_post_aggr_to,
        begin_edge_on_each_partition_from,
        begin_edge_on_each_partition_to,
        pre_post_aggr_from_splits,
        pre_post_aggr_to_splits,
    ) = DataProcessorForPreAggresive.divide_remote_edges_list(
        nodes_range_on_each_subgraph, remote_edges_list, world_size
    )

    (
        local_adj_t,
        adj_t_pre_post_aggr_from,
        adj_t_pre_post_aggr_to,
    ) = DataProcessorForPreAggresive.transform_edge_index_to_sparse_tensor(
        local_edges_list,
        remote_edges_list_pre_post_aggr_from,
        remote_edges_list_pre_post_aggr_to,
        begin_edge_on_each_partition_from,
        begin_edge_on_each_partition_to,
        num_local_nodes,
        nodes_range_on_each_subgraph[rank],
    )

    comm_splits = CommSplits(pre_post_aggr_from_splits, pre_post_aggr_to_splits, world_size, bits)
    comm_buf = CommBuffer(comm_splits, max_feat_len, bits)
    comm_buf_for_quantization = None

    if bits == 2:
        comm_buf_for_quantization = CommBufferForQuantization(comm_splits, max_feat_len, bits)

    distributed_graph = DistributedGraphForPre(
        local_adj_t,
        adj_t_pre_post_aggr_from,
        adj_t_pre_post_aggr_to,
        pre_post_aggr_from_splits,
        pre_post_aggr_to_splits,
        in_degrees,
        comm_splits,
        comm_buf,
        comm_buf_for_quantization,
    )

    # print("graph.local_adj_t = {}".format(distributed_graph.local_adj_t))
    # print("graph.adj_t_pre_post_aggr_from = {}".format(distributed_graph.adj_t_pre_post_aggr_from))
    # print("graph.adj_t_pre_post_aggr_to = {}".format(distributed_graph.adj_t_pre_post_aggr_to))
    # print("graph.pre_post_aggr_from_splits = {}".format(distributed_graph.pre_post_aggr_from_splits))
    # print("graph.pre_post_aggr_to_splits = {}".format(distributed_graph.pre_post_aggr_to_splits))
    # print("graph.in_degrees = {}".format(distributed_graph.in_degrees))
    # print("graph.send_buf.shape = {}".format(distributed_graph.comm_buf.send_buf.shape))
    # print("graph.recv_buf.shape = {}".format(distributed_graph.comm_buf.recv_buf.shape))

    return distributed_graph


def load_data(config):
    input_dir = config["input_dir"]
    graph_name = config["graph_name"]
    # is_fp16 = config['is_fp16']
    num_bits = config["num_bits"]
    # is_fp16 = True if num_bits != 32 else False
    is_pre_delay = config["is_pre_delay"]
    in_channels = config["in_channels"]
    hidden_channels = config["hidden_channels"]
    out_channels = config["out_channels"]
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    max_feat_len = max(in_channels, hidden_channels, out_channels)

    data = dict()

    (
        local_nodes_list,
        local_edges_list,
        remote_edges_list,
        nodes_range_on_each_subgraph,
        num_local_nodes,
    ) = load_graph_structures(input_dir, graph_name, rank)

    if is_pre_delay:
        # distributed_graph = get_distributed_graph_for_pre(local_edges_list, remote_edges_list, nodes_range_on_each_subgraph, \
        #                                                     num_local_nodes, max_feat_len, rank, world_size, is_fp16)
        distributed_graph = get_distributed_graph_for_pre_aggressive(
            local_edges_list,
            remote_edges_list,
            nodes_range_on_each_subgraph,
            num_local_nodes,
            max_feat_len,
            rank,
            world_size,
            num_bits,
        )
    else:
        distributed_graph = get_distributed_graph(
            local_edges_list,
            remote_edges_list,
            local_nodes_list,
            nodes_range_on_each_subgraph,
            num_local_nodes,
            max_feat_len,
            world_size,
            num_bits,
        )

    nodes_labels_list = load_nodes_labels(input_dir, graph_name, rank)
    nodes_features_list = load_nodes_features(input_dir, graph_name, rank)
    train_mask, valid_mask, test_mask = load_dataset_mask(input_dir, graph_name, rank)

    data["graph"] = distributed_graph
    data["nodes_features"] = nodes_features_list
    data["nodes_labels"] = nodes_labels_list
    data["nodes_train_masks"] = train_mask
    data["nodes_valid_masks"] = valid_mask
    data["nodes_test_masks"] = test_mask

    return data
