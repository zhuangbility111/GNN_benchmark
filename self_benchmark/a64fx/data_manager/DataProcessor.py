import torch
import numpy as np
import torch.distributed as dist
from torch_sparse import SparseTensor
import gc

class DataProcessor(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def sort_remote_edges_list_based_on_remote_nodes(remote_edges_list):
        remote_edges_row, remote_edges_col = remote_edges_list[0], remote_edges_list[1]
        sort_index = torch.argsort(remote_edges_row)
        remote_edges_list[0] = remote_edges_row[sort_index]
        remote_edges_list[1] = remote_edges_col[sort_index]
        return remote_edges_list

    @staticmethod
    def obtain_remote_nodes_list(remote_edges_list, num_local_nodes, num_nodes_on_each_subgraph, world_size):
        remote_nodes_list = list()
        range_of_remote_nodes_on_local_graph = torch.zeros(world_size+1, dtype=torch.int64)
        remote_nodes_num_from_each_subgraph = torch.zeros(world_size, dtype=torch.int64)
        remote_edges_row = remote_edges_list[0]

        part_idx = 0
        local_node_idx = num_local_nodes - 1
        prev_node = -1
        tmp_len = len(remote_edges_row)
        for i in range(0, tmp_len):
            # need to use the item() rather than the tensor as the tensor is a pointer
            cur_node = remote_edges_row[i].item()
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

        remote_nodes_list = torch.tensor(remote_nodes_list, dtype=torch.int64)

        return remote_nodes_list, range_of_remote_nodes_on_local_graph, remote_nodes_num_from_each_subgraph

    @staticmethod
    def obtain_local_nodes_required_by_other(local_nodes_list, remote_nodes_list, range_of_remote_nodes_on_local_graph, \
                                            remote_nodes_num_from_each_subgraph, world_size):
        # send the number of remote nodes we need to obtain from other subgrpah
        send_num_nodes = [torch.tensor([remote_nodes_num_from_each_subgraph[i]], dtype=torch.int64) for i in range(world_size)]
        recv_num_nodes = [torch.zeros(1, dtype=torch.int64) for i in range(world_size)]
        if world_size != 1:
            dist.all_to_all(recv_num_nodes, send_num_nodes)
        num_local_nodes_required_by_other = recv_num_nodes
        num_local_nodes_required_by_other = torch.cat(num_local_nodes_required_by_other, dim=0)
        # print("elapsed time of obtaining number of remote nodes(ms) = {}".format( \
        #         (obtain_number_remote_nodes_end - obtain_number_remote_nodes_start) * 1000))

        # then we need to send the nodes_list which include the id of remote nodes we want
        # and receive the nodes_list from other subgraphs
        send_nodes_list = [remote_nodes_list[range_of_remote_nodes_on_local_graph[i]: \
                        range_of_remote_nodes_on_local_graph[i+1]] for i in range(world_size)]
        recv_nodes_list = [torch.zeros(num_local_nodes_required_by_other[i], dtype=torch.int64) for i in range(world_size)]
        if world_size != 1:
            dist.all_to_all(recv_nodes_list, send_nodes_list)
        local_node_idx_begin = local_nodes_list[0][0]
        local_nodes_required_by_other = [i - local_node_idx_begin for i in recv_nodes_list]
        local_nodes_required_by_other = torch.cat(local_nodes_required_by_other, dim=0)
        return local_nodes_required_by_other, num_local_nodes_required_by_other

    @staticmethod
    def transform_edge_index_to_sparse_tensor(local_edges_list, remote_edges_list, num_local_nodes, num_remote_nodes):
        local_edges_list = SparseTensor(row=local_edges_list[1], col=local_edges_list[0], value=torch.ones(local_edges_list[1].shape[0], dtype=torch.float32), sparse_sizes=(num_local_nodes, num_local_nodes))
        tmp_col = remote_edges_list[0] - num_local_nodes
        remote_edges_list = SparseTensor(row=remote_edges_list[1], col=tmp_col, value=torch.ones(remote_edges_list[1].shape[0], dtype=torch.float32), sparse_sizes=(num_local_nodes, num_remote_nodes))

        return local_edges_list, remote_edges_list

class DataProcessorForPre(object):
    @staticmethod
    def get_in_degrees(local_edges_list, remote_edges_list, num_local_nodes, begin_idx_local_nodes):
        local_degs = torch.zeros((num_local_nodes), dtype=torch.int64)
        source = torch.ones((local_edges_list[1].shape[0]), dtype=torch.int64)
        local_degs.index_add_(dim=0, index=local_edges_list[1], source=source)
        source = torch.ones((remote_edges_list[1].shape[0]), dtype=torch.int64)
        tmp_index = remote_edges_list[1] - begin_idx_local_nodes
        local_degs.index_add_(dim=0, index=tmp_index, source=source)
        return local_degs.unsqueeze(-1)

    @staticmethod
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
    
    @staticmethod
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
        # print("num_diff_nodes = {}".format(num_diff_nodes))

        # communicate with other mpi ranks to get the status of pre_aggr or post_aggr 
        # and number of remote edges(pre_aggr) or remote src nodes(post_aggr)
        is_pre_post_aggr_to = [torch.zeros((3), dtype=torch.int64) for i in range(world_size)]
        dist.all_to_all(is_pre_post_aggr_to, is_pre_post_aggr_from)

        # communicate with other mpi ranks to get the remote edges(pre_aggr) 
        # or remote src nodes(post_aggr)
        remote_edges_pre_post_aggr_to = [torch.empty((indices[1]), dtype=torch.int64) for indices in is_pre_post_aggr_to]
        dist.all_to_all(remote_edges_pre_post_aggr_to, remote_edges_pre_post_aggr_from)

        remote_edges_list_pre_post_aggr_to, begin_edge_on_each_partition_to, \
        post_aggr_to_splits, pre_aggr_to_splits = \
            DataProcessorForPre.process_remote_edges_pre_post_aggr_to(is_pre_post_aggr_to, remote_edges_pre_post_aggr_to, world_size)

        del is_pre_post_aggr_from
        del is_pre_post_aggr_to
        del remote_edges_pre_post_aggr_from
        del remote_edges_pre_post_aggr_to

        return remote_edges_list_pre_post_aggr_from, remote_edges_list_pre_post_aggr_to, \
            begin_edge_on_each_partition_from, begin_edge_on_each_partition_to, \
            pre_aggr_from_splits, post_aggr_from_splits, \
            post_aggr_to_splits, pre_aggr_to_splits

    # to remap the nodes id in remote_nodes_list to local nodes id (from 0)
    # the remote nodes list must be ordered
    @staticmethod
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

    @staticmethod
    def transform_edge_index_to_sparse_tensor(local_edges_list, \
                                          remote_edges_list_pre_post_aggr_from, \
                                          remote_edges_list_pre_post_aggr_to, \
                                          begin_edge_on_each_partition_from, \
                                          begin_edge_on_each_partition_to, \
                                          num_local_nodes, \
                                          local_node_begin_idx):
        # local_edges_list has been localized
        local_adj_t = SparseTensor(row=local_edges_list[1], \
                                col=local_edges_list[0], \
                                value=torch.ones(local_edges_list[1].shape[0], dtype=torch.float32), \
                                sparse_sizes=(num_local_nodes, num_local_nodes))

        del local_edges_list
        gc.collect()

        # ----------------------------------------------------------

        # localize the dst nodes id (local nodes id)
        remote_edges_list_pre_post_aggr_from[1] -= local_node_begin_idx
        # remap (localize) the sorted src nodes id (remote nodes id) for construction of SparseTensor
        num_remote_nodes_from = DataProcessorForPre.remap_remote_nodes_id(remote_edges_list_pre_post_aggr_from[0], \
                                                                            begin_edge_on_each_partition_from)

        adj_t_pre_post_aggr_from = SparseTensor(row=remote_edges_list_pre_post_aggr_from[1], \
                                                col=remote_edges_list_pre_post_aggr_from[0], \
                                                value=torch.ones(remote_edges_list_pre_post_aggr_from[1].shape[0], dtype=torch.float32), \
                                                sparse_sizes=(num_local_nodes, num_remote_nodes_from))
        
        del remote_edges_list_pre_post_aggr_from
        del begin_edge_on_each_partition_from
        gc.collect()

        # ----------------------------------------------------------

        # localize the src nodes id (local nodes id)
        remote_edges_list_pre_post_aggr_to[0] -= local_node_begin_idx
        # remap (localize) the sorted dst nodes id (remote nodes id) for construction of SparseTensor
        num_remote_nodes_to = DataProcessorForPre.remap_remote_nodes_id(remote_edges_list_pre_post_aggr_to[1], \
                                                                        begin_edge_on_each_partition_to)

        adj_t_pre_post_aggr_to = SparseTensor(row=remote_edges_list_pre_post_aggr_to[1], \
                                            col=remote_edges_list_pre_post_aggr_to[0], \
                                            value=torch.ones(remote_edges_list_pre_post_aggr_to[1].shape[0], dtype=torch.float32), \
                                            sparse_sizes=(num_remote_nodes_to, num_local_nodes))
        del remote_edges_list_pre_post_aggr_to
        del begin_edge_on_each_partition_to
        gc.collect()
        # ----------------------------------------------------------

        return local_adj_t, adj_t_pre_post_aggr_from, adj_t_pre_post_aggr_to
