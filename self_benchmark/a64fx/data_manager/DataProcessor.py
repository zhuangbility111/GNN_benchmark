import torch
import torch.distributed as dist
from torch_sparse import SparseTensor
import gc


class DataProcessor(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_in_degrees(local_edges_list, remote_edges_list, num_local_nodes):
        local_degs = torch.zeros((num_local_nodes), dtype=torch.float32)
        source = torch.ones((local_edges_list[1].shape[0]), dtype=torch.float32)
        local_degs.index_add_(dim=0, index=local_edges_list[1], source=source)
        source = torch.ones((remote_edges_list[1].shape[0]), dtype=torch.float32)
        local_degs.index_add_(dim=0, index=remote_edges_list[1], source=source)
        return local_degs.clamp(min=1).unsqueeze(-1)

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
        range_of_remote_nodes_on_local_graph = torch.zeros(world_size + 1, dtype=torch.int64)
        remote_nodes_num_from_each_subgraph = torch.zeros(world_size, dtype=torch.int64)
        remote_edges_row = remote_edges_list[0]

        part_idx = 0
        local_node_idx = num_local_nodes - 1
        prev_node = -1
        tmp_len = remote_edges_row.shape[0]
        for i in range(0, tmp_len):
            # need to use the item() rather than the tensor as the tensor is a pointer
            cur_node = remote_edges_row[i].item()
            if cur_node != prev_node:
                remote_nodes_list.append(cur_node)
                local_node_idx += 1
                while cur_node >= num_nodes_on_each_subgraph[part_idx + 1]:
                    part_idx += 1
                    range_of_remote_nodes_on_local_graph[part_idx + 1] = range_of_remote_nodes_on_local_graph[
                        part_idx
                    ]
                range_of_remote_nodes_on_local_graph[part_idx + 1] += 1
                remote_nodes_num_from_each_subgraph[part_idx] += 1
            prev_node = cur_node
            remote_edges_row[i] = local_node_idx

        for i in range(part_idx + 1, world_size):
            range_of_remote_nodes_on_local_graph[i + 1] = range_of_remote_nodes_on_local_graph[i]

        remote_nodes_list = torch.tensor(remote_nodes_list, dtype=torch.int64)

        return remote_nodes_list, range_of_remote_nodes_on_local_graph, remote_nodes_num_from_each_subgraph

    @staticmethod
    def obtain_local_nodes_required_by_other(
        local_nodes_list,
        remote_nodes_list,
        range_of_remote_nodes_on_local_graph,
        remote_nodes_num_from_each_subgraph,
        world_size,
    ):
        # send the number of remote nodes we need to obtain from other subgrpah
        send_num_nodes = [
            torch.tensor([remote_nodes_num_from_each_subgraph[i]], dtype=torch.int64)
            for i in range(world_size)
        ]
        recv_num_nodes = [torch.zeros(1, dtype=torch.int64) for i in range(world_size)]
        if world_size != 1:
            dist.all_to_all(recv_num_nodes, send_num_nodes)
        num_local_nodes_required_by_other = recv_num_nodes
        num_local_nodes_required_by_other = torch.cat(num_local_nodes_required_by_other, dim=0)

        # then we need to send the nodes_list which include the id of remote nodes we want
        # and receive the nodes_list from other subgraphs
        send_nodes_list = [
            remote_nodes_list[
                range_of_remote_nodes_on_local_graph[i] : range_of_remote_nodes_on_local_graph[i + 1]
            ]
            for i in range(world_size)
        ]
        recv_nodes_list = [
            torch.zeros(int(num_local_nodes_required_by_other[i].item()), dtype=torch.int64)
            for i in range(world_size)
        ]
        if world_size != 1:
            dist.all_to_all(recv_nodes_list, send_nodes_list)
        local_node_idx_begin = local_nodes_list[0][0]
        local_nodes_required_by_other = [i - local_node_idx_begin for i in recv_nodes_list]
        local_nodes_required_by_other = torch.cat(local_nodes_required_by_other, dim=0)
        return local_nodes_required_by_other, num_local_nodes_required_by_other

    @staticmethod
    def transform_edge_index_to_sparse_tensor(
        local_edges_list, remote_edges_list, num_local_nodes, num_remote_nodes
    ):
        local_edges_list = SparseTensor(
            row=local_edges_list[1],
            col=local_edges_list[0],
            value=torch.ones(local_edges_list[1].shape[0], dtype=torch.float32),
            sparse_sizes=(num_local_nodes, num_local_nodes),
        )
        tmp_col = remote_edges_list[0] - num_local_nodes
        remote_edges_list = SparseTensor(
            row=remote_edges_list[1],
            col=tmp_col,
            value=torch.ones(remote_edges_list[1].shape[0], dtype=torch.float32),
            sparse_sizes=(num_local_nodes, num_remote_nodes),
        )

        return local_edges_list, remote_edges_list


class DataProcessorForPreAggresive(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_degrees(nodes_list):
        # count the number of unique nodes
        unique_nodes_list, counts = torch.unique(nodes_list, return_counts=True)
        # save the degree of each src node
        degrees = {unique_nodes_list[i].item(): counts[i].item() for i in range(len(unique_nodes_list))}
        return degrees

    @staticmethod
    def split_remote_edges_for_aggr_and_graph_exchange(
        begin_node_on_each_subgraph, remote_edges_list, world_size
    ):
        # collect remote edges which are requested from other MPI ranks
        remote_edges_sent_for_graph_exchange = []
        recv_splits_for_data_exchange = []
        begin_edge_on_each_partition_from = torch.zeros(world_size + 1, dtype=torch.int64)
        # collect remote edges which are used to compute aggregation with SPMM
        remote_edges_for_aggr_on_recv = [
            torch.empty((0), dtype=torch.int64),
            torch.empty((0), dtype=torch.int64),
        ]

        for i in range(world_size):
            # set the begin node idx and end node idx on current rank i
            begin_idx = begin_node_on_each_subgraph[i]
            end_idx = begin_node_on_each_subgraph[i + 1]

            src_in_remote_edges = remote_edges_list[0]
            dst_in_remote_edges = remote_edges_list[1]

            # get the remote edges which are from current rank i
            edge_idx = (src_in_remote_edges >= begin_idx) & (src_in_remote_edges < end_idx)
            src_in_remote_edges = src_in_remote_edges[edge_idx]
            dst_in_remote_edges = dst_in_remote_edges[edge_idx]

            is_pre = 1
            is_post = 0

            # decide pre_aggr or post_aggr for each remote edge
            pre_or_post_aggr_flags = DataProcessorForPreAggresive.decide_pre_or_post_aggr(
                src_in_remote_edges, dst_in_remote_edges, is_pre, is_post
            )

            pre_aggr_edge_idx = pre_or_post_aggr_flags == is_pre
            post_aggr_edge_idx = pre_or_post_aggr_flags == is_post

            DataProcessorForPreAggresive.collect_edges_sent_to_other_subgraphs(
                src_in_remote_edges,
                dst_in_remote_edges,
                pre_aggr_edge_idx,
                post_aggr_edge_idx,
                remote_edges_sent_for_graph_exchange,
            )

            # ----------------------------------------------------------------
            DataProcessorForPreAggresive.collect_edges_for_aggregation(
                src_in_remote_edges,
                dst_in_remote_edges,
                pre_aggr_edge_idx,
                post_aggr_edge_idx,
                remote_edges_for_aggr_on_recv,
                recv_splits_for_data_exchange,
                begin_edge_on_each_partition_from,
                i,
            )

        # ----------------------------------------------------------------
        begin_edge_on_each_partition_from[world_size] = remote_edges_for_aggr_on_recv[0].shape[0]

        return (
            remote_edges_sent_for_graph_exchange,
            remote_edges_for_aggr_on_recv,
            begin_edge_on_each_partition_from,
            recv_splits_for_data_exchange,
        )

    @staticmethod
    def split_remote_edges_recv_from_graph_exchange(remote_edges_recv_from_graph_exchange, world_size):
        remote_edges_for_aggr_for_send = [
            torch.empty((0), dtype=torch.int64),
            torch.empty((0), dtype=torch.int64),
        ]
        begin_edge_on_each_partition_to = torch.zeros(world_size + 1, dtype=torch.int64)
        send_splits_for_data_exchange = []
        for i in range(world_size):
            # the remote edges sent from other MPI ranks is divided into two parts
            # src nodes and dst nodes
            num_remote_edges = int(remote_edges_recv_from_graph_exchange[i].shape[0] / 2)
            src = remote_edges_recv_from_graph_exchange[i][:num_remote_edges]
            dst = remote_edges_recv_from_graph_exchange[i][num_remote_edges:]

            send_splits_for_data_exchange.append(torch.unique(dst).shape[0])

            # append the remote edges to the list for all_to_all communication
            remote_edges_for_aggr_for_send[0] = torch.cat((remote_edges_for_aggr_for_send[0], src), dim=0)
            remote_edges_for_aggr_for_send[1] = torch.cat((remote_edges_for_aggr_for_send[1], dst), dim=0)

            begin_edge_on_each_partition_to[i + 1] = begin_edge_on_each_partition_to[i] + dst.shape[0]
        begin_edge_on_each_partition_to[world_size] = remote_edges_for_aggr_for_send[0].shape[0]

        return remote_edges_for_aggr_for_send, begin_edge_on_each_partition_to, send_splits_for_data_exchange

    @staticmethod
    def decide_pre_or_post_aggr(src_in_remote_edges, dst_in_remote_edges, is_pre, is_post):
        # get out degrees of src nodes
        out_degrees = DataProcessorForPreAggresive.get_degrees(src_in_remote_edges)

        # get in degrees of dst nodes
        in_degrees = DataProcessorForPreAggresive.get_degrees(dst_in_remote_edges)

        pre_or_post_aggr_flags = torch.zeros(src_in_remote_edges.shape[0], dtype=torch.int64)

        # traverse the remote edges to decide pre_aggr or post_aggr
        for e_idx in range(src_in_remote_edges.shape[0]):
            src_node = src_in_remote_edges[e_idx].item()
            dst_node = dst_in_remote_edges[e_idx].item()
            # if the out degree of src node > in degree of dst node, then post_aggr
            if out_degrees[src_node] > in_degrees[dst_node]:
                pre_or_post_aggr_flags[e_idx] = is_post
            # else, pre_aggr
            else:
                pre_or_post_aggr_flags[e_idx] = is_pre

        return pre_or_post_aggr_flags

    # to collect the pre_aggr edges and post_aggr edges which will be sent to other MPI ranks
    @staticmethod
    def collect_edges_sent_to_other_subgraphs(
        src_in_remote_edges,
        dst_in_remote_edges,
        pre_aggr_edge_idx,
        post_aggr_edge_idx,
        remote_edges_sent_for_graph_exchange,
    ):
        # collect the remote edges which are pre_aggr
        src_in_pre = src_in_remote_edges[pre_aggr_edge_idx]
        dst_in_pre = dst_in_remote_edges[pre_aggr_edge_idx]

        # collect the remote nodes which are post_aggr
        src_in_post = torch.unique(src_in_remote_edges[post_aggr_edge_idx], sorted=True)

        # combine the remote edges which are pre_aggr and post_aggr
        # to send them to other MPI ranks
        src_to_send = torch.cat((src_in_pre, src_in_post), dim=0)
        dst_to_send = torch.cat((dst_in_pre, src_in_post), dim=0)

        # sort the remote edges based on the dst nodes (remote nodes on post_aggr and local nodes on pre_aggr)
        sort_index = torch.argsort(dst_to_send)
        src_to_send = src_to_send[sort_index]
        dst_to_send = dst_to_send[sort_index]

        # append the remote edges to the list for all_to_all communication
        # the remote edges is used to request the remote nodes (post_aggr)
        # or the local nodes (pre_aggr) from other MPI ranks
        remote_edges_sent_for_graph_exchange.append(torch.cat((src_to_send, dst_to_send), dim=0))

    # to collect the pre_aggr edges and post_aggr edges which will be recv from other MPI ranks
    @staticmethod
    def collect_edges_for_aggregation(
        src_in_remote_edges,
        dst_in_remote_edges,
        pre_aggr_edge_idx,
        post_aggr_edge_idx,
        remote_edges_for_aggr_on_recv,
        recv_splits_for_data_exchange,
        begin_edge_on_each_partition_from,
        cur_rank,
    ):
        # then construct the remote edges list for aggregation with SPMM later
        # sort the remote edges based on the src nodes (remote nodes on pre_aggr and local nodes on post_aggr)

        # collect the remote edges which are post_aggr
        src_in_post = src_in_remote_edges[post_aggr_edge_idx]
        dst_in_post = dst_in_remote_edges[post_aggr_edge_idx]

        dst_in_pre = dst_in_remote_edges[pre_aggr_edge_idx]

        # collect the remote nodes which are sent from other MPI ranks (the result of remote pre_aggr)
        dst_from_recv = torch.unique(dst_in_pre, sorted=True)

        # combine the src of remote edges which are pre_aggr and post_aggr
        src_from_recv = torch.cat((src_in_post, dst_from_recv), dim=0)

        # combine the dst of remote edges which are pre_aggr and post_aggr
        dst_from_recv = torch.cat((dst_in_post, dst_from_recv), dim=0)

        # sort the remote edges based on the src nodes (remote nodes on post_aggr and local nodes on pre_aggr)
        sort_index = torch.argsort(src_from_recv)
        src_from_recv = src_from_recv[sort_index]
        dst_from_recv = dst_from_recv[sort_index]

        # collect number of nodes sent from other subgraphs for all_to_all_single
        recv_splits_for_data_exchange.append(torch.unique(src_from_recv).shape[0])

        # collect the remote edges for aggregation with SPMM
        remote_edges_for_aggr_on_recv[0] = torch.cat((remote_edges_for_aggr_on_recv[0], src_from_recv), dim=0)
        remote_edges_for_aggr_on_recv[1] = torch.cat((remote_edges_for_aggr_on_recv[1], dst_from_recv), dim=0)

        # collect the begin edge idx and end edge idx on each partition
        begin_edge_on_each_partition_from[cur_rank + 1] = (
            begin_edge_on_each_partition_from[cur_rank] + src_from_recv.shape[0]
        )

    @staticmethod
    def communicate_to_get_num_remote_edges(remote_edges_sent_for_graph_exchange, world_size):
        # communicate with other mpi ranks to get the size of remote edges(pre_aggr and post_aggr)
        # num_remote_edges_pre_post_aggr_from = [torch.tensor([indices.shape[0]], dtype=torch.int64)
        #                                        for indices in remote_edges_pre_post_aggr_from]
        # num_remote_edges_pre_post_aggr_to = [torch.zeros(1, dtype=torch.int64) for _ in range(world_size)]
        num_remote_edges_send = torch.zeros(world_size, dtype=torch.int64)
        for i in range(world_size):
            num_remote_edges_send[i] = remote_edges_sent_for_graph_exchange[i].shape[0]
        num_remote_edges_recv = torch.zeros(world_size, dtype=torch.int64)
        send_splits = [1 for _ in range(world_size)]
        recv_splits = [1 for _ in range(world_size)]

        if world_size != 1:
            # dist.all_to_all(num_remote_edges_pre_post_aggr_to, num_remote_edges_pre_post_aggr_from)
            dist.all_to_all_single(
                num_remote_edges_recv,
                num_remote_edges_send,
                recv_splits,
                send_splits,
            )

        return num_remote_edges_send, num_remote_edges_recv

    @staticmethod
    def communicate_to_get_remote_edges(
        remote_edges_sent_for_graph_exchange,
        send_splits_for_graph_exchange,
        recv_splits_for_graph_exchange,
        world_size,
    ):
        # print("before transform, remote_edges_pre_post_aggr_from = {}".format(remote_edges_pre_post_aggr_from), flush=True)
        # communicate with other mpi ranks to get the remote edges(pre_aggr and post_aggr)
        # remote_edges_pre_post_aggr_to = [torch.empty((indices[0].item()), dtype=torch.int64)
        #                                  for indices in num_remote_edges_pre_post_aggr_to]
        remote_edges_recv_from_graph_exchange = torch.empty(
            (int(recv_splits_for_graph_exchange.sum().item())), dtype=torch.int64
        )
        send_splits = [indices.item() for indices in send_splits_for_graph_exchange]
        remote_edges_sent_for_graph_exchange = torch.cat(remote_edges_sent_for_graph_exchange, dim=0)
        recv_splits = [indices.item() for indices in recv_splits_for_graph_exchange]

        if world_size != 1:
            # dist.all_to_all(remote_edges_pre_post_aggr_to, remote_edges_pre_post_aggr_from)
            dist.all_to_all_single(
                remote_edges_recv_from_graph_exchange,
                remote_edges_sent_for_graph_exchange,
                recv_splits,
                send_splits,
            )

        remote_edges_recv_from_graph_exchange = torch.split(
            remote_edges_recv_from_graph_exchange, recv_splits, dim=0
        )
        return remote_edges_recv_from_graph_exchange

    @staticmethod
    def divide_remote_edges_list(begin_node_on_each_subgraph, remote_edges_list, world_size):
        (
            remote_edges_sent_for_graph_exchange,
            remote_edges_for_aggr_on_recv,
            begin_edge_on_each_partition_from,
            recv_splits_for_data_exchange,
        ) = DataProcessorForPreAggresive.split_remote_edges_for_aggr_and_graph_exchange(
            begin_node_on_each_subgraph, remote_edges_list, world_size
        )

        (
            send_splits_for_graph_exchange,
            recv_splits_for_graph_exchange,
        ) = DataProcessorForPreAggresive.communicate_to_get_num_remote_edges(
            remote_edges_sent_for_graph_exchange, world_size
        )

        remote_edges_recv_from_graph_exchange = DataProcessorForPreAggresive.communicate_to_get_remote_edges(
            remote_edges_sent_for_graph_exchange,
            send_splits_for_graph_exchange,
            recv_splits_for_graph_exchange,
            world_size,
        )

        (
            remote_edges_for_aggr_for_send,
            begin_edge_on_each_partition_to,
            send_splits_for_data_exchange,
        ) = DataProcessorForPreAggresive.split_remote_edges_recv_from_graph_exchange(
            remote_edges_recv_from_graph_exchange, world_size
        )

        del remote_edges_sent_for_graph_exchange
        del remote_edges_recv_from_graph_exchange

        return (
            remote_edges_for_aggr_on_recv,
            remote_edges_for_aggr_for_send,
            begin_edge_on_each_partition_from,
            begin_edge_on_each_partition_to,
            recv_splits_for_data_exchange,
            send_splits_for_data_exchange,
        )

    # to remap the nodes id in remote_nodes_list to local nodes id (from 0)
    # the remote nodes list must be ordered
    @staticmethod
    def remap_remote_nodes_id(remote_nodes_list, begin_edge_on_each_partition):
        local_node_idx = -1
        for rank in range(begin_edge_on_each_partition.shape[0] - 1):
            prev_node = -1
            num_nodes = begin_edge_on_each_partition[rank + 1] - begin_edge_on_each_partition[rank]
            begin_idx = begin_edge_on_each_partition[rank]
            for i in range(num_nodes):
                # Attention !!! remote_nodes_list[i] must be transformed to scalar !!!
                cur_node = remote_nodes_list[begin_idx + i].item()
                if cur_node != prev_node:
                    local_node_idx += 1
                prev_node = cur_node
                remote_nodes_list[begin_idx + i] = local_node_idx
        return local_node_idx + 1

    @staticmethod
    def transform_edge_index_to_sparse_tensor(
        local_edges_list,
        remote_edges_list_pre_post_aggr_from,
        remote_edges_list_pre_post_aggr_to,
        begin_edge_on_each_partition_from,
        begin_edge_on_each_partition_to,
        num_local_nodes,
        local_node_begin_idx,
    ):
        # local_edges_list has been localized
        local_adj_t = SparseTensor(
            row=local_edges_list[1],
            col=local_edges_list[0],
            value=torch.ones(local_edges_list[1].shape[0], dtype=torch.float32),
            sparse_sizes=(num_local_nodes, num_local_nodes),
        )

        del local_edges_list
        gc.collect()

        # ----------------------------------------------------------

        # localize the dst nodes id (local nodes id)
        remote_edges_list_pre_post_aggr_from[1] -= local_node_begin_idx
        # remap (localize) the sorted src nodes id (remote nodes id) for construction of SparseTensor
        num_remote_nodes_from = DataProcessorForPreAggresive.remap_remote_nodes_id(
            remote_edges_list_pre_post_aggr_from[0], begin_edge_on_each_partition_from
        )

        adj_t_pre_post_aggr_from = SparseTensor(
            row=remote_edges_list_pre_post_aggr_from[1],
            col=remote_edges_list_pre_post_aggr_from[0],
            value=torch.ones(remote_edges_list_pre_post_aggr_from[1].shape[0], dtype=torch.float32),
            sparse_sizes=(num_local_nodes, num_remote_nodes_from),
        )

        del remote_edges_list_pre_post_aggr_from
        del begin_edge_on_each_partition_from
        gc.collect()

        # ----------------------------------------------------------

        # localize the src nodes id (local nodes id)
        remote_edges_list_pre_post_aggr_to[0] -= local_node_begin_idx
        # remap (localize) the sorted dst nodes id (remote nodes id) for construction of SparseTensor
        num_remote_nodes_to = DataProcessorForPreAggresive.remap_remote_nodes_id(
            remote_edges_list_pre_post_aggr_to[1], begin_edge_on_each_partition_to
        )

        adj_t_pre_post_aggr_to = SparseTensor(
            row=remote_edges_list_pre_post_aggr_to[1],
            col=remote_edges_list_pre_post_aggr_to[0],
            value=torch.ones(remote_edges_list_pre_post_aggr_to[1].shape[0], dtype=torch.float32),
            sparse_sizes=(num_remote_nodes_to, num_local_nodes),
        )
        del remote_edges_list_pre_post_aggr_to
        del begin_edge_on_each_partition_to
        gc.collect()
        # ----------------------------------------------------------

        return local_adj_t, adj_t_pre_post_aggr_from, adj_t_pre_post_aggr_to
