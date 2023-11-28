import sys
sys.path.append("../")

import torch
import pytest
from data_manager.DataProcessor import DataProcessorForPreAggresive

@pytest.fixture
def data_processor_for_pre():
    return DataProcessorForPreAggresive()

def test_get_degrees(data_processor_for_pre):
    nodes_list = torch.tensor([], dtype=torch.int64)
    expected_output = {}
    output = data_processor_for_pre.get_degrees(nodes_list)
    assert output == expected_output, f"Test case 'empty_nodes_list' failed. Expected: {expected_output}, Got: {output}"

    nodes_list = torch.tensor([1], dtype=torch.int64)
    expected_output = {1: 1}
    output = data_processor_for_pre.get_degrees(nodes_list)
    assert output == expected_output, f"Test case 'single_node' failed. Expected: {expected_output}, Got: {output}"

    nodes_list = torch.tensor([1, 2, 3, 2, 1, 3, 3], dtype=torch.int64)
    expected_output = {1: 2, 2: 2, 3: 3}
    output = data_processor_for_pre.get_degrees(nodes_list)
    assert output == expected_output, f"Test case 'multiple_nodes' failed. Expected: {expected_output}, Got: {output}"

    nodes_list = torch.randint(0, 100, (10000,), dtype=torch.int64)
    expected_output = {node.item(): nodes_list.tolist().count(node) for node in torch.unique(nodes_list)}
    output = data_processor_for_pre.get_degrees(nodes_list)
    assert output == expected_output, f"Test case 'large_number_of_nodes' failed. Expected: {expected_output}, Got: {output}"

def test_decide_pre_or_post_aggr(data_processor_for_pre):
    is_pre = 1
    is_post = 0

    # Test case 1: Out degree > In degree, should return post_aggr flag
    src_in_remote_edges_1 = torch.tensor([1, 3, 1, 1])
    dst_in_remote_edges_1 = torch.tensor([2, 1, 2, 3])
    expected_result_1 = torch.tensor([is_post, is_pre, is_post, is_post])
    result_1 = data_processor_for_pre.decide_pre_or_post_aggr(src_in_remote_edges_1, dst_in_remote_edges_1, is_pre, is_post)
    assert torch.equal(result_1, expected_result_1), "Test case 1 failed"

    # Test case 2: Out degree <= In degree, should return pre_aggr flag
    src_in_remote_edges_2 = torch.tensor([2, 1, 2, 1, 0])
    dst_in_remote_edges_2 = torch.tensor([1, 4, 1, 1, 1])
    expected_result_2 = torch.tensor([is_pre, is_post, is_pre, is_pre, is_pre])
    result_2 = data_processor_for_pre.decide_pre_or_post_aggr(src_in_remote_edges_2, dst_in_remote_edges_2, is_pre, is_post)
    assert torch.equal(result_2, expected_result_2), "Test case 2 failed"

    # Test case 3: Empty input, should return empty result
    src_in_remote_edges_3 = torch.tensor([], dtype=torch.int64)
    dst_in_remote_edges_3 = torch.tensor([], dtype=torch.int64)
    expected_result_3 = torch.tensor([], dtype=torch.int64)
    result_3 = data_processor_for_pre.decide_pre_or_post_aggr(src_in_remote_edges_3, dst_in_remote_edges_3, is_pre, is_post)
    assert torch.equal(result_3, expected_result_3), "Test case 3 failed"

    src_in_remote_edges_4 = torch.tensor([0, 2, 2, 2, 4, 4, 5, 5])
    dst_in_remote_edges_4 = torch.tensor([1, 1, 2, 4, 4, 3, 6, 3])
    expected_result_4 = torch.tensor([is_pre, is_post, is_post, is_post, is_pre, is_pre, is_post, is_pre])
    result_4 = data_processor_for_pre.decide_pre_or_post_aggr(src_in_remote_edges_4, dst_in_remote_edges_4, is_pre, is_post)
    assert torch.equal(result_4, expected_result_4), "Test case 4 failed"

def test_collect_edges_sent_to_other_subgraphs(data_processor_for_pre):
    # Test case 1: Both pre-aggregated and post-aggregated edges
    src_in_remote_edges = torch.tensor([0, 2, 2, 2, 4, 4, 5, 5])
    dst_in_remote_edges = torch.tensor([1, 1, 2, 4, 4, 3, 6, 3])
    pre_aggr_edge_idx = torch.tensor([0, 4, 5, 7])
    post_aggr_edge_idx = torch.tensor([1, 2, 3, 6])
    remote_edges_pre_post_aggr_from = []
    expected_src_to_send = torch.tensor([0, 2, 4, 5, 4, 5])
    expected_dst_to_send = torch.tensor([1, 2, 3, 3, 4, 5])
    data_processor_for_pre.collect_edges_sent_to_other_subgraphs(src_in_remote_edges, dst_in_remote_edges, pre_aggr_edge_idx, post_aggr_edge_idx, remote_edges_pre_post_aggr_from)
    print(remote_edges_pre_post_aggr_from)
    num_post = torch.unique(src_in_remote_edges[post_aggr_edge_idx]).shape[0]
    src_in_remote_edges_pre_aggr = remote_edges_pre_post_aggr_from[0][: pre_aggr_edge_idx.shape[0] + num_post]
    dst_in_remote_edges_pre_aggr = remote_edges_pre_post_aggr_from[0][pre_aggr_edge_idx.shape[0] + num_post: ]
    assert torch.equal(src_in_remote_edges_pre_aggr, expected_src_to_send), "Test case 1 failed for src_in_remote_edges_pre_aggr"
    assert torch.equal(dst_in_remote_edges_pre_aggr, expected_dst_to_send), "Test case 1 failed for dst_in_remote_edges_pre_aggr"

    # Test case 2: Pre-aggregated edges only
    src_in_remote_edges = torch.tensor([0, 2, 2, 2, 4, 4, 5, 5])
    dst_in_remote_edges = torch.tensor([1, 1, 2, 4, 4, 3, 6, 3])
    pre_aggr_edge_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    post_aggr_edge_idx = torch.tensor([], dtype=torch.int64)
    remote_edges_pre_post_aggr_from = []
    expected_src_to_send = torch.tensor([0, 2, 2, 4, 5, 2, 4, 5])
    expected_dst_to_send = torch.tensor([1, 1, 2, 3, 3, 4, 4, 6])
    data_processor_for_pre.collect_edges_sent_to_other_subgraphs(src_in_remote_edges, dst_in_remote_edges, pre_aggr_edge_idx, post_aggr_edge_idx, remote_edges_pre_post_aggr_from)
    num_post = torch.unique(src_in_remote_edges[post_aggr_edge_idx]).shape[0]
    src_in_remote_edges_pre_aggr = remote_edges_pre_post_aggr_from[0][: pre_aggr_edge_idx.shape[0] + num_post]
    dst_in_remote_edges_pre_aggr = remote_edges_pre_post_aggr_from[0][pre_aggr_edge_idx.shape[0] + num_post: ]
    assert torch.equal(src_in_remote_edges_pre_aggr, expected_src_to_send), "Test case 2 failed for src_in_remote_edges_pre_aggr"
    assert torch.equal(dst_in_remote_edges_pre_aggr, expected_dst_to_send), "Test case 2 failed for dst_in_remote_edges_pre_aggr"

    # Test case 3: Post-aggregated edges only
    src_in_remote_edges = torch.tensor([0, 2, 9, 3, 4, 1, 5, 3])
    dst_in_remote_edges = torch.tensor([1, 1, 2, 4, 4, 3, 6, 3])
    pre_aggr_edge_idx = torch.tensor([], dtype=torch.int64)
    post_aggr_edge_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    remote_edges_pre_post_aggr_from = []
    expected_src_to_send = torch.tensor([0, 1, 2, 3, 4, 5, 9])
    expected_dst_to_send = torch.tensor([0, 1, 2, 3, 4, 5, 9])
    data_processor_for_pre.collect_edges_sent_to_other_subgraphs(src_in_remote_edges, dst_in_remote_edges, pre_aggr_edge_idx, post_aggr_edge_idx, remote_edges_pre_post_aggr_from)
    num_post = torch.unique(src_in_remote_edges[post_aggr_edge_idx]).shape[0]
    src_in_remote_edges_pre_aggr = remote_edges_pre_post_aggr_from[0][: pre_aggr_edge_idx.shape[0] + num_post]
    dst_in_remote_edges_pre_aggr = remote_edges_pre_post_aggr_from[0][pre_aggr_edge_idx.shape[0] + num_post: ]
    assert torch.equal(src_in_remote_edges_pre_aggr, expected_src_to_send), "Test case 3 failed for src_in_remote_edges_pre_aggr"
    assert torch.equal(dst_in_remote_edges_pre_aggr, expected_dst_to_send), "Test case 3 failed for dst_in_remote_edges_pre_aggr"

def test_collect_edges_for_aggregation(data_processor_for_pre):
    src_in_remote_edges = torch.tensor([1, 1, 2, 4, 4, 3, 6, 3])
    dst_in_remote_edges = torch.tensor([0, 2, 2, 2, 4, 4, 5, 5])
    pre_aggr_edge_idx = torch.tensor([1, 2, 3, 6])
    post_aggr_edge_idx = torch.tensor([0, 4, 5, 7])
    remote_edges_list_pre_post_aggr_from = [torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)]
    pre_post_aggr_from_splits = []
    begin_edge_on_each_partition_from = torch.zeros(2, dtype=torch.int64)
    cur_rank = 0

    data_processor_for_pre.collect_edges_for_aggregation(src_in_remote_edges, dst_in_remote_edges, pre_aggr_edge_idx, post_aggr_edge_idx,
                                                        remote_edges_list_pre_post_aggr_from, pre_post_aggr_from_splits,
                                                        begin_edge_on_each_partition_from, cur_rank)

    expected_src_from_recv = torch.tensor([1, 2, 3, 3, 4, 5])
    expected_dst_from_recv = torch.tensor([0, 2, 4, 5, 4, 5])
    expected_pre_post_aggr_from_splits = [5]
    expected_begin_edge_on_each_partition_from = torch.tensor([0, 6], dtype=torch.int64)

    assert torch.equal(remote_edges_list_pre_post_aggr_from[0], expected_src_from_recv), "Test case failed for src_from_recv"
    assert torch.equal(remote_edges_list_pre_post_aggr_from[1], expected_dst_from_recv), "Test case failed for dst_from_recv"
    assert pre_post_aggr_from_splits == expected_pre_post_aggr_from_splits, "Test case failed for pre_post_aggr_from_splits"
    assert torch.equal(begin_edge_on_each_partition_from, expected_begin_edge_on_each_partition_from), "Test case failed for begin_edge_on_each_partition_from"


    # post-aggregated edges only
    src_in_remote_edges = torch.tensor([1, 1, 2, 4, 4, 3, 6, 3])
    dst_in_remote_edges = torch.tensor([0, 2, 2, 2, 4, 4, 5, 5])
    pre_aggr_edge_idx = torch.tensor([], dtype=torch.int64) 
    post_aggr_edge_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    remote_edges_list_pre_post_aggr_from = [torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)]
    pre_post_aggr_from_splits = []
    begin_edge_on_each_partition_from = torch.zeros(2, dtype=torch.int64)
    cur_rank = 0

    data_processor_for_pre.collect_edges_for_aggregation(src_in_remote_edges, dst_in_remote_edges, pre_aggr_edge_idx, post_aggr_edge_idx,
                                                        remote_edges_list_pre_post_aggr_from, pre_post_aggr_from_splits,
                                                        begin_edge_on_each_partition_from, cur_rank)

    expected_src_from_recv = torch.tensor([1, 1, 2, 3, 3, 4, 4, 6])
    expected_dst_from_recv = torch.tensor([0, 2, 2, 4, 5, 2, 4, 5])
    expected_pre_post_aggr_from_splits = [5]
    expected_begin_edge_on_each_partition_from = torch.tensor([0, 8], dtype=torch.int64)

    assert torch.equal(remote_edges_list_pre_post_aggr_from[0], expected_src_from_recv), "Test case failed for src_from_recv"
    assert torch.equal(remote_edges_list_pre_post_aggr_from[1], expected_dst_from_recv), "Test case failed for dst_from_recv"
    assert pre_post_aggr_from_splits == expected_pre_post_aggr_from_splits, "Test case failed for pre_post_aggr_from_splits"
    assert torch.equal(begin_edge_on_each_partition_from, expected_begin_edge_on_each_partition_from), "Test case failed for begin_edge_on_each_partition_from"

    # pre-aggregated edges only
    src_in_remote_edges = torch.tensor([1, 1, 2, 4, 4, 3, 6, 3])
    dst_in_remote_edges = torch.tensor([0, 2, 9, 3, 4, 1, 5, 3])
    pre_aggr_edge_idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    post_aggr_edge_idx = torch.tensor([], dtype=torch.int64) 
    remote_edges_list_pre_post_aggr_from = [torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)]
    pre_post_aggr_from_splits = []
    begin_edge_on_each_partition_from = torch.zeros(2, dtype=torch.int64)
    cur_rank = 0

    data_processor_for_pre.collect_edges_for_aggregation(src_in_remote_edges, dst_in_remote_edges, pre_aggr_edge_idx, post_aggr_edge_idx,
                                                        remote_edges_list_pre_post_aggr_from, pre_post_aggr_from_splits,
                                                        begin_edge_on_each_partition_from, cur_rank)

    expected_src_from_recv = torch.tensor([0, 1, 2, 3, 4, 5, 9])
    expected_dst_from_recv = torch.tensor([0, 1, 2, 3, 4, 5, 9])
    expected_pre_post_aggr_from_splits = [7]
    expected_begin_edge_on_each_partition_from = torch.tensor([0, 7], dtype=torch.int64)

    assert torch.equal(remote_edges_list_pre_post_aggr_from[0], expected_src_from_recv), "Test case failed for src_from_recv"
    assert torch.equal(remote_edges_list_pre_post_aggr_from[1], expected_dst_from_recv), "Test case failed for dst_from_recv"
    assert pre_post_aggr_from_splits == expected_pre_post_aggr_from_splits, "Test case failed for pre_post_aggr_from_splits"
    assert torch.equal(begin_edge_on_each_partition_from, expected_begin_edge_on_each_partition_from), "Test case failed for begin_edge_on_each_partition_from"

def test_split_remote_edges_for_aggr_and_graph_exchange(data_processor_for_pre):
    begin_node_on_each_subgraph = torch.tensor([0, 2, 5, 10, 14], dtype=torch.int64)
    remote_edges_list = [
        torch.tensor([8, 9, 3, 6, 6, 7, 1, 0, 3, 5]),
        torch.tensor([9, 7, 10, 8, 4, 7, 10, 10, 8, 11])
    ]
    world_size = 4

    (
        remote_edges_sent_for_graph_exchange,
        remote_edges_for_aggr_on_recv,
        begin_edge_on_each_partition_from,
        recv_splits_for_data_exchange,
    ) = data_processor_for_pre.split_remote_edges_for_aggr_and_graph_exchange(
        begin_node_on_each_subgraph, remote_edges_list, world_size
    )

    # Test remote_edges_sent_for_graph_exchange
    expected_remote_edges_sent_for_graph_exchange = [
        torch.tensor([1, 0, 10, 10]),
        torch.tensor([3, 3]),
        torch.tensor([6, 9, 7, 8, 5, 6, 7, 7, 9, 11]),
        torch.tensor([], dtype=torch.int64)
    ]
    assert len(remote_edges_sent_for_graph_exchange) == len(expected_remote_edges_sent_for_graph_exchange)
    for i in range(len(remote_edges_sent_for_graph_exchange)):
        assert torch.equal(remote_edges_sent_for_graph_exchange[i], expected_remote_edges_sent_for_graph_exchange[i])

    # Test remote_edges_for_aggr_on_recv
    expected_remote_edges_for_aggr_on_recv = [
        torch.tensor([10, 3, 3, 6, 6, 7, 9, 11]),
        torch.tensor([10, 10, 8, 8, 4, 7, 9, 11]),
    ]
    assert len(remote_edges_for_aggr_on_recv) == len(expected_remote_edges_for_aggr_on_recv)
    for i in range(len(remote_edges_for_aggr_on_recv)):
        assert torch.equal(remote_edges_for_aggr_on_recv[i], expected_remote_edges_for_aggr_on_recv[i])

    # Test begin_edge_on_each_partition_from
    expected_begin_edge_on_each_partition_from = torch.tensor([0, 1, 3, 8, 8], dtype=torch.int64)
    assert torch.equal(begin_edge_on_each_partition_from, expected_begin_edge_on_each_partition_from)

    # Test recv_splits_for_data_exchange
    expected_recv_splits_for_data_exchange = [1, 1, 4, 0]
    assert recv_splits_for_data_exchange == expected_recv_splits_for_data_exchange


def test_split_remote_edges_recv_from_graph_exchange(data_processor_for_pre):
    remote_edges_pre_post_aggr_to = [
        torch.tensor([], dtype=torch.int64),
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([7, 8, 9, 10, 11, 12])
    ]
    world_size = 3

    remote_edges_list_pre_post_aggr_to, begin_edge_on_each_partition_to, pre_post_aggr_to_splits = data_processor_for_pre.split_remote_edges_recv_from_graph_exchange(remote_edges_pre_post_aggr_to, world_size)

    expected_remote_edges_list_pre_post_aggr_to = [
        torch.tensor([1, 2, 7, 8, 9]),
        torch.tensor([3, 4, 10, 11, 12])
    ]
    expected_begin_edge_on_each_partition_to = torch.tensor([0, 0, 2, 5], dtype=torch.int64)
    expected_pre_post_aggr_to_splits = [0, 2, 3]

    assert torch.equal(remote_edges_list_pre_post_aggr_to[0], expected_remote_edges_list_pre_post_aggr_to[0]), "Test case failed for remote_edges_list_pre_post_aggr_to[0]"
    assert torch.equal(remote_edges_list_pre_post_aggr_to[1], expected_remote_edges_list_pre_post_aggr_to[1]), "Test case failed for remote_edges_list_pre_post_aggr_to[1]"
    assert torch.equal(begin_edge_on_each_partition_to, expected_begin_edge_on_each_partition_to), "Test case failed for begin_edge_on_each_partition_to"
    assert pre_post_aggr_to_splits == expected_pre_post_aggr_to_splits, "Test case failed for pre_post_aggr_to_splits"


    remote_edges_pre_post_aggr_to = [
        torch.tensor([2, 2, 3, 4]),
        torch.tensor([], dtype=torch.int64),
        torch.tensor([7, 8, 9, 9])
    ]
    world_size = 3

    remote_edges_list_pre_post_aggr_to, begin_edge_on_each_partition_to, pre_post_aggr_to_splits = data_processor_for_pre.split_remote_edges_recv_from_graph_exchange(remote_edges_pre_post_aggr_to, world_size)

    expected_remote_edges_list_pre_post_aggr_to = [
        torch.tensor([2, 2, 7, 8]),
        torch.tensor([3, 4, 9, 9])
    ]
    expected_begin_edge_on_each_partition_to = torch.tensor([0, 2, 2, 4], dtype=torch.int64)
    expected_pre_post_aggr_to_splits = [2, 0, 1]

    assert torch.equal(remote_edges_list_pre_post_aggr_to[0], expected_remote_edges_list_pre_post_aggr_to[0]), "Test case failed for remote_edges_list_pre_post_aggr_to[0]"
    assert torch.equal(remote_edges_list_pre_post_aggr_to[1], expected_remote_edges_list_pre_post_aggr_to[1]), "Test case failed for remote_edges_list_pre_post_aggr_to[1]"
    assert torch.equal(begin_edge_on_each_partition_to, expected_begin_edge_on_each_partition_to), "Test case failed for begin_edge_on_each_partition_to"
    assert pre_post_aggr_to_splits == expected_pre_post_aggr_to_splits, "Test case failed for pre_post_aggr_to_splits"

    remote_edges_pre_post_aggr_to = [
        torch.tensor([], dtype=torch.int64),
        torch.tensor([], dtype=torch.int64),
    ]
    world_size = 2

    remote_edges_list_pre_post_aggr_to, begin_edge_on_each_partition_to, pre_post_aggr_to_splits = data_processor_for_pre.split_remote_edges_recv_from_graph_exchange(remote_edges_pre_post_aggr_to, world_size)

    expected_remote_edges_list_pre_post_aggr_to = [
        torch.tensor([], dtype=torch.int64),
        torch.tensor([], dtype=torch.int64)
    ]
    expected_begin_edge_on_each_partition_to = torch.tensor([0, 0, 0], dtype=torch.int64)
    expected_pre_post_aggr_to_splits = [0, 0]

    assert torch.equal(remote_edges_list_pre_post_aggr_to[0], expected_remote_edges_list_pre_post_aggr_to[0]), "Test case failed for remote_edges_list_pre_post_aggr_to[0]"
    assert torch.equal(remote_edges_list_pre_post_aggr_to[1], expected_remote_edges_list_pre_post_aggr_to[1]), "Test case failed for remote_edges_list_pre_post_aggr_to[1]"
    assert torch.equal(begin_edge_on_each_partition_to, expected_begin_edge_on_each_partition_to), "Test case failed for begin_edge_on_each_partition_to"
    assert pre_post_aggr_to_splits == expected_pre_post_aggr_to_splits, "Test case failed for pre_post_aggr_to_splits"

def test_remap_remote_nodes_id(data_processor_for_pre):
    remote_nodes_list = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    begin_node_on_each_partition = torch.tensor([0, 2, 4, 6, 8])

    expected_output = 4
    output = data_processor_for_pre.remap_remote_nodes_id(remote_nodes_list, begin_node_on_each_partition)
    assert output == expected_output, f"Test case 1 failed. Expected: {expected_output}, Got: {output}"
    assert torch.equal(remote_nodes_list, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])), "Test case 1 failed. The input remote_nodes_list should not be modified."

    remote_nodes_list = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    begin_node_on_each_partition = torch.tensor([0, 2, 4, 6, 8, 10])

    expected_output = 5
    output = data_processor_for_pre.remap_remote_nodes_id(remote_nodes_list, begin_node_on_each_partition)
    assert output == expected_output, f"Test case 2 failed. Expected: {expected_output}, Got: {output}"
    assert torch.equal(remote_nodes_list, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])), "Test case 2 failed. The input remote_nodes_list should not be modified."

    remote_nodes_list = torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3, 3])
    begin_node_on_each_partition = torch.tensor([0, 3, 5, 7, 10])

    expected_output = 4
    output = data_processor_for_pre.remap_remote_nodes_id(remote_nodes_list, begin_node_on_each_partition)
    assert output == expected_output, f"Test case 3 failed. Expected: {expected_output}, Got: {output}"
    assert torch.equal(remote_nodes_list, torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3, 3])), "Test case 3 failed. The input remote_nodes_list should not be modified."

    remote_nodes_list = torch.tensor([0, 1, 2, 3, 4, 5])
    begin_node_on_each_partition = torch.tensor([0, 1, 2, 3, 4, 5, 6])

    expected_output = 6
    output = data_processor_for_pre.remap_remote_nodes_id(remote_nodes_list, begin_node_on_each_partition)
    assert output == expected_output, f"Test case 4 failed. Expected: {expected_output}, Got: {output}"
    assert torch.equal(remote_nodes_list, torch.tensor([0, 1, 2, 3, 4, 5])), "Test case 4 failed. The input remote_nodes_list should not be modified."

    remote_nodes_list = torch.tensor([0, 0, 0, 0, 0])
    begin_node_on_each_partition = torch.tensor([0, 5])

    expected_output = 1
    output = data_processor_for_pre.remap_remote_nodes_id(remote_nodes_list, begin_node_on_each_partition)
    assert output == expected_output, f"Test case 5 failed. Expected: {expected_output}, Got: {output}"
    assert torch.equal(remote_nodes_list, torch.tensor([0, 0, 0, 0, 0])), "Test case 5 failed. The input remote_nodes_list should not be modified."


    remote_nodes_list = torch.tensor([0, 0, 2, 2, 4, 5, 10, 11, 11, 12])
    begin_node_on_each_partition = torch.tensor([0, 2, 4, 6, 9, 10])

    expected_output = 7
    output = data_processor_for_pre.remap_remote_nodes_id(remote_nodes_list, begin_node_on_each_partition)
    assert output == expected_output, f"Test case 2 failed. Expected: {expected_output}, Got: {output}"
    assert torch.equal(remote_nodes_list, torch.tensor([0, 0, 1, 1, 2, 3, 4, 5, 5, 6])), "Test case 6 failed. The input remote_nodes_list should not be modified."

    remote_nodes_list = torch.tensor([0, 0, 2, 2, 4, 5, 10, 11, 13, 13])
    begin_node_on_each_partition = torch.tensor([0, 0, 2, 4, 6, 8, 10])

    expected_output = 7
    output = data_processor_for_pre.remap_remote_nodes_id(remote_nodes_list, begin_node_on_each_partition)
    assert output == expected_output, f"Test case 2 failed. Expected: {expected_output}, Got: {output}"
    assert torch.equal(remote_nodes_list, torch.tensor([0, 0, 1, 1, 2, 3, 4, 5, 6, 6])), "Test case 7 failed. The input remote_nodes_list should not be modified."

    remote_nodes_list = torch.tensor([3, 3, 4, 7, 7, 9])
    begin_node_on_each_partition = torch.tensor([0, 0, 2, 3, 5, 6])

    expected_output = 4
    output = data_processor_for_pre.remap_remote_nodes_id(remote_nodes_list, begin_node_on_each_partition)
    assert output == expected_output, f"Test case 2 failed. Expected: {expected_output}, Got: {output}"
    assert torch.equal(remote_nodes_list, torch.tensor([0, 0, 1, 2, 2, 3])), "Test case 7 failed. The input remote_nodes_list should not be modified."
