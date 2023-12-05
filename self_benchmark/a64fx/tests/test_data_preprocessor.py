import sys
sys.path.append("../")
import torch
import pytest
from data_manager.DataProcessor import DataProcessor

@pytest.fixture
def data_processor():
    return DataProcessor()

def test_sort_remote_edges_list_based_on_remote_nodes(data_processor):
    remote_edges_list = torch.tensor([[3, 2, 1, 4, 0, 0], [6, 5, 8, 7, 9, 10]])
    sorted_edges_list = data_processor.sort_remote_edges_list_based_on_remote_nodes(remote_edges_list)
    assert torch.equal(sorted_edges_list, torch.tensor([[0, 0, 1, 2, 3, 4], [9, 10, 8, 5, 6, 7]]))

def test_obtain_remote_nodes_list(data_processor):
    remote_edges_list = torch.tensor([[2, 5, 5, 4, 4, 4, 1], [6, 5, 7, 8, 9, 10, 8]])
    sorted_edges_list = data_processor.sort_remote_edges_list_based_on_remote_nodes(remote_edges_list)
    num_local_nodes = 2
    num_nodes_on_each_subgraph = torch.tensor([0, 3, 6, 9])
    world_size = 3
    remote_nodes_list, range_of_remote_nodes_on_local_graph, remote_nodes_num_from_each_subgraph = \
        data_processor.obtain_remote_nodes_list(sorted_edges_list, num_local_nodes, num_nodes_on_each_subgraph, world_size)
    assert torch.equal(remote_edges_list, torch.tensor([[2, 3, 4, 4, 4, 5, 5], [8, 6, 8, 9, 10, 5, 7]]))
    assert torch.equal(remote_nodes_list, torch.tensor([1, 2, 4, 5]))
    assert torch.equal(range_of_remote_nodes_on_local_graph, torch.tensor([0, 2, 4, 4]))
    assert torch.equal(remote_nodes_num_from_each_subgraph, torch.tensor([2, 2, 0]))

# def test_obtain_local_nodes_required_by_other(data_processor):
#     local_nodes_list = [torch.tensor([1, 2, 3, 4, 5])]
#     remote_nodes_list = torch.tensor([2, 3, 4])
#     range_of_remote_nodes_on_local_graph = torch.tensor([0, 0, 1, 2])
#     remote_nodes_num_from_each_subgraph = torch.tensor([1, 2, 3])
#     world_size = 3
#     local_nodes_required_by_other, num_local_nodes_required_by_other = \
#         data_processor.obtain_local_nodes_required_by_other(local_nodes_list, remote_nodes_list,
#                                                             range_of_remote_nodes_on_local_graph,
#                                                             remote_nodes_num_from_each_subgraph, world_size)
#     assert local_nodes_required_by_other == torch.tensor([1, 2, 3])
#     assert num_local_nodes_required_by_other == torch.tensor([1, 2, 3])

def test_transform_edge_index_to_sparse_tensor(data_processor):
    local_edges_list = [torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])]
    remote_edges_list = [torch.tensor([7, 7, 7]), torch.tensor([2, 0, 1])]
    num_local_nodes = 5
    num_remote_nodes = 3
    local_adj_t, remote_adj_t = data_processor.transform_edge_index_to_sparse_tensor(local_edges_list, remote_edges_list,
                                                                        num_local_nodes, num_remote_nodes)
    expected_local_adj_t = torch.sparse_coo_tensor(
        torch.tensor([[2, 3, 4], [1, 2, 3]]),
        torch.tensor([1., 1., 1.]),
        size=(5, 5)
    )

    expected_remote_adj_t = torch.sparse_coo_tensor(
        torch.tensor([[2, 0, 1], [2, 2, 2]]),
        torch.tensor([1., 1., 1.]),
        size=(5, 3)
    )
    assert torch.equal(local_adj_t.to_dense(), expected_local_adj_t.to_dense())
    assert torch.equal(remote_adj_t.to_dense(), expected_remote_adj_t.to_dense())

    #########################################################

    local_edges_list = torch.tensor([[0, 1, 1], [0, 0, 1]])
    remote_edges_list = torch.tensor([[2, 5, 5, 4, 4, 4, 2], [0, 0, 0, 1, 0, 1, 0]])
    remote_edges_list = data_processor.sort_remote_edges_list_based_on_remote_nodes(remote_edges_list)
    num_local_nodes = 2
    num_remote_nodes = 3
    num_nodes_on_each_subgraph = torch.tensor([0, 3, 6, 9])
    world_size = 3
    remote_nodes_list, range_of_remote_nodes_on_local_graph, remote_nodes_num_from_each_subgraph = \
        data_processor.obtain_remote_nodes_list(remote_edges_list, num_local_nodes, num_nodes_on_each_subgraph, world_size)
    
    assert torch.equal(remote_edges_list, torch.tensor([[2, 2, 3, 3, 3, 4, 4], [0, 0, 1, 0, 1, 0, 0]]))
    
    local_adj_t, remote_adj_t = data_processor.transform_edge_index_to_sparse_tensor(local_edges_list, remote_edges_list,
                                                                        num_local_nodes, num_remote_nodes)
    
    expected_local_adj_t = torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 1], [0, 1, 1]]),
        torch.tensor([1., 1., 1.]),
        size=(2, 2)
    )

    expected_remote_adj_t = torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 1, 0], [0, 1, 1, 2]]),
        torch.tensor([1., 1., 1., 1.]),
        size=(2, 3)
    )

    assert torch.equal(local_adj_t.to_dense(), expected_local_adj_t.to_dense())
    assert torch.equal(remote_adj_t.to_dense(), expected_remote_adj_t.to_dense())
                                            

def test_get_in_degrees(data_processor):
    local_edges_list = [torch.tensor([1, 2, 3]), torch.tensor([0, 0, 1])]
    remote_edges_list = [torch.tensor([2, 3, 4]), torch.tensor([2, 3, 3])]
    num_local_nodes = 5
    local_degs = data_processor.get_in_degrees(local_edges_list, remote_edges_list, num_local_nodes)
    expected_local_degs = torch.tensor([[2.], [1.], [1.], [2.], [1.]])
    assert torch.equal(local_degs, expected_local_degs)