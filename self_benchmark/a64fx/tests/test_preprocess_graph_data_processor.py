# select the path for package
from os import remove
import sys
sys.path.append("../")

import pytest
import numpy as np
from graph_partition_scripts.preprocess_graph import Graph

@pytest.fixture
def graph():
    edge_index = np.array([[0, 0, 1, 2, 0, 3, 1], [1, 0, 2, 3, 1, 0, 2]])
    node_feat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    num_nodes = 6
    node_label = np.array([0, 1, 0, 1])
    train_idx = np.array([0, 2])
    valid_idx = np.array([1])
    test_idx = np.array([3])
    return Graph(edge_index, node_feat, num_nodes, node_label, train_idx, valid_idx, test_idx)

def test_get_in_degrees(graph):
    # Define a more complicated input
    local_edges_list = (np.array([0, 1, 2, 3, 4, 5]), np.array([0, 2, 0, 1, 2, 0, 1, 3, 4, 5]))
    num_local_nodes = 6

    # Define the expected output
    expected_local_degs = np.array([3, 2, 2, 1, 1, 1])

    # Call the function to get the actual output
    actual_local_degs = graph.get_in_degrees(local_edges_list, num_local_nodes)

    # Compare the actual and expected output
    assert np.array_equal(actual_local_degs, expected_local_degs)

def test_process_edge_index(graph):
    graph.process_edge_index()

    assert graph.edge_index[0].shape[0] == 4
    assert graph.edge_index[1].shape[0] == 4

    assert np.array_equal(graph.edge_index[0], np.array([0, 1, 2, 3]))
    assert np.array_equal(graph.edge_index[1], np.array([1, 2, 3, 0]))

    assert graph.removed_edge_index.shape[0] == 3
    assert graph.removed_edge_index.shape[1] == 4

    assert graph.edge_data.shape[0] == 4
    assert graph.edge_data.shape[1] == 4

def test_save_node_feat(graph):
    graph.save_node_feat("test_graph", "./output_dir")

    # Add assertions to check if the node features are saved successfully

def test_save_node_label(graph):
    graph.save_node_label("test_graph", "./output_dir")

    # Add assertions to check if the node labels are saved successfully

def test_save_node_mask(graph):
    graph.save_node_mask("test_graph", "./output_dir")

    # Add assertions to check if the node masks are saved successfully

def test_save_edge_index(graph):
    graph.save_edge_index("test_graph", "./output_dir")

    # Add assertions to check if the edge index is saved successfully

def test_save_nodes(graph):
    graph.save_nodes("test_graph", "./output_dir")

    # Add assertions to check if the nodes are saved successfully

def test_save_stats(graph):
    graph.save_stats("test_graph", "./output_dir")

    # Add assertions to check if the graph stats are saved successfully

# if __name__ == "__main__":
#     pytest.main()