import sys
sys.path.append("../")

import numpy as np
import pytest

from graph_partition_scripts.postprocess_graph_multi_proc import divide_edges_into_local_and_remote
from graph_partition_scripts.postprocess_graph_multi_proc import remap_dataset_mask

@pytest.mark.parametrize(
    "edges_list, node_idx_begin, node_idx_end, expected_local_edges, expected_remote_edges",
    [
        (
            np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
            0,
            4,
            np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]),
            np.array([[], []])
        ),
        (
            np.array([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]]),
            2,
            6,
            np.array([[0, 1, 2], [2, 3, 4]]),
            np.array([[0, 1], [0, 1]])
        ),
        (
            np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
            5,
            10,
            np.array([[], []]),
            np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        ),
    ]
)
def test_divide_edges_into_local_and_remote(edges_list, node_idx_begin, node_idx_end, expected_local_edges, expected_remote_edges):
    edges_list = edges_list.T
    local_edges, remote_edges = divide_edges_into_local_and_remote(edges_list, node_idx_begin, node_idx_end)
    assert np.array_equal(local_edges, expected_local_edges)
    assert np.array_equal(remote_edges, expected_remote_edges)


@pytest.mark.parametrize(
    "data_idx, node_ids, node_idx_begin, expected_result",
    [
        # Test case 1: data_idx is empty
        (np.array([], dtype=np.int64), np.array([[10, 10], [11, 20], [12, 30]], dtype=np.int64), 10, np.array([], dtype=np.int64)),
        
        # Test case 2: data_idx contains global ids that are present in node_ids
        (np.array([20, 10, 30], dtype=np.int64), np.array([[10, 10], [11, 20], [12, 30]], dtype=np.int64), 10, np.array([1, 0, 2], dtype=np.int64)),
        
        # Test case 3: data_idx contains global ids that are not present in node_ids
        (np.array([40, 50, 60], dtype=np.int64), np.array([[10, 10], [11, 20], [12, 30]], dtype=np.int64), 10, np.array([], dtype=np.int64)),
        
        # Test case 4: data_idx contains duplicate global ids
        (np.array([20, 10, 20, 30, 10], dtype=np.int64), np.array([[0, 10], [1, 20], [2, 30]], dtype=np.int64), 0, np.array([1, 0, 1, 2, 0], dtype=np.int64))
    ]
)
def test_remap_dataset_mask(data_idx, node_ids, node_idx_begin, expected_result):
    remap_dataset_mask(data_idx, node_ids, node_idx_begin, "test_file.npy")
    assert np.array_equal(np.load("test_file.npy"), expected_result)

if __name__ == '__main__':
    pytest.main()