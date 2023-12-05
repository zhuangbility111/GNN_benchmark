import sys
sys.path.append("../")
import pytest
from graph_partition_scripts.preprocess_graph import DataLoader

def test_load_ogbn_dataset():
    dataset = "ogbn-arxiv"
    raw_dir = "data/ogbn-arxiv"
    graph = DataLoader.load_ogbn_dataset(dataset, raw_dir)
    assert graph is not None
    assert graph.edge_index is not None
    assert graph.node_feat is not None
    assert graph.num_nodes is not None
    assert graph.node_label is not None
    assert graph.train_idx is not None
    assert graph.valid_idx is not None
    assert graph.test_idx is not None

# def test_load_reddit_dataset():
#     raw_dir = "data/reddit"
#     graph = DataLoader.load_reddit_dataset(raw_dir)
#     assert graph is not None
#     assert graph.edge_index is not None
#     assert graph.node_feat is not None
#     assert graph.num_nodes is not None
#     assert graph.node_label is not None
#     assert graph.train_idx is not None
#     assert graph.valid_idx is not None
#     assert graph.test_idx is not None

# def test_load_proteins_dataset():
#     graph = DataLoader.load_proteins_dataset()
#     assert graph is not None
#     assert graph.edge_index is not None
#     assert graph.node_feat is not None
#     assert graph.num_nodes is not None
#     assert graph.node_label is not None
#     assert graph.train_idx is not None
#     assert graph.valid_idx is not None
#     assert graph.test_idx is not None

def test_generate_data_for_graph():
    num_nodes = 1000
    feat_len = 128
    train_set_size = 800
    test_set_size = 100
    valid_set_size = 100
    num_labels = 10

    node_feat, node_label, train_idx, valid_idx, test_idx = DataLoader.generate_data_for_graph(
        num_nodes, feat_len, train_set_size, test_set_size, valid_set_size, num_labels
    )

    assert node_feat is not None
    assert node_label is not None
    assert train_idx is not None
    assert valid_idx is not None
    assert test_idx is not None
    assert node_feat.shape == (num_nodes, feat_len)
    assert node_label.shape == (num_nodes,)
    assert train_idx.shape == (num_nodes,)
    assert valid_idx.shape == (num_nodes,)
    assert test_idx.shape == (num_nodes,)
    assert sum(train_idx) == train_set_size
    assert sum(valid_idx) == valid_set_size
    assert sum(test_idx) == test_set_size