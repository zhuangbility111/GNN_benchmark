import argparse
import os
import gc
import torch
import numpy as np
from scipy.io import mmread
import requests
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.datasets import Reddit


class Graph(object):
    """
    Class to store the data such as edge index, node features, node labels, training, validation and test set
    """

    # constructor
    def __init__(
        self,
        edge_index: np.ndarray,
        node_feat: np.ndarray,
        num_nodes: int,
        node_label: np.ndarray,
        train_idx: np.ndarray,
        valid_idx: np.ndarray,
        test_idx: np.ndarray,
    ):
        self.edge_index = edge_index
        self.node_feat = node_feat
        self.num_nodes = num_nodes
        self.num_edges = edge_index[0].shape[0]
        self.num_node_weights = 0
        self.node_label = node_label
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx

    def get_in_degrees(self, local_edges_list, num_local_nodes):
        """
        to get the in-degree of each node for load balancing in graph partitioning
        """
        local_degs = torch.zeros((num_local_nodes), dtype=torch.int64)
        source = torch.ones((local_edges_list[1].shape[0]), dtype=torch.int64)
        index = torch.from_numpy(local_edges_list[1])
        local_degs.index_add_(dim=0, index=index, source=source)
        return local_degs

    def save_node_feat(self, graph_name: str, out_dir: str):
        """
        to save node features
        """
        # if the out_dir does not exist, create it
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        np.save(os.path.join(out_dir, "{}_nodes_feat.npy".format(graph_name)), self.node_feat)
        print("save node features successfully.")
        del self.node_feat
        gc.collect()

    def save_node_label(self, graph_name: str, out_dir: str):
        """
        to save node labels
        """
        np.save(os.path.join(out_dir, "{}_nodes_label.npy".format(graph_name)), self.node_label)
        print("save node labels successfully.")

    def save_node_mask(self, graph_name: str, out_dir: str):
        """
        to save node masks (training, validation and test set)
        """
        np.save(os.path.join(out_dir, "{}_nodes_train_idx.npy".format(graph_name)), self.train_idx)
        np.save(os.path.join(out_dir, "{}_nodes_valid_idx.npy".format(graph_name)), self.valid_idx)
        np.save(os.path.join(out_dir, "{}_nodes_test_idx.npy".format(graph_name)), self.test_idx)
        print("save training, valid, test idx successfully.")

    def process_edge_index(self):
        """
        to process edge index, remove self loop and duplicated edges
        """
        src_id, dst_id = self.edge_index
        original_edge_id = np.arange(src_id.shape[0], dtype=np.int64)
        print("length of src_idx before removing self loop = {}".format(src_id.shape[0]))

        # remove self loop
        self_loop_idx = src_id == dst_id
        not_self_loop_idx = src_id != dst_id

        self_loop_src_id = src_id[self_loop_idx]
        self_loop_dst_id = dst_id[self_loop_idx]
        self_loop_original_edge_id = original_edge_id[self_loop_idx]

        src_id = src_id[not_self_loop_idx]
        dst_id = dst_id[not_self_loop_idx]
        original_edge_id = original_edge_id[not_self_loop_idx]
        print("length of src_idx after removing self loop = {}".format(src_id.shape[0]))

        # remove duplicated edges
        print("length of src_idx before removing duplicated edges = {}".format(src_id.shape[0]))
        ids = src_id * self.num_nodes + dst_id
        uniq_ids, idx = np.unique(ids, return_index=True)
        duplicate_idx = np.setdiff1d(np.arange(ids.shape[0], dtype=np.int64), idx)
        duplicate_src_id = src_id[duplicate_idx]
        duplicate_dst_id = dst_id[duplicate_idx]
        duplicate_original_edge_id = original_edge_id[duplicate_idx]

        src_id = src_id[idx]
        dst_id = dst_id[idx]
        original_edge_id = original_edge_id[idx]
        print("length of src_idx after removing duplicated edges = {}".format(src_id.shape[0]))

        src_id = torch.from_numpy(src_id)
        dst_id = torch.from_numpy(dst_id)
        original_edge_id = torch.from_numpy(original_edge_id)
        edge_type = torch.zeros(src_id.shape[0], dtype=torch.int64)
        edge_data = torch.stack([src_id, dst_id, original_edge_id, edge_type], 1)

        self.edge_index = (src_id.numpy(), dst_id.numpy())
        self.num_edges = src_id.shape[0]
        self.edge_data = edge_data.numpy()

        self_loop_src_id = torch.from_numpy(self_loop_src_id)
        self_loop_dst_id = torch.from_numpy(self_loop_dst_id)
        self_loop_original_edge_id = torch.from_numpy(self_loop_original_edge_id)
        duplicate_src_id = torch.from_numpy(duplicate_src_id)
        duplicate_dst_id = torch.from_numpy(duplicate_dst_id)
        duplicate_original_edge_id = torch.from_numpy(duplicate_original_edge_id)

        removed_edge_data = torch.stack(
            [
                torch.cat([self_loop_src_id, duplicate_src_id]),
                torch.cat([self_loop_dst_id, duplicate_dst_id]),
                torch.cat([self_loop_original_edge_id, duplicate_original_edge_id]),
                torch.cat(
                    [
                        torch.zeros(self_loop_src_id.shape[0], dtype=torch.int64),
                        torch.zeros(duplicate_src_id.shape[0], dtype=torch.int64),
                    ]
                ),
            ],
            1,
        )

        self.removed_edge_index = removed_edge_data.numpy()

    def save_edge_index(self, graph_name: str, out_dir: str):
        """
        to save edge index to txt file for graph partitioning
        """
        self.process_edge_index()

        # save edge index
        np.savetxt(
            os.path.join(out_dir, "{}_edges.txt".format(graph_name)), self.edge_data, fmt="%d", delimiter=" "
        )

        # save removed edge index
        np.savetxt(
            os.path.join(out_dir, "{}_removed_edges.txt".format(graph_name)),
            self.removed_edge_index,
            fmt="%d",
            delimiter=" ",
        )

    def save_nodes(self, graph_name: str, out_dir: str):
        """
        to save nodes to txt file for graph partitioning
        """
        node_weight = []
        node_attr = []

        node_type = torch.zeros(self.num_nodes, dtype=torch.int64)
        node_attr.append(node_type)

        # append the in-degree of each node to node_weight, which will be used for load balancing in graph partitioning
        node_weight.append(self.get_in_degrees(self.edge_index, self.num_nodes).reshape(-1))

        # train_idx will also be append to node weight
        node_train_idx = torch.zeros(self.num_nodes, dtype=torch.int64)
        node_train_idx[self.train_idx] = 1
        node_weight.append(node_train_idx)

        # append the weight of each node to node_weight
        node_weight.append(torch.ones(self.num_nodes, dtype=torch.int64))

        self.num_node_weights = len(node_weight)
        node_attr.extend(node_weight)

        node_id = torch.arange(self.num_nodes, dtype=torch.int64)
        node_attr.append(node_id)

        # node_data = torch.stack([node_type, node_weight[0], node_weight[1], node_weight[2], node_id], 1)
        node_data = torch.stack(node_attr, 1)
        # print(node_data)
        # print(node_data.shape)
        np.savetxt(
            os.path.join(out_dir, "{}_nodes.txt".format(graph_name)),
            node_data.numpy(),
            fmt="%d",
            delimiter=" ",
        )

    def save_stats(self, graph_name: str, out_dir: str):
        """
        to save graph stats to txt file for graph partitioning
        """
        graph_stats = [self.num_nodes, self.num_edges, self.num_node_weights]
        print(graph_stats)
        with open(os.path.join(out_dir, "{}_stats.txt".format(graph_name)), "w") as f:
            for i in graph_stats:
                f.write(str(i))
                f.write(" ")


class DataLoader(object):
    """
    Utility class to load graph data
    """

    @staticmethod
    def load_ogbn_dataset(dataset: str, raw_dir: str) -> Graph:
        """
        to download and load ogbn dataset, return the loaded graph and node labels
        """
        data = NodePropPredDataset(name=dataset, root=raw_dir)
        graph, node_label = data[0]
        edge_index = graph["edge_index"]
        num_nodes = graph["num_nodes"]
        node_feat = graph["node_feat"]
        data_mask = data.get_idx_split()
        train_idx, valid_idx, test_idx = data_mask["train"], data_mask["valid"], data_mask["test"]
        graph = Graph(edge_index, node_feat, num_nodes, node_label, train_idx, valid_idx, test_idx)
        return graph

    @staticmethod
    def load_reddit_dataset(raw_dir: str) -> Graph:
        """
        to download and load reddit dataset, return the loaded graph and node labels
        """
        data = Reddit(root=raw_dir)[0]
        edge_index = data.edge_index.numpy()
        node_feat = data.x.numpy()
        num_nodes = node_feat.shape[0]
        node_label = data.y.numpy()
        train_idx = data.train_mask.nonzero().squeeze().numpy()
        valid_idx = data.val_mask.nonzero().squeeze().numpy()
        test_idx = data.test_mask.nonzero().squeeze().numpy()

        graph = Graph(edge_index, node_feat, num_nodes, node_label, train_idx, valid_idx, test_idx)
        return graph

    @staticmethod
    def generate_data_for_graph(
        num_nodes: int,
        feat_len: int,
        train_set_size: int,
        test_set_size: int,
        valid_set_size: int,
        num_labels: int,
    ):
        """
        to generate the node features, labels, training, validation and testing set for the graph which only has edge index
        """
        node_feat = torch.randn((num_nodes, feat_len), dtype=torch.float32).numpy()
        train_idx = torch.zeros(num_nodes, dtype=torch.bool)
        test_idx = torch.zeros(num_nodes, dtype=torch.bool)
        valid_idx = torch.zeros(num_nodes, dtype=torch.bool)
        node_label = torch.zeros(num_nodes, dtype=torch.int64).numpy()

        train_idx[:train_set_size] = True
        test_idx[train_set_size : train_set_size + test_set_size] = True
        valid_idx[train_set_size + test_set_size : train_set_size + test_set_size + valid_set_size] = True

        train_idx = train_idx.nonzero().squeeze().numpy()
        test_idx = test_idx.nonzero().squeeze().numpy()
        valid_idx = valid_idx.nonzero().squeeze().numpy()

        node_label[train_idx] = np.random.randint(0, num_labels, train_set_size)

        return (node_feat, node_label, train_idx, valid_idx, test_idx)
    
    @staticmethod
    def download_proteins_dataset(raw_dir: str):
        """
        to download proteins dataset
        """
        print("Downloading dataset...")
        print("This might a take while..")
        url = "https://portal.nersc.gov/project/m1982/GNN/"
        file_name = "subgraph3_iso_vs_iso_30_70length_ALL.m100.propermm.mtx"
        url = url + file_name
        try:
            r = requests.get(url)
        except:
            print("Error: can't download Proteins dataset!! Aborting..")
        
        with open(os.path.join(raw_dir, "proteins.mtx"), "wb") as handle:
            handle.write(r.content)
        print("Downloaded Proteins dataset successfully.")

    @staticmethod
    def load_proteins_dataset(raw_dir: str) -> Graph:
        """
        to load proteins dataset, return the loaded graph and node labels
        """

        if not os.path.exists(os.path.join(raw_dir, "proteins.mtx")):
            DataLoader.download_proteins_dataset(raw_dir)

        data = mmread(os.path.join(raw_dir, "proteins.mtx"))
        coo = data.tocoo()
        src_idx = torch.tensor(coo.row, dtype=torch.int64).numpy().reshape(-1)
        dst_idx = torch.tensor(coo.col, dtype=torch.int64).numpy().reshape(-1)
        edge_index = np.stack((src_idx, dst_idx), axis=0)

        num_nodes = 8745542
        # arbitrary number
        feat_len = 128
        train_set_size = 1000000
        test_set_size = 500000
        valid_set_size = 5000
        num_labels = 256

        node_feat, node_label, train_idx, valid_idx, test_idx = DataLoader.generate_data_for_graph(
            num_nodes, feat_len, train_set_size, test_set_size, valid_set_size, num_labels
        )

        graph = Graph(edge_index, node_feat, num_nodes, node_label, train_idx, valid_idx, test_idx)
        return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed_dir", type=str, default="./", help="The path of processed data directory."
    )
    parser.add_argument("--raw_dir", type=str, default="./", help="The path of raw dataset directory.")
    parser.add_argument("--dataset", type=str, help="The name of input dataset.")
    args = parser.parse_args()
    processed_dir = args.processed_dir
    raw_dir = args.raw_dir
    dataset = args.dataset
    graph_name = dataset

    # load graph
    if dataset[:4] == "ogbn":
        graph = DataLoader.load_ogbn_dataset(dataset, raw_dir)
    elif dataset == "reddit":
        graph = DataLoader.load_reddit_dataset(raw_dir)
    elif dataset == "proteins":
        graph = DataLoader.load_proteins_dataset(raw_dir)
    else:
        raise ValueError("Invalid dataset name")

    graph.save_node_feat(graph_name, processed_dir)
    graph.save_node_label(graph_name, processed_dir)
    graph.save_node_mask(graph_name, processed_dir)
    graph.save_edge_index(graph_name, processed_dir)
    graph.save_nodes(graph_name, processed_dir)
    graph.save_stats(graph_name, processed_dir)
    print("save graph data successfully.")
