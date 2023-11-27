from fileinput import filename
import numpy as np
import argparse
import pandas as pd
import os
import time
from multiprocessing import Process


def divide_edges_into_local_and_remote(
    edges_list: np.ndarray, node_idx_begin: int, node_idx_end: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    divide the edges into local and remote based on if the src_nodes are local or remote
    if the edges are local, then the src_nodes and dst_nodes are in the range of [node_idx_begin, node_idx_end]
    if the edges are remote, then the src_nodes are in the range of [node_idx_begin, node_idx_end], and the dst_nodes are original id
    """
    edges_list = edges_list.T
    src_nodes, dst_nodes = edges_list
    assert dst_nodes.min() >= node_idx_begin and dst_nodes.max() <= node_idx_end
    local_idx = (src_nodes >= node_idx_begin) & (src_nodes <= node_idx_end)
    remote_idx = ~local_idx

    local_src_nodes = src_nodes[local_idx]
    local_dst_nodes = dst_nodes[local_idx]

    # localize the node id
    local_src_nodes -= node_idx_begin
    local_dst_nodes -= node_idx_begin

    remote_src_nodes = src_nodes[remote_idx]
    remote_dst_nodes = dst_nodes[remote_idx]

    # localize the node id
    remote_dst_nodes -= node_idx_begin

    local_edges_list = np.concatenate(
        (local_src_nodes.reshape(1, -1), local_dst_nodes.reshape(1, -1)), axis=0
    )
    remote_edges_list = np.concatenate(
        (remote_src_nodes.reshape(1, -1), remote_dst_nodes.reshape(1, -1)), axis=0
    )

    return local_edges_list, remote_edges_list


def load_node_ids(file_name: str) -> np.ndarray:
    """
    load the txt file of node_id_list
    """
    # [0, 5], 0 -> transformed node id, 5 -> original node id
    node_ids = pd.read_csv(
        file_name,
        sep=" ",
        header=None,
        usecols=[0, 4],
        # usecols=[0, 5],
        dtype="int64",
    ).values
    return node_ids


def save_node_ids(file_name: str, node_ids: np.ndarray):
    """
    save the node_id_list to npy file
    """
    np.save(file_name, node_ids)


def save_node_feats(file_name: str, node_feats: np.ndarray, node_ids: np.ndarray):
    """
    save the node_feats to npy file
    """
    local_node_feats = node_feats[node_ids[:, 1]]
    np.save(file_name, local_node_feats)


def save_node_labels(file_name: str, node_labels: np.ndarray, node_ids: np.ndarray):
    """
    save the node_labels to npy file
    """
    local_node_labels = node_labels[node_ids[:, 1]].reshape(-1)
    local_node_labels = np.nan_to_num(local_node_labels, nan=-1).astype(np.int64)
    np.save(file_name, local_node_labels)


def save_edge_index(
    input_file_name: str,
    output_file_name_local_edges: str,
    output_file_name_remote_edges: str,
    node_ids: np.ndarray,
):
    """
    divide the edges into local and remote and save them to npy file
    """
    begin_idx = node_ids[0][0]
    end_idx = node_ids[-1][0]
    edges_list = pd.read_csv(
        input_file_name,
        sep=" ",
        header=None,
        usecols=[0, 1],
        dtype="int64",
    ).values
    local_edge_index, remote_edge_index = divide_edges_into_local_and_remote(edges_list, begin_idx, end_idx)
    np.save(output_file_name_local_edges, local_edge_index)
    np.save(output_file_name_remote_edges, remote_edge_index)


def split_nodes_feats(
    in_raw_dir: str, in_partition_dir: str, out_dir: str, graph_name: str, begin_part: int, end_part: int
):
    """
    split the node ids, node feats, node labels and edge index into each partition
    """
    node_feats = np.load(os.path.join(in_raw_dir, "{}_nodes_feat.npy".format(graph_name)))
    node_labels = np.load(os.path.join(in_raw_dir, "{}_nodes_label.npy".format(graph_name)))

    for i in range(begin_part, end_part):
        node_ids = load_node_ids(
            os.path.join(in_partition_dir, "p{:0>3d}-{}_nodes.txt".format(i, graph_name))
        )

        save_node_ids(os.path.join(out_dir, "p{:0>3d}-{}_nodes.npy".format(i, graph_name)), node_ids)
        save_node_feats(
            os.path.join(out_dir, "p{:0>3d}-{}_nodes_feat.npy".format(i, graph_name)), node_feats, node_ids
        )
        save_node_labels(
            os.path.join(out_dir, "p{:0>3d}-{}_nodes_label.npy".format(i, graph_name)), node_labels, node_ids
        )
        save_edge_index(
            os.path.join(in_partition_dir, "p{:0>3d}-{}_edges.txt".format(i, graph_name)),
            os.path.join(out_dir, "p{:0>3d}-{}_local_edges.npy".format(i, graph_name)),
            os.path.join(out_dir, "p{:0>3d}-{}_remote_edges.npy".format(i, graph_name)),
            node_ids,
        )

        print("partition {} over".format(i), flush=True)


# def compare_array(train_idx: np.ndarray, node_ids: np.ndarray, node_idx_begin: int) -> np.ndarray:
#     """
#     compare two array and remap the elem in train_idx according to the mapping in nodes_id_list
#     global id is mapped to local id in train_idx
#     """
#     local_train_idx = []
#     train_idx.sort()
#     idx_in_mask = 0
#     idx_in_node = 0
#     len_mask = train_idx.shape[0]
#     len_node_list = node_ids.shape[0]
#     while idx_in_mask < len_mask and idx_in_node < len_node_list:
#         if train_idx[idx_in_mask] < node_ids[idx_in_node][1]:
#             idx_in_mask += 1
#         elif train_idx[idx_in_mask] > node_ids[idx_in_node][1]:
#             idx_in_node += 1
#         else:
#             local_train_idx.append(node_ids[idx_in_node][0] - node_idx_begin)
#             idx_in_mask += 1
#             idx_in_node += 1

#     return np.array(local_train_idx, dtype=np.int64)


def remap_dataset_mask(data_idx: np.ndarray, node_ids: np.ndarray, node_idx_begin: int, file_name: str):
    """
    remap the data_idx from global id to local id and save it to file_name
    """
    # local_data_idx = compare_array(data_idx, node_ids, node_idx_begin)
    # construct a dict based on node_ids
    node_ids_dict = {}
    for i in range(node_ids.shape[0]):
        node_ids_dict[node_ids[i][1]] = node_ids[i][0] - node_idx_begin
    local_data_idx = []
    for i in range(data_idx.shape[0]):
        if data_idx[i] in node_ids_dict:
            local_data_idx.append(node_ids_dict[data_idx[i]])
        # else:
            # print("data_idx {} not in node_ids_dict".format(data_idx[i]))
    local_data_idx = np.array(local_data_idx, dtype=np.int64)
    np.save(file_name, local_data_idx)


def split_node_datamask(
    in_raw_dir: str, in_partition_dir: str, out_dir: str, graph_name: str, begin_part: int, end_part: int
):
    """
    split the dataset mask into each partition
    """
    remap_start = time.perf_counter()
    train_idx = np.load(os.path.join(in_raw_dir, "{}_nodes_train_idx.npy".format(graph_name)))
    valid_idx = np.load(os.path.join(in_raw_dir, "{}_nodes_valid_idx.npy".format(graph_name)))
    test_idx = np.load(os.path.join(in_raw_dir, "{}_nodes_test_idx.npy".format(graph_name)))
    for i in range(begin_part, end_part):
        node_ids = np.load(os.path.join(out_dir, "p{:0>3d}-{}_nodes.npy".format(i, graph_name)))
        node_idx_begin = node_ids[0][0]
        # node_ids = node_ids[node_ids[:, 1].argsort()]

        remap_dataset_mask(
            train_idx,
            node_ids,
            node_idx_begin,
            os.path.join(out_dir, "p{:0>3d}-{}_nodes_train_idx.npy".format(i, graph_name)),
        )
        remap_dataset_mask(
            valid_idx,
            node_ids,
            node_idx_begin,
            os.path.join(out_dir, "p{:0>3d}-{}_nodes_valid_idx.npy".format(i, graph_name)),
        )
        remap_dataset_mask(
            test_idx,
            node_ids,
            node_idx_begin,
            os.path.join(out_dir, "p{:0>3d}-{}_nodes_test_idx.npy".format(i, graph_name)),
        )

    remap_end = time.perf_counter()
    print("elapsed time of ramapping dataset mask(ms) = {}".format((remap_end - remap_start) * 1000))


def combined_func(
    in_raw_dir: str, in_partition_dir: str, out_dir: str, graph_name: str, begin_part: int, end_part: int
):
    split_nodes_feats(in_raw_dir, in_partition_dir, out_dir, graph_name, begin_part, end_part)
    split_node_datamask(in_raw_dir, in_partition_dir, out_dir, graph_name, begin_part, end_part)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", type=str, default="./", help="The path of output dir.")
    parser.add_argument("-ir", "--in_raw_dir", type=str, default="./", help="The path of raw data.")
    parser.add_argument(
        "-ip", "--in_partition_dir", type=str, default="./", help="The path of partitioned data."
    )
    parser.add_argument("-g", "--graph_name", type=str, help="The name of graph.")
    parser.add_argument("-b", "--begin_partition", type=int, help="The id of beginning partition.")
    parser.add_argument("-e", "--end_partition", type=int, help="The id of ending partition.")
    parser.add_argument("-p", "--num_process", type=int, default=8, help="The number of process.")
    args = parser.parse_args()
    out_dir = args.out_dir
    in_raw_dir = args.in_raw_dir
    in_partition_dir = args.in_partition_dir
    graph_name = args.graph_name
    num_process = args.num_process
    begin_part = args.begin_partition
    end_part = args.end_partition
    num_partition = end_part - begin_part
    print("begin_part = {}, end_part = {}".format(begin_part, end_part))
    step = int((end_part - begin_part + num_process - 1) / num_process)
    process_list = []

    for pid in range(num_process):
        p = Process(
            target=combined_func,
            args=(
                in_raw_dir,
                in_partition_dir,
                out_dir,
                graph_name,
                begin_part + pid * step,
                min((begin_part + (pid + 1) * step), end_part),
            ),
        )
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    print("All process over!!!")

    node_range_on_each_part = np.zeros(num_partition + 1, dtype=np.int64)
    for i in range(num_partition):
        tmp = np.load(os.path.join(out_dir, "p{:0>3d}-{}_nodes.npy".format(i, graph_name)))
        node_range_on_each_part[i + 1] = tmp[-1][0] + 1
    np.savetxt(
        os.path.join(out_dir, "begin_node_on_each_partition.txt"),
        node_range_on_each_part.reshape(1, -1),
        fmt="%d",
        delimiter=" ",
    )
