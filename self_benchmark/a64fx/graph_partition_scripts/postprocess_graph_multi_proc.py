import numpy as np
import argparse
import pandas as pd
import os
import time
from multiprocessing import Process

def divide_edges_into_local_and_remote(edges_list, node_idx_begin, node_idx_end):
    edges_list = edges_list.T
    src_nodes, dst_nodes = edges_list
    local_idx = ((src_nodes >= node_idx_begin) & (src_nodes <= node_idx_end))
    remote_idx = ~local_idx

    local_src_nodes = src_nodes[local_idx]
    local_dst_nodes = dst_nodes[local_idx]

    local_src_nodes -= node_idx_begin
    local_dst_nodes -= node_idx_begin

    remote_src_nodes = src_nodes[remote_idx]
    remote_dst_nodes = dst_nodes[remote_idx]

    remote_dst_nodes -= node_idx_begin

    local_edges_list = np.concatenate((local_src_nodes.reshape(1,-1), local_dst_nodes.reshape(1,-1)), axis=0)
    remote_edges_list = np.concatenate((remote_src_nodes.reshape(1,-1), remote_dst_nodes.reshape(1,-1)), axis=0)

    return local_edges_list, remote_edges_list

def split_nodes_feats(dir_path, graph_name, begin_part, end_part):
    # node_range_on_each_part = []
    # node_range_on_each_part.append(0)
    for i in range(begin_part, end_part):
        # node_id_list = np.loadtxt(os.path.join(dir_path, "p{:0>3d}-{}_nodes.txt".format(i, graph_name)), dtype='int64', delimiter = ' ', usecols=(0, 3))
        node_id_list = pd.read_csv(os.path.join("./", "p{:0>3d}-{}_nodes.txt".format(i, graph_name)), sep=" ", header=None, usecols=[0, 3], dtype='int64').values
        # save the node_id_list to npy file
        np.save(os.path.join(dir_path, "p{:0>3d}-{}_nodes.npy".format(i, graph_name)), node_id_list)
        begin_idx = node_id_list[0][0]
        num_nodes = node_id_list.shape[0]
        end_idx = node_id_list[num_nodes-1][0]

        node_feat_list = np.load(os.path.join("./", "{}_nodes_feat.npy".format(graph_name)), mmap_mode='r')
        local_node_feat_list = node_feat_list[node_id_list[:, 1]]
        np.save(os.path.join(dir_path, "p{:0>3d}-{}_nodes_feat.npy".format(i, graph_name)), local_node_feat_list)
        print(local_node_feat_list.shape)

        node_label_list = np.load(os.path.join("./", "{}_nodes_label.npy".format(graph_name)), mmap_mode='r')
        local_node_label_list = node_label_list[node_id_list[:, 1]].reshape(-1)
        np.save(os.path.join(dir_path, "p{:0>3d}-{}_nodes_label.npy".format(i, graph_name)), local_node_label_list)
        print(local_node_label_list.shape)

        edges_list = pd.read_csv(os.path.join("./", "p{:0>3d}-{}_edges.txt".format(i, graph_name)), sep=" ", header=None, usecols=[0, 1], dtype='int64').values
        local_edges_list, remote_edges_list = divide_edges_into_local_and_remote(edges_list, begin_idx, end_idx)
        print(local_edges_list)
        print(remote_edges_list)
        np.save(os.path.join(dir_path, "p{:0>3d}-{}_local_edges.npy".format(i, graph_name)), local_edges_list)
        np.save(os.path.join(dir_path, "p{:0>3d}-{}_remote_edges.npy".format(i, graph_name)), remote_edges_list)

# compare two array and remap the elem in train_idx according to the mapping in nodes_id_list
# global id is mapped to local id in train_idx
def compare_array(train_idx, nodes_id_list, node_idx_begin):
    local_train_idx = []
    train_idx.sort()
    idx_in_mask = 0
    idx_in_node = 0
    len_mask = train_idx.shape[0]
    len_node_list = nodes_id_list.shape[0]
    while idx_in_mask < len_mask and idx_in_node < len_node_list:
        if train_idx[idx_in_mask] < nodes_id_list[idx_in_node][1]:
            idx_in_mask += 1
        elif train_idx[idx_in_mask] > nodes_id_list[idx_in_node][1]:
            idx_in_node += 1
        else:
            local_train_idx.append(nodes_id_list[idx_in_node][0] - node_idx_begin)
            idx_in_mask += 1
            idx_in_node += 1

    return np.array(local_train_idx, dtype=np.int64)

def split_node_datamask(dir_path, graph_name, begin_part, end_part):
    remap_start = time.perf_counter()
    train_idx = np.load(os.path.join("./", "{}_nodes_train_idx.npy".format(graph_name)))
    valid_idx = np.load(os.path.join("./", "{}_nodes_valid_idx.npy".format(graph_name)))
    test_idx = np.load(os.path.join("./", "{}_nodes_test_idx.npy".format(graph_name)))
    for i in range(begin_part, end_part):
        # nodes_id_list = pd.read_csv(os.path.join("./", "p{:0>3d}-{}_nodes.txt".format(i, graph_name)), sep=" ", header=None, usecols=[0, 3]).values
        nodes_id_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes.npy".format(i, graph_name)))
        node_idx_begin = nodes_id_list[0][0]
        nodes_id_list = nodes_id_list[nodes_id_list[:,1].argsort()]

        # remap training mask
        local_train_idx = compare_array(train_idx, nodes_id_list, node_idx_begin)

        # remap validated mask
        local_valid_idx = compare_array(valid_idx, nodes_id_list, node_idx_begin)

        # remap test mask
        local_test_idx = compare_array(test_idx, nodes_id_list, node_idx_begin)

        # np.savetxt(os.path.join(dir_path, "p{:0>3d}-{}_nodes_train_idx.txt".format(i, graph_name)), local_train_idx, fmt = "%d", delimiter = ' ')
        np.save(os.path.join(dir_path, "p{:0>3d}-{}_nodes_train_idx.npy".format(i, graph_name)), local_train_idx)
        # np.savetxt(os.path.join(dir_path, "p{:0>3d}-{}_nodes_valid_idx.txt".format(i, graph_name)), local_valid_idx, fmt = "%d", delimiter = ' ')
        np.save(os.path.join(dir_path, "p{:0>3d}-{}_nodes_valid_idx.npy".format(i, graph_name)), local_valid_idx)
        # np.savetxt(os.path.join(dir_path, "p{:0>3d}-{}_nodes_test_idx.txt".format(i, graph_name)), local_test_idx, fmt = "%d", delimiter = ' ')
        np.save(os.path.join(dir_path, "p{:0>3d}-{}_nodes_test_idx.npy".format(i, graph_name)), local_test_idx)

    remap_end = time.perf_counter()
    print("elapsed time of ramapping dataset mask(ms) = {}".format((remap_end - remap_start) * 1000))

def combined_func(dir_path, graph_name, begin_part, end_part):
    split_nodes_feats(dir_path, graph_name, begin_part, end_part)
    split_node_datamask(dir_path, graph_name, begin_part, end_part)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir_path', type=str, default='./', help='The path of father directory.')
    parser.add_argument('-g', '--graph_name', type=str, help='The name of graph.')
    parser.add_argument('-b', '--begin_partition', type=int, help='The id of beginning partition.')
    parser.add_argument('-e', '--end_partition', type=int, help='The id of ending partition.')
    parser.add_argument('-p', '--num_process', type=int, default=16, help='The number of process.')
    args = parser.parse_args()
    dir_path = args.dir_path
    graph_name = args.graph_name
    num_process = args.num_process
    begin_part = args.begin_partition
    end_part = args.end_partition
    num_partition = end_part - begin_part
    print("begin_part = {}, end_part = {}".format(begin_part, end_part))
    step = int((end_part - begin_part + num_process - 1) / num_process)
    process_list = []

    for pid in range(num_process):
        p = Process(target=combined_func,
                    args=(dir_path, graph_name, begin_part + pid*step, min((begin_part + (pid+1)*step), end_part)))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    print("All process over!!!")

    node_range_on_each_part = np.zeros(num_partition+1, dtype=np.int64)
    for i in range(num_partition):
        tmp = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes.npy".format(i, graph_name)))
        node_range_on_each_part[i+1] = tmp[-1][0] + 1
    np.savetxt(os.path.join(dir_path, "begin_node_on_each_partition.txt"), node_range_on_each_part.reshape(1, -1), fmt = "%d", delimiter = ' ')