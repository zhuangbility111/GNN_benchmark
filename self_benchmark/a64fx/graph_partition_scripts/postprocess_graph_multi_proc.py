import numpy as np
import argparse
import pandas as pd
import os
import time
from multiprocessing import Process

def divide_edges_into_local_and_remote(edges_list, node_idx_begin, node_idx_end):
    '''
    local_edges_list = [[], []]
    remote_edges_list = [[], []]
    for edge in edges_list:
        # only the src node of each edges can be the remote node
        if edge[0] >= node_idx_begin and edge[0] <= node_idx_end:
            edge[0] -= node_idx_begin
            edge[1] -= node_idx_begin
            local_edges_list[0].append(edge[0])
            local_edges_list[1].append(edge[1])
        else:
            edge[1] -= node_idx_begin
            remote_edges_list[0].append(edge[0])
            remote_edges_list[1].append(edge[1])

    # convert list to numpy.array
    local_edges_list = np.array(local_edges_list, dtype= np.int64)
    remote_edges_list = np.array(remote_edges_list, dtype=np.int64)
    return local_edges_list, remote_edges_list
    '''
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

def split_nodes_feats(dir_path, graph_name, begin_part, end_part, len_feature):
        # node_range_on_each_part = []
        # node_range_on_each_part.append(0)
        for i in range(begin_part, end_part):
                # node_id_list = np.loadtxt(os.path.join(dir_path, "p{:0>3d}-{}_nodes.txt".format(i, graph_name)), dtype='int64', delimiter = ' ', usecols=(0, 3))
                node_id_list = pd.read_csv(os.path.join("./", "p{:0>3d}-{}_nodes.txt".format(i, graph_name)), sep=" ", header=None, usecols=[0, 3]).values
                # save the node_id_list to npy file
                np.save(os.path.join(dir_path, "p{:0>3d}-{}_nodes.npy".format(i, graph_name)), node_id_list)
                begin_idx = node_id_list[0][0]
                num_nodes = node_id_list.shape[0]
                end_idx = node_id_list[num_nodes-1][0]
                # node_range_on_each_part.append(node_range_on_each_part[i] + num_nodes)
                print(node_id_list.shape)
                # node_id_list.sort()
                node_id_list = node_id_list[node_id_list[:, 1].argsort()]
                # node_feat_list = []
                node_feat_list = np.empty([num_nodes, len_feature], dtype=np.float32)
                print(node_feat_list.dtype)
                idx_in_node_list = 0
                with open(os.path.join("./", "{}_nodes_feat.txt".format(graph_name))) as file:
                        for idx, line in enumerate(file):
                                if idx_in_node_list == len(node_id_list):
                                        break
                                elif idx == node_id_list[idx_in_node_list][1]:
                                        # node_feat_list.append(line.split(" "))
                                        node_feat_list[node_id_list[idx_in_node_list][0] - begin_idx] = line.split(" ")
                                        idx_in_node_list += 1
                # node_feat_list = np.array(node_feat_list, dtype="float32")
                print(node_feat_list.shape)
                np.save(os.path.join(dir_path, "p{:0>3d}-{}_nodes_feat.npy".format(i, graph_name)), node_feat_list)

                # node_label_list = []
                node_label_list = np.empty([num_nodes], dtype=np.int64)
                idx_in_node_list = 0
                with open(os.path.join("./", "{}_nodes_label.txt".format(graph_name))) as file:
                        for idx, line in enumerate(file):
                                if idx_in_node_list == len(node_id_list):
                                        break
                                elif idx == node_id_list[idx_in_node_list][1]:
                                        # node_label_list.append(line.split(" "))
                                        node_label_list[node_id_list[idx_in_node_list][0] - begin_idx] = line.split(" ")[0]
                                        idx_in_node_list += 1
                # node_label_list = np.array(node_label_list, dtype="int")
                print(node_label_list.shape)
                np.save(os.path.join(dir_path, "p{:0>3d}-{}_nodes_label.npy".format(i, graph_name)), node_label_list)
                # np.savetxt(os.path.join(dir_path, "part{}".format(i), "nodes_id.txt"), node_id_list, delimiter = ' ')

                # edges_list = np.load(os.path.join(dir_path, "p{:0>3d}-{}_edges.npy".format(i, graph_name)))
                edges_list = pd.read_csv(os.path.join("./", "p{:0>3d}-{}_edges.txt".format(i, graph_name)), sep=" ", header=None, usecols=[0, 1]).values
                local_edges_list, remote_edges_list = divide_edges_into_local_and_remote(edges_list, begin_idx, end_idx)
                print(local_edges_list)
                print(remote_edges_list)
                np.save(os.path.join(dir_path, "p{:0>3d}-{}_local_edges.npy".format(i, graph_name)), local_edges_list)
                np.save(os.path.join(dir_path, "p{:0>3d}-{}_remote_edges.npy".format(i, graph_name)), remote_edges_list)
        
        # node_range_on_each_part = np.array(node_range_on_each_part).reshape(1, -1)
        # np.savetxt(os.path.join(dir_path, "begin_node_on_each_partition.txt"), node_range_on_each_part, fmt = "%d", delimiter = ' ')

def split_nodes_feats_v1(dir_path, graph_name, begin_part, end_part, len_feature):
    # node_range_on_each_part = []
    # node_range_on_each_part.append(0)
    for i in range(begin_part, end_part):
        # node_id_list = np.loadtxt(os.path.join(dir_path, "p{:0>3d}-{}_nodes.txt".format(i, graph_name)), dtype='int64', delimiter = ' ', usecols=(0, 3))
        node_id_list = pd.read_csv(os.path.join("./", "p{:0>3d}-{}_nodes.txt".format(i, graph_name)), sep=" ", header=None, usecols=[0, 3]).values
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

        edges_list = pd.read_csv(os.path.join("./", "p{:0>3d}-{}_edges.txt".format(i, graph_name)), sep=" ", header=None, usecols=[0, 1]).values
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
    train_idx = pd.read_csv(os.path.join("./", "{}_nodes_train_idx.txt".format(graph_name)), sep=" ", header=None).values
    valid_idx = pd.read_csv(os.path.join("./", "{}_nodes_valid_idx.txt".format(graph_name)), sep=" ", header=None).values
    test_idx = pd.read_csv(os.path.join("./", "{}_nodes_test_idx.txt".format(graph_name)), sep=" ", header=None).values
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
    
def combined_func(dir_path, graph_name, begin_part, end_part, len_feature):
        # split_nodes_feats(dir_path, graph_name, begin_part, end_part, len_feature)
        split_nodes_feats_v1(dir_path, graph_name, begin_part, end_part, len_feature)
        split_node_datamask(dir_path, graph_name, begin_part, end_part)

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--dir_path', type=str, default='./', help='The path of father directory.')
        parser.add_argument('-g', '--graph_name', type=str, help='The name of graph.')
        parser.add_argument('-n', '--num_partition', type=int, help='The number of partitioning.')
        parser.add_argument('-l', '--len_feature', type=int, help='The length of feature vector.')
        args = parser.parse_args()
        dir_path = args.dir_path
        graph_name = args.graph_name
        num_partition = args.num_partition
        len_feature = args.len_feature
        num_process = 16
        begin_part = 0
        end_part = num_partition
        step = int((end_part - begin_part + num_process - 1) / num_process)
        process_list = []
        for pid in range(num_process):
            p = Process(target=combined_func, 
                        args=(dir_path, graph_name, begin_part + pid*step, min((begin_part + (pid+1)*step), end_part), len_feature))
            p.start()
            process_list.append(p)

        for pid in range(num_process):
            p.join()

        print("All process over!!!")
        node_range_on_each_part = np.zeros(num_partition+1, dtype=np.int64)
        for i in range(num_partition): 
            tmp = np.load(os.path.join(dir_path, "p{:0>3d}-{}_nodes.npy".format(i, graph_name)))
            node_range_on_each_part[i+1] = tmp[-1][0] + 1
        np.savetxt(os.path.join(dir_path, "begin_node_on_each_partition.txt"), node_range_on_each_part.reshape(1, -1), fmt = "%d", delimiter = ' ')

