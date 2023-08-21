from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# import for finding connected components
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import argparse

def get_X(data):
    n = int(data.shape[0])
    # collect the upper triangle of data comm matrix as input X for cluster algorithm
    size = int((1 + n) * n / 2)
    X = np.zeros(size, dtype = np.float32)
    k = 0
    for i in range(n):
        for j in range(i, n):
            X[k] += (data[i, j] + data[j, i])
            k += 1
    print("k = ", k)
    return X.reshape(-1, 1)

def adjust_cluster_order(labels, X):
    labels_and_maximums = []
    for i in range(np.max(labels)+1):
        # (original cluster label, maximum value in this cluster)
        labels_and_maximums.append((i, np.max(X[labels == i])))
    # sort the original cluster label by maximum value in each cluster
    labels_and_maximums.sort(key=lambda x: x[1])
    new_labels = np.zeros_like(labels)
    for i in range(len(labels_and_maximums)):
        # remap the original cluster label to new cluster label
        new_labels[labels == labels_and_maximums[i][0]] = i

    return new_labels

def cluster_for_result(X, cluster_num):
    # X_normalized = (X - X.min()) / (X.max() - X.min())
    X_normalized = X
    
    kmeans = KMeans(n_clusters=cluster_num, random_state=0)
    kmeans.fit(X_normalized)

    # 获取每个点所属的簇的标签
    labels = kmeans.labels_

    # 获取每个簇的中心点
    centroids = kmeans.cluster_centers_

    # 反归一化处理，将中心点转换回原始数据范围
    # centroids_denormalized = centroids * (X.max() - X.min()) + X.min()
    centroids_denormalized = centroids

    # 绘制聚类结果
    # plt.scatter(X, np.zeros_like(X), c=labels, cmap='viridis')
    # plt.scatter(centroids_denormalized, np.zeros_like(centroids_denormalized), marker='x', color='red', s=100)
    # plt.title('Clustering Result')
    # plt.xlabel('Data')
    # plt.yticks([])
    # plt.show()
    # plt.savefig('cluster_result_{}_normalize.png'.format(cluster_num))
    
    labels = adjust_cluster_order(labels, X)

    for i in range(cluster_num):
        if i < cluster_num - 1:
            labels[labels == i+1] = i
            print("cluster {} has {} edges".format(i, np.sum(labels == i)))
            print("comm data volume range: [{}, {}]".format(np.min(X[labels == i]), np.max(X[labels == i])))
    
    return labels

def recover_cluster_result(label_list, data_volume_matrix):
    n = int(data_volume_matrix.shape[0])
    k = 0
    label_matrix = np.zeros_like(data_volume_matrix)
    # recover the cluster result (the upper triangle of data comm matrix) to original data comm matrix
    for i in range(n):
        for j in range(i, n):
            # the cluster algorithm will group the edges in which data volume is 0 to the cluster 0
            # so we need to remove these edges from cluster 0 as these edges (volume = 0) don't exist in the original graph
            # if data[i, j] == 0 and data[j, i] == 0, means that there is no edge between vertex i and vertex j
            # so i and j are not in the same communicator
            if data_volume_matrix[i, j] == 0 and data_volume_matrix[j, i] == 0:
                label_matrix[i, j] = -1
                label_matrix[j, i] = -1
            # but if data[i, j] != 0 or data[j, i] != 0, means that i and j need to be the same communicator for communication
            else:
                label_matrix[i, j] = label_list[k]
                label_matrix[j, i] = label_list[k]
            k += 1

    return label_matrix

def divide_group_into_p2p_and_collective(data_volume_matrix, degs_threshold=4):
    in_degs = np.sum((data_volume_matrix != 0), axis=0)
    out_degs = np.sum((data_volume_matrix != 0), axis=1)
    total_degs = in_degs + out_degs
    print("in_degs = {}".format(in_degs))
    print("out_degs = {}".format(out_degs))
    print("in_degs = {}".format(in_degs))


    num_procs = data_volume_matrix.shape[0]
    # get the idx of procs which have degree <= degs_threshold
    p2p_ranks_list = (total_degs <= degs_threshold).nonzero()[0]
    # get the idx of procs which have degree > degs_threshold
    collective_ranks_list = (total_degs > degs_threshold).nonzero()[0]

    print("p2p_ranks_list = {}".format(p2p_ranks_list))
    print("collective_ranks_list = {}".format(collective_ranks_list))

    # collective_label_matrix = np.zeros_like(data_volume_matrix)
    collective_label_matrix = np.full_like(data_volume_matrix, -1)
    p2p_label_matrix = np.zeros_like(data_volume_matrix)

    for src_rank in range(num_procs):
        for dst_rank in range(num_procs):
            # alltoallv comm needs all procs to participate in
            # so no matter if the comm is necessary or not, we set the label to 0
            if src_rank in collective_ranks_list and dst_rank in collective_ranks_list:
                collective_label_matrix[src_rank, dst_rank] = 0
                p2p_label_matrix[src_rank, dst_rank] = 0
            else:
                collective_label_matrix[src_rank, dst_rank] = -1
                # need to send data to dst_rank by p2p comm
                # so we only set the label when the p2p comm is necessary
                if src_rank != dst_rank and data_volume_matrix[src_rank, dst_rank] != 0:
                    p2p_label_matrix[src_rank, dst_rank] = 1

    return p2p_label_matrix, collective_label_matrix

def remove_large_procs_from_small_clusters(collective_label_matrix, comm_matrix):
    p2p_label_matrix = np.zeros_like(collective_label_matrix)
    new_collective_label_matrix = np.copy(collective_label_matrix)
    # for i in range(num_clusters):
    #     row_idx, col_idx = np.where(collective_label_matrix == i)
    #     if clusters_dict.get(i) is None:
    #         clusters_dict[i] = set()
    #     clusters_dict[i].update(row_idx)
    #     clusters_dict[i].update(col_idx)

    num_procs = collective_label_matrix.shape[0]
    group_idx = np.zeros(num_procs, dtype=np.int32)
    for cur_proc in range(num_procs):
        idx = np.max(collective_label_matrix[cur_proc, :])
        group_idx[cur_proc] = idx

    print("group_idx = {}".format(group_idx))

    for src_proc in range(num_procs):
        '''
        idx = ((collective_label_matrix[cur_proc, :] != group_idx[cur_proc]) & \
                (collective_label_matrix[cur_proc, :] != -1)).nonzero()[0]
        
        for target_proc in idx:
            if group_idx[target_proc] == group_idx[cur_proc]:
                new_collective_label_matrix[cur_proc, target_proc] = group_idx[cur_proc]
                new_collective_label_matrix[target_proc, cur_proc] = group_idx[cur_proc]
            else:
                new_collective_label_matrix[cur_proc, target_proc] = -1
                new_collective_label_matrix[target_proc, cur_proc] = -1
                p2p_label_matrix[cur_proc, target_proc] = 1
                p2p_label_matrix[target_proc, cur_proc] = 1
        '''
        for dst_proc in range(num_procs):
            if group_idx[dst_proc] == group_idx[src_proc]:
                new_collective_label_matrix[src_proc, dst_proc] = group_idx[src_proc]
                # new_collective_label_matrix[dst_proc, src_proc] = group_idx[src_proc]
            else:
                if collective_label_matrix[src_proc, dst_proc] != -1:
                    new_collective_label_matrix[src_proc, dst_proc] = -1
                    if comm_matrix[src_proc, dst_proc] != 0:
                    # new_collective_label_matrix[dst_proc, src_proc] = -1
                        p2p_label_matrix[src_proc, dst_proc] = 1

    print("new_collective_label_matrix = {}".format(new_collective_label_matrix))

    return new_collective_label_matrix, p2p_label_matrix

def find_connected_components(label_matrix):
    max_label = np.max(label_matrix)
    new_global_rank_to_labels = []
    cluster_dict = dict()
    label_idx_begin = 0
    for i in range(max_label + 1):
        # convert the subgraph to a sparse matrix
        row_idx, col_idx = np.where(label_matrix == i)
        global_rank = np.unique(np.concatenate([row_idx, col_idx], axis=0))
        if global_rank.shape[0] == 0:
            new_global_rank_to_labels.append(dict())
            continue
        global_rank_to_local_rank = np.zeros(np.max(global_rank) + 1)
        for k in range(global_rank.shape[0]):
            global_rank_to_local_rank[global_rank[k]] = k
        
        row_idx = global_rank_to_local_rank[row_idx]
        col_idx = global_rank_to_local_rank[col_idx]
        values = np.ones_like(row_idx, dtype=np.int32)

        subgraph = csr_matrix((values, (row_idx, col_idx)), shape=(global_rank.shape[0], global_rank.shape[0]))
        # find the connected components in the subgraph
        n_components, labels = connected_components(csgraph=subgraph, directed=True, connection='weak', return_labels=True)
        print("cluster {} has {} connected components".format(i, n_components))
        print("global_rank = {}".format(global_rank))
        print("num_procs = {}".format(global_rank.shape[0]))
        print("connected components: ", labels)

        labels += label_idx_begin
        label_idx_begin += n_components

        tmp = dict()
        for i in range(labels.shape[0]):
            tmp[global_rank[i]] = labels[i]
        new_global_rank_to_labels.append(tmp)

        unique_labels = np.unique(labels)
        for label in unique_labels:
            cluster_dict[label] = global_rank[labels == label]

    for i in range(label_matrix.shape[0]):
        for j in range(label_matrix.shape[1]):
            if label_matrix[i, j] != -1:
                label_matrix[i, j] = new_global_rank_to_labels[int(label_matrix[i, j])][i]
    
    return label_matrix, cluster_dict

def check_result(collective_label_matrix, p2p_label_matrix, data_volume_matrix):
    num_procs = data_volume_matrix.shape[0]

    for i in range(num_procs):
        original_send_data_volume = np.sum(data_volume_matrix[i]) 
        original_recv_data_volume = np.sum(data_volume_matrix[:, i])

        # collect the nonzero elements in the row i of collective_label_matrix
        
        collective_send_data_volume = np.sum(data_volume_matrix[i, collective_label_matrix[i, :] != -1])
        collective_recv_data_volume = np.sum(data_volume_matrix[collective_label_matrix[:, i] != -1, i])

        p2p_send_data_volume = np.sum(data_volume_matrix[i, p2p_label_matrix[i, :] == 1])
        p2p_recv_data_volume = np.sum(data_volume_matrix[p2p_label_matrix[:, i] == 1, i])

        assert original_send_data_volume == collective_send_data_volume + p2p_send_data_volume
        assert original_recv_data_volume == collective_recv_data_volume + p2p_recv_data_volume

    print("check group result passed!")

def save_labels(label_list, data_volume_matrix, num_clusters, num_procs):
    # recover the cluster result (the upper triangle of data comm matrix) to original data comm matrix
    label_matrix = recover_cluster_result(label_list, data_volume_matrix)

    # divide the graph into p2p group and collective group based on the degree of each vertex
    # p2p_label_matrix, collective_label_matrix = divide_group_into_p2p_and_collective(data_volume_matrix)

    collective_label_matrix, p2p_label_matrix = remove_large_procs_from_small_clusters(label_matrix, data_volume_matrix)

    # label_matrix, cluster_dict = find_connected_components(label_matrix)
    collective_label_matrix, collective_cluster_dict = find_connected_components(collective_label_matrix)

    # print("cluster_dict: ", cluster_dict)
    # for key in collective_cluster_dict:
    #     max_degs = -1
    #     min_degs = 99999
    #     for rank in collective_cluster_dict[key]:
    #         # print("rank {} has {} edges(out-degs)".format(rank, np.sum(label_matrix[rank, :] == key)))
    #         degs = np.sum(label_matrix[rank, :] == key)
    #         min_degs = min(min_degs, degs)
    #         max_degs = max(max_degs, degs)
    #     print("cluster {} has {} processes, the min-degs and max-degs is {} and {}".format(key, len(collective_cluster_dict[key]), min_degs, max_degs))

    np.save('collective_group_labels({}clusters_{}procs).npy'.format(num_clusters, num_procs), collective_label_matrix)
    np.save('p2p_group_labels({}clusters_{}procs).npy'.format(num_clusters, num_procs), p2p_label_matrix)

    check_result(collective_label_matrix, p2p_label_matrix, data_volume_matrix)

    with open('ranks_list_in_each_collective_group({}clusters_{}procs).txt'.format(num_clusters, num_procs), 'w') as f:
        num_keys = len(collective_cluster_dict.keys())
        for i in range(num_keys):
            ranks_list = list(collective_cluster_dict[i])
            ranks_list.sort()
            # print(ranks_list)
            delimiter = ','
            f.write(delimiter.join(map(str, ranks_list)))
            if i != num_keys - 1:
                f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clusters', type=int, default=5)
    parser.add_argument('--num_procs', type=int, default=512)
    args = parser.parse_args()

    num_clusters = args.num_clusters
    num_procs = args.num_procs

    data = np.load('global_comm_pattern_{}proc.npy'.format(num_procs))
    X = get_X(data)

    labels = cluster_for_result(X, num_clusters)

    save_labels(labels, data, num_clusters, num_procs)
