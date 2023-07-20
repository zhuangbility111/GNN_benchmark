from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# import for finding connected components
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


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
    plt.scatter(X, np.zeros_like(X), c=labels, cmap='viridis')
    plt.scatter(centroids_denormalized, np.zeros_like(centroids_denormalized), marker='x', color='red', s=100)
    plt.title('Clustering Result')
    plt.xlabel('Data')
    plt.yticks([])
    plt.show()
    plt.savefig('cluster_result_{}_normalize.png'.format(cluster_num))
    
    labels = adjust_cluster_order(labels, X)

    for i in range(cluster_num):
        print("cluster {} has {} edges".format(i, np.sum(labels == i)))
        print("comm data volume range: [{}, {}]".format(np.min(X[labels == i]), np.max(X[labels == i])))
    
    return labels

def find_connected_components(label_matrix):
    max_label = np.max(label_matrix)
    new_global_rank_to_labels = []
    cluster_dict = dict()
    label_idx_begin = 0
    for i in range(max_label + 1):
        # convert the subgraph to a sparse matrix
        row_idx, col_idx = np.where(label_matrix == i)
        global_rank = np.unique(np.concatenate([row_idx, col_idx], axis=0))
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

def save_labels(label_list, data_volume_matrix):
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

    label_matrix, cluster_dict= find_connected_components(label_matrix)

    print("cluster_dict: ", cluster_dict)
    for key in cluster_dict:
        print("cluster {} has {} processes".format(key, len(cluster_dict[key])))

    np.save('global_comm_pattern_512proc_clustered_label.npy', label_matrix)

    with open('ranks_list_in_each_clusters_512proc.txt', 'w') as f:
        num_keys = len(cluster_dict.keys())
        for i in range(num_keys):
            ranks_list = list(cluster_dict[i])
            ranks_list.sort()
            print(ranks_list)
            delimiter = ','
            f.write(delimiter.join(map(str, ranks_list)))
            if i != num_keys - 1:
                f.write('\n')


if __name__ == "__main__":
    data = np.load('global_comm_pattern_512proc.npy')
    X = get_X(data)

    labels = cluster_for_result(X, 5)

    save_labels(labels, data)

    # sse = []
    # # silhouette_scores = []
    # for k in range(2, 20):
    #     kmeans = KMeans(n_clusters=k, random_state=42)
    #     kmeans.fit(X)
    #     sse.append(kmeans.inertia_)
    #     # silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    #     print("Kmeans on K = {} finish!".format(k))

    # plt.plot(range(2, 20), sse, 'go-')
    # plt.xlabel('Number of Clusters (k)')
    # plt.ylabel('SSE')
    # plt.savefig('kmeans_elbow_random42.png')

    # plt.plot(range(2, 20), silhouette_scores, 'ro-')
    # plt.xlabel('Number of Clusters (k)')
    # plt.ylabel('Silhouette Coefficient')
    # plt.savefig('kmeans_silhouette.png')
