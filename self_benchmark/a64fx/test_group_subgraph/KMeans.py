from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

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

def save_labels(labels, data):
    n = int(data.shape[0])
    k = 0
    cluster_dict = {}
    # recover the cluster result (the upper triangle of data comm matrix) to original data comm matrix
    for i in range(n):
        for j in range(i, n):
            # the cluster algorithm will group the edges in which data volume is 0 to the cluster 0
            # so we need to remove these edges from cluster 0 as these edges (volume = 0) don't exist in the original graph
            # if data[i, j] == 0 and data[j, i] == 0, means that there is no edge between vertex i and vertex j
            # so i and j are not in the same communicator
            if data[i, j] == 0 and data[j, i] == 0:
                data[i, j] = -1
                data[j, i] = -1
            # but if data[i, j] != 0 or data[j, i] != 0, means that i and j need to be the same communicator for communication
            else:
                data[i, j] = labels[k]
                data[j, i] = labels[k]
                if labels[k] not in cluster_dict:
                    cluster_dict[labels[k]] = set()
                cluster_dict[labels[k]].add(i)
                cluster_dict[labels[k]].add(j)
            k += 1

    print("cluster_dict: ", cluster_dict)
    for key in cluster_dict:
        print("cluster {} has {} processes".format(key, len(cluster_dict[key])))

    np.save('global_comm_pattern_512proc_clustered_label.npy', data)

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

    labels = cluster_for_result(X, 20)

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
