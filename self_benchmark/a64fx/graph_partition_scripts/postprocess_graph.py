import numpy as np
import argparse
import os

def split_nodes_feats(dir_path, graph_name, num_partition, len_feature):
	node_range_on_each_part = []
	node_range_on_each_part.append(0)
	for i in range(num_partition):
		node_id_list = np.loadtxt(os.path.join(dir_path, "p{:0>3d}-{}_nodes.txt".format(i, graph_name)), dtype='int64', delimiter = ' ', usecols=(0, 3))
		begin_idx = node_id_list[0][0]
		num_nodes = node_id_list.shape[0]
		node_range_on_each_part.append(node_range_on_each_part[i] + num_nodes)
		print(node_id_list.shape)
		# node_id_list.sort()
		node_id_list = node_id_list[node_id_list[:, 1].argsort()]
		# node_feat_list = []
		node_feat_list = np.empty([num_nodes, len_feature], dtype = float)
		idx_in_node_list = 0
		with open(os.path.join(dir_path, "{}_nodes_feat.txt".format(graph_name))) as file:
			for idx, line in enumerate(file):
				if idx_in_node_list == len(node_id_list):
					break
				elif idx == node_id_list[idx_in_node_list][1]:
					# node_feat_list.append(line.split(" "))
					node_feat_list[node_id_list[idx_in_node_list][0] - begin_idx] = line.split(" ")
					idx_in_node_list += 1
		# node_feat_list = np.array(node_feat_list, dtype="float32")
		print(node_feat_list.shape)
		np.savetxt(os.path.join(dir_path, "p{:0>3d}-{}_nodes_feat.txt".format(i, graph_name)), node_feat_list, delimiter = ' ')

		# node_label_list = []
		node_label_list = np.empty([num_nodes], dtype=int)
		idx_in_node_list = 0
		with open(os.path.join(dir_path, "{}_nodes_label.txt".format(graph_name))) as file:
			for idx, line in enumerate(file):
				if idx_in_node_list == len(node_id_list):
					break
				elif idx == node_id_list[idx_in_node_list][1]:
					# node_label_list.append(line.split(" "))
					node_label_list[node_id_list[idx_in_node_list][0] - begin_idx] = line.split(" ")[0]
					idx_in_node_list += 1
		# node_label_list = np.array(node_label_list, dtype="int")
		print(node_label_list.shape)
		np.savetxt(os.path.join(dir_path, "p{:0>3d}-{}_nodes_label.txt".format(i, graph_name)), node_label_list, fmt = "%d", delimiter = ' ')
		# np.savetxt(os.path.join(dir_path, "part{}".format(i), "nodes_id.txt"), node_id_list, delimiter = ' ')
	
	node_range_on_each_part = np.array(node_range_on_each_part).reshape(1, -1)
	np.savetxt(os.path.join(dir_path, "begin_node_on_each_partition.txt"), node_range_on_each_part, fmt = "%d", delimiter = ' ')

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
	split_nodes_feats(dir_path, graph_name, num_partition, len_feature)
