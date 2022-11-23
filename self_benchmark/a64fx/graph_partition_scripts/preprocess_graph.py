from ogb.nodeproppred import NodePropPredDataset
import torch
import numpy as np
import time
import argparse
import os
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', type=str, default='./', help='The path of father directory.')
parser.add_argument('--dataset', type=str, help='The name of input dataset.')
parser.add_argument('--graph_name', type=str, help='The name of graph.')
args = parser.parse_args()
dir_path = args.dir_path
dataset = args.dataset
graph_name = args.graph_name

dataset = NodePropPredDataset(name = dataset)
graph, node_label = dataset[0]
num_nodes = graph['num_nodes']

###### process nodes labels ######
print(node_label)
print(node_label.shape)
np.savetxt(os.path.join(dir_path, "{}_nodes_label.txt".format(graph_name)), node_label, fmt="%d", delimiter=' ')

###### process nodes features ######
node_feat = graph['node_feat']
np.savetxt(os.path.join(dir_path, "{}_nodes_feat.txt".format(graph_name)), node_feat, delimiter=' ')
print(num_nodes)
print(node_feat.shape)
del node_feat
gc.collect()

###### process edge ######
edge_index = graph['edge_index']
src_id, dst_id = edge_index
original_edge_id = np.arange(len(src_id))
print("length of src_id before removing self loop = {}".format(len(src_id)))

# remove self loop
self_loop_idx = src_id == dst_id
not_self_loop_idx = src_id != dst_id

self_loop_src_id = src_id[self_loop_idx]
self_loop_dst_id = dst_id[self_loop_idx]
self_loop_original_edge_id = original_edge_id[self_loop_idx]

src_id = src_id[not_self_loop_idx]
dst_id = dst_id[not_self_loop_idx]
original_edge_id = original_edge_id[not_self_loop_idx]
print("length of src_id after removing self loop = {}".format(len(src_id)))

print("length of src_id before removing duplicated edges = {}".format(len(src_id)))
start_time = time.time()
ids = (src_id * num_nodes + dst_id)
uniq_ids, idx = np.unique(ids, return_index=True)
duplicate_idx = np.setdiff1d(np.arange(len(ids)), idx)
duplicate_src_id = src_id[duplicate_idx]
duplicate_dst_id = dst_id[duplicate_idx]
duplicate_original_edge_id = original_edge_id[duplicate_idx]

src_id = src_id[idx]
dst_id = dst_id[idx]
original_edge_id = original_edge_id[idx]
end_time = time.time()
print("length of src_id after removing duplicated edges = {}".format(len(src_id)))
print("elapsed time of removing duplicated edges = {}ms".format((end_time - start_time)*1000.0))

src_id = torch.from_numpy(src_id)
dst_id = torch.from_numpy(dst_id)
original_edge_id = torch.from_numpy(original_edge_id)
edge_type = torch.zeros(len(src_id))
edge_data = torch.stack([src_id, dst_id, original_edge_id, edge_type], 1)
print(edge_data)
print(edge_data.shape)
np.savetxt(os.path.join(dir_path, "{}_edges.txt".format(graph_name)), edge_data.numpy(), fmt='%d', delimiter=' ')

self_loop_src_id = torch.from_numpy(self_loop_src_id)
self_loop_dst_id = torch.from_numpy(self_loop_dst_id)
self_loop_original_edge_id = torch.from_numpy(self_loop_original_edge_id)
duplicate_src_id = torch.from_numpy(duplicate_src_id)
duplicate_dst_id = torch.from_numpy(duplicate_dst_id)
duplicate_original_edge_id = torch.from_numpy(duplicate_original_edge_id)

removed_edge_data = torch.stack([torch.cat([self_loop_src_id, duplicate_src_id]),
				 torch.cat([self_loop_dst_id, duplicate_dst_id]),
				 torch.cat([self_loop_original_edge_id, duplicate_original_edge_id]),
				 torch.cat([torch.zeros(len(self_loop_src_id)), torch.zeros(len(duplicate_src_id))])],
				1)

print(removed_edge_data)
print(removed_edge_data.shape)
np.savetxt(os.path.join(dir_path, "{}_removed_edges.txt".format(graph_name)), removed_edge_data.numpy(), fmt='%d', delimiter=' ')

###### process node ######
node_type = torch.zeros(num_nodes)
node_weight = torch.ones(num_nodes)
node_id = torch.arange(num_nodes)
node_data = torch.stack([node_type, node_weight, node_id], 1)
print(node_data)
print(node_data.shape)
np.savetxt(os.path.join(dir_path, "{}_nodes.txt".format(graph_name)), node_data.numpy(), fmt='%d', delimiter=' ')

###### process stat file ######
num_node_weights = 1
graph_stats = [num_nodes, len(src_id), num_node_weights]
print(graph_stats)
with open (os.path.join(dir_path, "{}_stats.txt".format(graph_name)), 'w') as f:
	for i in graph_stats:
		f.write(str(i))
		f.write(" ")
