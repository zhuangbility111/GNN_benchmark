from ogb.nodeproppred import NodePropPredDataset
import torch
import numpy as np
import time
import argparse
import os
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default='./', help='The path of output directory.')
parser.add_argument('--in_dir', type=str, default='./', help='The path of input (dataset) directory.')
parser.add_argument('--dataset', type=str, help='The name of input dataset.')
parser.add_argument('--graph_name', type=str, help='The name of graph.')
args = parser.parse_args()
out_dir = args.out_dir
in_dir = args.in_dir
dataset = args.dataset
graph_name = args.graph_name

dataset = NodePropPredDataset(name = dataset, root=in_dir)
graph, node_label = dataset[0]
num_nodes = graph['num_nodes']

###### process nodes labels ######
print(node_label)
print(node_label.shape)
# np.savetxt(os.path.join(out_dir, "{}_nodes_label.txt".format(graph_name)), node_label, fmt="%d", delimiter=' ')
np.save(os.path.join(out_dir, "{}_nodes_label.npy".format(graph_name)), node_label)

###### process nodes features ######
node_feat = graph['node_feat']
# np.savetxt(os.path.join(out_dir, "{}_nodes_feat.npy".format(graph_name)), node_feat)
np.save(os.path.join(out_dir, "{}_nodes_feat.npy".format(graph_name)), node_feat)
print(num_nodes)
print(node_feat.shape)
del node_feat
gc.collect()

###### process node mask (training, test, validated) ######
dataset_mask = dataset.get_idx_split()
train_idx, valid_idx, test_idx = dataset_mask["train"], dataset_mask["valid"], dataset_mask["test"]
# np.savetxt(os.path.join(out_dir, "{}_nodes_train_idx.txt".format(graph_name)), train_idx, fmt="%d", delimiter=' ')
np.save(os.path.join(out_dir, "{}_nodes_train_idx.npy".format(graph_name)), train_idx)
# np.savetxt(os.path.join(out_dir, "{}_nodes_valid_idx.txt".format(graph_name)), valid_idx, fmt="%d", delimiter=' ')
np.save(os.path.join(out_dir, "{}_nodes_valid_idx.npy".format(graph_name)), valid_idx)
# np.savetxt(os.path.join(out_dir, "{}_nodes_test_idx.txt".format(graph_name)), test_idx, fmt="%d", delimiter=' ')
np.save(os.path.join(out_dir, "{}_nodes_test_idx.npy".format(graph_name)), test_idx)
print("save training, valid, test idx successfully.")

###### process edge ######
edge_index = graph['edge_index']
src_id, dst_id = edge_index
original_edge_id = np.arange(len(src_id), dtype=np.int64)
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
duplicate_idx = np.setdiff1d(np.arange(len(ids), dtype=np.int64), idx)
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
edge_type = torch.zeros(len(src_id), dtype=torch.int64)
edge_data = torch.stack([src_id, dst_id, original_edge_id, edge_type], 1)
print(edge_data)
print(edge_data.shape)
np.savetxt(os.path.join(out_dir, "{}_edges.txt".format(graph_name)), edge_data.numpy(), fmt='%d', delimiter=' ')

self_loop_src_id = torch.from_numpy(self_loop_src_id)
self_loop_dst_id = torch.from_numpy(self_loop_dst_id)
self_loop_original_edge_id = torch.from_numpy(self_loop_original_edge_id)
duplicate_src_id = torch.from_numpy(duplicate_src_id)
duplicate_dst_id = torch.from_numpy(duplicate_dst_id)
duplicate_original_edge_id = torch.from_numpy(duplicate_original_edge_id)

removed_edge_data = torch.stack([torch.cat([self_loop_src_id, duplicate_src_id]),
				 torch.cat([self_loop_dst_id, duplicate_dst_id]),
				 torch.cat([self_loop_original_edge_id, duplicate_original_edge_id]),
				 torch.cat([torch.zeros(len(self_loop_src_id), dtype=torch.int64), torch.zeros(len(duplicate_src_id), dtype=torch.int64)])],
				1)

print(removed_edge_data)
print(removed_edge_data.shape)
np.savetxt(os.path.join(out_dir, "{}_removed_edges.txt".format(graph_name)), removed_edge_data.numpy(), fmt='%d', delimiter=' ')

###### process node ######
node_weight = []
node_type = torch.zeros(num_nodes, dtype=torch.int64)
node_weight.append(torch.ones(num_nodes, dtype=torch.int64))
# train_idx will also be used as node weight
node_train_idx = torch.zeros(num_nodes, dtype=torch.int64)
node_train_idx[train_idx] = 1
node_weight.append(node_train_idx)
node_id = torch.arange(num_nodes, dtype=torch.int64)
# node_data = torch.stack([node_type, node_weight, node_id], 1)
node_data = torch.stack([node_type, node_weight[0], node_weight[1], node_id], 1)
print(node_data)
print(node_data.shape)
np.savetxt(os.path.join(out_dir, "{}_nodes.txt".format(graph_name)), node_data.numpy(), fmt='%d', delimiter=' ')

###### process stat file ######
num_node_weights = len(node_weight)
graph_stats = [num_nodes, len(src_id), num_node_weights]
print(graph_stats)
with open (os.path.join(out_dir, "{}_stats.txt".format(graph_name)), 'w') as f:
	for i in graph_stats:
		f.write(str(i))
		f.write(" ")
