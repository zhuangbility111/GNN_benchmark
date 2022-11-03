import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn import DistGCNConv
from torch_geometric.nn import DistGCNConvGrad
from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
import time
import argparse
import os

class DistGCNGrad(torch.nn.Module):
    def __init__(self, local_nodes_required_by_other, 
                 remote_nodes_id_list, 
                 num_remote_nodes_on_part, 
                 local_range_nodes_on_part,
                 rank,
                 num_part,
                 cached):
        super().__init__()
        num_node_features = 128
        num_classes = 40
        num_hidden_channels = 256
        self.conv1 = DistGCNConvGrad(num_node_features, num_hidden_channels, 
                                 local_nodes_required_by_other, 
                                 remote_nodes_id_list,
                                 num_remote_nodes_on_part,
                                 local_range_nodes_on_part,
                                 rank,
                                 num_part,
                                 cached=cached)
        self.conv2 = DistGCNConvGrad(num_hidden_channels, num_classes,
                                 local_nodes_required_by_other, 
                                 remote_nodes_id_list,
                                 num_remote_nodes_on_part,
                                 local_range_nodes_on_part,
                                 rank,
                                 num_part,
                                 cached=cached)

    # forward() for tensor
    def forward(self, x, local_edges_list, remote_edges_list):
        x = self.conv1(x, local_edges_list, remote_edges_list)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, local_edges_list, remote_edges_list)
        return F.log_softmax(x, dim=1)

class GCN(torch.nn.Module):
    def __init__(self, cached):
        super().__init__()
        num_node_features = 128
        num_classes = 40
        num_hidden_channels = 256
        self.conv1 = GCNConv(num_node_features, num_hidden_channels, cached=cached)
        self.conv2 = GCNConv(num_hidden_channels, num_classes, cached=cached)

    # forward() for tensor
    def forward(self, x, edge):
        # x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge)
        return F.log_softmax(x, dim=1)

# compare two array and remap the elem in train_idx according to the mapping in nodes_id_list
# global id is mapped to local id in train_idx
def compare_array(train_idx, nodes_id_list, nodes_id_begin):
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
            local_train_idx.append(nodes_id_list[idx_in_node][0].item() - nodes_id_begin)
            idx_in_mask += 1
            idx_in_node += 1
    
    local_train_idx = torch.Tensor(local_train_idx).long()
    return local_train_idx

# To remap the training mask, test mask and validated mask according to new node id assignment
def remap_dataset_mask(dataset_mask, nodes_id_list, rank):
    remap_start = time.perf_counter()
    train_idx, valid_idx, test_idx = dataset_mask["train"], dataset_mask["valid"], dataset_mask["test"]
    nodes_id_begin = nodes_id_list[0][0]
    # sort the original nodes_list according to the order of global id
    # this one could be eliminated by graph partitioning
    nodes_id_list = nodes_id_list[nodes_id_list[:,1].argsort()]

    # remap training mask
    local_train_idx = compare_array(train_idx, nodes_id_list, nodes_id_begin)

    # remap validated mask
    local_valid_idx = compare_array(valid_idx, nodes_id_list, nodes_id_begin)

    # remap test mask
    local_test_idx = compare_array(test_idx, nodes_id_list, nodes_id_begin)
    remap_end = time.perf_counter()
    if rank == 0:
        print("elapsed time of ramapping dataset mask(ms) = {}".format((remap_end - remap_start) * 1000))

    return local_train_idx, local_valid_idx, local_test_idx

def load_graph_data(dir_path, graph_name, rank, world_size):
    # load vertices on subgraph
    load_nodes_start = time.perf_counter()
    local_nodes_id_list = np.loadtxt(os.path.join(dir_path, "p{:0>3d}-{}_nodes.txt".format(rank, graph_name)), dtype='int64', delimiter=' ', usecols=(0, 3))
    # node_id_list.sort()
    # print(local_nodes_id_list.shape)
    nodes_id_begin = local_nodes_id_list[0][0]
    nodes_id_end = local_nodes_id_list[local_nodes_id_list.shape[0]-1][0]
    print("nodes_id_range: {} - {}".format(nodes_id_begin, nodes_id_end))
    num_local_nodes = nodes_id_end - nodes_id_begin + 1
    load_nodes_end = time.perf_counter()
    time_load_nodes = load_nodes_end - load_nodes_start

    # load features of vertices on subgraph
    nodes_feat_list = np.loadtxt(os.path.join(dir_path, "p{:0>3d}-{}_nodes_feat.txt".format(rank, graph_name)), dtype='float32', delimiter=' ')
    load_nodes_feats_end = time.perf_counter()
    time_load_nodes_feats = load_nodes_feats_end - load_nodes_end

    # load labels of vertices on subgraph
    nodes_label_list = np.loadtxt(os.path.join(dir_path, "p{:0>3d}-{}_nodes_label.txt".format(rank, graph_name)), dtype='int64', delimiter=' ')
    load_nodes_labels_end = time.perf_counter()
    time_load_nodes_labels = load_nodes_labels_end - load_nodes_feats_end

    # load edges on subgraph
    edges_list = np.loadtxt(os.path.join(dir_path, "p{:0>3d}-{}_edges.txt".format(rank, graph_name)), dtype='int64', delimiter=' ')
    load_edges_list_end = time.perf_counter()
    time_load_edges_list = load_edges_list_end - load_nodes_labels_end

    # load number of nodes on each subgraph
    range_nodes_on_part = np.loadtxt(os.path.join(dir_path, "begin_node_on_each_partition.txt"), dtype='int64', delimiter=' ')
    load_number_nodes_end = time.perf_counter()
    time_load_number_nodes = load_number_nodes_end - load_edges_list_end

    # divide the global edges list into the local edges list and the remote edges list
    local_edges_list = [[], []]
    remote_edges_list = [[], []]
    for edge in edges_list:
        # only the src node of each edges can be the remote node
        if edge[0] >= nodes_id_begin and edge[0] <= nodes_id_end:
            edge[0] -= nodes_id_begin
            edge[1] -= nodes_id_begin
            local_edges_list[0].append(edge[0])
            local_edges_list[1].append(edge[1])
        else:
            edge[1] -= nodes_id_begin
            remote_edges_list[0].append(edge[0])
            remote_edges_list[1].append(edge[1])
    
    local_edges_list = np.array(local_edges_list, dtype= np.int64)
    remote_edges_list = np.array(remote_edges_list, dtype=np.int64)
    divide_edges_list_end = time.perf_counter()
    time_divide_edges_list = divide_edges_list_end - load_number_nodes_end

    # sort remote_edges_list based on the src(remote) nodes' global id
    remote_edges_row, remote_edges_col = remote_edges_list[0], remote_edges_list[1]
    sort_index = np.argsort(remote_edges_row)
    remote_edges_list[0] = remote_edges_row[sort_index]
    remote_edges_list[1] = remote_edges_col[sort_index]
    sort_remote_edges_list_end = time.perf_counter()
    time_sort_remote_edges_list = sort_remote_edges_list_end - divide_edges_list_end

    # obatain remote nodes list 
    # remove duplicated nodes
    # remap the global id of remote nodes to local id on remote_edge_list
    remote_nodes_id_list = []
    local_range_nodes_on_part = torch.zeros(world_size+1, dtype=torch.int64)
    num_remote_nodes_on_part = torch.zeros(world_size, dtype=torch.int64)
    remote_edges_row = remote_edges_list[0]

    part_idx = 0
    local_node_idx = num_local_nodes - 1
    prev_node = -1
    tmp_len = len(remote_edges_row)
    for i in range(0, tmp_len):
        cur_node = remote_edges_row[i]
        if cur_node != prev_node:
            remote_nodes_id_list.append(cur_node)
            local_node_idx += 1
            while cur_node >= range_nodes_on_part[part_idx+1]:
                part_idx += 1
                local_range_nodes_on_part[part_idx+1] = local_range_nodes_on_part[part_idx]
            local_range_nodes_on_part[part_idx+1] += 1
            num_remote_nodes_on_part[part_idx] += 1
        prev_node = cur_node
        remote_edges_row[i] = local_node_idx

    for i in range(part_idx+1, world_size):
        local_range_nodes_on_part[i+1] = local_range_nodes_on_part[i]

    remote_nodes_id_list = np.array(remote_nodes_id_list)
    obtain_remote_nodes_list_end = time.perf_counter()
    time_obtain_remote_nodes_list = obtain_remote_nodes_list_end - sort_remote_edges_list_end
    time_load_and_preprocessing_graph = obtain_remote_nodes_list_end - load_nodes_start
    print("elapsed time of loading nodes(ms) = {}".format(time_load_nodes * 1000))
    print("elapsed time of loading nodes feats(ms) = {}".format(time_load_nodes_feats * 1000))
    print("elapsed time of loading nodes labels(ms) = {}".format(time_load_nodes_labels * 1000))
    print("elapsed time of loading edges(ms) = {}".format(time_load_edges_list * 1000))
    print("elapsed time of loading number of nodes(ms) = {}".format(time_load_number_nodes * 1000))
    print("elapsed time of dividing edges(ms) = {}".format(time_divide_edges_list * 1000))
    print("elapsed time of sorting edges(ms) = {}".format(time_sort_remote_edges_list * 1000))
    print("elapsed time of obtaining remote nodes(ms) = {}".format(time_obtain_remote_nodes_list * 1000))
    print("elapsed time of whole process of loading graph(ms) = {}".format(time_load_and_preprocessing_graph * 1000))
    print("number of remote nodes = {}".format(remote_nodes_id_list.shape[0]))

    return torch.from_numpy(local_nodes_id_list), torch.from_numpy(nodes_feat_list), \
           torch.from_numpy(nodes_label_list), \
           torch.from_numpy(remote_nodes_id_list), \
           local_range_nodes_on_part, num_remote_nodes_on_part, \
           torch.from_numpy(local_edges_list), torch.from_numpy(remote_edges_list)

def init_dist_group():
    if dist.is_mpi_available():
        print("mpi in torch.distributed is available!")
        dist.init_process_group(backend="mpi")
    assert torch.distributed.is_initialized()
    print(f"dist_info RANK: {dist.get_rank()}, SIZE: {dist.get_world_size()}")
    # number of process in this MPI group
    world_size = dist.get_world_size() 
    # mpi rank in this MPI group
    rank = dist.get_rank()
    # assert 
    return (rank, world_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor_type', type=str, default='tensor')
    parser.add_argument('--use_profiler', type=str, default='false')
    parser.add_argument('--cached', type=str, default='true')
    args = parser.parse_args()
    tensor_type = args.tensor_type
    use_profiler = args.use_profiler
    if args.cached == 'false':
        cached = False
    elif args.cached == 'true':
        cached = True

    rank, world_size = init_dist_group()
    torch.set_num_threads(12)
    print("Rank = {}, Number of threads = {}".format(rank, torch.get_num_threads()))

    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv') 

    # obtain graph information
    local_nodes_id_list, nodes_feat_list, nodes_label_list, \
        remote_nodes_id_list, local_range_nodes_on_part, num_remote_nodes_on_part, \
        local_edges_list, remote_edges_list = load_graph_data("./arxiv_graph_1_part/", "arxiv", rank, world_size)

    # obtain training, validated, testing mask
    local_train_mask, local_valid_mask, local_test_mask = remap_dataset_mask(dataset.get_idx_split(), local_nodes_id_list, rank)

    # send the number of remote nodes we need to obtain from other subgrpah
    obtain_number_remote_nodes_start = time.perf_counter()
    send_num_nodes = [torch.tensor([num_remote_nodes_on_part[i]], dtype=torch.int64) for i in range(world_size)]
    recv_num_nodes = [torch.zeros(1, dtype=torch.int64) for i in range(world_size)]
    dist.all_to_all(recv_num_nodes, send_num_nodes)
    num_local_nodes_required_by_other = recv_num_nodes
    obtain_number_remote_nodes_end = time.perf_counter()
    print("elapsed time of obtaining number of remote nodes(ms) = {}".format((obtain_number_remote_nodes_end - obtain_number_remote_nodes_start) * 1000))
    print("num_local_nodes_required_by_other:")
    print(num_local_nodes_required_by_other)

    # then we need to send the nodes_list which include the id of remote nodes we want
    # and receive the nodes_list from other subgraphs
    obtain_remote_nodes_list_start = time.perf_counter()
    send_nodes_list = [remote_nodes_id_list[local_range_nodes_on_part[i]: local_range_nodes_on_part[i+1]]
                            for i in range(world_size)]
    recv_nodes_list = [torch.zeros(num_local_nodes_required_by_other[i], dtype=torch.int64) for i in range(world_size)]
    dist.all_to_all(recv_nodes_list, send_nodes_list)
    local_nodes_id_begin = local_nodes_id_list[0][0]
    local_nodes_required_by_other = [i - local_nodes_id_begin for i in recv_nodes_list]
    obtain_remote_nodes_list_end = time.perf_counter()
    print("elapsed time of obtaining list of remote nodes(ms) = {}".format((obtain_remote_nodes_list_end - obtain_remote_nodes_list_start) * 1000))
    
    # start training
    start = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(cached).to(device)
    '''
    model = DistGCNGrad(local_nodes_required_by_other,
                            remote_nodes_id_list,
                            num_remote_nodes_on_part,
                            local_range_nodes_on_part,
                            rank,
                            world_size,
                            cached).to(device)
    '''
    '''
    # load model parameters 
    model.load_state_dict(torch.load('./GCNNet.pt'))
    for param in model.parameters():
        print(param)
    '''
    
    # model = torch.nn.parallel.DistributedDataParallel(model)
    for name, paramer in model.named_parameters():
        print(name, paramer)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()

    total_forward_dur = 0
    total_backward_dur = 0
    total_share_grad_dur = 0
    total_update_weight_dur = 0

    for epoch in range(200):
        optimizer.zero_grad()
        forward_start = time.perf_counter()
        if tensor_type == 'tensor':
            # out = model(nodes_feat_list, local_edges_list, remote_edges_list)
            out = model(nodes_feat_list, local_edges_list)
        elif tensor_type == 'sparse_tensor':
            print("not support yet.")
            # out = model(nodes_feat_list, data.adj_t)
        backward_start = time.perf_counter()
        loss = F.nll_loss(out[local_train_mask], nodes_label_list[local_train_mask])
        loss.backward()

        share_grad_start = time.perf_counter()
        # communicate gradients
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= float(world_size)

        update_weight_start = time.perf_counter()
        optimizer.step()
        print("rank: {}, epoch: {}, loss: {}".format(rank, epoch, loss.item()))
        update_weight_end = time.perf_counter()
        total_forward_dur += backward_start - forward_start
        total_backward_dur += share_grad_start - backward_start
        total_share_grad_dur += update_weight_start - share_grad_start
        total_update_weight_dur += update_weight_end - update_weight_start
    end = time.perf_counter()
    total_training_dur = (end - start)
    
    total_forward_dur = torch.tensor([total_forward_dur])
    total_backward_dur = torch.tensor([total_backward_dur])
    total_share_grad_dur = torch.tensor([total_share_grad_dur])
    total_update_weight_dur = torch.tensor([total_update_weight_dur])
    total_training_dur = torch.tensor([total_training_dur])

    dist.reduce(total_forward_dur, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total_backward_dur, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total_share_grad_dur, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total_update_weight_dur, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total_training_dur, 0, op=dist.ReduceOp.SUM)

    print("training end.")
    print("forward_time(ms): {}".format(total_forward_dur[0] / float(world_size) * 1000))
    print("backward_time(ms): {}".format(total_backward_dur[0] / float(world_size) * 1000))
    print("share_grad_time(ms): {}".format(total_share_grad_dur[0] / float(world_size) * 1000))
    print("update_weight_time(ms): {}".format(total_update_weight_dur[0] / float(world_size) * 1000))
    print("total_training_time(ms): {}".format(total_training_dur[0] / float(world_size) * 1000))

    '''
    # save model parameters
    if rank == 0:
        for name, parameters in model.named_parameters():
            print(name,':',parameters.size())
        torch.save(model.state_dict(), 'GCNNet_mpi.pt')
    '''

    # check accuracy
    model.eval()
    predict_result = []
    if tensor_type == 'tensor':
        # out, accs = model(nodes_feat_list, local_edges_list, remote_edges_list), []
        out, accs = model(nodes_feat_list, local_edges_list), []
    elif tensor_type == 'sparse_tensor':
        print("not support yet.")
    for mask in (local_train_mask, local_valid_mask, local_test_mask):
        num_correct_data = (out[mask].argmax(-1) == nodes_label_list[mask]).sum()
        num_data = mask.size(0)
        print("local num_correct_data = {}, local num_entire_dataset = {}".format(num_correct_data, num_data))
        predict_result.append(num_correct_data) 
        predict_result.append(num_data)
        # acc = float((out[mask].argmax(-1) == nodes_label_list[mask]).sum() / mask.size(0))
        # accs.append(acc)
    predict_result = torch.tensor(predict_result)
    dist.reduce(predict_result, 0, op=dist.ReduceOp.SUM)

    if rank == 0:
        train_acc = float(predict_result[0] / predict_result[1])
        val_acc = float(predict_result[2] / predict_result[3])
        test_acc = float(predict_result[4] / predict_result[5])
        print("size of correct training sample = {}, size of correct valid sample = {}, size of correct test sample = {}".format(predict_result[0], predict_result[2], predict_result[4]))
        print("size of all training sample = {}, size of all valid sample = {}, size of all test sample = {}".format(predict_result[1], predict_result[3], predict_result[5]))
        # print(f'Rank: {rank}, World_size: {world_size}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
        print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
