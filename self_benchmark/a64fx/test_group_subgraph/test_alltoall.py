import torch
import torch.distributed as dist
import os
import numpy as np
import time

import argparse

from mpi4py import MPI

torch.set_num_threads(1)
mpi4py_available = True

def init_dist_group():
    comm = None
    if dist.is_mpi_available():
        if mpi4py_available:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            world_size = comm.Get_size()
        print("mpi in mpi4py is available!")
        dist.init_process_group(backend="mpi")
        print("mpi in torch.distributed is available!")
    else:
        try:
            import torch_ccl
        except ImportError as e:
            print(e)
        world_size = int(os.environ.get("PMI_SIZE", -1))
        rank = int(os.environ.get("PMI_RANK", -1))
        print("PMI_SIZE = {}".format(world_size))
        print("PMI_RANK = {}".format(rank))
        print("use ccl backend for torch.distributed package on x86 cpu.")
        dist_url = "env://"
        dist.init_process_group(backend="ccl", init_method="env://", 
                                world_size=world_size, rank=rank)
    # assert torch.distributed.is_initialized()
    # print(f"dist_info RANK: {dist.get_rank()}, SIZE: {dist.get_world_size()}")
    # number of process in this MPI group
    world_size = dist.get_world_size() 
    # mpi rank in this MPI group
    rank = dist.get_rank()
    return (comm, rank, world_size)

def init_subcommunicator(rank, labels, ranks_list_in_each_cluster):
    groups = []
    num_labels = np.max(labels) + 1
    for i in range(num_labels):
        if mpi4py_available:
            key = 0
            if rank in ranks_list_in_each_cluster[i]:
                color = i
            else: 
                color = MPI.UNDEFINED
            groups.append(MPI.COMM_WORLD.Split(color, key))
        else: 
            groups.append(dist.new_group(ranks_list_in_each_cluster[i], backend="mpi"))
    return groups

def construct_result(recv_labels, data_recv_from_other_rank, recv_bufs_list, num_communicators):
    data_range = dict()
    data_idx = dict()
    for label in range(num_communicators):
        idx = (recv_labels == label).nonzero()[0]
        if idx.shape[0] > 0:
            data_volume = np.zeros(idx.shape[0] + 1, dtype=np.int64)
            for j in range(idx.shape[0]):
                data_volume[j+1] = data_volume[j] + data_recv_from_other_rank[idx[j]]
            
            data_range[label] = data_volume
            data_idx[label] = 0

    res = []
    for recv_rank in range(data_recv_from_other_rank.shape[0]):
        label = recv_labels[recv_rank]
        if label >= 0:
            begin_idx = data_range[label][data_idx[label]]
            end_idx = data_range[label][data_idx[label] + 1]
            res.append(recv_bufs_list[label][begin_idx: end_idx])
            data_idx[label] += 1
            # print("begin idx and end idx in rank {} from label {} = {}, {}".format(recv_rank, label, begin_idx, end_idx))
    
    # print(res)
    res = torch.cat(res, dim=0)
    return res

def construct_send_buf(send_labels, data_send_to_other_rank, send_buf):
    send_bufs_list = dict()
    idx_send_buf = 0
    for send_rank in range(data_send_to_other_rank.shape[0]):
        label = send_labels[send_rank]
        # label < 0 means current rank doesn't need to send data to `send_rank` as they are not in the same communicator
        if label >= 0:
            # to ensure whether the label is in the dict's keys
            if label not in send_bufs_list.keys():
                send_bufs_list[label] = []
            send_bufs_list[label].append(send_buf[idx_send_buf: idx_send_buf+data_send_to_other_rank[send_rank]])
            idx_send_buf += data_send_to_other_rank[send_rank]

    for label in send_bufs_list.keys():
        send_bufs_list[label] = torch.cat(send_bufs_list[label], dim=0)
    
    return send_bufs_list
    
def test_group_alltoall(rank, groups, labels, comm_matrix, global_rank_to_group_rank, send_buf, feat_len):
    recv_bufs_list = dict()

    data_send_to_other_rank = comm_matrix[rank]
    data_recv_from_other_rank = comm_matrix[:, rank]

    send_labels = labels[rank]
    recv_labels = labels[:, rank]

    num_communicators = np.max(recv_labels) + 1

    print("num_communicators = {}".format(num_communicators))

    send_bufs_list = construct_send_buf(send_labels, data_send_to_other_rank, send_buf)

    for label in range(num_communicators):
        recv_rank = (recv_labels == label).nonzero()[0]
        if recv_rank.shape[0] > 0:
            recv_data_volume = np.sum(data_recv_from_other_rank[recv_rank])
            recv_bufs_list[label] = torch.zeros((recv_data_volume, feat_len), dtype=torch.float32)

    repeat = 21
    total_comm_time_list = np.zeros(repeat, dtype=np.float32)

    for n in range(repeat):
        total_comm_time = 0.0
        handle_list = []
        for label in range(num_communicators):
            group_size = dist.get_world_size(group=groups[label])
            if group_size != -1:
                send_splits = np.zeros(group_size, dtype=np.int64)
                recv_splits = np.zeros(group_size, dtype=np.int64)

                send_rank = (send_labels == label).nonzero()[0]
                recv_rank = (recv_labels == label).nonzero()[0]
                
                send_splits[global_rank_to_group_rank[label][send_rank]] = data_send_to_other_rank[send_rank]
                recv_splits[global_rank_to_group_rank[label][recv_rank]] = data_recv_from_other_rank[recv_rank]

                # print("world_size in group {} = {}".format(i, dist.get_world_size(group=groups[i])), flush=True)
                # print("send_splits.size() = {}".format(len(send_splits)), flush=True)
                # print("recv_splits.size() = {}".format(len(recv_splits)), flush=True)
                assert dist.get_world_size(group=groups[label]) == len(send_splits) and \
                        dist.get_world_size(group=groups[label]) == len(recv_splits)
                
                dist.barrier(group=groups[label])
                begin = time.perf_counter()
                handle = dist.all_to_all_single(recv_bufs_list[label], send_bufs_list[label], \
                                                recv_splits.tolist(), send_splits.tolist(), \
                                                groups[label], async_op=True)
                
                handle.wait()
                end = time.perf_counter()
                print("elasped time of all_to_all in group {} = {}ms".format(label, (end - begin)*1000.0), flush=True)
                total_comm_time += (end - begin)*1000.0
                # handle_list.append(handle)

        '''
        for handle in handle_list:
            print("handle.wait() start!", flush=True)
            handle.wait()
            print("handle.wait() finish!", flush=True)
        '''

        print("total_comm_time = {}ms".format(total_comm_time))
        total_comm_time_list[n] = total_comm_time
    
    print("average elapsed time of group all_to_all in ({}) repeats = {}ms".format(repeat - 1, np.mean(total_comm_time_list[1:])))

    res = construct_result(recv_labels, data_recv_from_other_rank, recv_bufs_list, num_communicators)

    return res

def p2p_communicate(rank, comm, p2p_labels_matrix, comm_matirx, \
                    send_data_range_list, recv_data_range_list, \
                    send_buf, recv_buf, feat_len):
    p2p_send_to_ranks_list   = (p2p_labels_matrix[rank, :] == 1).nonzero()[0]
    p2p_recv_from_ranks_list = (p2p_labels_matrix[:, rank] == 1).nonzero()[0]

    send_bufs_list = list()
    recv_bufs_list = list()
    for send_to_rank in p2p_send_to_ranks_list:
        send_bufs_list.append(send_buf[send_data_range_list[send_to_rank]: send_data_range_list[send_to_rank+1]])

    for recv_from_rank in p2p_recv_from_ranks_list:
        recv_bufs_list.append(recv_buf[recv_data_range_list[recv_from_rank]: recv_data_range_list[recv_from_rank+1]])

    send_handles = list()
    recv_handles = list()

    begin = time.perf_counter()
    # send data to other ranks
    for i in range(p2p_send_to_ranks_list.shape[0]):
        if send_bufs_list[i].shape[0] > 0:
            if mpi4py_available:
                send_handles.append(comm.Isend(send_bufs_list[i], dest=p2p_send_to_ranks_list[i], tag=11))
    
    # recv data from other ranks
    for i in range(p2p_recv_from_ranks_list.shape[0]):
        if recv_bufs_list[i].shape[0] > 0:
            if mpi4py_available:
                recv_handles.append(comm.Irecv(recv_bufs_list[i], source=p2p_recv_from_ranks_list[i], tag=11))
    
    MPI.Request.Waitall(send_handles)
    MPI.Request.Waitall(recv_handles)
    end = time.perf_counter()
    print("p2p comm time = {}ms".format((end - begin) * 1000.0), flush=True)
    # print("p2p is finished!")

# def collective_communicate(rank, groups, collective_labels_matrix, \
#                            comm_matrix, global_rank_to_group_rank, \
#                            send_data_range_list, recv_data_range_list, \
#                            send_buf, recv_buf, feat_len=128):
#     send_labels = collective_labels_matrix[rank]
#     recv_labels = collective_labels_matrix[:, rank]

#     data_send_to_other_rank = comm_matrix[rank]
#     data_recv_from_other_rank = comm_matrix[:, rank]

#     unique_send_labels = np.unique(send_labels)
            
#     send_bufs_list = list()
#     recv_bufs_list = list()

#     for label in unique_send_labels:
#         group_size = dist.get_world_size(group=groups[label])
#         if group_size != -1 and label >= 0:

#             send_splits = np.zeros(group_size, dtype=np.int64)
#             recv_splits = np.zeros(group_size, dtype=np.int64)

#             send_rank = (send_labels == label).nonzero()[0]
#             recv_rank = (recv_labels == label).nonzero()[0]

#             for i in send_rank:
#                 send_bufs_list.append(send_buf[send_data_range_list[i]: send_data_range_list[i+1]])
#             send_bufs_list = torch.cat(send_bufs_list, dim=0)
            
#             for i in recv_rank:
#                 recv_bufs_list.append(recv_buf[recv_data_range_list[i]: recv_data_range_list[i+1]])
#             recv_bufs_list = torch.cat(recv_bufs_list, dim=0)

#             send_splits[global_rank_to_group_rank[label][send_rank]] = data_send_to_other_rank[send_rank]
#             recv_splits[global_rank_to_group_rank[label][recv_rank]] = data_recv_from_other_rank[recv_rank]

#             assert dist.get_world_size(group=groups[label]) == len(send_splits) and \
#                     dist.get_world_size(group=groups[label]) == len(recv_splits)

#             # dist.barrier(group=groups[label])
#             repeat = 21
#             total_comm_time_list = np.zeros(repeat, dtype=np.float32)
#             for n in range(repeat):
#                 dist.barrier(group=groups[label])
#                 begin = time.perf_counter()
#                 # handle = dist.all_to_all(send_bufs_list, recv_bufs_list, groups[label], async_op=True)
#                 handle = dist.all_to_all_single(recv_bufs_list, send_bufs_list, \
#                                                 recv_splits.tolist(), send_splits.tolist(), \
#                                                 groups[label], async_op=True)
#                 handle.wait()
#                 end = time.perf_counter()
#                 print("total group_alltoall time = {}ms".format((end - begin) * 1000.0), flush=True)
#                 total_comm_time_list[n] = (end - begin) * 1000.0
#             print("average elapsed time of group all_to_all in ({}) repeats = {}ms".format(repeat - 1, np.mean(total_comm_time_list[1:])))

def collective_communicate_mpi4py(rank, groups, collective_labels_matrix, \
                                    comm_matrix, global_rank_to_group_rank, \
                                    send_data_range_list, recv_data_range_list, \
                                    send_buf, recv_buf, feat_len):
                                
    data_send_to_other_rank = comm_matrix[rank]
    data_recv_from_other_rank = comm_matrix[:, rank]

    send_labels = collective_labels_matrix[rank]
    recv_labels = collective_labels_matrix[:, rank]
    
    num_communicators = np.max(recv_labels) + 1

    print("num_communicators = {}".format(num_communicators))

    for label in range(num_communicators):
        if groups[label] != MPI.COMM_NULL:

            send_labels[rank] = label
            recv_labels[rank] = label

            # local_size = groups[label].Get_size()
            send_rank = (send_labels == label).nonzero()[0]
            recv_rank = (recv_labels == label).nonzero()[0]

            send_displs = send_data_range_list[send_rank] * feat_len
            recv_displs = recv_data_range_list[recv_rank] * feat_len

            send_counts = data_send_to_other_rank[send_rank] * feat_len
            recv_counts = data_recv_from_other_rank[recv_rank] * feat_len

            # print("label = {}, shape of send_buf = {}, shape of send_displs = {}, shape of send_counts = {}".format(label, send_buf.shape, send_displs.shape, send_counts.shape), flush=True)
            # print("groups[label] = {}".format(groups[label]), flush=True)

            begin = time.perf_counter()
            groups[label].Alltoallv([send_buf, send_counts, send_displs, MPI.FLOAT], \
                                    [recv_buf, recv_counts, recv_displs, MPI.FLOAT],)
            end = time.perf_counter()
            print("label = {}, collective alltoall time = {}ms".format(label, (end - begin) * 1000.0), flush=True)

    return None

def collective_communicate(rank, groups, collective_labels_matrix, \
                           comm_matrix, global_rank_to_group_rank, \
                           send_data_range_list, recv_data_range_list, \
                           send_buf, recv_buf, feat_len):
    if mpi4py_available:
        return collective_communicate_mpi4py(rank, groups, collective_labels_matrix, \
                                                comm_matrix, global_rank_to_group_rank, \
                                                send_data_range_list, recv_data_range_list, \
                                                send_buf, recv_buf, feat_len)
    return test_group_alltoall(rank, groups, collective_labels_matrix, comm_matrix, global_rank_to_group_rank, send_buf, feat_len)
    
# p2p and collective communication are mixed together
def test_group_alltoall_v2(rank, groups, comm, collective_labels_matrix, p2p_labels_matrix, \
                           comm_matrix, global_rank_to_group_rank, send_buf, feat_len):
    send_data_range_list = np.zeros(comm_matrix.shape[0] + 1, dtype=np.int64)
    recv_data_range_list = np.zeros(comm_matrix.shape[0] + 1, dtype=np.int64)

    recv_buf = torch.zeros((comm_matrix[:, rank].sum(), feat_len), dtype=torch.float32)

    for i in range(comm_matrix.shape[0]):
        send_data_range_list[i+1] = send_data_range_list[i] + comm_matrix[rank, i]
        recv_data_range_list[i+1] = recv_data_range_list[i] + comm_matrix[i, rank]
    
    repeat = 21
    # total_comm_time_list = np.zeros(repeat, dtype=np.float32)

    p2p_timer = np.zeros(repeat, dtype=np.float32)
    collective_timer = np.zeros(repeat, dtype=np.float32)
    total_timer = np.zeros(repeat, dtype=np.float32)
    
    for n in range(repeat):
        p2p_begin = time.perf_counter()
        p2p_communicate(rank, comm, p2p_labels_matrix, comm_matrix, \
                        send_data_range_list, recv_data_range_list, \
                        send_buf, recv_buf, feat_len)

        p2p_end = time.perf_counter()
    
        collective_begin = time.perf_counter()
    
        collective_communicate(rank, groups, collective_labels_matrix, \
                                comm_matrix, global_rank_to_group_rank, \
                                send_data_range_list, recv_data_range_list, \
                                send_buf, recv_buf, feat_len)

        # total_comm_time_list[n] = (end - begin)*1000.0

        collective_end = time.perf_counter()
        
        p2p_timer[n] = (p2p_end - p2p_begin) * 1000.0
        collective_timer[n] = (collective_end - collective_begin) * 1000.0
        total_timer[n] = (collective_end - p2p_begin) * 1000.0
    
    print("average elapsed time of p2p comm in ({}) repeats = {}ms".format(repeat - 1, np.mean(p2p_timer[1:])))
    print("average elapsed time of collective comm in ({}) repeats = {}ms".format(repeat - 1, np.mean(collective_timer[1:])))
    print("average elapsed time of total group all_to_all in ({}) repeats = {}ms".format(repeat - 1, np.mean(total_timer[1:])))

    return recv_buf

    
def test_original_alltoall(rank, comm, comm_matrix, send_buf, feat_len):
    send_splits = comm_matrix[rank].tolist()
    recv_splits = comm_matrix[:, rank].tolist()

    recv_buf = torch.zeros((sum(recv_splits), feat_len), dtype=torch.float32)

    send_counts = comm_matrix[rank] * feat_len
    recv_counts = comm_matrix[:, rank] * feat_len

    send_displs = np.zeros(comm_matrix.shape[0], dtype=np.int64)
    recv_displs = np.zeros(comm_matrix.shape[0], dtype=np.int64)

    for i in range(1, comm_matrix.shape[0]):
        send_displs[i] = send_displs[i-1] + comm_matrix[rank, i-1] * feat_len
        recv_displs[i] = recv_displs[i-1] + comm_matrix[i-1, rank] * feat_len

    repeat = 21
    total_comm_time_list = np.zeros(repeat, dtype=np.float32)
    
    for i in range(repeat):
        begin = time.perf_counter()
        if mpi4py_available:
            comm.Alltoallv([send_buf, send_counts, send_displs, MPI.FLOAT], \
                            [recv_buf, recv_counts, recv_displs, MPI.FLOAT],)
        else:
            handle = dist.all_to_all_single(recv_buf, send_buf, recv_splits, send_splits, async_op=True)
            handle.wait()

        end = time.perf_counter()
        print("elasped time of all_to_all in rank {} = {}ms".format(rank, (end - begin)*1000.0))
        total_comm_time_list[i] = (end - begin)*1000.0
    print("average elapsed time of original all_to_all in ({}) repeats = {}ms".format(repeat - 1, np.mean(total_comm_time_list[1:])))

    return recv_buf
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clusters", type=int, default=5)
    args = parser.parse_args()

    num_clusters = args.num_clusters

    comm, rank, world_size = init_dist_group()

    # load the collective communication pattern
    collective_label_matrix = np.load('collective_group_labels({}clusters_{}procs).npy'.format(num_clusters, world_size))
    # load the p2p communication pattern
    p2p_label_matrix = np.load('p2p_group_labels({}clusters_{}procs).npy'.format(num_clusters, world_size))
    # load the global communication volume
    comm_matirx = np.load('global_comm_pattern_{}proc.npy'.format(world_size))

    global_rank_to_group_rank = np.zeros((np.max(collective_label_matrix)+1, world_size), dtype=np.int64)

    ranks_list_in_each_cluster = []
    with open('ranks_list_in_each_collective_group({}clusters_{}procs).txt'.format(num_clusters, world_size), 'r') as f:
        cluster_id = 0
        for line in f:
            if line != '\n':
                ranks_list = list(map(int, line.split(',')))
                ranks_list_in_each_cluster.append(ranks_list)
                global_rank_to_group_rank[cluster_id][ranks_list] = np.arange(len(ranks_list), dtype=np.int64)
                cluster_id += 1

    feat_len = 64

    # print(label_matrix)
    # init subcommunicators for collective communication
    groups = init_subcommunicator(rank, collective_label_matrix, ranks_list_in_each_cluster)

    # prepare the send buffer
    send_buf = torch.rand((comm_matirx[rank].sum(), feat_len), dtype=torch.float32)

    res = test_group_alltoall_v2(rank, groups, comm, collective_label_matrix, p2p_label_matrix, comm_matirx, global_rank_to_group_rank, send_buf, feat_len)

    res_ref = test_original_alltoall(rank, comm, comm_matirx, send_buf, feat_len)

    # res = test_group_alltoall(rank, groups, collective_label_matrix, comm_matirx, global_rank_to_group_rank, send_buf, feat_len)

    err_idx = torch.nonzero(res != res_ref)
    print("err_idx = {}".format(err_idx))

    is_passed = 0
    if torch.allclose(res_ref, res):
        is_passed = 1
    
    # wrap the is_passed flag into a tensor
    is_passed = torch.tensor([is_passed], dtype=torch.int32)

    dist.reduce(is_passed, dst=0, op=dist.ReduceOp.PRODUCT)
    
    if rank == 0:
        if is_passed[0].item() == 1:
            print("test passed!")
        else:
            print("test failed!")
