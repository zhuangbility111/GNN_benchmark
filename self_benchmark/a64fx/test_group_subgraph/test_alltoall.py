import torch
import torch.distributed as dist
import os
import numpy as np
import time

torch.set_num_threads(1)

def init_dist_group():
    if dist.is_mpi_available():
        print("mpi in torch.distributed is available!")
        dist.init_process_group(backend="mpi")
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
    assert torch.distributed.is_initialized()
    print(f"dist_info RANK: {dist.get_rank()}, SIZE: {dist.get_world_size()}")
    # number of process in this MPI group
    world_size = dist.get_world_size() 
    # mpi rank in this MPI group
    rank = dist.get_rank()
    return (rank, world_size)

def init_subcommunicator(rank, world_size, labels, ranks_list_in_each_cluster):
    groups = []
    num_labels = np.max(labels) + 1
    for i in range(num_labels):
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
    
def test_group_alltoall(rank, groups, labels, comm_matrix, global_rank_to_group_rank, send_buf, feat_len=128):
    recv_bufs_list = dict()

    data_send_to_other_rank = comm_matrix[rank]
    data_recv_from_other_rank = comm_matrix[:, rank]

    send_labels = labels[rank]
    recv_labels = labels[:, rank]

    num_communicators = np.max(recv_labels) + 1

    send_bufs_list = construct_send_buf(send_labels, data_send_to_other_rank, send_buf)

    for label in range(num_communicators):
        recv_rank = (recv_labels == label).nonzero()[0]
        if recv_rank.shape[0] > 0:
            recv_data_volume = np.sum(data_recv_from_other_rank[recv_rank])
            recv_bufs_list[label] = torch.zeros((recv_data_volume, feat_len), dtype=torch.float32)

    repeat = 11
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

    
def test_original_alltoall(rank, comm_matrix, send_buf, feat_len=128):
    send_splits = comm_matrix[rank].tolist()
    recv_splits = comm_matrix[:, rank].tolist()

    recv_buf = torch.zeros((sum(recv_splits), feat_len), dtype=torch.float32)

    repeat = 11
    total_comm_time_list = np.zeros(repeat, dtype=np.float32)
    
    for i in range(repeat):
        begin = time.perf_counter()
        handle = dist.all_to_all_single(recv_buf, send_buf, recv_splits, send_splits, async_op=True)
        handle.wait()
        end = time.perf_counter()

        total_comm_time_list[i] = (end - begin)*1000.0
    print("average elapsed time of original all_to_all in ({}) repeats = {}ms".format(repeat - 1, np.mean(total_comm_time_list[1:])))

    return recv_buf
        
if __name__ == "__main__":
    rank, world_size = init_dist_group()

    label_matrix = np.load('global_comm_pattern_512proc_clustered_label.npy')
    comm_matirx = np.load('global_comm_pattern_512proc.npy')

    global_rank_to_group_rank = np.zeros((np.max(label_matrix)+1, world_size), dtype=np.int64)
    ranks_list_in_each_cluster = []
    with open('ranks_list_in_each_clusters_512proc.txt', 'r') as f:
        cluster_id = 0
        for line in f:
            if line != '\n':
                ranks_list = list(map(int, line.split(',')))
                ranks_list_in_each_cluster.append(ranks_list)
                global_rank_to_group_rank[cluster_id][ranks_list] = np.arange(len(ranks_list), dtype=np.int64)
                cluster_id += 1

    feat_len = 64

    groups = init_subcommunicator(rank, world_size, label_matrix, ranks_list_in_each_cluster)

    send_buf = torch.rand((comm_matirx[rank].sum(), feat_len), dtype=torch.float32)
    res_ref = test_original_alltoall(rank, comm_matirx, send_buf, feat_len)

    res = test_group_alltoall(rank, groups, label_matrix, comm_matirx, global_rank_to_group_rank, send_buf, feat_len)

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

    # repeat = 1
    # comm_time = []
    # barrier_time = []

    # if rank == 0:
    #     send_buf = torch.zeros((100 + world_size - 1,2), dtype=torch.float16)
    # else:
    #     send_buf = torch.zeros((world_size,2), dtype=torch.float16)
            
    # for i in range(send_buf.shape[0]):
    #     send_buf[i][0] = world_size*rank + i
    #     send_buf[i][1] = world_size*rank + i
    # print("send_buf = {}".format(send_buf))
    # recv_buf = torch.zeros((world_size, 2), dtype=torch.float16)
    # if rank == 3:
    #     recv_buf = torch.zeros((100 + world_size - 1, 2), dtype=torch.float16)
    # send_splits = [1 for i in range(world_size)]
    # recv_splits = [1 for i in range(world_size)]

    # if rank == 0:
    #     send_splits[3] = 100
    # elif rank == 3:
    #     recv_splits[0] = 100

    # for i in range(repeat):
    #     barrier_begin = time.perf_counter()
    #     dist.barrier()
    #     comm_begin = time.perf_counter()
    #     dist.all_to_all_single(recv_buf, send_buf, recv_splits, send_splits)
    #     comm_end = time.perf_counter()
    #     barrier_time.append((comm_begin - barrier_begin)*1000.0)
    #     comm_time.append((comm_end - comm_begin)*1000.0)
    #     print("number of send data = {}".format(sum(send_splits)))
    #     print("number of recv data = {}".format(sum(recv_splits)))
    #     print("recv_buf = {}".format(recv_buf))

    # for i in range(repeat):
    #     print("elapsed time of barrier {} = {}ms".format(i, barrier_time[i]))
    #     print("elapsed time of communication {} = {}ms".format(i, comm_time[i]))

