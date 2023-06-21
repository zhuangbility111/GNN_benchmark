import torch
import torch.distributed as dist
import os
import time

try:
    import torch_ccl
except ImportError as e:
    print(e)


def init_dist_group():
    if dist.is_mpi_available():
        print("mpi in torch.distributed is available!")
        dist.init_process_group(backend="mpi")
    else:
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

def test_alltoall_in_half(rank, world_size):
    send_buf = torch.zeros((world_size+1), dtype=torch.float16)
    recv_buf = torch.zeros((world_size+1), dtype=torch.float16)
    for i in range(world_size+1):
        send_buf[i] = float(rank + rank*0.1)
    print("before communication send_buf = {}".format(send_buf))
    print("before communication recv_buf = {}".format(recv_buf))
    send_splits = [1 for i in range(world_size)]
    recv_splits = [1 for i in range(world_size)]

    if rank+1 >= world_size:
        send_splits[0] = 2
    else:
        send_splits[rank+1] = 2

    if rank-1 >= 0:
        recv_splits[rank-1] = 2
    else:
        recv_splits[world_size-1] = 2
    
    dist.all_to_all_single(recv_buf, send_buf, recv_splits, send_splits)
    print("after communication send_buf = {}".format(send_buf))
    print("after communication recv_buf = {}".format(recv_buf))

if __name__ == "__main__":
    rank, world_size = init_dist_group()
    test_alltoall_in_half(rank, world_size)


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

