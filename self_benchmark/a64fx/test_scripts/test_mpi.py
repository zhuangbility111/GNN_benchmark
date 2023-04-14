import torch
import torch.distributed as dist
import time


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
    return (rank, world_size)

if __name__ == "__main__":
    rank, world_size = init_dist_group()
    repeat = 20
    comm_time = []
    barrier_time = []

    if rank == 0:
        send_buf = torch.zeros((1000000 + world_size - 1,2), dtype=torch.int64)
    else:
        send_buf = torch.zeros((world_size,2), dtype=torch.int64)
            
    for i in range(send_buf.shape[0]):
        send_buf[i][0] = world_size*rank + i
        send_buf[i][1] = world_size*rank + i
    print("send_buf = {}".format(send_buf))
    recv_buf = torch.zeros((world_size, 2), dtype=torch.int64)
    if rank == 4:
        recv_buf = torch.zeros((1000000 + world_size - 1, 2), dtype=torch.int64)
    send_splits = [1 for i in range(world_size)]
    recv_splits = [1 for i in range(world_size)]

    if rank == 0:
        send_splits[4] = 1000000
    elif rank == 4:
        recv_splits[0] = 1000000

    for i in range(repeat):
        barrier_begin = time.perf_counter()
        dist.barrier()
        comm_begin = time.perf_counter()
        dist.all_to_all_single(recv_buf, send_buf, recv_splits, send_splits)
        comm_end = time.perf_counter()
        barrier_time.append((comm_begin - barrier_begin)*1000.0)
        comm_time.append((comm_end - comm_begin)*1000.0)
        print("number of send data = {}".format(sum(send_splits)))
        print("number of recv data = {}".format(sum(recv_splits)))
        print("recv_buf = {}".format(recv_buf))

    for i in range(repeat):
        print("elapsed time of barrier {} = {}ms".format(i, barrier_time[i]))
        print("elapsed time of communication {} = {}ms".format(i, comm_time[i]))
