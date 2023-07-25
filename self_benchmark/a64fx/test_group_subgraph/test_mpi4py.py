import torch
import torch.distributed as dist
from mpi4py import MPI                                                    
import numpy as np


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

print("world_size = {}, rank = {}".format(world_size, rank))
print("MPI vendor = {}".format(MPI.get_vendor()))
print("MPI version = {}".format(MPI.Get_version()))

def test_async_sendrecv():
    if rank == 0:
        req_list = []
        data = []
        for i in range(10):
            data.append(torch.ones(i+1))
            req = comm.Isend(data[i], dest=1, tag=11)
            req_list.append(req)
        MPI.Request.Waitall(req_list)
        for i in range(10):
            print("process {} send {}...".format(rank, data[i]))
    else:
        req_list = []
        data = []
        for i in range(10):
            data.append(torch.zeros(i+1))
            req = comm.Irecv(data[i], source=0, tag=11)
            req_list.append(req)
        MPI.Request.Waitall(req_list)
        for i in range(10):
            print("process {} recv {}...".format(rank, data[i]))

def test_alltoallv():
    send_counts = np.zeros(world_size, dtype=np.int32)
    recv_counts = np.zeros(world_size, dtype=np.int32)
    send_displs = np.zeros(world_size, dtype=np.int32)
    recv_displs = np.zeros(world_size, dtype=np.int32)

    send_buf = np.zeros(world_size * world_size, dtype=np.int32)
    recv_buf = np.zeros(world_size * world_size, dtype=np.int32)

    # initialize sned_buf and recv_buf
    for i in range(world_size * world_size):
        send_buf[i] = i + 100*rank
        recv_buf[i] = -i

    # initialize send_counts and recv_counts and send_displs and recv_displs
    for i in range(world_size):
        send_counts[i] = i
        recv_counts[i] = rank
        send_displs[i] = (i * (i+1)) / 2
        recv_displs[i] = i * rank

    req = comm.Ialltoallv([send_buf, send_counts, send_displs, MPI.INT], \
                            [recv_buf, recv_counts, recv_displs, MPI.INT])
    
    print("rank = {}, recv_buf = {}".format(rank, recv_buf))

    req.Wait()

    # check the result
    for i in range(world_size):
        for j in range(rank):
            assert recv_buf[recv_displs[i] + j] == i * 100 + (rank*(rank+1))/2 + j

    print("rank = {}, test_alltoallv passed".format(rank))

def test_subcommunicator():
    ranks_list = [[0,1,2,3], [4,5,6,7]]

    groups = []
    for i in range(2):
        key = 0
        if rank in ranks_list[i]:
            color = i
        else:
            color = MPI.UNDEFINED

        print("color = {}".format(color))
        
        groups.append(MPI.COMM_WORLD.Split(color, key))

    print(groups, flush=True)
    print(comm)

    for i in range(2):
        if groups[i] != MPI.COMM_NULL:
            send_counts = np.array([1,1,1,1], dtype=np.int32)
            recv_counts = np.array([1,1,1,1], dtype=np.int32)

            send_displs = np.array([0,1,2,3], dtype=np.int32)
            recv_displs = np.array([0,1,2,3], dtype=np.int32)

            send_buf = np.arange(4, dtype=np.float32)
            recv_buf = np.zeros(4, dtype=np.float32)

            groups[i].Alltoallv([send_buf, send_counts, send_displs, MPI.FLOAT], \
                                [recv_buf, recv_counts, recv_displs, MPI.FLOAT],)

            print("i = {}, alltoallv is finished".format(i), flush=True)

# test_async_sendrecv()
# test_alltoallv()
test_subcommunicator()

