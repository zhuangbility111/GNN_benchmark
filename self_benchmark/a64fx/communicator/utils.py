import os
import torch.distributed as dist
import torch

def init_dist_group():
    if dist.is_mpi_available(): 
        # backend with mpi
        print("mpi in torch.distributed is available!")
        dist.init_process_group(backend="mpi")
    else:
        # backend with torch_ccl
        import torch_ccl
        world_size = int(os.environ.get("PMI_SIZE", -1))
        rank = int(os.environ.get("PMI_RANK", -1))
        print("use ccl backend for torch.distributed package on x86 cpu.")
        dist.init_process_group(backend="ccl", init_method="env://", 
                                world_size=world_size, rank=rank)

    print(f"dist_info RANK: {dist.get_rank()}, SIZE: {dist.get_world_size()}")
    # number of process in this MPI group
    world_size = dist.get_world_size() 
    # mpi rank in this MPI group
    rank = dist.get_rank()

    return (rank, world_size)