#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=small-s1"
#PJM -L elapse=01:00:00
#PJM -g ra000012
#PJM -L "node=8x4x8,freq=2000"
#PJM --mpi "proc=1024"
#PJM -j
#PJM -S

source ~/gnn/gnn/pytorch/config_env.sh
mpirun env
# export OMP_NUM_THREADS=11
# mpirun -np 256 python dist_pyg_test.py --tensor_type=sparse_tensor
# LD_PRELOAD=/home/ra000012/a04083/gnn/gnn/pytorch/scripts/fujitsu/lib/libtcmalloc.so mpirun -np 1024 -stdout-proc ogbn_papers100M_log/1024proc/0/stdout -stderr-proc ogbn_papers100M_log/1024proc/0/stderr python dist_pyg_test.py --graph_name=papers100M --model=gcn --is_async=false
# LD_PRELOAD=/home/ra000012/a04083/gnn/gnn/pytorch/scripts/fujitsu/lib/libtcmalloc.so mpirun -np 1024 -stdout-proc ogbn_papers100M_log/1024proc/1/stdout -stderr-proc ogbn_papers100M_log/1024proc/1/stderr python dist_pyg_test.py --graph_name=papers100M --model=sage --is_async=true
LD_PRELOAD=/home/ra000012/a04083/gnn/gnn/pytorch/scripts/fujitsu/lib/libtcmalloc.so mpirun -np 1024 -stdout-proc ogbn_papers100M_log_test/1024proc/2/stdout -stderr-proc ogbn_papers100M_log_test/1024proc/2/stderr python dist_pyg_test_pre.py --tensor_type=sparse_tensor
