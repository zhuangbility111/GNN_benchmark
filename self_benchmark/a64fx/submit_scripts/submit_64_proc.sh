#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=small-s1"
#PJM -L elapse=01:00:00
#PJM -g ra000012
#PJM -L "node=2x2x4,freq=2000"
#PJM --mpi "proc=64"
#PJM -j
#PJM -S

source ~/gnn/gnn/pytorch/config_env.sh
mpirun env
# export OMP_NUM_THREADS=11
# mpirun -np 64 python dist_pyg_test.py --tensor_type=sparse_tensor
# LD_PRELOAD=/home/ra000012/a04083/gnn/gnn/pytorch/scripts/fujitsu/lib/libtcmalloc.so mpirun -np 64 -stdout-proc ogbn_products_log/64proc/0/stdout -stderr-proc ogbn_products_log/64proc/0/stderr python dist_pyg_test.py --graph_name=products --model=gcn --is_async=false
# LD_PRELOAD=/home/ra000012/a04083/gnn/gnn/pytorch/scripts/fujitsu/lib/libtcmalloc.so mpirun -np 64 -stdout-proc ogbn_products_log/64proc/1/stdout -stderr-proc ogbn_products_log/64proc/1/stderr python dist_pyg_test.py --graph_name=products --model=sage --is_async=true
LD_PRELOAD=/home/ra000012/a04083/gnn/gnn/pytorch/scripts/fujitsu/lib/libtcmalloc.so mpirun -np 64 -stdout-proc ogbn_products_log/64proc/2/stdout -stderr-proc ogbn_products_log/64proc/2/stderr python dist_pyg_test_pre.py --tensor_type=sparse_tensor
