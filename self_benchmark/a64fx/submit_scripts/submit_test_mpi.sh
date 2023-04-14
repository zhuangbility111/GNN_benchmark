#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=small-s1"
#PJM -L elapse=00:30:00
#PJM -g ra000012
#PJM -L "node=4x8x8,freq=2000"
#PJM --mpi "proc=1024"
#PJM -j
#PJM -S

source ~/gnn/gnn/pytorch/config_env.sh
mpirun env
export OMP_NUM_THREADS=12
LD_PRELOAD=/home/ra000012/a04083/gnn/gnn/pytorch/scripts/fujitsu/lib/libtcmalloc.so mpirun -np 1024 python test_mpi.py
