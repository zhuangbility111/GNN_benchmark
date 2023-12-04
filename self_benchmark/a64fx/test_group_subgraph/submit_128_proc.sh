#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=small-s1"
#PJM -L elapse=00:05:00
#PJM -g ra000012
#PJM -L "node=4x2x4,freq=2000"
#PJM --mpi "proc=128"
#PJM -j
#PJM -S

source ~/gnn/gnn/pytorch/config_env.sh
dir_stdout="./log/${PJM_MPI_PROC}_proc"

python KMeans.py --num_procs=${PJM_MPI_PROC}
LD_PRELOAD=/home/ra000012/a04083/gnn/gnn/pytorch/scripts/fujitsu/lib/libtcmalloc.so mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/stdout -stderr-proc ${dir_stdout}/stderr python test_alltoall.py