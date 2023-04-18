#!/bin/bash
#PJM -L "rscunit=rscunit_ft01,rscgrp=small-s1"
#PJM -L elapse=01:00:00
#PJM -g ra000012
#PJM -L "node=2x2x2,freq=2000"
#PJM --mpi "proc=32"
#PJM -j
#PJM -S

source ~/gnn/gnn/pytorch/config_env.sh
graph_name="arxiv"
dir_stdout="../log/ogbn_${graph_name}_log/${PJM_MPI_PROC}proc/"
input_dir="../dataset/ogbn_${graph_name}_new/${graph_name}_graph_${PJM_MPI_PROC}_part/"

LD_PRELOAD=/home/ra000012/a04083/gnn/gnn/pytorch/scripts/fujitsu/lib/libtcmalloc.so mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}0/stdout -stderr-proc ${dir_stdout}0/stderr python ../dist_pyg_test.py --graph_name=${graph_name} --model=gcn --is_async=false --input_dir=${input_dir}
LD_PRELOAD=/home/ra000012/a04083/gnn/gnn/pytorch/scripts/fujitsu/lib/libtcmalloc.so mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}1/stdout -stderr-proc ${dir_stdout}1/stderr python ../dist_pyg_test.py --graph_name=${graph_name} --model=sage --is_async=true --input_dir=${input_dir}
LD_PRELOAD=/home/ra000012/a04083/gnn/gnn/pytorch/scripts/fujitsu/lib/libtcmalloc.so mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}2/stdout -stderr-proc ${dir_stdout}2/stderr python ../dist_pyg_test_pre_post_1_comm.py --graph_name=${graph_name} --model=sage --is_async=true --input_dir=${input_dir}