#!/bin/bash

#$-l rt_F=8
#$-cwd
#$-l h_rt=00:15:00


source ~/gcn.work/dgl_intel_setting_1/env_torch_1.10.0.sh
source /etc/profile.d/modules.sh
# load mpi library
module load intel-mpi/2021.8
export FI_PROVIDER=tcp
random_seed=0
is_fp16=true
is_async=true
graph_name=products
num_epochs=10

# number of total processes 
NP=16

# number of processes per node
NPP=2

tcmalloc_path=/home/aaa10008ku/gcn.work/dgl_intel_setting_1/sub407/miniconda3/envs/torch-1.10/lib/libtcmalloc.so
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../../dist_pyg_test.py --graph_name=${graph_name} --model=sage --is_async=${is_async} --input_dir=../../dataset/ogbn_${graph_name}_new/ogbn_${graph_name}_${NP}_part --random_seed=${random_seed} --is_fp16=${is_fp16} --num_epochs=${num_epochs}
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../../dist_pyg_test_pre_post_1_comm.py --graph_name=${graph_name} --model=sage --is_async=${is_async} --input_dir=../../dataset/ogbn_${graph_name}_new/ogbn_${graph_name}_${NP}_part --random_seed=${random_seed} --is_fp16=${is_fp16} --num_epochs=${num_epochs}
