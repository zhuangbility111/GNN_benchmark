#!/bin/sh

#$-l rt_F=4
#$-cwd
#$-l h_rt=00:15:00

# rt_f is the maximum number of nodes, that want to use
# h_rt is the maximum time, that our module will run at 
#         the run time of our module should be less than this value#         the format h_rt is Hour:Minute:Second

# source ~/gcn.work/dgl_intel_setting_1/env.sh
source ~/gcn.work/dgl_intel_setting_1/env_torch_1.10.0.sh
source /etc/profile.d/modules.sh
# source ~/gcn.work/dgl_intel_setting_1/sub407/miniconda3/bin/activate sub407

# load mpi library
module load intel-mpi/2021.8
# module load intel-mkl/2023.0.0
export FI_PROVIDER=tcp
random_seed=0
is_fp16=true
is_async=true
graph_name="products"
num_epochs=8

# mpi run
# MPIRUN=mpiexec

# number of total processes 
NP=4

# number of processes per node
NPP=1

tcmalloc_path=/home/aaa10008ku/gcn.work/dgl_intel_setting_1/sub407/miniconda3/envs/torch-1.10/lib/libtcmalloc.so
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../dist_pyg_test.py --graph_name=${graph_name} --model=sage --is_async=${is_async} --input_dir=../dataset/ogbn_${graph_name}_new/ogbn_${graph_name}_${NP}_part --random_seed=${random_seed} --is_fp16=${is_fp16} --num_epochs=${num_epochs}
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../dist_pyg_test_pre_post_1_comm.py --graph_name=${graph_name} --model=sage --is_async=${is_async} --input_dir=../dataset/ogbn_${graph_name}_new/ogbn_${graph_name}_${NP}_part --random_seed=${random_seed} --is_fp16=${is_fp16} --num_epochs=${num_epochs}
