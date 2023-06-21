#!/bin/sh

#$-l rt_F=4
#$-cwd
#$-l h_rt=00:15:00

# rt_f is the maximum number of nodes, that want to use
# h_rt is the maximum time, that our module will run at 
#         the run time of our module should be less than this value#         the format h_rt is Hour:Minute:Second

# source ~/gcn.work/dgl_intel_setting_1/env.sh
source source ~/gcn.work/dgl_intel_setting_1/env_torch_1.10.0.sh
source /etc/profile.d/modules.sh
# source ~/gcn.work/dgl_intel_setting_1/sub407/miniconda3/bin/activate sub407

# load mpi library
module load intel-mpi/2021.8
# module load intel-mkl/2023.0.0
export FI_PROVIDER=tcp
num_nodes=4

# mpi run
# MPIRUN=mpiexec

# number of total processes 
# NP=16

# number of processes per node
# NPP=1

# HOME_DIR=$(echo ~)
# ImgJ_path=$HOME_DIR/run_ThunderSTORM.sh

# cmd="$MPIRUN -np $NP -ppn $NPP ./bin/bioMpi $ImgJ_path"
# echo run_mu_command: $cmd
# eval $cmd
sh run_dist.sh -n ${num_nodes} -ppn 1 python test_half_alltoall.py
