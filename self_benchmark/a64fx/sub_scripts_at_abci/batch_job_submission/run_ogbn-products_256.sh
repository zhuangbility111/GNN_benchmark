#!/bin/bash

#$-l rt_F=128
#$-cwd
#$-l h_rt=02:00:00

source ~/gcn.work/dgl_intel_setting_1/env_torch_1.10.0.sh
source /etc/profile.d/modules.sh
# load mpi library
module load intel-mpi/2021.8
export FI_PROVIDER=tcp

# number of total processes 
NP=256

# number of processes per node
NPP=2

tcmalloc_path=/home/aaa10008ku/gcn.work/dgl_intel_setting_1/sub407/miniconda3/envs/torch-1.10/lib/libtcmalloc.so
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../../train.py --config=../../config/abci/ogbn-products.yaml --num_bits=32
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../../train.py --config=../../config/abci/ogbn-products.yaml --num_bits=32 --is_pre_delay=true
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../../train.py --config=../../config/abci/ogbn-products.yaml --num_bits=2
LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../../train.py --config=../../config/abci/ogbn-products.yaml --num_bits=2 --is_pre_delay=true
