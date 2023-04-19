#!/bin/bash
#SBATCH -J parmetis_test
#SBATCH -p baode_cpu_long
#SBATCH -N 2

#SBATCH --cpus-per-task=1

source /public/software/profile.d/mpi_mpich-gnu-3.2.sh
# python postprocess_graph.py -d ./ -g products -n 2 -l 100
 python postprocess_graph_multi_proc_1.py -d ./papers100M_graph_4096_part_new/ -g papers100M -b 0 -e 4096 -p 16