#!/bin/bash

graph_name="products"
num_epochs=5

# write a for loop range from 1 to 16
for ((i=1; i<=16; i=i*2))
do
    # run the script with the number of threads
    OMP_NUM_THREADS=$i numactl -N 0 -m 0 python ../dist_pyg_test.py --graph_name=${graph_name} --model=sage --is_async=false --input_dir=../dataset/ogbn_${graph_name}_new/ogbn_${graph_name}_1_part --random_seed=0 --is_fp16=false --num_epochs=${num_epochs} &> ./experiment_result/${graph_name}_sage_${i}_threads.log
done
