#!/bin/bash

# graph_names=("proteins" "reddit" "ogbn-products" "ogbn-papers100M")
graph_names=("uk-2007-02")

for graph in "${graph_names[@]}"; do
	echo "Processing file: $graph"
	python collect_performance_data.py -i $graph/log/barrier_with_no_asnyc_v1/ -o ../experiment_result/multi_nodes/ -g $graph
done
