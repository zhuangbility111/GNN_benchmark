# General and Scalable Framework for GCN Training on CPU-powered Supercomputers
To run the code on ABCI (Intel Xeon) supercomputer:
1. switch branch to `cpu_dev`: git checkout cpu_dev
2. go to the directory `self_benchmark/a64fx/sub_scripts_at_abci/batch_job_submission/`
3. run the submit scripts `submit_{graph_name}_all.sh` to submit the job, the `graph_name` needs to be replaced with the specific name of the graph. The command to run code on ogbn-papers100M dataset: sh submit_ogbn-papers100M_all.sh

To run the code on Fugaku (ARM Fujitsu A64FX) supercomputer:
1. switch branch to `fugaku_dev`: git checkout cpu_dev
2. go to the directory `self_benchmark/a64fx/sub_scripts_at_fugaku/{graph_name}/submit_scripts`, the `{graph_name}` needs to be changed with the specific name of the graph, for example, `ogbn-papers100M`
3. run the submit scripts `submit_{graph_name}_all.sh` to submit the job, the `graph_name` needs to be replaced with the specific name of the graph. The command to run code on ogbn-papers100M dataset: sh submit_ogbn-papers100M_all.sh
