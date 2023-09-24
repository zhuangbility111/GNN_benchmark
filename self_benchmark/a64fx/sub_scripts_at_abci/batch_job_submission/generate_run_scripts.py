import os
import argparse


def generate_job_submission_script(num_nodes, graph_name, num_processes_per_node=2):
    script = "#!/bin/bash\n"
    script += "\n"
    script += "#$-l rt_F={}\n".format(num_nodes)
    script += "#$-cwd\n"
    script += "#$-l h_rt=02:00:00\n"
    script += "\n"

    script += "source ~/gcn.work/dgl_intel_setting_1/env_torch_1.10.0.sh\n"
    script += "source /etc/profile.d/modules.sh\n"

    script += "# load mpi library\n"
    script += "module load intel-mpi/2021.8\n"
    script += "export FI_PROVIDER=tcp\n"
    script += "\n"

    script += "# number of total processes \n"
    script += "NP={}\n".format(num_nodes * num_processes_per_node)
    script += "\n"

    script += "# number of processes per node\n"
    script += "NPP={}\n".format(num_processes_per_node)
    script += "\n"

    script += "tcmalloc_path=/home/aaa10008ku/gcn.work/dgl_intel_setting_1/sub407/miniconda3/envs/torch-1.10/lib/libtcmalloc.so\n"
    script += "LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../../train.py --config=../../config/abci/{}.yaml --num_bits=32\n".format(
        graph_name
    )
    script += "LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../../train.py --config=../../config/abci/{}.yaml --num_bits=32 --is_pre_delay=true\n".format(
        graph_name
    )
    # script += "LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../../train.py --config=../../config/abci/{}.yaml --num_bits=-1\n".format(
    #     graph_name
    # )
    # script += "LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../../train.py --config=../../config/abci/{}.yaml --num_bits=-1 --is_pre_delay=true\n".format(
    #     graph_name
    # )
    script += "LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../../train.py --config=../../config/abci/{}.yaml --num_bits=2\n".format(
        graph_name
    )
    script += "LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../../train.py --config=../../config/abci/{}.yaml --num_bits=2 --is_pre_delay=true\n".format(
        graph_name
    )

    return script


parser = argparse.ArgumentParser()
parser.add_argument("--graph_name", type=str, default="ogbn-products")
args = parser.parse_args()
graph_name = args.graph_name

# Generate job submission scripts for different number of nodes
num_nodes_list = [1, 2, 4, 8, 16, 32, 64, 128]
num_processes_per_node = 2

for num_nodes in num_nodes_list:
    script = generate_job_submission_script(num_nodes, graph_name, num_processes_per_node)
    filename = "run_{}_{}.sh".format(graph_name, num_nodes * num_processes_per_node)

    with open(filename, "w") as f:
        f.write(script)

    print("Generated job submission script: {}".format(filename))

script = "GROUP_ID=gac50544\n"
for num_nodes in num_nodes_list:
    script += "qsub -g $GROUP_ID -o {}/log/ -e {}/log/ ./run_{}_{}.sh\n".format(
        graph_name, graph_name, graph_name, num_nodes * num_processes_per_node
    )
filename = "submit_{}_all.sh".format(graph_name)
with open(filename, "w") as f:
    f.write(script)
