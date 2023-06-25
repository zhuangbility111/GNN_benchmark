import os

def generate_job_submission_script(num_nodes, num_processes_per_node=2):
    script = "#!/bin/bash\n"
    script += "\n"
    script += "#$-l rt_F={}\n".format(num_nodes)
    script += "#$-cwd\n"
    script += "#$-l h_rt=00:15:00\n"
    script += "\n\n"

    script += "source ~/gcn.work/dgl_intel_setting_1/env_torch_1.10.0.sh\n"
    script += "source /etc/profile.d/modules.sh\n"

    script += "# load mpi library\n"
    script += "module load intel-mpi/2021.8\n"
    script += "export FI_PROVIDER=tcp\n"
    script += "random_seed=0\n"
    script += "is_fp16=true\n"
    script += "is_async=true\n"
    script += "graph_name=products\n"
    script += "num_epochs=10\n"
    script += "\n"

    script += "# number of total processes \n"
    script += "NP={}\n".format(num_nodes * num_processes_per_node)
    script += "\n"

    script += "# number of processes per node\n"
    script += "NPP={}\n".format(num_processes_per_node)
    script += "\n"
    
    params = "--graph_name=${graph_name} --model=sage --is_async=${is_async} --input_dir=../../dataset/ogbn_${graph_name}_new/ogbn_${graph_name}_${NP}_part --random_seed=${random_seed} --is_fp16=${is_fp16} --num_epochs=${num_epochs}"
    script += "tcmalloc_path=/home/aaa10008ku/gcn.work/dgl_intel_setting_1/sub407/miniconda3/envs/torch-1.10/lib/libtcmalloc.so\n"
    script += "LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../../dist_pyg_test.py {}\n".format(params)
    script += "LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD sh run_dist.sh -n $NP -ppn $NPP python ../../dist_pyg_test_pre_post_1_comm.py {}\n".format(params)

    return script

# Generate job submission scripts for different number of nodes
num_nodes_list = [1, 2, 4, 8, 16, 32, 64, 128]

for num_nodes in num_nodes_list:
    script = generate_job_submission_script(num_nodes)
    filename = "run_{}.sh".format(num_nodes)

    with open(filename, 'w') as f:
        f.write(script)

    print("Generated job submission script: {}".format(filename))

script = "GROUP_ID=gac50544\n"
for num_nodes in num_nodes_list:
    script += "qsub -g $GROUP_ID ./run_{}.sh\n".format(num_nodes)
filename = "submit_all.sh"
with open(filename, 'w') as f:
    f.write(script)