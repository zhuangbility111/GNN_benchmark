import os
import argparse

node_configures = {
    1: "1x1x1",
    2: "1x1x1",
    4: "1x1x1",
    8: "2x1x1",
    16: "2x1x2",
    32: "2x2x2",
    64: "2x2x4",
    128: "4x2x4",
    256: "4x4x4",
    512: "4x4x8",
    1024: "8x4x8",
    2048: "8x8x8",
    4096: "8x8x16",
    8192: "16x8x16",
    16384: "16x16x16",
    32768: "16x16x32",
    65536: "32x16x32",
    131072: "32x32x32",
}

def generate_job_submission_script(num_mpi_processes, node_hours, graph_name):
    srcgrp = "small-s1"
    if int(node_hours) > 1:
        srcgrp = "small-s2"
    if num_mpi_processes >= 2048:
        srcgrp = "large"
    script = "#!/bin/bash\n"
    script += "\n"
    script += "#PJM -L \"rscunit=rscunit_ft01,rscgrp={}\"\n".format(srcgrp)
    script += "#PJM -L elapse={}:00:00 \n".format(node_hours)
    script += "#PJM -g ra000012 \n"

    script += "#PJM -L \"node={}:torus:strict-io,freq=2000\"\n".format(node_configures[num_mpi_processes])
    script += "#PJM --mpi \"proc={}\"\n".format(num_mpi_processes)
    script += "#PJM -j\n"
    script += "#PJM -S\n"

    script += "\n"
    script += "source ~/gnn/gnn/pytorch/config_env.sh\n"
    script += "graph_name={}\n".format(graph_name)
    script += "dir_stdout=../log/barrier_with_no_asnyc_v1/${PJM_MPI_PROC}proc/"

    script += "\n"
    script += "tcmalloc_path=/home/ra000012/a04083/gnn/gnn/pytorch/scripts/fujitsu/lib/libtcmalloc.so\n"
    script += "date\n"
    script += "LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori/stdout -stderr-proc ${dir_stdout}/ori/stderr python ../../../train.py --config=../../../config/fugaku/${graph_name}.yaml --num_bits=32\n"
    script += "LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+pre/stdout -stderr-proc ${dir_stdout}/ori+pre/stderr python ../../../train.py --config=../../../config/fugaku/${graph_name}.yaml --num_bits=32 --is_pre_delay=true\n"
    script += "LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+int2/stdout -stderr-proc ${dir_stdout}/ori+int2/stderr python ../../../train.py --config=../../../config/fugaku/${graph_name}.yaml --num_bits=2\n"
    script += "LD_PRELOAD=$tcmalloc_path:$LD_PRELOAD mpirun -np ${PJM_MPI_PROC} -stdout-proc ${dir_stdout}/ori+pre+int2/stdout -stderr-proc ${dir_stdout}/ori+pre+int2/stderr python ../../../train.py --config=../../../config/fugaku/${graph_name}.yaml --num_bits=2 --is_pre_delay=true\n"
    script += "date\n"

    return script


parser = argparse.ArgumentParser()
parser.add_argument("--graph_name", "-g", type=str, default="ogbn-products")
args = parser.parse_args()
graph_name = args.graph_name

output_dir = os.path.join("./", graph_name, "submit_scripts")
if os.path.exists(output_dir) == False:
    os.makedirs(output_dir)

# Generate job submission scripts for different number of nodes
num_mpi_processes_list = list()
node_hours_list = list()

if graph_name == "ogbn-products":
    num_mpi_processes_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    node_hours_list = ["8", "6", "4", "2", "1", "1", "1", "1", "1", "1", "1"]
elif graph_name == "ogbn-papers100M":
    num_mpi_processes_list = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    node_hours_list = ["12", "10", "6", "3", "1", "1", "1", "1", "1", "1", "1", "1"]
elif graph_name == "proteins":
    num_mpi_processes_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    node_hours_list = ["8", "6", "4", "2", "1", "1", "1", "1", "1", "1", "1"]
elif graph_name == "reddit":
    num_mpi_processes_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    node_hours_list = ["8", "6", "4", "4", "2", "1", "1", "1", "1", "1", "1", "1"]
elif graph_name == "uk-2007-02":
    num_mpi_processes_list = [512, 1024, 2048, 4096, 8192, 16384]
    node_hours_list = ["3", "2", "1", "1", "1", "1"]


for i in range(len(num_mpi_processes_list)):
    script = generate_job_submission_script(num_mpi_processes_list[i], node_hours_list[i], graph_name)
    filename = os.path.join(output_dir, "run_{}_{}.sh".format(graph_name, num_mpi_processes_list[i]))

    with open(filename, "w") as f:
        f.write(script)

    print("Generated job submission script: {}".format(filename))

script = "#!/bin/bash\n"
for num_nodes in num_mpi_processes_list:
    script += "pjsub run_{}_{}.sh\n".format(graph_name, num_nodes)
filename = os.path.join(output_dir, "submit_{}_all.sh".format(graph_name))
with open(filename, "w") as f:
    f.write(script)
