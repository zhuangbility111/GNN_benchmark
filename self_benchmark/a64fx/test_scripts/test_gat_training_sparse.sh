OMP_NUM_THREADS=1  numactl -C 12 -m 4 python gat_training_sparse.py
OMP_NUM_THREADS=12 numactl -C 12-23 -m 4 python gat_training_sparse.py
OMP_NUM_THREADS=24 numactl -C 12-35 -m 4-5 python gat_training_sparse.py
OMP_NUM_THREADS=36 numactl -C 12-47 -m 4-6 python gat_training_sparse.py
OMP_NUM_THREADS=48 numactl -C 12-59 -m 4-7 python gat_training_sparse.py
