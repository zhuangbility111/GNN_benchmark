export XOS_MMM_L_PAGING_POLICY=demand:demand:demand

OMP_NUM_THREADS=1 numactl -C 12 -m 4 python test_index_add.py
OMP_NUM_THREADS=2 numactl -C 12-13 -m 4 python test_index_add.py
OMP_NUM_THREADS=4 numactl -C 12-15 -m 4 python test_index_add.py
OMP_NUM_THREADS=8 numactl -C 12-19 -m 4 python test_index_add.py
OMP_NUM_THREADS=12 numactl -C 12-23 -m 4 python test_index_add.py
OMP_NUM_THREADS=24 numactl -C 12-35 -m 4-5 python test_index_add.py
OMP_NUM_THREADS=36 numactl -C 12-47 -m 4-6 python test_index_add.py
OMP_NUM_THREADS=48 numactl -C 12-59 -m 4-7 python test_index_add.py
