mpirun -np 1 -std output.log python dist_pyg_test.py --tensor_type=sparse_tensor
mpirun -np 2 -std output.log python dist_pyg_test.py --tensor_type=sparse_tensor
mpirun -np 4 -std output.log python dist_pyg_test.py --tensor_type=sparse_tensor
mpirun -np 8 -std output.log python dist_pyg_test.py --tensor_type=sparse_tensor
