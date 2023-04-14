import torch
import time

'''
print(torch.__version__)
print(torch.__config__.parallel_info())

torchdevice = torch.device('cpu')
# torch.set_num_interop_threads(2)
if torch.cuda.is_available():
    torchdevice = torch.device('cuda')
    print('Default GPU is ' + torch.cuda.get_device_name(torch.device('cuda')))
print('Running on ' + str(torchdevice))

import torch_sparse

# Convenience wrapper around torch_sparse index_select
def ts_index_select(A,sdim,idx):
    Ats = torch_sparse.SparseTensor.from_torch_sparse_coo_tensor(A)
    Ats_select = torch_sparse.index_select(Ats,sdim,idx)
    row, col, value = Ats_select.coo()
    As_select = torch.sparse_coo_tensor(torch.stack([row, col], dim=0), value, (Ats_select.size(0), Ats_select.size(1)))
    return As_select

# Dimension of the square sparse matrix
# n = 1000000
n = 1000000
# Number of non-zero elements (up to duplicates)
# nnz = 100000
nnz = 100000
# Number of selected indices (up to duplicates)
# m = 10000
m = 10000

repeat = 100

rowidx = torch.randint(low=0, high=n, size=(nnz,), device=torchdevice)
colidx = torch.randint(low=0, high=n, size=(nnz,), device=torchdevice)
itemidx = torch.vstack((rowidx,colidx))
xvalues = torch.randn(nnz, device=torchdevice)
SparseX = torch.sparse_coo_tensor(itemidx, xvalues, size=(n,n)).coalesce()
print('SparseX:',SparseX)

selectrowidx = torch.unique(torch.randint(low=0, high=n, size=(m,), device=torchdevice), sorted=True)

print('\nRunning index_select from torch_sparse')
start = time.perf_counter()
for _ in range(repeat):
    SparseXsub2 = ts_index_select(SparseX,0,selectrowidx)
end = time.perf_counter()
print("torch_sparse time(s): {}".format((end - start)/repeat))

print('\nRunning index_select from PyTorch')
start = time.perf_counter()
for _ in range(repeat):
    SparseXsub1 = SparseX.index_select(0,selectrowidx)
end = time.perf_counter()
print("torch time(s): {}".format((end - start)/repeat))
'''

import numpy as np

INDEX = 100
NELE = 500000
a = torch.rand(INDEX, NELE)
index = np.random.randint(INDEX-1, size=INDEX*4)
b = torch.from_numpy(index)

# print('a:',a)
repeat = 10
res = a.index_select(0, b)
start = time.time()
for _ in range(repeat):
    res = a.index_select(0, b)
end = time.time()
elapsed_time = end - start

bandwidth = 2*4*INDEX*4*NELE*repeat / elapsed_time / 1000.0 / 1000.0 / 1000.0
print("the number of cpu threads: {}, time: {}, bandwidth: {} GB/s".format(torch.get_num_threads(), elapsed_time, bandwidth))

