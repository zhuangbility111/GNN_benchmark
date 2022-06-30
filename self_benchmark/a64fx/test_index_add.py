import torch
import time
# print(torch.__version__)
# print(torch.__config__.parallel_info())

torchdevice = torch.device('cpu')

row = 2000
col = 10
src_row = 13000
repeat = 100

output = torch.ones(row, col)
src = torch.rand(src_row, col) * 5

import numpy as np
tmp = np.random.randint(row, size=src_row)
index = torch.from_numpy(tmp)

# print(index.size())
# print(src.size())
    
output.index_add_(0, index, src, alpha=1)

start = time.perf_counter()
for _ in range(repeat):
    output.index_add_(0, index, src, alpha=1)
end = time.perf_counter()
elapsed_time = end - start 
bandwidth = 2 * 4 * repeat * src_row * col / elapsed_time / 1000.0 / 1000.0 / 1000.0

print("index_add_ thread numbers: {}, repeat: {}, total time(s): {}, bandwidth: {} GB/s".format(torch.get_num_threads(), repeat, elapsed_time, bandwidth))
# print(output)
