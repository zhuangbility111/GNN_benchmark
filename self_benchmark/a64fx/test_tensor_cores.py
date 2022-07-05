import torch
import time

a_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
b_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
ab_full = a_full @ b_full
mean = ab_full.abs().mean()

a = a_full.float()
b = b_full.float()

# Do matmul at TF32 mode.
start = time.perf_counter()
ab_tf32 = a @ b 
end = time.perf_counter()
elapsed_time_fp32 = end - start 
error_tf32 = (ab_tf32 - ab_full).abs().max()  # 0.1747
relative_error_tf32 = error_tf32 / mean  # 0.0022

# Do matmul with TF32 disabled.
torch.backends.cuda.matmul.allow_tf32 = False
start = time.perf_counter()
ab_fp32 = a @ b 
end = time.perf_counter()
elapsed_time_tf32 = end - start 
error_fp32 = (ab_fp32 - ab_full).abs().max()  # 0.0031
relative_error_fp32 = error_fp32 / mean  # 0.000039

print("fp32: error: {:e}, relative_error: {:e}, elapsed_time(s): {:e}".format(error_fp32, relative_error_fp32, elapsed_time_fp32))
print("tf32: error: {:e}, relative_error: {:e}, elapsed_time(s): {:e}".format(error_tf32, relative_error_tf32, elapsed_time_tf32))
