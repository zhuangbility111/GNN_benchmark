import torch
import quantization_cpu
import math

torch.set_num_threads(12)

row = 12341
col = 47
size = (row, col)

# data_fp32 = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 2.0, 1.0, 4.0, 5.0]])
data_fp32 = torch.randn(size, dtype=torch.float32)
bits = 4
print(data_fp32)
scale = (data_fp32.max(dim=1)[0] - data_fp32.min(dim=1)[0] + 10e-20) / (2**bits - 1)
zero_point = data_fp32.min(dim=1)[0] / scale * (-1)
data_int8_ref = torch.quantize_per_channel(data_fp32, scale, zero_point, 0, torch.quint4x2)
print("data_int8_ref: ", data_int8_ref.int_repr())

# data_int8 = torch.empty_like(data_int8_ref.int_repr())
data_int8 = torch.empty((math.ceil(row / float(8 / bits)), col), dtype=torch.uint8)
quantization_cpu.quantize_tensor(data_fp32, data_int8, data_fp32.min(dim=1)[0], scale, bits)
print("data_int8: ", data_int8)
print(data_int8.shape)

data_fp32_dequant_ref = torch.dequantize(data_int8_ref)
print("data_fp32_dequant_ref: ", data_fp32_dequant_ref)
data_fp32_dequant = torch.empty_like(data_fp32_dequant_ref)
quantization_cpu.dequantize_tensor(data_int8, data_fp32_dequant, data_fp32.min(dim=1)[0], scale, bits)
print("data_fp32_dequant: ", data_fp32_dequant)

torch.set_printoptions(precision=10)
atol = 1e-06
rtol = 1e-05
# idx_of_diff = torch.where(torch.abs(data_int8 - data_int8_ref.int_repr()) > (atol + rtol * torch.abs(data_int8_ref.int_repr())))
# print(idx_of_diff)
# print("ref_int8[idx_of_diff] = {}".format(data_int8_ref.int_repr()[idx_of_diff]))
# print("our_int8[idx_of_diff] = {}".format(data_int8[idx_of_diff]))
# assert(torch.allclose(data_int8_ref.int_repr(), data_int8))
idx_of_diff = torch.where(torch.abs(data_fp32_dequant - data_fp32_dequant_ref) > \
                                 (atol + rtol * torch.abs(data_fp32_dequant_ref)))
print("ref_fp32[idx_of_diff] = {}".format(data_fp32_dequant_ref[idx_of_diff]))
print("our_fp32[idx_of_diff] = {}".format(data_fp32_dequant[idx_of_diff]))
# assert(torch.allclose(data_fp32_dequant_ref, data_fp32_dequant, atol=atol, rtol=rtol))

# check performance
import time
import numpy as np

warmup = 2
repeat = warmup + 10
quantization_time_ref = np.zeros(repeat)
dequantization_time_ref = np.zeros(repeat)

quantization_time_ours = np.zeros(repeat)
dequantization_time_ours = np.zeros(repeat)

for i in range(repeat):
    start = time.perf_counter()
    data_int8 = torch.empty((math.ceil(row / float(8 / bits)), col), dtype=torch.uint8)
    quantization_cpu.quantize_tensor(data_fp32, data_int8, data_fp32.min(dim=1)[0], scale, bits)
    end = time.perf_counter()
    quantization_time_ours[i] = (end - start) * 1000.0

    start = time.perf_counter()
    data_fp32_ours = torch.empty_like(data_fp32)
    quantization_cpu.dequantize_tensor(data_int8, data_fp32_ours, data_fp32.min(dim=1)[0], scale, bits)
    end = time.perf_counter()
    dequantization_time_ours[i] = (end - start) * 1000.0

for i in range(repeat):
    start = time.perf_counter()
    data_int8_ref = torch.quantize_per_channel(data_fp32, scale, zero_point, 0, torch.quint8)
    data_int8 = data_int8_ref.int_repr()
    end = time.perf_counter()
    quantization_time_ref[i] = (end - start) * 1000.0

    start = time.perf_counter()
    data_fp32_ref = torch.dequantize(data_int8_ref)
    end = time.perf_counter()
    dequantization_time_ref[i] = (end - start) * 1000.0

print("quantization_time_ref: ", np.mean(quantization_time_ref[warmup:]))
print("dequantization_time_ref: ", np.mean(dequantization_time_ref[warmup:]))
print("quantization_time_ours: ", np.mean(quantization_time_ours[warmup:]))
print("dequantization_time_ours: ", np.mean(dequantization_time_ours[warmup:]))
