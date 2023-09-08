import torch
import quantization_cpu
import math
import time
import numpy as np

torch.set_num_threads(20)

num_nodes = 50000
feat_len = 256

def run_torch_quantize_per_channel(data_fp32, scale, zero_point, bits):
    data_int8_ref = None
    if bits == 8:
        data_int8_ref = torch.quantize_per_channel(data_fp32, scale, zero_point, 0, torch.quint8)
    elif bits == 4:
        data_int8_ref = torch.quantize_per_channel(data_fp32, scale, zero_point, 0, torch.quint4x2)
    else:
        print("bits = {} is not supported on torch_quantize_per_channel".format(bits))
    return data_int8_ref

def run_torch_dequantize(data_int8_ref):
    return torch.dequantize(data_int8_ref)

def run_quantization_cpu_quantize_tensor(data_fp32, min_val, scale, bits):
    data_int8 = torch.empty((math.ceil(num_nodes / float(8 / bits)), feat_len), dtype=torch.uint8)
    inner_quantization_begin = time.perf_counter()
    quantization_cpu.quantize_tensor(data_fp32, data_int8, min_val, scale, bits)
    inner_quantization_end = time.perf_counter()
    print("inner_quantization_time (ms): ", (inner_quantization_end - inner_quantization_begin) * 1000.0)
    return data_int8

def run_quantization_cpu_dequantize_tensor(data_int8, min_val, scale, bits):
    data_fp32_dequant = torch.empty((num_nodes, feat_len), dtype=torch.float32)
    quantization_cpu.dequantize_tensor(data_int8, data_fp32_dequant, min_val, scale, bits)
    return data_fp32_dequant

def run_quantization_cpu_quantize_tensor_v1(data_fp32, bits):
    prepare_begin = time.perf_counter()
    nodes_num_bits_tensor = torch.full((num_nodes, ), bits, dtype=torch.int32)

    # prefix sum of nodes_num_bits_tensor
    quantized_nodes_feat_range = torch.full((num_nodes+1, ), feat_len, dtype=torch.int64)
    quantized_nodes_feat_range[0] = 0
    quantized_nodes_feat_range[1:] = torch.ceil(quantized_nodes_feat_range[1:] * nodes_num_bits_tensor / 8.0)
    quantized_nodes_feat_range = torch.cumsum(quantized_nodes_feat_range, 0)
    # for i in range(0, num_nodes):
    #     quantized_nodes_feat_range[i+1] = quantized_nodes_feat_range[i] + \
    #                 torch.ceil((nodes_num_bits_tensor[i]) * feat_len / 8.0)
    data_int8 = torch.empty(quantized_nodes_feat_range[-1], dtype=torch.uint8)
    quantized_params = torch.empty((num_nodes, 2), dtype=torch.float32)
    prepare_end = time.perf_counter()
    inner_quantization_begin = time.perf_counter()
    quantization_cpu.quantize_tensor_v1(data_fp32, data_int8, 
                                        quantized_nodes_feat_range, 
                                        nodes_num_bits_tensor, quantized_params)
    inner_quantization_end = time.perf_counter()
    print("prepare_time (ms): ", (prepare_end - prepare_begin) * 1000.0)
    print("inner_quantization_time (ms): ", (inner_quantization_end - inner_quantization_begin) * 1000.0)
    return data_int8, quantized_nodes_feat_range, nodes_num_bits_tensor, quantized_params

def run_quantization_cpu_dequantize_tensor_v1(data_int8, quantized_nodes_feat_range, 
                                              nodes_num_bits_tensor, quantized_params):
    data_fp32_dequant = torch.empty((num_nodes, feat_len), dtype=torch.float32)
    quantization_cpu.dequantize_tensor_v1(data_int8, data_fp32_dequant, quantized_nodes_feat_range, 
                                          nodes_num_bits_tensor, quantized_params)
    return data_fp32_dequant


def test_correctness_for_quantize_tensor(data_fp32, min_val, zero_point, scale, bits):
    data_int8_ref = run_torch_quantize_per_channel(data_fp32, scale, zero_point, bits)
    data_fp32_dequant_ref = run_torch_dequantize(data_int8_ref)
    data_int8 = run_quantization_cpu_quantize_tensor(data_fp32, min_val, scale, bits)
    data_fp32_dequant = run_quantization_cpu_dequantize_tensor(data_int8, min_val, scale, bits)
    atol = 1e-06
    rtol = 1e-05
    idx_of_diff = torch.where(torch.abs(data_fp32_dequant - data_fp32_dequant_ref) > \
                                 (atol + rtol * torch.abs(data_fp32_dequant_ref)))
    torch.set_printoptions(precision=10)
    print(f"ref_fp32[idx_of_diff] = {data_fp32_dequant_ref[idx_of_diff]}")
    print(f"our_fp32[idx_of_diff] = {data_fp32_dequant[idx_of_diff]}")
    # print(f"ref_fp32 = {data_fp32_dequant_ref}")
    # print(f"our_fp32 = {data_fp32_dequant}")
    # print(f"our_int8 = {data_int8}")
    # idx_of_diff = torch.where(torch.abs(data_int8 - data_int8_ref.int_repr()) > (atol + rtol * torch.abs(data_int8_ref.int_repr())))
    # print(idx_of_diff)
    # print("ref_int8[idx_of_diff] = {}".format(data_int8_ref.int_repr()[idx_of_diff]))
    # print("our_int8[idx_of_diff] = {}".format(data_int8[idx_of_diff]))
    # assert(torch.allclose(data_int8_ref.int_repr(), data_int8))

def test_perf_for_quantize_tensor(data_fp32, min_val, zero_point, scale, bits, warmup, repeat):
    # check performance
    repeat = warmup + repeat
    quantization_time_ref = np.zeros(repeat)
    dequantization_time_ref = np.zeros(repeat)

    quantization_time_ours = np.zeros(repeat)
    dequantization_time_ours = np.zeros(repeat)

    for i in range(repeat):
        start = time.perf_counter()
        data_int8 = run_quantization_cpu_quantize_tensor(data_fp32, min_val, scale, bits)
        end = time.perf_counter()
        quantization_time_ours[i] = (end - start) * 1000.0

        start = time.perf_counter()
        run_quantization_cpu_dequantize_tensor(data_int8, min_val, scale, bits)
        end = time.perf_counter()
        dequantization_time_ours[i] = (end - start) * 1000.0

    for i in range(repeat):
        start = time.perf_counter()
        data_int8_ref = torch.quantize_per_channel(data_fp32, scale, zero_point, 0, torch.quint4x2)
        end = time.perf_counter()
        quantization_time_ref[i] = (end - start) * 1000.0

        start = time.perf_counter()
        torch.dequantize(data_int8_ref)
        end = time.perf_counter()
        dequantization_time_ref[i] = (end - start) * 1000.0

    print("quantization_time_ref: ", np.mean(quantization_time_ref[warmup:]))
    print("dequantization_time_ref: ", np.mean(dequantization_time_ref[warmup:]))
    print("quantization_time_ours: ", np.mean(quantization_time_ours[warmup:]))
    print("dequantization_time_ours: ", np.mean(dequantization_time_ours[warmup:]))

def test_correctness_for_quantize_tensor_v1(data_fp32, zero_point, scale, bits):
    data_int8_ref = run_torch_quantize_per_channel(data_fp32, scale, zero_point, bits)
    data_fp32_dequant_ref = run_torch_dequantize(data_int8_ref)
    data_int8, quantized_nodes_feat_range, nodes_num_bits_tensor, quantized_params  = \
        run_quantization_cpu_quantize_tensor_v1(data_fp32, bits)
    data_fp32_dequant = run_quantization_cpu_dequantize_tensor_v1(data_int8, quantized_nodes_feat_range, \
                                                                  nodes_num_bits_tensor, quantized_params)
    atol = 1e-06
    rtol = 1e-05
    torch.set_printoptions(precision=10)
    # idx_of_diff = torch.where(torch.abs(data_int8 - data_int8_ref.int_repr()) >
    #                            (atol + rtol * torch.abs(data_int8_ref.int_repr())))
    # print("ref_int8[idx_of_diff] = {}".format(data_int8_ref.int_repr()[idx_of_diff]))
    # print("our_int8[idx_of_diff] = {}".format(data_int8[idx_of_diff]))
    # assert(torch.allclose(data_int8_ref.int_repr(), data_int8))
    idx_of_diff = torch.where(torch.abs(data_fp32_dequant - data_fp32_dequant_ref) > \
                                 (atol + rtol * torch.abs(data_fp32_dequant_ref)))
    # print(f"ref_fp32 = {data_fp32_dequant_ref}")
    # print(f"our_fp32 = {data_fp32_dequant}")
    print(f"ref_fp32[idx_of_diff] = {data_fp32_dequant_ref[idx_of_diff]}")
    print(f"our_fp32[idx_of_diff] = {data_fp32_dequant[idx_of_diff]}")

def test_perf_for_quantize_tensor_v1(data_fp32, zero_point, scale, bits, warmup, repeat):
    # check performance
    repeat = warmup + repeat
    quantization_time_ref = np.zeros(repeat)
    dequantization_time_ref = np.zeros(repeat)

    quantization_time_ours = np.zeros(repeat)
    dequantization_time_ours = np.zeros(repeat)

    for i in range(repeat):
        start = time.perf_counter()
        data_int8, quantized_nodes_feat_range, nodes_num_bits_tensor, quantized_params = \
            run_quantization_cpu_quantize_tensor_v1(data_fp32, bits)
        end = time.perf_counter()
        quantization_time_ours[i] = (end - start) * 1000.0

        start = time.perf_counter()
        run_quantization_cpu_dequantize_tensor_v1(data_int8, quantized_nodes_feat_range, \
                                                  nodes_num_bits_tensor, quantized_params)
        end = time.perf_counter()
        dequantization_time_ours[i] = (end - start) * 1000.0

    for i in range(repeat):
        start = time.perf_counter()
        data_int8_ref = torch.quantize_per_channel(data_fp32, scale, zero_point, 0, torch.quint4x2)
        end = time.perf_counter()
        quantization_time_ref[i] = (end - start) * 1000.0

        start = time.perf_counter()
        torch.dequantize(data_int8_ref)
        end = time.perf_counter()
        dequantization_time_ref[i] = (end - start) * 1000.0

    print("quantization_time_ref: ", np.mean(quantization_time_ref[warmup:]))
    print("dequantization_time_ref: ", np.mean(dequantization_time_ref[warmup:]))
    print("quantization_time_ours: ", np.mean(quantization_time_ours[warmup:]))
    print("dequantization_time_ours: ", np.mean(dequantization_time_ours[warmup:]))

if __name__ == "__main__":
    # data_fp32 = torch.Tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 2.0, 1.0, 4.0, 5.0]])
    data_fp32 = torch.randn((num_nodes, feat_len), dtype=torch.float32)
    # data_fp32 = torch.empty((num_nodes, feat_len), dtype=torch.float32)
    # for i in range(num_nodes):
    #     data_fp32[i] = torch.arange(feat_len, dtype=torch.float32)

    bits = 8

    # print(data_fp32)

    min_val = data_fp32.min(dim=1)[0]
    scale = (data_fp32.max(dim=1)[0] - data_fp32.min(dim=1)[0] + 10e-20) / (2**bits - 1)
    zero_point = data_fp32.min(dim=1)[0] / scale * (-1)

    test_correctness_for_quantize_tensor(data_fp32, min_val, zero_point, scale, bits)
    test_perf_for_quantize_tensor(data_fp32, min_val, zero_point, scale, bits, 2, 10)
    test_correctness_for_quantize_tensor_v1(data_fp32, zero_point, scale, bits)
    test_perf_for_quantize_tensor_v1(data_fp32, zero_point, scale, bits, 2, 10)
