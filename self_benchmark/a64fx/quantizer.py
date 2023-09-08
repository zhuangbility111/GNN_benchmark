import time
import math
import torch
import torch.distributed as dist
import quantization_cpu
from time_recorder import TimeRecorder

class Quantizer(object):
    def __init__(self):
        pass

    @staticmethod
    def quantize_fp32_to_intX(data_fp32, data_int8, num_bits=8):
        total_quantize_begin = time.perf_counter()
        zero_point = data_fp32.min(dim=1)[0]
        scale = (data_fp32.max(dim=1)[0] - zero_point + 10e-20) / (2**num_bits - 1)
        inner_quantize_begin = time.perf_counter()
        quantization_cpu.quantize_tensor(data_fp32, data_int8, zero_point, scale, num_bits)
        inner_quantize_end = time.perf_counter()
        # print(data_int8)
        quant_param = torch.stack((scale, zero_point), dim=1)
        total_quantize_end = time.perf_counter()
        # print("inner quantize data(ms): {}".format((quantize_end - quantize_begin) * 1000.0))
        TimeRecorder.print_time(dist.get_rank(), "data_fp32.shape", data_fp32.shape)
        TimeRecorder.print_time(dist.get_rank(), "inner prepare data for quantization(ms): ", (inner_quantize_begin - total_quantize_begin) * 1000.0)
        TimeRecorder.print_time(dist.get_rank(), "inner inner quantize data(ms): ", (inner_quantize_end - inner_quantize_begin) * 1000.0)
        TimeRecorder.print_time(dist.get_rank(), "inner stack data(ms): ", (total_quantize_end - inner_quantize_end) * 1000.0)
        TimeRecorder.print_time(dist.get_rank(), "inner quantize data(ms): ", (total_quantize_end - total_quantize_begin) * 1000.0)
        return quant_param

    @staticmethod
    def dequantize_intX_to_fp32(data_int8, data_fp32, scale, zero_point, num_bits=8):
        # data_fp32.copy_((data_int8 - zero_point.view(-1, 1)) * scale.view(-1, 1))
        quantization_cpu.dequantize_tensor(data_int8, data_fp32, zero_point, scale, num_bits)

    @staticmethod
    def get_quantized_buffer_size(num_comm_nodes, num_bits):
        return math.ceil(num_comm_nodes / float(8 / num_bits))

class Quantizer_v1(object):
    @staticmethod
    def quantize_fp32_to_intX(data_fp32, data_int8, quantized_nodes_feat_range, nodes_num_bits_tensor):
        zero_points = torch.empty((data_fp32.size(0)), dtype=torch.float32)
        scales = torch.empty((data_fp32.size(0)), dtype=torch.float32)
        quantization_cpu.quantize_tensor_v1(data_fp32, data_int8, 
                                            quantized_nodes_feat_range, nodes_num_bits_tensor, 
                                            zero_points, scales)
        return zero_points, scales
    
    @staticmethod
    def dequantize_intX_to_fp32(data_int8, data_fp32, quantized_nodes_feat_range, node_num_bits,
                                   zero_points, scales):
        quantization_cpu.dequantize_tensor_v1(data_int8, data_fp32, 
                                              quantized_nodes_feat_range, node_num_bits, zero_points, scales)
        
    @staticmethod
    def get_quantized_nodes_feat_range(num_nodes: int, feat_len: int, nodes_num_bits_tensor: Tensor):
        # get the total bits of each node's quantized feature (size = num_nodes)
        quantized_nodes_feat_bits = torch.ceil(nodes_num_bits_tensor * feat_len / 8.0)
        # get the range of each node's quantized feature (start from 0) (size = num_nodes + 1)
        quantized_nodes_feat_range = torch.empty((num_nodes+1), dtype=torch.int32)
        torch.cumsum(quantized_nodes_feat_bits, dim=0, out=quantized_nodes_feat_range[1:])
        return quantized_nodes_feat_bits, quantized_nodes_feat_range
