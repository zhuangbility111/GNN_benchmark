import time
import math
import torch
from torch import Tensor
import torch.distributed as dist
import quantization_cpu
from time_recorder import TimeRecorder

class Quantizer(object):
    def __init__(self):
        pass

    @staticmethod
    def quantize_fp32_to_intX(data_fp32, data_intX, quant_params, num_bits=8):
        # total_quantize_begin = time.perf_counter()
        # inner_quantize_begin = time.perf_counter()
        quantization_cpu.quantize_tensor(data_fp32, data_intX, quant_params, num_bits)
        # inner_quantize_end = time.perf_counter()
        # print(data_int8)
        # total_quantize_end = time.perf_counter()
        # print("inner quantize data(ms): {}".format((quantize_end - quantize_begin) * 1000.0))
        # print("inner inner quantize data(ms): {}".format((inner_quantize_end - inner_quantize_begin) * 1000.0))
        # print("quantized data shape: {}".format(data_fp32.shape))
        # print("inner quantize data(ms): {}".format((total_quantize_end - total_quantize_begin) * 1000.0))
        # TimeRecorder.print_time(
        #     dist.get_rank(),
        #     "inner inner quantize data(ms): ",
        #     (inner_quantize_end - inner_quantize_begin) * 1000.0,
        # )
        # TimeRecorder.print_time(
        #     dist.get_rank(),
        #     "quantized data shape: ",
        #     data_fp32.shape,
        # )
        # TimeRecorder.print_time(
        #     dist.get_rank(), "inner quantize data(ms): ", (total_quantize_end - total_quantize_begin) * 1000.0
        # )

    @staticmethod
    def dequantize_intX_to_fp32(data_intX, data_fp32, dequant_params, num_bits=8):
        # data_fp32.copy_((data_int8 - zero_point.view(-1, 1)) * scale.view(-1, 1))
        quantization_cpu.dequantize_tensor(data_intX, data_fp32, dequant_params, num_bits)

    @staticmethod
    def get_quantized_buffer_size(num_comm_nodes, num_bits, feat_len):
        return math.ceil(num_comm_nodes / float(8 / num_bits)) * feat_len


class Quantizer_v1(object):
    @staticmethod
    def quantize_fp32_to_intX(data_fp32, data_int8, quantized_nodes_feat_range, quantized_params):
        # zero_points = torch.empty((data_fp32.size(0)), dtype=torch.float32)
        # scales = torch.empty((data_fp32.size(0)), dtype=torch.float32)
        quantization_cpu.quantize_tensor_v1(
            data_fp32, data_int8, quantized_nodes_feat_range, quantized_params
        )

    @staticmethod
    def dequantize_intX_to_fp32(data_int8, data_fp32, quantized_nodes_feat_range, dequantized_params):
        dequantization_begin = time.perf_counter()
        quantization_cpu.dequantize_tensor_v1(
            data_int8, data_fp32, quantized_nodes_feat_range, dequantized_params
        )
        dequantization_end = time.perf_counter()
        TimeRecorder.ctx.record_dequantization_time(dequantization_end - dequantization_begin)

    @staticmethod
    def get_quantized_nodes_feat_range(num_nodes: int, feat_len: int, nodes_num_bits_tensor: Tensor):
        # get the total bits of each node's quantized feature (size = num_nodes)
        quantized_nodes_feat_len = torch.ceil(nodes_num_bits_tensor * feat_len / 8.0)
        # get the range of each node's quantized feature (start from 0) (size = num_nodes + 1)
        quantized_nodes_feat_range = torch.empty((num_nodes + 1), dtype=torch.int64)
        quantized_nodes_feat_range[0] = 0
        torch.cumsum(quantized_nodes_feat_len, dim=0, out=quantized_nodes_feat_range[1:])
        return quantized_nodes_feat_range

class Quantizer_for_all_procs(object):
    def __init__(self, world_size, num_bits) -> None:
        self.num_bits = num_bits
        self.world_size = world_size
        Quantizer_for_all_procs.ctx = self

    def quantize_fp32_to_intX(self, data_fp32, data_int8, quantized_params, quantized_work_range_per_proc):
        quantization_cpu.quantize_tensor_for_all_procs(data_fp32, data_int8, quantized_params, 
                                                       quantized_work_range_per_proc, 
                                                       self.world_size, self.num_bits)

    def dequantize_intX_to_fp32(self, data_int8, data_fp32, quantized_params, dequantized_work_range_per_proc):
        quantization_cpu.dequantize_tensor_for_all_procs(data_int8, data_fp32, quantized_params, 
                                                         dequantized_work_range_per_proc, 
                                                         self.world_size, self.num_bits)
    
    def get_quantized_splits(self, num_comm_nodes):
        return math.ceil(num_comm_nodes / float(8 / self.num_bits))
