import os
import time
import torch
import torch.distributed as dist
from assigner import Assigner
from time_recorder import TimeRecorder
from quantizer import Quantizer, Quantizer_v1


class Communicator(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def init_dist_group():
        if dist.is_mpi_available():
            # backend with mpi
            print("mpi in torch.distributed is available!")
            dist.init_process_group(backend="mpi")
            # fugaku, reserve 1 thread for asynchronous
            torch.set_num_threads(11)
        else:
            # backend with torch_ccl
            import torch_ccl

            world_size = int(os.environ.get("PMI_SIZE", -1))
            rank = int(os.environ.get("PMI_RANK", -1))
            print("use ccl backend for torch.distributed package on x86 cpu.")
            dist.init_process_group(backend="ccl", init_method="env://", world_size=world_size, rank=rank)

        print(f"dist_info RANK: {dist.get_rank()}, SIZE: {dist.get_world_size()}")
        # number of process in this MPI group
        world_size = dist.get_world_size()
        # mpi rank in this MPI group
        rank = dist.get_rank()

        return (rank, world_size)

    @staticmethod
    def comm_with_fp32(recv_buf, send_buf, recv_splits, send_splits):
        barrier_begin = time.perf_counter()
        dist.barrier()
        barrier_end = time.perf_counter()
        comm_begin = time.perf_counter()
        handle = dist.all_to_all_single(recv_buf, send_buf, recv_splits, send_splits, async_op=False)
        comm_end = time.perf_counter()
        TimeRecorder.ctx.record_barrier_time(barrier_end - barrier_begin)
        TimeRecorder.ctx.record_communication_time(comm_end - comm_begin)
        return handle

    @staticmethod
    def comm_with_fp16(recv_buf, send_buf, recv_buf_fp16, send_buf_fp16, recv_splits, send_splits):
        quantization_begin = time.perf_counter()
        send_buf_fp16.copy_(send_buf)
        quantization_end = time.perf_counter()
        barrier_begin = time.perf_counter()
        dist.barrier()
        barrier_end = time.perf_counter()
        comm_begin = time.perf_counter()
        handle = dist.all_to_all_single(
            recv_buf_fp16, send_buf_fp16, recv_splits, send_splits, async_op=False
        )
        comm_end = time.perf_counter()
        dequantization_begin = time.perf_counter()
        recv_buf.copy_(recv_buf_fp16)
        dequantization_end = time.perf_counter()
        TimeRecorder.ctx.record_quantization_time(quantization_end - quantization_begin)
        TimeRecorder.ctx.record_barrier_time(barrier_end - barrier_begin)
        TimeRecorder.ctx.record_communication_time(comm_end - comm_begin)
        TimeRecorder.ctx.record_dequantization_time(dequantization_end - dequantization_begin)
        return handle

    @staticmethod
    def comm_with_quantization(
        recv_buf,
        send_buf,
        recv_splits,
        send_splits,
        recv_quant_data_buf_list,
        send_quant_data_buf_list,
        recv_quant_param_buf_list,
        send_quant_param_buf_list,
        num_bits,
        world_size,
    ):
        # send_nodes_feat_fp16_buf.copy_(send_nodes_feat_buf)
        quantize_begin = time.perf_counter()

        send_begin_idx = 0
        send_end_idx = 0

        for rank in range(world_size):
            send_begin_idx = send_end_idx
            send_end_idx += send_splits[rank]

            # prepare the quantized buffer for communication
            num_send_nodes = Quantizer.get_quantized_buffer_size(send_splits[rank], num_bits)
            num_recv_nodes = Quantizer.get_quantized_buffer_size(recv_splits[rank], num_bits)
            send_quant_data_buf_list.append(torch.empty(num_send_nodes, send_buf.size(-1), dtype=torch.uint8))
            recv_quant_data_buf_list.append(torch.empty(num_recv_nodes, recv_buf.size(-1), dtype=torch.uint8))

            if num_send_nodes > 0:
                # quantize the data
                send_quant_param = Quantizer.quantize_fp32_to_intX(
                    send_buf[send_begin_idx:send_end_idx], send_quant_data_buf_list[rank], num_bits
                )
            else:
                send_quant_param = torch.empty((send_splits[rank], 2), dtype=torch.float32)
            recv_quant_param = torch.empty((recv_splits[rank], 2), dtype=torch.float32)

            # collect the quantized params (scale and zero_point)
            send_quant_param_buf_list.append(send_quant_param)
            recv_quant_param_buf_list.append(recv_quant_param)

        rank = dist.get_rank()
        quantize_end = time.perf_counter()
        # print_time(rank, "outer quantize data(ms): ", (quantize_end - quantize_begin) * 1000.0)
        # TimeRecorder.print_time(rank, "outer quantize data (ms): ", (quantize_end - quantize_begin) * 1000.0)

        barrier_begin = time.perf_counter()
        dist.barrier()
        barrier_end = time.perf_counter()
        # print_time(rank, "barrier (ms): ", (barrier_end - barrier_begin) * 1000.0)
        # comm for quantized params (scale and zero_point)
        comm_begin = time.perf_counter()
        dist.all_to_all(recv_quant_param_buf_list, send_quant_param_buf_list, async_op=False)
        # print_time(rank, "inner comm for param (ms): ", (comm_for_param_end - comm_for_param_begin) * 1000.0)
        # comm for quantized data
        handle = dist.all_to_all(recv_quant_data_buf_list, send_quant_data_buf_list, async_op=False)
        comm_end = time.perf_counter()

        TimeRecorder.ctx.record_quantization_time(quantize_end - quantize_begin)
        TimeRecorder.ctx.record_barrier_time(barrier_end - barrier_begin)
        TimeRecorder.ctx.record_communication_time(comm_end - comm_begin)
        # print_time(rank, "inner comm for data (ms): ", (comm_end - comm_for_data_begin) * 1000.0)
        # print_time(rank, "inner total comm (ms): ", (comm_end - quantize_begin) * 1000.0)

        return handle

    @staticmethod
    def get_splits_for_comm_quantized_data(
        recv_splits_tensor,
        send_splits_tensor,
        dequantized_nodes_feat_range,
        quantized_nodes_feat_range,
        world_size,
    ):
        # prepare the buffer for receiving the quantized data based on the quantized params[0]
        # (range of quantized feature)
        quantized_recv_splits = list()
        quantized_send_splits = list()

        recv_node_idx_begin = torch.empty((world_size + 1), dtype=torch.int32)
        recv_node_idx_begin[0] = 0
        recv_node_idx_begin[1:] = recv_splits_tensor.cumsum(dim=0)

        send_node_idx_begin = torch.empty((world_size + 1), dtype=torch.int32)
        send_node_idx_begin[0] = 0
        send_node_idx_begin[1:] = send_splits_tensor.cumsum(dim=0)

        for rank in range(world_size):
            quantized_recv_splits.append(
                (
                    dequantized_nodes_feat_range[recv_node_idx_begin[rank + 1]]
                    - dequantized_nodes_feat_range[recv_node_idx_begin[rank]]
                ).item()
            )
            quantized_send_splits.append(
                (
                    quantized_nodes_feat_range[send_node_idx_begin[rank + 1]]
                    - quantized_nodes_feat_range[send_node_idx_begin[rank]]
                ).item()
            )

        return quantized_recv_splits, quantized_send_splits

    @staticmethod
    def comm_with_quantization_v1(
        recv_buf, send_buf, recv_splits, send_splits, nodes_num_bits_tensor, world_size
    ):
        prepare_params_begin = time.perf_counter()
        recv_splits_tensor = torch.tensor(recv_splits, dtype=torch.int32)
        send_splits_tensor = torch.tensor(send_splits, dtype=torch.int32)
        num_send_nodes = send_splits_tensor.sum().item()
        num_recv_nodes = recv_splits_tensor.sum().item()

        # create quantized params buffer for communication
        # [zero_points, scales, nodes_num_bits]
        send_params = torch.empty((num_send_nodes, (2 + 1)), dtype=torch.float32)

        # to get the random bits for each node
        send_params[:, 0] = nodes_num_bits_tensor

        # get the range of each node's quantized feature
        quantized_nodes_feat_range = Quantizer_v1.get_quantized_nodes_feat_range(
            num_send_nodes, send_buf.size(1), nodes_num_bits_tensor
        )

        quantized_send_buf = torch.empty(quantized_nodes_feat_range[-1], dtype=torch.uint8)
        prepare_params_end = time.perf_counter()

        quantization_begin = time.perf_counter()
        # quantize the data
        Quantizer_v1.quantize_fp32_to_intX(
            send_buf, quantized_send_buf, quantized_nodes_feat_range, send_params
        )
        quantization_end = time.perf_counter()

        barrier_begin = time.perf_counter()
        dist.barrier()
        barrier_end = time.perf_counter()

        # comm_for_param_begin = time.perf_counter()
        # prepare the buffer for receiving the quantized params
        recv_params = torch.empty((num_recv_nodes, (2 + 1)), dtype=torch.float32)

        comm_for_data_begin = time.perf_counter()
        # communication for quantized params
        dist.all_to_all_single(recv_params, send_params, recv_splits, send_splits, async_op=False)
        dequantized_params = recv_params

        # get the range of each node's dequantized feature
        dequantized_nodes_feat_range = Quantizer_v1.get_quantized_nodes_feat_range(
            num_recv_nodes, send_buf.size(1), dequantized_params[:, 0]
        )

        # get the splits for communication of quantized data
        quantized_recv_splits, quantized_send_splits = Communicator.get_splits_for_comm_quantized_data(
            recv_splits_tensor,
            send_splits_tensor,
            dequantized_nodes_feat_range,
            quantized_nodes_feat_range,
            world_size,
        )

        quantized_recv_buf = torch.empty((dequantized_nodes_feat_range[-1]), dtype=torch.uint8)

        # communication for quantized data
        dist.all_to_all_single(
            quantized_recv_buf,
            quantized_send_buf,
            quantized_recv_splits,
            quantized_send_splits,
            async_op=False,
        )
        comm_for_data_end = time.perf_counter()

        # TimeRecorder.print_time(
        #     dist.get_rank(),
        #     "inner prepare params (ms): ",
        #     (prepare_params_end - prepare_params_begin) * 1000.0,
        # )
        # TimeRecorder.print_time(
        #     dist.get_rank(), "inner quantization (ms): ", (quantization_end - quantization_begin) * 1000.0
        # )
        # TimeRecorder.print_time(
        #     dist.get_rank(), "inner barrier (ms): ", (barrier_end - barrier_begin) * 1000.0
        # )
        # TimeRecorder.print_time(
        #     dist.get_rank(), "inner comm for data (ms): ", (comm_for_data_end - comm_for_data_begin) * 1000.0
        # )

        TimeRecorder.ctx.record_barrier_time(barrier_end - barrier_begin)
        TimeRecorder.ctx.record_quantization_time(
            quantization_end - quantization_begin + prepare_params_end - prepare_params_begin
        )
        TimeRecorder.ctx.record_communication_time(comm_for_data_end - comm_for_data_begin)

        return (quantized_recv_buf, dequantized_nodes_feat_range, dequantized_params)

    @staticmethod
    def convert_data_to_fp32(
        recv_quant_data_buf_list, recv_buf, recv_splits, recv_quant_param_buf_list, num_bits, world_size
    ):
        dequantize_begin = time.perf_counter()
        begin_idx = 0
        end_idx = 0
        for rank in range(world_size):
            begin_idx = end_idx
            end_idx += recv_splits[rank]
            scale = recv_quant_param_buf_list[rank][:, 0]
            zero_point = recv_quant_param_buf_list[rank][:, 1]
            if end_idx - begin_idx > 0:
                Quantizer.dequantize_intX_to_fp32(
                    recv_quant_data_buf_list[rank], recv_buf[begin_idx:end_idx], scale, zero_point, num_bits
                )
        dequantize_end = time.perf_counter()
        # TimeRecorder.print_time(
        #     dist.get_rank(), "inner dequantize data (ms): ", (dequantize_end - dequantize_begin) * 1000.0
        # )
        TimeRecorder.ctx.record_dequantization_time(dequantize_end - dequantize_begin)

    @staticmethod
    def convert_data_to_fp32_v1(quantized_recv_buf, recv_buf, quantized_nodes_feat_range, dequantized_params):
        dequantization_begin = time.perf_counter()
        Quantizer_v1.dequantize_intX_to_fp32(
            quantized_recv_buf, recv_buf, quantized_nodes_feat_range, dequantized_params
        )
        dequantization_end = time.perf_counter()
        TimeRecorder.ctx.record_dequantization_time(dequantization_end - dequantization_begin)
