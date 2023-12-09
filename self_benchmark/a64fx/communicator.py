import os
import time
import torch
import torch.distributed as dist
from assigner import Assigner
from time_recorder import TimeRecorder
from quantizer import QuantizerForPureBits, QuantizerForMixedBits, Quantizer_for_all_procs
from data_manager import CommBuffer, CommBufferForQuantization
from data_manager import CommSplits


def diff(out, ref, num_of_out, num_of_ref, atol=1e-05, rtol=1e-05):
    torch.set_printoptions(precision=10)
    idx_of_diff = torch.where(torch.abs(out - ref) > (atol + rtol * torch.abs(ref)))
    print(f"{num_of_ref}[idx_of_diff] = {ref[idx_of_diff]}")
    print(f"{num_of_out}[idx_of_diff] = {out[idx_of_diff]}")


class Communicator(object):
    def __init__(self, num_bits=32, is_async=True) -> None:
        self.num_bits = num_bits
        self.is_async = is_async
        self.world_size = -1
        self.rank = -1

        Communicator.ctx = self

    def init_dist_group(self):
        # fugaku
        if dist.is_mpi_available():
            # backend with mpi
            print("mpi in torch.distributed is available!")
            dist.init_process_group(backend="mpi")
            if self.is_async:
                # fugaku, reserve 1 thread for asynchronous
                torch.set_num_threads(11)
            else:
                torch.set_num_threads(12)
            print("num_threads: ", torch.get_num_threads())
        # abci
        else:
            # backend with torch_ccl
            import torch_ccl

            world_size = int(os.environ.get("PMI_SIZE", -1))
            rank = int(os.environ.get("PMI_RANK", -1))
            print("use ccl backend for torch.distributed package on x86 cpu.")
            dist.init_process_group(backend="ccl", init_method="env://", world_size=world_size, rank=rank)

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        return (self.rank, self.world_size)

    def comm(
        self, comm_splits, comm_buf, comm_buf_for_quantization, layer, is_training, direction
    ):
        quantized_buf = None
        dequantized_nodes_feat_range = None
        dequantized_params = None
        comm_handle = None

        if self.num_bits == 32 or is_training is False:
            comm_handle = self.comm_with_fp32(comm_splits, comm_buf, direction)
        elif self.num_bits == 16:
            comm_handle = self.comm_with_fp16(comm_splits, comm_buf, direction)
            
        elif self.num_bits == 2:
            # comm_handle, quantized_buf, dequantized_nodes_feat_range, dequantized_params = self.comm_with_pure_quantization(
            #     recv_buf, send_buf, recv_splits, send_splits
            # )
            comm_handle = self.comm_with_int2_quantization_for_all_procs(comm_splits, comm_buf, comm_buf_for_quantization, direction)
        # else:
            # nodes_num_bits_tensor = Assigner.ctx.get_node_dataformat(layer)
            # (
            #     comm_handle,
            #     quantized_buf,
            #     dequantized_nodes_feat_range,
            #     dequantized_params,
            # ) = self.comm_with_mix_quantization(
            #     recv_buf, send_buf, recv_splits, send_splits, nodes_num_bits_tensor
            # )
        return comm_handle

    def comm_with_fp32(self, comm_splits: CommSplits, comm_buf: CommBuffer, direction: str):
        if direction == "forward":
            recv_buf = comm_buf.recv_buf
            send_buf = comm_buf.send_buf
            recv_splits = comm_splits.recv_splits
            send_splits = comm_splits.send_splits
        else:
            recv_buf = comm_buf.send_buf
            send_buf = comm_buf.recv_buf
            recv_splits = comm_splits.send_splits
            send_splits = comm_splits.recv_splits

        barrier_begin = time.perf_counter()
        dist.barrier()
        barrier_end = time.perf_counter()
        comm_begin = time.perf_counter()
        comm_handle = dist.all_to_all_single(recv_buf, send_buf, recv_splits, send_splits, async_op=self.is_async)
        comm_end = time.perf_counter()
        TimeRecorder.ctx.record_barrier_time(barrier_end - barrier_begin)
        TimeRecorder.ctx.record_communication_time(comm_end - comm_begin)
        TimeRecorder.print_time(
            dist.get_rank(), "inner barrier (ms): ", (barrier_end - barrier_begin) * 1000.0
        )
        TimeRecorder.print_time(dist.get_rank(), "inner comm (ms): ", (comm_end - comm_begin) * 1000.0)
        return comm_handle

    def comm_with_fp16(self, comm_splits: CommSplits, comm_buf: CommBuffer, direction: str):
        if direction == "forward":
            recv_buf = comm_buf.recv_buf
            send_buf = comm_buf.send_buf
            recv_buf_fp16 = comm_buf.recv_buf_fp16
            send_buf_fp16 = comm_buf.send_buf_fp16
            recv_splits = comm_splits.recv_splits
            send_splits = comm_splits.send_splits
        else:
            recv_buf = comm_buf.send_buf
            send_buf = comm_buf.recv_buf
            recv_buf_fp16 = comm_buf.send_buf_fp16
            send_buf_fp16 = comm_buf.recv_buf_fp16
            recv_splits = comm_splits.send_splits
            send_splits = comm_splits.recv_splits

        quantization_begin = time.perf_counter()
        send_buf_fp16.copy_(send_buf)
        quantization_end = time.perf_counter()
        barrier_begin = time.perf_counter()
        # dist.barrier()
        barrier_end = time.perf_counter()
        comm_begin = time.perf_counter()
        comm_handle = dist.all_to_all_single(recv_buf_fp16, send_buf_fp16, recv_splits, send_splits, async_op=self.is_async)
        comm_end = time.perf_counter()
        dequantization_begin = time.perf_counter()
        recv_buf.copy_(recv_buf_fp16)
        dequantization_end = time.perf_counter()
        TimeRecorder.ctx.record_quantization_time(quantization_end - quantization_begin)
        TimeRecorder.ctx.record_barrier_time(barrier_end - barrier_begin)
        TimeRecorder.ctx.record_communication_time(comm_end - comm_begin)
        TimeRecorder.ctx.record_dequantization_time(dequantization_end - dequantization_begin)
        return comm_handle

    def comm_with_pure_quantization(
        self,
        recv_buf,
        send_buf,
        recv_splits,
        send_splits,
    ):
        # send_nodes_feat_fp16_buf.copy_(send_nodes_feat_buf)

        prepare_params_begin = time.perf_counter()
        quantized_recv_splits = list()
        quantized_send_splits = list()

        for rank in range(self.world_size):
            # prepare the quantized buffer for communication
            int8_recv_buf_size = QuantizerForPureBits.get_quantized_buffer_size(recv_splits[rank], self.num_bits, recv_buf.size(-1))
            int8_send_buf_size = QuantizerForPureBits.get_quantized_buffer_size(send_splits[rank], self.num_bits, send_buf.size(-1))

            quantized_recv_splits.append(int8_recv_buf_size)
            quantized_send_splits.append(int8_send_buf_size)
        
        quantized_recv_data_buf = torch.empty(sum(quantized_recv_splits), dtype=torch.uint8)
        quantized_send_data_buf = torch.empty(sum(quantized_send_splits), dtype=torch.uint8)

        quantized_recv_params_buf = torch.empty((sum(recv_splits), 2), dtype=torch.float32)
        quantized_send_params_buf = torch.empty((sum(send_splits), 2), dtype=torch.float32)
        prepare_params_end = time.perf_counter()
        
        quantize_begin = time.perf_counter()
        send_begin_idx = 0
        send_end_idx = 0
        quantized_send_begin_idx = 0
        quantized_send_end_idx = 0
        for rank in range(self.world_size):
            send_begin_idx = send_end_idx
            send_end_idx += send_splits[rank]
            num_send_nodes = send_end_idx - send_begin_idx

            quantized_send_begin_idx = quantized_send_end_idx
            quantized_send_end_idx += quantized_send_splits[rank]

            if num_send_nodes > 0:
                # quantize the data
                QuantizerForPureBits.quantize_fp32_to_intX(
                    send_buf[send_begin_idx: send_end_idx], 
                    quantized_send_data_buf[quantized_send_begin_idx: quantized_send_end_idx], 
                    quantized_send_params_buf[send_begin_idx: send_end_idx],
                    self.num_bits
                )

        rank = dist.get_rank()
        quantize_end = time.perf_counter()
        # print_time(rank, "outer quantize data(ms): ", (quantize_end - quantize_begin) * 1000.0)
        # TimeRecorder.print_time(rank, "outer quantize data (ms): ", (quantize_end - quantize_begin) * 1000.0)

        barrier_begin = time.perf_counter()
        # dist.barrier()
        barrier_end = time.perf_counter()
        # print_time(rank, "barrier (ms): ", (barrier_end - barrier_begin) * 1000.0)
        # comm for quantized params (scale and zero_point)
        comm_param_begin = time.perf_counter()
        dist.all_to_all_single(quantized_recv_params_buf, quantized_send_params_buf, recv_splits, send_splits, async_op=False)
        comm_param_end = time.perf_counter()
        comm_data_begin = time.perf_counter()
        # comm_handle = None
        # dist.all_to_all(recv_quant_param_buf_list, send_quant_param_buf_list, async_op=False)
        # print_time(rank, "inner comm for param (ms): ", (comm_for_param_end - comm_for_param_begin) * 1000.0)
        # comm for quantized data
        # comm_handle = dist.all_to_all(recv_quant_data_buf_list, send_quant_data_buf_list, async_op=False)
        comm_handle = dist.all_to_all_single(quantized_recv_data_buf, quantized_send_data_buf, quantized_recv_splits, quantized_send_splits, async_op=self.is_async)
        comm_data_end = time.perf_counter()

        TimeRecorder.ctx.record_quantization_time(quantize_end - quantize_begin)
        TimeRecorder.ctx.record_barrier_time(barrier_end - barrier_begin)
        TimeRecorder.ctx.record_communication_time(comm_data_end - comm_param_begin)
        # print("rank:{}, send_buf shape: {}".format(rank, send_buf.shape), flush=True)
        # print("rank:{}, recv_buf shape (ms): {}".format(rank, recv_buf.shape), flush=True)
        print("rank:{}, prepare params (ms): {}".format(rank, (prepare_params_end - prepare_params_begin) * 1000.0), flush=True)
        print("rank:{}, quantization (ms): {}".format(rank, (quantize_end - quantize_begin) * 1000.0), flush=True)
        # print("rank:{}, barrier (ms): {}".format(rank, (barrier_end - barrier_begin) * 1000.0), flush=True)
        print("rank:{}, comm for param (ms): {}".format(rank, (comm_param_end - comm_param_begin) * 1000.0), flush=True)
        print("rank:{}, comm for data (ms): {}".format(rank, (comm_data_end - comm_data_begin) * 1000.0), flush=True)


        return comm_handle, quantized_recv_data_buf, quantized_recv_splits, quantized_recv_params_buf

    def comm_with_int2_quantization_for_all_procs(
        self,
        comm_splits: CommSplits,
        comm_buf: CommBuffer,
        comm_buf_for_quantization: CommBufferForQuantization,
        direction: str,
    ):
        prepare_params_begin = time.perf_counter()
        if direction == "forward":
            recv_splits = comm_splits.recv_splits
            send_splits = comm_splits.send_splits

            recv_splits_int2 = comm_splits.recv_splits_int2
            send_splits_int2 = comm_splits.send_splits_int2

            recv_buf = comm_buf.recv_buf
            send_buf = comm_buf.send_buf

            quantized_recv_data_buf = comm_buf_for_quantization.quantized_recv_data_buf
            quantized_send_data_buf = comm_buf_for_quantization.quantized_send_data_buf

            quantized_recv_params_buf_bf16 = comm_buf_for_quantization.quantized_recv_params_buf_bf16
            quantized_send_params_buf_bf16 = comm_buf_for_quantization.quantized_send_params_buf_bf16
            quantized_recv_params_buf_fp32 = comm_buf_for_quantization.quantized_recv_params_buf_fp32
            quantized_send_params_buf_fp32 = comm_buf_for_quantization.quantized_send_params_buf_fp32

            quantized_work_range_per_proc = comm_buf_for_quantization.quantized_work_range_per_proc
        else:
            recv_splits = comm_splits.send_splits
            send_splits = comm_splits.recv_splits

            recv_splits_int2 = comm_splits.send_splits_int2
            send_splits_int2 = comm_splits.recv_splits_int2

            recv_buf = comm_buf.send_buf
            send_buf = comm_buf.recv_buf

            quantized_recv_data_buf = comm_buf_for_quantization.quantized_send_data_buf
            quantized_send_data_buf = comm_buf_for_quantization.quantized_recv_data_buf
            
            quantized_recv_params_buf_bf16 = comm_buf_for_quantization.quantized_send_params_buf_bf16
            quantized_send_params_buf_bf16 = comm_buf_for_quantization.quantized_recv_params_buf_bf16
            quantized_recv_params_buf_fp32 = comm_buf_for_quantization.quantized_send_params_buf_fp32
            quantized_send_params_buf_fp32 = comm_buf_for_quantization.quantized_recv_params_buf_fp32

            quantized_work_range_per_proc = comm_buf_for_quantization.dequantized_work_range_per_proc

        prepare_params_end = time.perf_counter()
        
        quantize_begin = time.perf_counter()
        Quantizer_for_all_procs.ctx.quantize_fp32_to_intX(send_buf, 
                                                          quantized_send_data_buf, 
                                                          quantized_send_params_buf_fp32, 
                                                          quantized_work_range_per_proc)

        rank = dist.get_rank()
        quantize_end = time.perf_counter()
        # print_time(rank, "outer quantize data(ms): ", (quantize_end - quantize_begin) * 1000.0)
        # TimeRecorder.print_time(rank, "outer quantize data (ms): ", (quantize_end - quantize_begin) * 1000.0)

        barrier_begin = time.perf_counter()
        dist.barrier()
        barrier_end = time.perf_counter()
        # print_time(rank, "barrier (ms): ", (barrier_end - barrier_begin) * 1000.0)
        # comm for quantized params (scale and zero_point)
        comm_param_begin = time.perf_counter()
        quantized_send_params_buf_bf16.copy_(quantized_send_params_buf_fp32)
        dist.all_to_all_single(quantized_recv_params_buf_bf16, 
                               quantized_send_params_buf_bf16, 
                               recv_splits, 
                               send_splits, 
                               async_op=False)
        quantized_recv_params_buf_fp32.copy_(quantized_recv_params_buf_bf16)
        comm_param_end = time.perf_counter()

        comm_data_begin = time.perf_counter()
        comm_handle = dist.all_to_all_single(quantized_recv_data_buf, 
                                             quantized_send_data_buf, 
                                             recv_splits_int2, 
                                             send_splits_int2, 
                                             async_op=self.is_async)
        comm_data_end = time.perf_counter()

        TimeRecorder.ctx.record_quantization_time(quantize_end - quantize_begin)
        TimeRecorder.ctx.record_barrier_time(barrier_end - barrier_begin)
        TimeRecorder.ctx.record_communication_time(comm_data_end - comm_param_begin)
        # print("rank:{}, send_buf shape: {}".format(rank, send_buf.shape), flush=True)
        # print("rank:{}, recv_buf shape (ms): {}".format(rank, recv_buf.shape), flush=True)
        print("rank:{}, prepare params (ms): {}".format(rank, (prepare_params_end - prepare_params_begin) * 1000.0))
        print("rank:{}, quantization (ms): {}".format(rank, (quantize_end - quantize_begin) * 1000.0))
        print("rank:{}, barrier (ms): {}".format(rank, (barrier_end - barrier_begin) * 1000.0), flush=True)
        print("rank:{}, comm for param (ms): {}".format(rank, (comm_param_end - comm_param_begin) * 1000.0))
        print("rank:{}, comm for data (ms): {}".format(rank, (comm_data_end - comm_data_begin) * 1000.0), flush=True)

        return comm_handle

    def get_splits_for_comm_quantized_data(
        self,
        recv_splits_tensor,
        send_splits_tensor,
        dequantized_nodes_feat_range,
        quantized_nodes_feat_range,
    ):
        begin = time.perf_counter()
        # prepare the buffer for receiving the quantized data based on the quantized params[0]
        # (range of quantized feature)
        quantized_recv_splits = list()
        quantized_send_splits = list()

        cumsum0_begin = time.perf_counter()
        recv_node_idx_begin = torch.empty((self.world_size + 1), dtype=torch.int32)
        recv_node_idx_begin[0] = 0
        recv_node_idx_begin[1:] = recv_splits_tensor.cumsum(dim=0)
        cumsum0_end = time.perf_counter()

        cumsum1_begin = time.perf_counter()
        send_node_idx_begin = torch.empty((self.world_size + 1), dtype=torch.int32)
        send_node_idx_begin[0] = 0
        send_node_idx_begin[1:] = send_splits_tensor.cumsum(dim=0)
        cumsum1_end = time.perf_counter()

        append_begin = time.perf_counter()
        for rank in range(self.world_size):
            quantized_recv_splits.append(
                # (
                #     dequantized_nodes_feat_range[recv_node_idx_begin[rank + 1]]
                #     - dequantized_nodes_feat_range[recv_node_idx_begin[rank]]
                # ).item()
                (
                    dequantized_nodes_feat_range[recv_node_idx_begin[rank + 1]]
                    - dequantized_nodes_feat_range[recv_node_idx_begin[rank]]
                )
            )
            quantized_send_splits.append(
                # (
                #     quantized_nodes_feat_range[send_node_idx_begin[rank + 1]]
                #     - quantized_nodes_feat_range[send_node_idx_begin[rank]]
                # ).item()
                (
                    dequantized_nodes_feat_range[recv_node_idx_begin[rank + 1]]
                    - dequantized_nodes_feat_range[recv_node_idx_begin[rank]]
                )
            )
        append_end = time.perf_counter()
        end = time.perf_counter()
        TimeRecorder.print_time(dist.get_rank(), "inner cumsum0 (ms): ", (cumsum0_end - cumsum0_begin) * 1000.0)
        TimeRecorder.print_time(dist.get_rank(), "inner cumsum1 (ms): ", (cumsum1_end - cumsum1_begin) * 1000.0)
        TimeRecorder.print_time(dist.get_rank(), "inner append (ms): ", (append_end - append_begin) * 1000.0)
        TimeRecorder.print_time(dist.get_rank(), "inner inner get splits for quant (ms): ", (end - begin) * 1000.0)

        return quantized_recv_splits, quantized_send_splits

    def comm_with_mixed_quantization(self, recv_buf, send_buf, recv_splits, send_splits, nodes_num_bits_tensor):
        prepare_params_begin = time.perf_counter()
        recv_splits_tensor = torch.tensor(recv_splits, dtype=torch.int32)
        send_splits_tensor = torch.tensor(send_splits, dtype=torch.int32)
        num_send_nodes = send_splits_tensor.sum().item()
        num_recv_nodes = recv_splits_tensor.sum().item()

        # create quantized params buffer for communication
        # [nodes_num_bits, zero_points, scales]
        send_params = torch.empty((num_send_nodes, 2 + 1), dtype=torch.float32)

        # to get the random bits for each node
        send_params[:, 0] = nodes_num_bits_tensor

        # get the range of each node's quantized feature
        quantized_nodes_feat_range = QuantizerForMixedBits.get_quantized_nodes_feat_range(
            int(num_send_nodes), send_buf.size(1), nodes_num_bits_tensor
        )

        quantized_send_buf = torch.empty(quantized_nodes_feat_range[-1].item(), dtype=torch.uint8)
        prepare_params_end = time.perf_counter()

        quantization_begin = time.perf_counter()
        # quantize the data
        QuantizerForMixedBits.quantize_fp32_to_intX(
            send_buf, quantized_send_buf, quantized_nodes_feat_range, send_params
        )

        quantization_end = time.perf_counter()

        barrier_begin = time.perf_counter()
        # dist.barrier()
        barrier_end = time.perf_counter()

        # comm_for_param_begin = time.perf_counter()
        # prepare the buffer for receiving the quantized params
        # recv_params = torch.empty((num_recv_nodes, 2 + 1), dtype=torch.float32)
        recv_params = torch.empty((num_recv_nodes, 2 + 1), dtype=torch.bfloat16)
        send_params = send_params.to(torch.bfloat16)

        comm_for_param_begin = time.perf_counter()
        # communication for quantized params
        dist.all_to_all_single(recv_params, send_params, recv_splits, send_splits, async_op=False)
        dequantized_params = recv_params.to(torch.float32)
        comm_for_param_end = time.perf_counter()

        # get the range of each node's dequantized feature
        dequantized_nodes_feat_range = QuantizerForMixedBits.get_quantized_nodes_feat_range(
            int(num_recv_nodes), send_buf.size(1), dequantized_params[:, 0]
        )

        get_splits_for_quant_begin = time.perf_counter()
        # get the splits for communication of quantized data
        quantized_recv_splits, quantized_send_splits = self.get_splits_for_comm_quantized_data(
            recv_splits_tensor,
            send_splits_tensor,
            dequantized_nodes_feat_range,
            quantized_nodes_feat_range,
        )

        quantized_recv_buf = torch.empty(dequantized_nodes_feat_range[-1].item(), dtype=torch.uint8)
        get_splits_for_quant_end = time.perf_counter()
        TimeRecorder.print_time(dist.get_rank(), "inner get splits for quant (ms): ", (get_splits_for_quant_end - get_splits_for_quant_begin) * 1000.0)

        comm_for_data_begin = time.perf_counter()
        # communication for quantized data
        comm_handle = dist.all_to_all_single(
            quantized_recv_buf,
            quantized_send_buf,
            quantized_recv_splits,
            quantized_send_splits,
            async_op=self.is_async,
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

        TimeRecorder.print_time(
            dist.get_rank(), "inner quantization (ms): ", (quantization_end - quantization_begin) * 1000.0
        )
        TimeRecorder.print_time(
            dist.get_rank(), "inner comm data(ms): ", (comm_for_data_end - comm_for_data_begin) * 1000.0
        )
        TimeRecorder.print_time(
            dist.get_rank(), "inner comm param(ms): ", (comm_for_param_end - comm_for_param_begin) * 1000.0
        )

        return (comm_handle, quantized_recv_buf, dequantized_nodes_feat_range, dequantized_params)

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
                QuantizerForPureBits.dequantize_intX_to_fp32(
                    recv_quant_data_buf_list[rank], recv_buf[begin_idx:end_idx], scale, zero_point, num_bits
                )
        dequantize_end = time.perf_counter()
        # TimeRecorder.print_time(
        #     dist.get_rank(), "inner dequantize data (ms): ", (dequantize_end - dequantize_begin) * 1000.0
        # )
        TimeRecorder.ctx.record_dequantization_time(dequantize_end - dequantize_begin)
