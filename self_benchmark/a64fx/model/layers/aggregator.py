import torch
import time
import sys
import torch.distributed as dist
from torch_geometric.nn.spmm_kernel import SPMM_forward, SPMM_backward

sys.path.append("../../")

from communicator import Communicator
from assigner import Assigner
from time_recorder import TimeRecorder
from quantizer import Quantizer, Quantizer_v1, Quantizer_for_all_procs
from data_manager import DistributedGraph, DistributedGraphForPre
from typing import Union

class Aggregator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                graph: Union[DistributedGraph, DistributedGraphForPre], 
                local_nodes_feat, 
                layer, 
                num_bits: int, 
                is_pre_delay: bool, 
                is_training: bool):
        ctx.graph = graph
        ctx.num_bits = num_bits
        ctx.is_pre_delay = is_pre_delay
        ctx.layer = layer

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        prepare_comm_begin = time.perf_counter()

        graph.comm_buf.resize_buffer(graph.comm_splits, local_nodes_feat.size(-1), num_bits)
        if num_bits != 32 and num_bits != 16 and graph.comm_buf_for_quantization is not None:
            graph.comm_buf_for_quantization.resize_buffer(graph.comm_splits, local_nodes_feat.size(-1), num_bits)
        
        comm_begin = time.perf_counter()
        TimeRecorder.print_time(rank, "prepare comm data (ms): ", (comm_begin - prepare_comm_begin) * 1000.0)
        comm_handle = None

        send_buf = graph.comm_buf.send_buf
        # zero the send buffer
        send_buf.zero_()
            

        if world_size > 1:
            if is_pre_delay:  # pre aggregation
                SPMM_forward(graph.adj_t_pre_post_aggr_to, local_nodes_feat, send_buf)
            else:  # no pre aggregation
                torch.index_select(local_nodes_feat, 0, graph.idx_nodes_send_to_others, out=send_buf)
            
            layer = f"forward{layer}"
            comm_handle= Communicator.ctx.comm(
                graph.comm_splits,
                graph.comm_buf,
                graph.comm_buf_for_quantization,
                layer,
                is_training,
                "forward",
            )

        comm_end = time.perf_counter()
        TimeRecorder.print_time(rank, "outer total comm (ms): ", (comm_end - comm_begin) * 1000.0)
        # print("outer total comm (ms): {}".format((comm_end - comm_begin) * 1000.0))
        local_aggregate_begin = time.perf_counter()
        out = torch.zeros([graph.local_adj_t.sparse_size(0), local_nodes_feat.size(-1)], dtype=torch.float32)
        # aggregate message from local nodes
        # out = SPMM_forward(graph.local_adj_t, local_nodes_feat, out)
        SPMM_forward(graph.local_adj_t, local_nodes_feat, out)

        async_wait_begin = time.perf_counter()
        print("rank:{}, spmm time (ms): {}".format(rank, (async_wait_begin - local_aggregate_begin) * 1000.0))
        if comm_handle is not None:
            comm_handle.wait()

        convert_data_begin = time.perf_counter()
        # print("wait (ms): {}".format((convert_data_begin - async_wait_begin) * 1000.0))
        num_recv_nodes = sum(graph.comm_splits.recv_splits)
        if world_size > 1 and num_recv_nodes != 0 and num_bits != 32 and num_bits != 16 and is_training:
            if num_bits == 2 or num_bits == 4 or num_bits == 8:
                Quantizer_for_all_procs.ctx.dequantize_intX_to_fp32(graph.comm_buf_for_quantization.quantized_recv_data_buf, 
                                                                    graph.comm_buf.recv_buf, 
                                                                    graph.comm_buf_for_quantization.quantized_recv_params_buf_fp32,
                                                                    graph.comm_buf_for_quantization.dequantized_work_range_per_proc)
            # else:
            #     Quantizer_v1.dequantize_intX_to_fp32(
            #         quantized_recv_buf, recv_buf, dequantized_nodes_feat_range, dequantized_params
            #     )

        convert_data_end = time.perf_counter()
        # print_time(rank, "inner convert data (ms): ", (convert_data_end - convert_data_begin) * 1000.0)

        remote_aggregate_begin = time.perf_counter()
        remote_nodes_feat = graph.comm_buf.recv_buf
        # aggregate message from remote nodes
        if world_size > 1 and remote_nodes_feat.size(0) != 0:
            if is_pre_delay:  # post aggregation
                SPMM_forward(graph.adj_t_pre_post_aggr_from, remote_nodes_feat, out)
            else:
                SPMM_forward(graph.remote_adj_t, remote_nodes_feat, out)

        sum_message_end = time.perf_counter()

        print("rank:{}, aggregate local nodes (ms): {}".format(rank, (remote_aggregate_begin - local_aggregate_begin) * 1000.0))
        TimeRecorder.print_time(
            rank, "inner propagate forward (ms): ", (sum_message_end - prepare_comm_begin) * 1000.0
        )
        TimeRecorder.ctx.record_dequantization_time(convert_data_end - convert_data_begin)
        TimeRecorder.ctx.record_total_convolution_time(sum_message_end - prepare_comm_begin)
        TimeRecorder.ctx.next_layer()
        # print("inner propagate forward (ms): {}".format((sum_message_end - prepare_comm_begin) * 1000.0))

        # return local_out
        return out

    @staticmethod
    def backward(ctx, local_out_grad):
        graph = ctx.graph
        num_bits = ctx.num_bits
        is_pre_delay = ctx.is_pre_delay
        layer = f"backward{ctx.layer}"

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # scatter gradient to remote nodes
        backward_begin = time.perf_counter()

        # need to use the reverse buf for backward
        graph.comm_buf.resize_buffer(graph.comm_splits, local_out_grad.size(-1), num_bits)
        if num_bits != 32 and num_bits != 16 and graph.comm_buf_for_quantization is not None:
            graph.comm_buf_for_quantization.resize_buffer(graph.comm_splits, local_out_grad.size(-1), num_bits)


        # need to use the reverse buf for backward
        remote_nodes_grad_buf = graph.comm_buf.recv_buf

        remote_nodes_grad_buf.zero_()
        if remote_nodes_grad_buf.size(0) != 0:
            if is_pre_delay:  # pre aggregation
                SPMM_backward(graph.adj_t_pre_post_aggr_from, local_out_grad, remote_nodes_grad_buf)
            else:  # no pre aggregation
                SPMM_backward(graph.remote_adj_t, local_out_grad, remote_nodes_grad_buf)

        comm_handle = None
        # communicate to obtain the local node grads from other subgraph
        if world_size > 1:
            comm_handle = Communicator.ctx.comm(
                graph.comm_splits,
                graph.comm_buf,
                graph.comm_buf_for_quantization,
                layer,
                True,
                "backward",
            )

        # scatter gradient to local nodes
        local_nodes_grad = torch.zeros(
            [graph.local_adj_t.sparse_size(-1), local_out_grad.size(-1)], dtype=torch.float32
        )
        SPMM_backward(graph.local_adj_t, local_out_grad, local_nodes_grad)

        if comm_handle is not None:
            comm_handle.wait()

        dequantization_begin = time.perf_counter()
        num_recv_nodes = sum(graph.comm_splits.send_splits)
        # convert communication data to fp32
        if world_size > 1 and num_recv_nodes != 0 and num_bits != 32 and num_bits != 16:
            if num_bits == 2 or num_bits == 4 or num_bits == 8:
                Quantizer_for_all_procs.ctx.dequantize_intX_to_fp32(
                    graph.comm_buf_for_quantization.quantized_send_data_buf, 
                    graph.comm_buf.send_buf,
                    graph.comm_buf_for_quantization.quantized_send_params_buf_fp32,
                    graph.comm_buf_for_quantization.quantized_work_range_per_proc
                )
        dequantization_end = time.perf_counter()

        local_nodes_grad_buf = graph.comm_buf.send_buf
        # then accumulate the local node grads
        if local_nodes_grad_buf.size(0) != 0:
            if is_pre_delay:  # post aggregation
                SPMM_backward(graph.adj_t_pre_post_aggr_to, local_nodes_grad_buf, local_nodes_grad)
            else:
                local_nodes_grad.index_add_(
                    dim=0, index=graph.idx_nodes_send_to_others, source=local_nodes_grad_buf
                )

        backward_end = time.perf_counter()
        TimeRecorder.ctx.record_dequantization_time(dequantization_end - dequantization_begin)
        TimeRecorder.ctx.record_total_convolution_time(backward_end - backward_begin)
        TimeRecorder.ctx.next_layer()

        return None, local_nodes_grad, None, None, None, None
