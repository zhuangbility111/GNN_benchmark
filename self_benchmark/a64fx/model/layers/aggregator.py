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

import quantization_cpu
import math


class Aggregator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph, local_nodes_feat, layer, num_bits, is_pre_delay, is_training):
        ctx.graph = graph
        ctx.num_bits = num_bits
        ctx.is_pre_delay = is_pre_delay
        ctx.layer = layer

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        prepare_comm_begin = time.perf_counter()
        if is_pre_delay:  # pre-delay aggregation
            send_splits = graph.pre_post_aggr_to_splits
            recv_splits = graph.pre_post_aggr_from_splits
        else:  # no pre-delay aggregation
            send_splits = graph.num_nodes_send_to_others
            recv_splits = graph.num_nodes_recv_from_others

        num_send_nodes = sum(send_splits)
        num_recv_nodes = sum(recv_splits)

        graph.comm_buf.resize_buffer(
            (num_send_nodes, local_nodes_feat.size(-1)), (num_recv_nodes, local_nodes_feat.size(-1))
        )

        send_buf = graph.comm_buf.send_buf
        recv_buf = graph.comm_buf.recv_buf
        send_buf_fp16 = graph.comm_buf.send_buf_fp16
        recv_buf_fp16 = graph.comm_buf.recv_buf_fp16

        comm_begin = time.perf_counter()
        TimeRecorder.print_time(rank, "prepare comm data (ms): ", (comm_begin - prepare_comm_begin) * 1000.0)
        comm_handle = None

        # zero the send buffer
        send_buf.zero_()

        if world_size > 1:
            if is_pre_delay:  # pre aggregation
                SPMM_forward(graph.adj_t_pre_post_aggr_to, local_nodes_feat, send_buf)
            else:  # no pre aggregation
                torch.index_select(local_nodes_feat, 0, graph.idx_nodes_send_to_others, out=send_buf)

            layer = f"forward{layer}"
            (
                comm_handle,
                quantized_recv_buf,
                dequantized_nodes_feat_range,
                dequantized_params,
            ) = Communicator.ctx.comm(
                recv_buf,
                send_buf,
                recv_buf_fp16,
                send_buf_fp16,
                recv_splits,
                send_splits,
                layer,
                is_training,
                "forward"
            )

        comm_end = time.perf_counter()
        TimeRecorder.print_time(rank, "outer total comm (ms): ", (comm_end - comm_begin) * 1000.0)
        # print("outer total comm (ms): {}".format((comm_end - comm_begin) * 1000.0))
        out = torch.zeros([graph.local_adj_t.sparse_size(0), local_nodes_feat.size(-1)], dtype=torch.float32)
        local_aggregate_begin = time.perf_counter()
        # aggregate message from local nodes
        SPMM_forward(graph.local_adj_t, local_nodes_feat, out)

        async_wait_begin = time.perf_counter()
        if comm_handle is not None:
            comm_handle.wait()

        convert_data_begin = time.perf_counter()
        # print("wait (ms): {}".format((convert_data_begin - async_wait_begin) * 1000.0))
        if world_size > 1 and num_recv_nodes != 0 and num_bits != 32 and num_bits != 16 and is_training:
            if num_bits == 2 or num_bits == 4 or num_bits == 8:
                # begin_idx = 0
                # end_idx = 0
                # quantized_begin_idx = 0
                # quantized_end_idx = 0
                # for i in range(world_size):
                #     begin_idx = end_idx
                #     end_idx += recv_splits[i]

                #     quantized_begin_idx = quantized_end_idx
                #     quantized_end_idx += dequantized_nodes_feat_range[i]

                #     Quantizer.dequantize_intX_to_fp32(
                #         quantized_recv_buf[quantized_begin_idx: quantized_end_idx], 
                #         recv_buf[begin_idx: end_idx], 
                #         dequantized_params[begin_idx: end_idx],
                #         num_bits
                #     )
                dequantized_work_range_per_proc = torch.empty((len(recv_splits) + 1), dtype=torch.int32)
                dequantized_work_range_per_proc[0] = 0
                dequantized_work_range_per_proc[1:] = torch.tensor(recv_splits, dtype=torch.int32).cumsum(0)
                Quantizer_for_all_procs.ctx.dequantize_intX_to_fp32(quantized_recv_buf, recv_buf, dequantized_params, dequantized_work_range_per_proc)
            else:
                Quantizer_v1.dequantize_intX_to_fp32(
                    quantized_recv_buf, recv_buf, dequantized_nodes_feat_range, dequantized_params
                )

        convert_data_end = time.perf_counter()
        # print_time(rank, "inner convert data (ms): ", (convert_data_end - convert_data_begin) * 1000.0)

        remote_aggregate_begin = time.perf_counter()
        remote_nodes_feat = recv_buf
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

        if is_pre_delay:  # pre-delay aggregation
            send_splits = graph.pre_post_aggr_from_splits
            recv_splits = graph.pre_post_aggr_to_splits
        else:  # no pre-delay aggregation
            send_splits = graph.num_nodes_recv_from_others
            recv_splits = graph.num_nodes_send_to_others

        num_send_nodes = sum(send_splits)
        num_recv_nodes = sum(recv_splits)

        # need to use the reverse buf for backward
        graph.comm_buf.resize_buffer(
            (num_recv_nodes, local_out_grad.size(-1)), (num_send_nodes, local_out_grad.size(-1))
        )

        # need to use the reverse buf for backward
        send_buf = graph.comm_buf.recv_buf
        recv_buf = graph.comm_buf.send_buf
        send_buf_fp16 = graph.comm_buf.recv_buf_fp16
        recv_buf_fp16 = graph.comm_buf.send_buf_fp16

        remote_nodes_grad_buf = send_buf
        local_nodes_grad_buf = recv_buf

        remote_nodes_grad_buf.zero_()
        if remote_nodes_grad_buf.size(0) != 0:
            if is_pre_delay:  # pre aggregation
                SPMM_backward(graph.adj_t_pre_post_aggr_from, local_out_grad, remote_nodes_grad_buf)
            else:  # no pre aggregation
                SPMM_backward(graph.remote_adj_t, local_out_grad, remote_nodes_grad_buf)

        comm_handle = None
        # communicate to obtain the local node grads from other subgraph
        if world_size > 1:
            (
                comm_handle,
                quantized_recv_buf,
                dequantized_nodes_grad_range,
                dequantized_params,
            ) = Communicator.ctx.comm(
                recv_buf,
                send_buf,
                recv_buf_fp16,
                send_buf_fp16,
                recv_splits,
                send_splits,
                layer,
                True,
                "backward"
            )

        # scatter gradient to local nodes
        local_nodes_grad = torch.zeros(
            [graph.local_adj_t.sparse_size(-1), local_out_grad.size(-1)], dtype=torch.float32
        )
        SPMM_backward(graph.local_adj_t, local_out_grad, local_nodes_grad)

        if comm_handle is not None:
            comm_handle.wait()

        dequantization_begin = time.perf_counter()
        # convert communication data to fp32
        if world_size > 1 and num_recv_nodes != 0 and num_bits != 32 and num_bits != 16:
            if num_bits == 2 or num_bits == 4 or num_bits == 8:
                # begin_idx = 0
                # end_idx = 0
                # quantized_begin_idx = 0
                # quantized_end_idx = 0
                # for i in range(world_size):
                #     begin_idx = end_idx
                #     end_idx += recv_splits[i]

                #     quantized_begin_idx = quantized_end_idx
                #     quantized_end_idx += dequantized_nodes_grad_range[i]

                #     Quantizer.dequantize_intX_to_fp32(
                #         quantized_recv_buf[quantized_begin_idx: quantized_end_idx], 
                #         recv_buf[begin_idx: end_idx], 
                #         dequantized_params[begin_idx: end_idx],
                #         num_bits
                #     )

                dequantized_work_range_per_proc = torch.empty((len(recv_splits) + 1), dtype=torch.int32)
                dequantized_work_range_per_proc[0] = 0
                dequantized_work_range_per_proc[1:] = torch.tensor(recv_splits, dtype=torch.int32).cumsum(0)
                Quantizer_for_all_procs.ctx.dequantize_intX_to_fp32(quantized_recv_buf, recv_buf, dequantized_params, dequantized_work_range_per_proc)
            else:
                Quantizer_v1.dequantize_intX_to_fp32(
                    quantized_recv_buf,
                    recv_buf,
                    dequantized_nodes_grad_range,
                    dequantized_params,
                )
        dequantization_end = time.perf_counter()

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
