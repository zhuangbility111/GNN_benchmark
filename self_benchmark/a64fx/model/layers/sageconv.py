import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import fill_diag
from torch_sparse import sum as sparsesum

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.spmm_kernel import SPMM_forward, SPMM_backward

import quantization_cpu
import math

import time
import sys

sys.path.append("../../")

from time_recorder import TimeRecorder
from communicator import Communicator
from assigner import Assigner


def get_deg(local_adj_t, remote_adj_t, add_self_loops=False):
    if not local_adj_t.has_value():
        local_adj_t = local_adj_t.fill_value(1.0)

    if not remote_adj_t.has_value():
        remote_adj_t = remote_adj_t.fill_value(1.0)

    if add_self_loops:
        local_adj_t = fill_diag(local_adj_t, 1.0)

    local_deg = sparsesum(local_adj_t, dim=1)
    if remote_adj_t.size(0) != 0:
        local_deg += sparsesum(remote_adj_t, dim=1)

    return local_deg.unsqueeze(-1)


class Aggregate_for_local_and_remote(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph, local_nodes_feat, num_bits):
        ctx.graph = graph
        ctx.num_bits = num_bits

        prepare_comm_begin = time.perf_counter()

        num_recv_nodes = sum(graph.num_nodes_recv_from_others)
        num_send_nodes = sum(graph.num_nodes_send_to_others)

        graph.comm_buf.resize_buffer(
            (num_send_nodes, local_nodes_feat.size(-1)), (num_recv_nodes, local_nodes_feat.size(-1))
        )

        send_nodes_feat_buf = graph.comm_buf.send_buf
        recv_nodes_feat_buf = graph.comm_buf.recv_buf

        send_nodes_feat_fp16_buf = graph.comm_buf.send_buf_fp16
        recv_nodes_feat_fp16_buf = graph.comm_buf.recv_buf_fp16

        world_size = dist.get_world_size()

        comm_begin = time.perf_counter()
        if world_size > 1:
            # collect the local node feats to send to other subgraph
            torch.index_select(local_nodes_feat, 0, graph.idx_nodes_send_to_others, out=send_nodes_feat_buf)

            if num_bits == 32:
                handle = Communicator.comm_with_fp32(
                    recv_nodes_feat_buf,
                    send_nodes_feat_buf,
                    graph.num_nodes_recv_from_others,
                    graph.num_nodes_send_to_others,
                )
            elif num_bits == 16:
                handle = Communicator.comm_with_fp16(
                    recv_nodes_feat_buf,
                    send_nodes_feat_buf,
                    recv_nodes_feat_fp16_buf,
                    send_nodes_feat_fp16_buf,
                    graph.num_nodes_recv_from_others,
                    graph.num_nodes_send_to_others,
                )
            else:
                nodes_num_bits_tensor = torch.empty((num_send_nodes), dtype=torch.float32)
                Assigner.assign_node_dataformat_randomly(nodes_num_bits_tensor, num_bits)
                (
                    quantized_buf,
                    dequantized_nodes_feat_range,
                    dequantized_params,
                ) = Communicator.comm_with_quantization_v1(
                    recv_nodes_feat_buf,
                    send_nodes_feat_buf,
                    graph.num_nodes_recv_from_others,
                    graph.num_nodes_send_to_others,
                    nodes_num_bits_tensor,
                    world_size,
                )
                handle = None
        else:
            handle = None

        comm_end = time.perf_counter()
        rank = dist.get_rank()
        TimeRecorder.print_time(rank, "outer total comm (ms): ", (comm_end - comm_begin) * 1000.0)
        # print("outer total comm (ms): {}".format((comm_end - comm_begin) * 1000.0))
        out = torch.zeros([graph.local_adj_t.sparse_size(0), local_nodes_feat.size(-1)], dtype=torch.float)
        local_aggregate_begin = time.perf_counter()
        # aggregate message from local nodes
        SPMM_forward(graph.local_adj_t, local_nodes_feat, out)

        async_wait_begin = time.perf_counter()
        if handle is not None:
            handle.wait()

        convert_data_begin = time.perf_counter()
        # print("wait (ms): {}".format((convert_data_begin - async_wait_begin) * 1000.0))
        if num_bits != 32 and num_bits != 16 and num_recv_nodes != 0:
            Communicator.convert_data_to_fp32_v1(
                quantized_buf, recv_nodes_feat_buf, dequantized_nodes_feat_range, dequantized_params
            )

        convert_data_end = time.perf_counter()
        # print_time(rank, "inner convert data (ms): ", (convert_data_end - convert_data_begin) * 1000.0)
        # print("inner convert data (ms): {}".format((convert_data_end - convert_data_begin) * 1000.0))

        remote_aggregate_begin = time.perf_counter()
        remote_nodes_feat = recv_nodes_feat_buf
        # aggregate message from remote nodes
        if remote_nodes_feat.size(0) != 0:
            SPMM_forward(graph.remote_adj_t, remote_nodes_feat, out)

        sum_message_end = time.perf_counter()

        TimeRecorder.print_time(
            rank, "inner propagate forward (ms): ", (sum_message_end - prepare_comm_begin) * 1000.0
        )
        TimeRecorder.ctx.record_total_convolution_time(sum_message_end - prepare_comm_begin)
        TimeRecorder.ctx.next_layer()
        # print("inner propagate forward (ms): {}".format((sum_message_end - prepare_comm_begin) * 1000.0))

        # return local_out
        return out

    @staticmethod
    def backward(ctx, local_out_grad):
        graph = ctx.graph
        num_bits = ctx.num_bits

        if ctx.needs_input_grad[1]:
            # scatter gradient to remote nodes
            backward_begin = time.perf_counter()
            num_send_nodes = sum(graph.num_nodes_recv_from_others)
            num_recv_nodes = sum(graph.num_nodes_send_to_others)

            graph.comm_buf.resize_buffer(
                (num_recv_nodes, local_out_grad.size(-1)), (num_send_nodes, local_out_grad.size(-1))
            )

            remote_nodes_grad_buf = graph.comm_buf.recv_buf
            local_nodes_grad_buf = graph.comm_buf.send_buf

            remote_nodes_grad_fp16_buf = graph.comm_buf.recv_buf_fp16
            local_nodes_grad_fp16_buf = graph.comm_buf.send_buf_fp16

            remote_nodes_grad_buf.zero_()
            if remote_nodes_grad_buf.size(0) != 0:
                SPMM_backward(graph.remote_adj_t, local_out_grad, remote_nodes_grad_buf)

            world_size = dist.get_world_size()

            # communicate to obtain the local node grads from other subgraph
            if world_size > 1:
                if num_bits == 32:
                    handle = Communicator.comm_with_fp32(
                        local_nodes_grad_buf,
                        remote_nodes_grad_buf,
                        graph.num_nodes_send_to_others,
                        graph.num_nodes_recv_from_others,
                    )
                elif num_bits == 16:
                    handle = Communicator.comm_with_fp16(
                        local_nodes_grad_buf,
                        remote_nodes_grad_buf,
                        local_nodes_grad_fp16_buf,
                        remote_nodes_grad_fp16_buf,
                        graph.num_nodes_send_to_others,
                        graph.num_nodes_recv_from_others,
                    )
                else:
                    nodes_num_bits_tensor = torch.empty((num_send_nodes), dtype=torch.float32)
                    Assigner.assign_node_dataformat_randomly(nodes_num_bits_tensor, num_bits)
                    (
                        quantized_buf,
                        dequantized_nodes_feat_range,
                        dequantized_params,
                    ) = Communicator.comm_with_quantization_v1(
                        local_nodes_grad_buf,
                        remote_nodes_grad_buf,
                        graph.num_nodes_send_to_others,
                        graph.num_nodes_recv_from_others,
                        nodes_num_bits_tensor,
                        world_size,
                    )
                    handle = None
            else:
                handle = None

            # scatter gradient to local nodes
            local_nodes_grad = torch.zeros(
                [graph.local_adj_t.sparse_size(-1), local_out_grad.size(-1)], dtype=torch.float
            )
            SPMM_backward(graph.local_adj_t, local_out_grad, local_nodes_grad)

            if handle is not None:
                handle.wait()

            # convert communication data to fp32
            if num_bits != 32 and num_bits != 16 and num_recv_nodes != 0:
                Communicator.convert_data_to_fp32_v1(
                    quantized_buf, local_nodes_grad_buf, dequantized_nodes_feat_range, dequantized_params
                )

            index_add_begin = time.perf_counter()
            # then accumulate the local node grads
            local_nodes_grad_from = local_nodes_grad_buf
            if local_nodes_grad_from.size(0) != 0:
                local_nodes_grad.index_add_(
                    dim=0, index=graph.idx_nodes_send_to_others, source=local_nodes_grad_from
                )

            index_add_end = time.perf_counter()
            backward_end = time.perf_counter()
            TimeRecorder.ctx.record_total_convolution_time(backward_end - backward_begin)
            TimeRecorder.ctx.next_layer()

        return None, local_nodes_grad, None


def aggregate_for_local_and_remote(graph, local_nodes_feat, num_bits):
    return Aggregate_for_local_and_remote.apply(graph, local_nodes_feat, num_bits)


class DistSAGEConvGrad(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bits: int = 32,
        add_self_loops: bool = False,
        normalize: bool = True,
        bias: bool = True,
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.local_deg = None
        self.num_bits = num_bits

        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer="glorot")

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def propagate(self, graph, local_nodes_feat, num_bits):
        local_out = aggregate_for_local_and_remote(graph, local_nodes_feat, num_bits)
        return local_out

    def forward(self, graph, local_nodes_feat) -> Tensor:
        """"""
        norm_begin = time.perf_counter()

        # communication first
        # linear_first = self.in_channels > self.out_channels

        linear_begin = time.perf_counter()
        # if linear_first:
        local_nodes_feat = self.lin(local_nodes_feat)

        propagate_begin = time.perf_counter()
        out = self.propagate(graph, local_nodes_feat, self.num_bits)

        add_bias_begin = time.perf_counter()
        TimeRecorder.print_time(
            dist.get_rank(), "outer propagate forward (ms): ", (add_bias_begin - propagate_begin) * 1000.0
        )
        # print("outer propagate forward (ms): {}".format((add_bias_begin - propagate_begin) * 1000.0))
        out += local_nodes_feat

        if self.normalize:
            local_deg = self.local_deg
            if local_deg is None:
                local_deg = get_deg(graph.local_adj_t, graph.remote_adj_t, self.add_self_loops)
                self.local_deg = local_deg
            out /= local_deg + 1

        # if not linear_first:
        #     out = self.lin(out)

        if self.bias is not None:
            out += self.bias
        add_bias_end = time.perf_counter()

        return out
