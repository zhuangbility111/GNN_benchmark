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

class Quantizer(object):
    def __init__(self):
        pass

    @staticmethod
    def quantize_fp32_to_intX(data_fp32, data_int8, num_bits=8):
        quantize_begin = time.perf_counter()
        scale = (data_fp32.max(dim=1)[0] - data_fp32.min(dim=1)[0] + 10e-20) / (2**num_bits - 1)
        # zero_indices = torch.nonzero(scale == 0).squeeze()
        # print("zero_indices = {}".format(zero_indices))
        # zero_point = data_fp32.min(dim=1)[0] / scale * (-1)
        zero_point = data_fp32.min(dim=1)[0]
        # ref = torch.quantize_per_channel(data_fp32, scale, zero_point, 0, torch.quint8).int_repr()
        # data_int8.copy_(torch.quantize_per_channel(data_fp32, scale, zero_point, 0, torch.quint8).int_repr())
        quantization_cpu.quantize_tensor(data_fp32, data_int8, zero_point, scale, num_bits)
        print(data_int8)
        quant_param = torch.stack((scale, zero_point), dim=1)
        quantize_end = time.perf_counter()
        print("inner quantize data(ms): {}".format((quantize_end - quantize_begin) * 1000.0))
        return quant_param

    @staticmethod
    def dequantize_intX_to_fp32(data_int8, data_fp32, scale, zero_point, num_bits=8):
        # data_fp32.copy_((data_int8 - zero_point.view(-1, 1)) * scale.view(-1, 1))
        quantization_cpu.dequantize_tensor(data_int8, data_fp32, zero_point, scale, num_bits)
        print(data_fp32)
        # data_fp32.copy_(torch.dequantize(data_int8))

def get_deg(local_adj_t, remote_adj_t, add_self_loops=False):
    if not local_adj_t.has_value():
        local_adj_t = local_adj_t.fill_value(1.)

    if not remote_adj_t.has_value():
        remote_adj_t = remote_adj_t.fill_value(1.)

    if add_self_loops:
        local_adj_t = fill_diag(local_adj_t, 1.)
    
    local_deg = sparsesum(local_adj_t, dim=1)
    if remote_adj_t.size(0) != 0:
        local_deg += sparsesum(remote_adj_t, dim=1)

    return local_deg.unsqueeze(-1)
    
def comm_for_remote_nodes(recv_nodes_buf, send_nodes_buf,
                          recv_nodes_quant_buf, send_nodes_quant_buf, 
                          recv_nodes_splits, send_nodes_splits,
                          num_bits):

    prepare_send_node_begin = time.perf_counter()

    prepare_recv_node_begin = time.perf_counter()

    barrier_begin = time.perf_counter()
    # dist.barrier()

    comm_begin = time.perf_counter()
    # handle = dist.all_to_all_single(recv_node_feats, send_node_feats, recv_node_feats_splits, send_node_feats_splits, async_op=True)
    if recv_nodes_quant_buf is not None and send_nodes_quant_buf is not None:
        # convert communication data to fp16
        transform_begin = time.perf_counter()
        # send_nodes_feat_fp16_buf.copy_(send_nodes_feat_buf)
        quantize_begin = time.perf_counter()
        send_quant_param = Quantizer.quantize_fp32_to_intX(send_nodes_buf, send_nodes_quant_buf, num_bits)
        quantize_end = time.perf_counter()
        print("outer quantize data(ms): {}".format((quantize_end - quantize_begin) * 1000.0))
        recv_quant_param = torch.empty((sum(recv_nodes_splits), 2), dtype=torch.float32)

        transform_end = time.perf_counter()
        dist.all_to_all_single(recv_quant_param, send_quant_param, recv_nodes_splits, send_nodes_splits, async_op=False)
        handle = dist.all_to_all_single(recv_nodes_quant_buf, send_nodes_quant_buf, recv_nodes_splits, send_nodes_splits, async_op=True)
        # dist.all_to_all_single(recv_nodes_feat_fp16_buf, send_nodes_feat_fp16_buf, recv_nodes_feat_splits, send_nodes_feat_splits, async_op=False)
    else:
        handle = dist.all_to_all_single(recv_nodes_buf, send_nodes_buf, recv_nodes_splits, send_nodes_splits, async_op=True)
        # dist.all_to_all_single(recv_nodes_feat_buf, send_nodes_feat_buf, recv_nodes_feat_splits, send_nodes_feat_splits, async_op=False)

    comm_end = time.perf_counter()

    print("inner comm data (ms): {}".format((comm_end - prepare_send_node_begin) * 1000.0))
    return handle, recv_quant_param[:, 0], recv_quant_param[:, 1]

class Aggregate_for_local_and_remote(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph, local_nodes_feat, num_bits):
        ctx.graph = graph
        ctx.num_bits = num_bits
    
        prepare_comm_begin = time.perf_counter()
        
        num_recv_nodes = sum(graph.num_nodes_recv_from_others)
        num_send_nodes = sum(graph.num_nodes_send_to_others)

        send_nodes_feat_buf = graph.comm_buf.send_buf
        recv_nodes_feat_buf = graph.comm_buf.recv_buf

        send_nodes_feat_fp16_buf = graph.comm_buf.send_buf_fp16
        recv_nodes_feat_fp16_buf = graph.comm_buf.recv_buf_fp16

        send_nodes_feat_buf.resize_(num_send_nodes, local_nodes_feat.size(-1))
        recv_nodes_feat_buf.resize_(num_recv_nodes, local_nodes_feat.size(-1))

        if send_nodes_feat_fp16_buf is not None and recv_nodes_feat_fp16_buf is not None and \
           num_send_nodes != 0 and num_recv_nodes != 0:
            num_send_nodes = math.ceil(num_send_nodes / float(8 / num_bits))
            num_recv_nodes = math.ceil(num_recv_nodes / float(8 / num_bits))
            send_nodes_feat_fp16_buf.resize_(num_send_nodes, local_nodes_feat.size(-1))
            recv_nodes_feat_fp16_buf.resize_(num_recv_nodes, local_nodes_feat.size(-1))

        comm_begin = time.perf_counter()
        if num_send_nodes != 0 and num_recv_nodes != 0:
            torch.index_select(local_nodes_feat, 0, graph.idx_nodes_send_to_others, out=send_nodes_feat_buf)
            handle, recv_scale, recv_zero_point = \
                    comm_for_remote_nodes(recv_nodes_feat_buf, send_nodes_feat_buf,
                                          recv_nodes_feat_fp16_buf, send_nodes_feat_fp16_buf,
                                          graph.num_nodes_recv_from_others, graph.num_nodes_send_to_others,
                                          num_bits)
        else:
            handle = None
        
        allocate_out_begin = time.perf_counter()
        print("outer comm data (ms): {}".format((allocate_out_begin - comm_begin) * 1000.0))
        out = torch.zeros([graph.local_adj_t.sparse_size(0), local_nodes_feat.size(-1)], dtype=torch.float)
        local_aggregate_begin = time.perf_counter()
        # aggregate message from local nodes
        SPMM_forward(graph.local_adj_t, local_nodes_feat, out)

        async_wait_begin = time.perf_counter()
        if handle is not None:
            handle.wait()
 
        if recv_nodes_feat_fp16_buf is not None and num_recv_nodes != 0:
            # convert communication data to fp32
            # recv_nodes_feat_buf.copy_(recv_nodes_feat_fp16_buf)
            Quantizer.dequantize_intX_to_fp32(recv_nodes_feat_fp16_buf, recv_nodes_feat_buf, recv_scale, recv_zero_point)

        remote_aggregate_begin = time.perf_counter()
        remote_nodes_feat = recv_nodes_feat_buf
        # aggregate message from remote nodes
        if remote_nodes_feat.size(0) != 0:
            SPMM_forward(graph.remote_adj_t, remote_nodes_feat, out)

        sum_message_begin = time.perf_counter()

        sum_message_end = time.perf_counter()

        # rank = dist.get_rank()
        # if rank == 0:
        #     print('#########')
        #     print("Time of prepare comm_forward(ms): {}".format((comm_begin - prepare_comm_begin) * 1000.0))
        #     print("Time of comm_forward(ms): {}".format((allocate_out_begin - comm_begin) * 1000.0))
        #     print("Time of allocate out(ms): {}".format((local_aggregate_begin - allocate_out_begin) * 1000.0))
        #     print("Time of local aggregate(ms): {}".format((async_wait_begin - local_aggregate_begin) * 1000.0))
        #     print("Time of async wait(ms): {}".format((remote_aggregate_begin - async_wait_begin) * 1000.0))
        #     print("Time of remote aggregate(ms): {}".format((sum_message_begin - remote_aggregate_begin) * 1000.0))
        #     print("Time of sum up message(ms): {}".format((sum_message_end - sum_message_begin) * 1000.0))
        #     print("Time of 1 dist conv forward(inner)(ms): {}".format((sum_message_end - prepare_comm_begin) * 1000.0))
        #     print('#########')

        print("inner propagate forward (ms): {}".format((sum_message_end - prepare_comm_begin) * 1000.0))

        # return local_out
        return out

    @staticmethod
    def backward(ctx, local_out_grad):
        graph = ctx.graph
        num_bits = ctx.num_bits

        if ctx.needs_input_grad[1]:
            # scatter gradient to remote nodes
            remote_nodes_grad_buf = graph.comm_buf.recv_buf
            local_nodes_grad_buf = graph.comm_buf.send_buf

            remote_nodes_grad_fp16_buf = graph.comm_buf.recv_buf_fp16
            local_nodes_grad_fp16_buf = graph.comm_buf.send_buf_fp16

            num_send_nodes = sum(graph.num_nodes_recv_from_others)
            num_recv_nodes = sum(graph.num_nodes_send_to_others)

            remote_nodes_grad_buf.resize_(num_send_nodes, local_out_grad.size(-1))
            local_nodes_grad_buf.resize_(num_recv_nodes, local_out_grad.size(-1))

            if remote_nodes_grad_fp16_buf is not None and local_nodes_grad_fp16_buf is not None and \
               num_send_nodes != 0 and num_recv_nodes != 0:
                num_send_nodes = math.ceil(num_send_nodes / float(8 / num_bits))
                num_recv_nodes = math.ceil(num_recv_nodes / float(8 / num_bits))
                remote_nodes_grad_fp16_buf.resize_(num_send_nodes, local_out_grad.size(-1))
                local_nodes_grad_fp16_buf.resize_(num_recv_nodes, local_out_grad.size(-1))

            remote_nodes_grad_buf.zero_()
            if remote_nodes_grad_buf.size(0) != 0:
                SPMM_backward(graph.remote_adj_t, local_out_grad, remote_nodes_grad_buf)

            # communicate to obtain the local node grads from other subgraph
            if num_send_nodes != 0 and num_recv_nodes != 0:
                handle, recv_scale, recv_zero_point = \
                        comm_for_remote_nodes(local_nodes_grad_buf, remote_nodes_grad_buf,
                                              local_nodes_grad_fp16_buf, remote_nodes_grad_fp16_buf,
                                              graph.num_nodes_send_to_others, graph.num_nodes_recv_from_others, 
                                              num_bits)
            else:
                handle = None
            
            # scatter gradient to local nodes
            local_nodes_grad = torch.zeros([graph.local_adj_t.sparse_size(-1), local_out_grad.size(-1)], dtype=torch.float)
            SPMM_backward(graph.local_adj_t, local_out_grad, local_nodes_grad)

            if handle is not None:
                handle.wait()

            if local_nodes_grad_fp16_buf is not None and num_recv_nodes != 0:
                # convert communication data to fp32
                # local_nodes_grad_buf.copy_(local_nodes_grad_fp16_buf)
                Quantizer.dequantize_intX_to_fp32(local_nodes_grad_fp16_buf, local_nodes_grad_buf, recv_scale, recv_zero_point, num_bits)

            index_add_begin = time.perf_counter()
            # then accumulate the local node grads
            local_nodes_grad_from = local_nodes_grad_buf
            if local_nodes_grad_from.size(0) != 0:
                local_nodes_grad.index_add_(dim=0, index=graph.idx_nodes_send_to_others,
                                            source=local_nodes_grad_from)

            index_add_end = time.perf_counter()
            # rank = dist.get_rank()
            # if rank == 0:
            #     print('#########')
            #     print("Time of scatter gradient to local nodes(ms): {}".format((index_add_end - index_add_begin) * 1000.0))
            #     print('#########')

        return None, local_nodes_grad, None

def aggregate_for_local_and_remote(graph, local_nodes_feat, num_bits):
    return Aggregate_for_local_and_remote.apply(graph, local_nodes_feat, num_bits)

class DistSAGEConvGrad(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, num_bits: int = 32, \
                 add_self_loops: bool = False, normalize: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.local_deg = None
        self.num_bits = num_bits

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

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
        print("outer propagate forward (ms): {}".format((add_bias_begin - propagate_begin) * 1000.0))
        out += local_nodes_feat

        if self.normalize:
            local_deg = self.local_deg
            if local_deg is None:
                local_deg = get_deg(graph.local_adj_t, graph.remote_adj_t, self.add_self_loops)
                self.local_deg = local_deg
            out /= (local_deg + 1)

        # if not linear_first:
        #     out = self.lin(out)

        if self.bias is not None:
            out += self.bias
        add_bias_end = time.perf_counter()

        # rank = dist.get_rank()
        # if rank == 0:
        #     print("**************")
        #     # print("Time of norm(ms): {}".format((linear_begin - norm_begin) * 1000.0))
        #     print("Time of linear(ms): {}".format((propagate_begin -linear_begin) * 1000.0))
        #     print("Time of propagate(ms): {}".format((add_bias_begin - propagate_begin) * 1000.0))
        #     # print("Time of add_bias(ms): {}".format((add_bias_end - add_bias_begin) * 1000.0))
        #     print("Time of 1 dist conv forward(ms): {}".format((add_bias_end - norm_begin) * 1000.0))
        #     print("**************")

        return out
