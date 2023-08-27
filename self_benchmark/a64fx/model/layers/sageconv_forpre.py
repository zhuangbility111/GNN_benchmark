import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import Parameter

from torch_sparse import fill_diag
from torch_sparse import sum as sparsesum

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.spmm_kernel import SPMM_forward, SPMM_backward

import time

class DistributedAggregation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, graph, local_nodes_feat):
        ctx.graph = graph

        resize_buffer_begin = time.perf_counter()
        send_nodes_feat_buf = graph.comm_buf.send_buf
        recv_nodes_feat_buf = graph.comm_buf.recv_buf

        send_nodes_feat_fp16_buf = graph.comm_buf.send_buf_fp16
        recv_nodes_feat_fp16_buf = graph.comm_buf.recv_buf_fp16

        num_recv_nodes = sum(graph.pre_post_aggr_from_splits)
        num_send_nodes = sum(graph.pre_post_aggr_to_splits)

        send_nodes_feat_buf.resize_(num_send_nodes, local_nodes_feat.size(-1))
        recv_nodes_feat_buf.resize_(num_recv_nodes, local_nodes_feat.size(-1))

        if send_nodes_feat_fp16_buf is not None and recv_nodes_feat_fp16_buf is not None and \
           num_send_nodes != 0 and num_recv_nodes != 0:
            send_nodes_feat_fp16_buf.resize_(num_send_nodes, local_nodes_feat.size(-1))
            recv_nodes_feat_fp16_buf.resize_(num_recv_nodes, local_nodes_feat.size(-1))

        create_out_memory_begin = time.perf_counter()
        out = torch.zeros([graph.local_adj_t.sparse_size(0), local_nodes_feat.size(-1)], dtype=torch.float)
        send_nodes_feat_buf.zero_()

        pre_aggr_to_begin = time.perf_counter()
        SPMM_forward(graph.adj_t_pre_post_aggr_to, local_nodes_feat, send_nodes_feat_buf)

        barrier_begin = time.perf_counter()
        rank = dist.get_rank()
        if rank == 0:
            print("pre aggr time = ", barrier_begin - resize_buffer_begin)
        # dist.barrier()

        comm_pre_aggr_to_begin = time.perf_counter()
        # communication in fp16
        if recv_nodes_feat_fp16_buf is not None and send_nodes_feat_fp16_buf is not None:
            # convert fp32 to fp16
            send_nodes_feat_fp16_buf.copy_(send_nodes_feat_buf)
            handle = dist.all_to_all_single(recv_nodes_feat_fp16_buf, send_nodes_feat_fp16_buf, \
                                            graph.pre_post_aggr_from_splits, graph.pre_post_aggr_to_splits, async_op=True)
        # communication in fp32
        else:
            handle = dist.all_to_all_single(recv_nodes_feat_buf, send_nodes_feat_buf, \
                                            graph.pre_post_aggr_from_splits, graph.pre_post_aggr_to_splits, async_op=True)

        local_aggr_begin = time.perf_counter()
        SPMM_forward(graph.local_adj_t, local_nodes_feat, out)

        async_wait_begin = time.perf_counter()
        if handle is not None:
            handle.wait()

        if rank == 0:
            print("communication time = ", async_wait_begin - comm_pre_aggr_to_begin)

        if recv_nodes_feat_fp16_buf is not None and send_nodes_feat_fp16_buf is not None:
            # recover fp16 to fp32
            recv_nodes_feat_buf.copy_(recv_nodes_feat_fp16_buf)

        post_aggr_from_begin = time.perf_counter()
        if recv_nodes_feat_buf.size(0) > 0:
            SPMM_forward(graph.adj_t_pre_post_aggr_from, recv_nodes_feat_buf, out)
        post_aggr_from_end = time.perf_counter()

        if rank == 0:
            print("post aggr time = ", post_aggr_from_end - post_aggr_from_begin)
 
        # rank = dist.get_rank()
        # if rank == 0:
        #     print('$$$$')
        #     # print("Time of resize buffer(ms): {}".format((create_out_memory_begin - resize_buffer_begin) * 1000.0))
        #     print("Time of create out memory(ms): {}".format((pre_aggr_to_begin - create_out_memory_begin) * 1000.0))
        #     print("Time of pre_aggr_to (ms): {}".format((barrier_begin - pre_aggr_to_begin) * 1000.0))
        #     print("Time of barrier (ms): {}".format((comm_pre_aggr_to_begin - barrier_begin) * 1000.0))
        #     print("Time of comm pre_aggr_to result (ms): {}".format((local_aggr_begin - comm_pre_aggr_to_begin) * 1000.0))
        #     print("Time of local aggr (ms): {}".format((async_wait_begin - local_aggr_begin) * 1000.0))
        #     print("Time of async wait (ms): {}".format((post_aggr_from_begin - async_wait_begin) * 1000.0))
        #     print("Time of post_aggr_from (ms): {}".format((post_aggr_from_end - post_aggr_from_begin) * 1000.0))
        #     print('$$$$')

        return out
        
    @staticmethod
    def backward(ctx, local_out_grad):
        graph = ctx.graph

        remote_nodes_grad_buf = graph.comm_buf.recv_buf
        local_nodes_grad_buf = graph.comm_buf.send_buf

        remote_nodes_grad_fp16_buf = graph.comm_buf.recv_buf_fp16
        local_nodes_grad_fp16_buf = graph.comm_buf.send_buf_fp16

        num_recv_nodes = sum(graph.pre_post_aggr_to_splits)
        num_send_nodes = sum(graph.pre_post_aggr_from_splits)

        remote_nodes_grad_buf.resize_(num_send_nodes, local_out_grad.size(-1))
        local_nodes_grad_buf.resize_(num_recv_nodes, local_out_grad.size(-1))

        if remote_nodes_grad_fp16_buf is not None and local_nodes_grad_fp16_buf is not None and \
            num_send_nodes != 0 and num_recv_nodes != 0:
            remote_nodes_grad_fp16_buf.resize_(num_send_nodes, local_out_grad.size(-1))
            local_nodes_grad_fp16_buf.resize_(num_recv_nodes, local_out_grad.size(-1))


        local_nodes_grad = torch.zeros([graph.local_adj_t.sparse_size(-1), local_out_grad.size(-1)], dtype=torch.float)
        
        # 1. collect input's grad (remote nodes' grad) of post-aggregation from
        remote_nodes_grad_buf.zero_()
        SPMM_backward(graph.adj_t_pre_post_aggr_from, local_out_grad, remote_nodes_grad_buf)

        if remote_nodes_grad_fp16_buf is not None and local_nodes_grad_fp16_buf is not None:
            # convert fp32 to fp16
            remote_nodes_grad_fp16_buf.copy_(remote_nodes_grad_buf)
            handle = dist.all_to_all_single(local_nodes_grad_fp16_buf, remote_nodes_grad_fp16_buf, \
                                            graph.pre_post_aggr_to_splits, graph.pre_post_aggr_from_splits, async_op=True)
        else:
            handle = dist.all_to_all_single(local_nodes_grad_buf, remote_nodes_grad_buf, \
                                            graph.pre_post_aggr_to_splits, graph.pre_post_aggr_from_splits, async_op=True)

        # 2.2 collect input's grad (local nodes' grad) of local-aggregation
        SPMM_backward(graph.local_adj_t, local_out_grad, local_nodes_grad)

        if handle is not None:
            handle.wait()

        # recover fp16 to fp32
        if remote_nodes_grad_fp16_buf is not None and local_nodes_grad_fp16_buf is not None:
            local_nodes_grad_buf.copy_(local_nodes_grad_fp16_buf)

        # 4. collect input's grad (local nodes' grad) of pre-aggregation to
        SPMM_backward(graph.adj_t_pre_post_aggr_to, local_nodes_grad_buf, local_nodes_grad)

        return None, local_nodes_grad

def aggregate_for_local_and_remote(graph, local_nodes_feat: Tensor):
    return DistributedAggregation.apply(graph, local_nodes_feat)

class DistSAGEConvGradWithPre(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 is_fp16: bool = False, add_self_loops: bool = False, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.is_fp16 = is_fp16

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

    def propagate(self, graph, **kwargs):
        # prepare the local nodes' feature which are required by other subgraphs
        local_nodes_feat = kwargs['x']

        local_out = aggregate_for_local_and_remote(graph, local_nodes_feat)
        return local_out

    def forward(self, graph, x: Tensor) -> Tensor:
        """"""
        norm_begin = time.perf_counter()
        linear_begin = time.perf_counter()

        linear_first = self.in_channels > self.out_channels
        if linear_first:
            # neural operation on nodes
            x = self.lin(x)

        propagate_begin = time.perf_counter()
        # if isinstance(local_edge_index, SparseTensor):
        out = self.propagate(graph, x=x)
        add_bias_begin = time.perf_counter()
        out += x
        out /= (graph.in_degrees + 1)

        if not linear_first:
            out = self.lin(out)

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

