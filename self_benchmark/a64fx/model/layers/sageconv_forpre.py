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


def aggregate_for_local_and_remote(graph, local_nodes_feat: Tensor):
    return DistributedAggregation.apply(graph, local_nodes_feat)


class DistSAGEConvGradWithPre(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_fp16: bool = False,
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
        self.is_fp16 = is_fp16

        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer="glorot")

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def propagate(self, graph, **kwargs):
        # prepare the local nodes' feature which are required by other subgraphs
        local_nodes_feat = kwargs["x"]

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
        out /= graph.in_degrees + 1

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
