import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import LayerNorm
import time

# from torch_geometric.nn import DistSAGEConvGradWithPre
# from torch_geometric.nn import DistSAGEConvGrad
from .layers.sageconv_forpre import DistSAGEConvGradWithPre
from .layers.sageconv import DistSAGEConvGrad


class DistSAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=3,
        dropout=0.5,
        num_bits=32,
        is_pre_delay=False,
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.convs.append(DistSAGEConvGrad(in_channels, hidden_channels, num_bits, is_pre_delay))
        self.norms.append(LayerNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(DistSAGEConvGrad(hidden_channels, hidden_channels, num_bits, is_pre_delay))
            self.norms.append(LayerNorm(hidden_channels))
        self.convs.append(DistSAGEConvGrad(hidden_channels, out_channels, num_bits, is_pre_delay))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, graph, nodes_feats):
        total_conv_time = 0.0
        total_relu_time = 0.0
        total_dropout_time = 0.0
        for i, conv in enumerate(self.convs[:-1]):
            conv_begin = time.perf_counter()
            nodes_feats = conv(graph, nodes_feats, i)
            dropout_begin = time.perf_counter()
            nodes_feats = F.dropout(nodes_feats, p=self.dropout, training=self.training)
            nodes_feats = self.norms[i](nodes_feats)
            relu_begin = time.perf_counter()
            nodes_feats = F.relu(nodes_feats)
            relu_end = time.perf_counter()
            total_conv_time = dropout_begin - conv_begin
            total_relu_time = relu_end - relu_begin
            total_dropout_time = relu_begin - dropout_begin
            rank = dist.get_rank()
            if rank == 0:
                print("----------------------------------------")
                print("Time of conv(ms): {:.4f}".format(total_conv_time * 1000.0))
                print("Time of relu(ms): {:.4f}".format(total_relu_time * 1000.0))
                print("Time of dropout(ms): {:.4f}".format(total_dropout_time * 1000.0))
                print("----------------------------------------")

        conv_begin = time.perf_counter()
        nodes_feats = self.convs[-1](graph, nodes_feats, len(self.convs) - 1)
        return F.log_softmax(nodes_feats, dim=1)
