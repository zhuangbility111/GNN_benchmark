import torch
import torch.distributed as dist
import torch.nn.functional as F
import time
from torch_geometric.nn import DistSAGEConvGradWithPre
from torch_geometric.nn import DistSAGEConvGrad

class DistSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, 
                 num_layers=3, dropout=0.5,
                 is_fp16=False, is_pre_delay=False):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        if not is_pre_delay:
            self.convs.append(DistSAGEConvGrad(in_channels, hidden_channels, is_fp16))
            for _ in range(num_layers - 2):
                self.convs.append(DistSAGEConvGrad(hidden_channels, hidden_channels, is_fp16))
            self.convs.append(DistSAGEConvGrad(hidden_channels, out_channels, is_fp16))
        else:
            self.convs.append(DistSAGEConvGradWithPre(in_channels, hidden_channels, is_fp16))
            for _ in range(num_layers - 2):
                self.convs.append(DistSAGEConvGradWithPre(hidden_channels, hidden_channels, is_fp16))
            self.convs.append(DistSAGEConvGradWithPre(hidden_channels, out_channels, is_fp16))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, graph, nodes_feats):
        total_conv_time = 0.0
        total_relu_time = 0.0
        total_dropout_time = 0.0
        for i, conv in enumerate(self.convs[:-1]):
            conv_begin = time.perf_counter()
            nodes_feats = conv(graph, nodes_feats)
            relu_begin = time.perf_counter()
            nodes_feats = F.relu(nodes_feats)
            dropout_begin = time.perf_counter()
            nodes_feats = F.dropout(nodes_feats, p=self.dropout, training=self.training)
            dropout_end = time.perf_counter()
            # total_conv_time += relu_begin - conv_begin
            total_conv_time = relu_begin - conv_begin
            # total_relu_time += dropout_begin - relu_begin
            total_relu_time = dropout_begin - relu_begin
            # total_dropout_time += dropout_end - dropout_begin
            total_dropout_time = dropout_end - dropout_begin
            rank = dist.get_rank()
            if rank == 0:
                print("----------------------------------------")
                print("Time of conv(ms): {:.4f}".format(total_conv_time * 1000.0))
                print("Time of relu(ms): {:.4f}".format(total_relu_time * 1000.0))
                print("Time of dropout(ms): {:.4f}".format(total_dropout_time * 1000.0))
                print("----------------------------------------")

        conv_begin = time.perf_counter()
        nodes_feats = self.convs[-1](graph, nodes_feats)
        # total_conv_time += time.perf_counter() - conv_begin
        return F.log_softmax(nodes_feats, dim=1)