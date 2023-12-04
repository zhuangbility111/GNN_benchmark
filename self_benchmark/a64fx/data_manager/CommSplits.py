import torch
import sys

sys.path.append("../")
from quantizer import Quantizer_for_all_procs

class CommSplits(object):
    def __init__(self, recv_splits: list, send_splits: list, world_size: int, bits: int) -> None:
        self.recv_splits = recv_splits
        self.send_splits = send_splits
        if bits == 2:
            self.recv_splits_int2 = list()
            self.send_splits_int2 = list()
            for rank in range(world_size):
                # prepare the quantized buffer for communication
                int2_recv_buf_size = Quantizer_for_all_procs.ctx.get_quantized_splits(recv_splits[rank])
                int2_send_buf_size = Quantizer_for_all_procs.ctx.get_quantized_splits(send_splits[rank])
                self.recv_splits_int2.append(int2_recv_buf_size)
                self.send_splits_int2.append(int2_send_buf_size)
        