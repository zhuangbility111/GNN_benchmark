import torch
from . import CommSplits

class CommBuffer(object):
    def __init__(self, comm_splits: CommSplits, feat_len: int, bits: int) -> None:
        size_send_buf = (sum(comm_splits.send_splits), feat_len)
        size_recv_buf = (sum(comm_splits.recv_splits), feat_len)
        self.send_buf = torch.zeros(size_send_buf, dtype=torch.float32)
        self.recv_buf = torch.zeros(size_recv_buf, dtype=torch.float32)
        self.send_buf_fp16 = None
        self.recv_buf_fp16 = None
        if bits == 16:
            self.send_buf_fp16 = torch.zeros(size_send_buf, dtype=torch.bfloat16)
            self.recv_buf_fp16 = torch.zeros(size_recv_buf, dtype=torch.bfloat16)

    def resize_buffer(self, comm_splits: CommSplits, feat_len: int, bits: int) -> None:
        size_send_buf = (sum(comm_splits.send_splits), feat_len)
        size_recv_buf = (sum(comm_splits.recv_splits), feat_len)
        # resize the fp32 message buffer
        self.send_buf.resize_(size_send_buf)
        self.recv_buf.resize_(size_recv_buf)

        if bits == 16 and self.send_buf_fp16 is not None and self.recv_buf_fp16 is not None:
            self.send_buf_fp16.resize_(size_send_buf)
            self.recv_buf_fp16.resize_(size_recv_buf)

class CommBufferForQuantization(object):
    def __init__(self, comm_splits: CommSplits, feat_len: int, bits: int) -> None:
        if bits == 2:
            self.quantized_send_data_buf = torch.zeros((sum(comm_splits.send_splits_int2), feat_len), dtype=torch.uint8)
            self.quantized_recv_data_buf = torch.zeros((sum(comm_splits.recv_splits_int2), feat_len), dtype=torch.uint8)
            self.quantized_send_params_buf_fp32 = torch.zeros((sum(comm_splits.send_splits), 2), dtype=torch.float32)
            self.quantized_recv_params_buf_fp32 = torch.zeros((sum(comm_splits.recv_splits), 2), dtype=torch.float32)
            self.quantized_send_params_buf_bf16 = torch.zeros((sum(comm_splits.send_splits), 2), dtype=torch.bfloat16)
            self.quantized_recv_params_buf_bf16 = torch.zeros((sum(comm_splits.recv_splits), 2), dtype=torch.bfloat16)

            self.quantized_work_range_per_proc = torch.empty((len(comm_splits.send_splits) + 1), dtype=torch.int32)
            self.quantized_work_range_per_proc[0] = 0
            self.quantized_work_range_per_proc[1:] = torch.tensor(comm_splits.send_splits, dtype=torch.int32).cumsum(0)

            self.dequantized_work_range_per_proc = torch.empty((len(comm_splits.recv_splits) + 1), dtype=torch.int32)
            self.dequantized_work_range_per_proc[0] = 0
            self.dequantized_work_range_per_proc[1:] = torch.tensor(comm_splits.recv_splits, dtype=torch.int32).cumsum(0)

    def resize_buffer(self, comm_splits: CommSplits, feat_len: int, bits: int) -> None:
        if bits == 2:
        # resize the fp32 message buffer
            self.quantized_send_data_buf.resize_((sum(comm_splits.send_splits_int2), feat_len))
            self.quantized_recv_data_buf.resize_((sum(comm_splits.recv_splits_int2), feat_len))
