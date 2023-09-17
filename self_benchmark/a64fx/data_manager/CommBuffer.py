import torch


class CommBuffer(object):
    def __init__(self, size_send_buf: tuple, size_recv_buf: tuple, is_fp16=False) -> None:
        self.send_buf = torch.zeros(size_send_buf, dtype=torch.float32)
        self.recv_buf = torch.zeros(size_recv_buf, dtype=torch.float32)
        self.send_buf_fp16 = None
        self.recv_buf_fp16 = None
        if is_fp16:
            self.send_buf_fp16 = torch.zeros(size_send_buf, dtype=torch.bfloat16)
            self.recv_buf_fp16 = torch.zeros(size_recv_buf, dtype=torch.bfloat16)
            # self.send_buf_fp16 = torch.zeros(size_send_buf, dtype=torch.uint8)
            # self.recv_buf_fp16 = torch.zeros(size_recv_buf, dtype=torch.uint8)

    def resize_buffer(self, size_send_buf: tuple, size_recv_buf: tuple) -> None:
        # resize the fp32 message buffer
        self.send_buf.resize_(size_send_buf)
        self.recv_buf.resize_(size_recv_buf)

        # resize the fp16 message buffer
        if self.send_buf_fp16 is not None and self.recv_buf_fp16 is not None:
            self.send_buf_fp16.resize_(size_send_buf)
            self.recv_buf_fp16.resize_(size_recv_buf)
