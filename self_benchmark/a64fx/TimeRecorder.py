import torch

class TimeRecorder(object):

    def __init__(self, num_layer, num_epoch) -> None:
        self.barrier_time = torch.zeros((num_layer * 2, num_epoch + 1))
        self.quantization_time = torch.zeros((num_layer * 2, num_epoch + 1))
        self.communication_time = torch.zeros((num_layer * 2, num_epoch + 1))
        self.dequantization_time = torch.zeros((num_layer * 2, num_epoch + 1))
        self.total_covolution_time = torch.zeros((num_layer * 2, num_epoch + 1))
        self.cur_epoch = 0
        self.cur_layer = 0
        TimeRecorder.ctx = self

    def record_barrier_time(self, time: float) -> None:
        self.barrier_time[self.cur_layer, self.cur_epoch] = time
    
    def record_quantization_time(self, time: float) -> None:
        self.quantization_time[self.cur_layer, self.cur_epoch] = time
    
    def record_communication_time(self, time: float) -> None:
        self.communication_time[self.cur_layer, self.cur_epoch] = time

    def record_dequantization_time(self, time: float) -> None:
        self.dequantization_time[self.cur_layer, self.cur_epoch] = time

    def record_total_convolution_time(self, time: float) -> None:
        self.total_covolution_time[self.cur_layer, self.cur_epoch] = time

    def next_layer(self) -> None:
        self.cur_layer += 1
    
    def next_epoch(self) -> None:
        self.cur_epoch += 1
        self.cur_layer = 0

    # time unit: ms
    def get_total_barrier_time(self) -> float:
        return torch.sum(self.barrier_time) * 1000.0
    
    def get_total_quantization_time(self) -> float:
        return torch.sum(self.quantization_time) * 1000.0
    
    def get_total_communication_time(self) -> float:
        return torch.sum(self.communication_time) * 1000.0
    
    def get_total_dequantization_time(self) -> float:
        return torch.sum(self.dequantization_time) * 1000.0
    
    def get_total_convolution_time(self) -> float:
        return torch.sum(self.total_covolution_time) * 1000.0

