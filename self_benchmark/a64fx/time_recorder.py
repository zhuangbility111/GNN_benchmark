import torch


class TimeRecorder(object):
    def __init__(self, num_layer, num_epoch) -> None:
        self.barrier_time = torch.zeros((num_layer * 2, num_epoch))
        self.quantization_time = torch.zeros((num_layer * 2, num_epoch))
        self.communication_time = torch.zeros((num_layer * 2, num_epoch))
        self.dequantization_time = torch.zeros((num_layer * 2, num_epoch))
        self.total_covolution_time = torch.zeros((num_layer * 2, num_epoch))
        self.cur_epoch = 0
        self.cur_layer = 0
        self._is_training = True
        TimeRecorder.ctx = self

    def record_barrier_time(self, time: float) -> None:
        if self._is_training:
            self.barrier_time[self.cur_layer, self.cur_epoch] = time

    def record_quantization_time(self, time: float) -> None:
        if self._is_training:
            self.quantization_time[self.cur_layer, self.cur_epoch] = time

    def record_communication_time(self, time: float) -> None:
        if self._is_training:
            self.communication_time[self.cur_layer, self.cur_epoch] = time

    def record_dequantization_time(self, time: float) -> None:
        if self._is_training:
            self.dequantization_time[self.cur_layer, self.cur_epoch] = time

    def record_total_convolution_time(self, time: float) -> None:
        if self._is_training:
            self.total_covolution_time[self.cur_layer, self.cur_epoch] = time

    def next_layer(self) -> None:
        if self._is_training:
            self.cur_layer += 1

    def next_epoch(self) -> None:
        if self._is_training:
            self.cur_epoch += 1
            self.cur_layer = 0

    def set_is_training(self, is_training: bool) -> None:
        self._is_training = is_training

    # time unit: ms
    def get_total_barrier_time(self) -> float:
        return torch.sum(self.barrier_time).item() * 1000.0

    def get_total_quantization_time(self) -> float:
        return torch.sum(self.quantization_time).item() * 1000.0

    def get_total_communication_time(self) -> float:
        return torch.sum(self.communication_time).item() * 1000.0

    def get_total_dequantization_time(self) -> float:
        return torch.sum(self.dequantization_time).item() * 1000.0

    def get_total_convolution_time(self) -> float:
        return torch.sum(self.total_covolution_time).item() * 1000.0

    @staticmethod
    def print_time(rank, message, time):
        if rank == 0:
            print("{}: {}".format(message, time))
