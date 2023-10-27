import torch
import numpy as np
import torch.distributed as dist

class TimeRecorder(object):
    def __init__(self, num_layer, num_epoch) -> None:
        self.barrier_time = torch.zeros((num_layer * 2, num_epoch))
        self.quantization_time = torch.zeros((num_layer * 2, num_epoch))
        self.communication_time = torch.zeros((num_layer * 2, num_epoch))
        self.dequantization_time = torch.zeros((num_layer * 2, num_epoch))
        self.prepare_comm_time = torch.zeros((num_layer * 2, num_epoch))
        self.pre_aggregate_time = torch.zeros((num_layer * 2, num_epoch))
        self.local_aggregate_time = torch.zeros((num_layer * 2, num_epoch))
        self.remote_aggregate_time = torch.zeros((num_layer * 2, num_epoch))
        self.total_covolution_time = torch.zeros((num_layer * 2, num_epoch))
        self.total_training_time = torch.zeros(num_epoch)
        self.cur_epoch = 0
        self.cur_layer = 0
        self._is_training = True
        TimeRecorder.ctx = self

    def record_barrier_time(self, time: float) -> None:
        if self._is_training:
            self.barrier_time[self.cur_layer, self.cur_epoch] += time

    def record_quantization_time(self, time: float) -> None:
        if self._is_training:
            self.quantization_time[self.cur_layer, self.cur_epoch] += time

    def record_communication_time(self, time: float) -> None:
        if self._is_training:
            self.communication_time[self.cur_layer, self.cur_epoch] += time

    def record_dequantization_time(self, time: float) -> None:
        if self._is_training:
            self.dequantization_time[self.cur_layer, self.cur_epoch] += time

    def record_total_convolution_time(self, time: float) -> None:
        if self._is_training:
            self.total_covolution_time[self.cur_layer, self.cur_epoch] += time

    def record_prepare_comm_time(self, time: float) -> None:
        if self._is_training:
            self.prepare_comm_time[self.cur_layer, self.cur_epoch] += time

    def record_pre_aggregate_time(self, time: float) -> None:
        if self._is_training:
            self.pre_aggregate_time[self.cur_layer, self.cur_epoch] += time
    
    def record_local_aggregate_time(self, time: float) -> None:
        if self._is_training:
            self.local_aggregate_time[self.cur_layer, self.cur_epoch] += time
    
    def record_remote_aggregate_time(self, time: float) -> None:
        if self._is_training:
            self.remote_aggregate_time[self.cur_layer, self.cur_epoch] += time
    
    def record_total_training_time(self, time: float) -> None:
        if self._is_training:
            self.total_training_time[self.cur_epoch] += time

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
    
    def get_total_prepare_comm_time(self) -> float:
        return torch.sum(self.prepare_comm_time).item() * 1000.0
    
    def get_total_pre_aggregate_time(self) -> float:
        return torch.sum(self.pre_aggregate_time).item() * 1000.0
    
    def get_total_local_aggregate_time(self) -> float:
        return torch.sum(self.local_aggregate_time).item() * 1000.0
    
    def get_total_remote_aggregate_time(self) -> float:
        return torch.sum(self.remote_aggregate_time).item() * 1000.0
    
    def get_total_training_time(self) -> float:
        return torch.sum(self.total_training_time).item() * 1000.0
    
    # save the time to file
    def save_time_to_file(self, graph_name, world_size) -> None:

        # convert all time from s to ms
        self.barrier_time *= 1000.0
        self.quantization_time *= 1000.0
        self.communication_time *= 1000.0
        self.dequantization_time *= 1000.0
        self.total_covolution_time *= 1000.0
        self.prepare_comm_time *= 1000.0
        self.pre_aggregate_time *= 1000.0
        self.local_aggregate_time *= 1000.0
        self.remote_aggregate_time *= 1000.0
        self.total_training_time *= 1000.0

        # use reduce to get the average time of all mpi processes
        dist.reduce(self.barrier_time, 0, op=dist.ReduceOp.SUM)
        dist.reduce(self.quantization_time, 0, op=dist.ReduceOp.SUM)
        dist.reduce(self.communication_time, 0, op=dist.ReduceOp.SUM)
        dist.reduce(self.dequantization_time, 0, op=dist.ReduceOp.SUM)
        dist.reduce(self.total_covolution_time, 0, op=dist.ReduceOp.SUM)
        dist.reduce(self.prepare_comm_time, 0, op=dist.ReduceOp.SUM)
        dist.reduce(self.pre_aggregate_time, 0, op=dist.ReduceOp.SUM)
        dist.reduce(self.local_aggregate_time, 0, op=dist.ReduceOp.SUM)
        dist.reduce(self.remote_aggregate_time, 0, op=dist.ReduceOp.SUM)
        dist.reduce(self.total_training_time, 0, op=dist.ReduceOp.SUM)


        if dist.get_rank() == 0:
            # get the average time
            self.barrier_time /= float(world_size)
            self.quantization_time /= float(world_size)
            self.communication_time /= float(world_size)
            self.dequantization_time /= float(world_size)
            self.total_covolution_time /= float(world_size)
            self.prepare_comm_time /= float(world_size)
            self.pre_aggregate_time /= float(world_size)
            self.local_aggregate_time /= float(world_size)
            self.remote_aggregate_time /= float(world_size)
            self.total_training_time /= float(world_size)


            # convert all time to numpy
            barrier_time = self.barrier_time.numpy()
            quantization_time = self.quantization_time.numpy()
            communication_time = self.communication_time.numpy()
            dequantization_time = self.dequantization_time.numpy()
            total_covolution_time = self.total_covolution_time.numpy()
            prepare_comm_time = self.prepare_comm_time.numpy()
            pre_aggregate_time = self.pre_aggregate_time.numpy()
            local_aggregate_time = self.local_aggregate_time.numpy()
            remote_aggregate_time = self.remote_aggregate_time.numpy()
            total_training_time = self.total_training_time.numpy()

            # save all array to npy file
            np.save("barrier_time_{}_{}.npy".format(graph_name, world_size), barrier_time)
            np.save("quantization_time_{}_{}.npy".format(graph_name, world_size), quantization_time)
            np.save("communication_time_{}_{}.npy".format(graph_name, world_size), communication_time)
            np.save("dequantization_time_{}_{}.npy".format(graph_name, world_size), dequantization_time)
            np.save("total_covolution_time_{}_{}.npy".format(graph_name, world_size), total_covolution_time)
            np.save("prepare_comm_time_{}_{}.npy".format(graph_name, world_size), prepare_comm_time)
            np.save("pre_aggregate_time_{}_{}.npy".format(graph_name, world_size), pre_aggregate_time)
            np.save("local_aggregate_time_{}_{}.npy".format(graph_name, world_size), local_aggregate_time)
            np.save("remote_aggregate_time_{}_{}.npy".format(graph_name, world_size), remote_aggregate_time)
            np.save("total_training_time_{}_{}.npy".format(graph_name, world_size), total_training_time)

    @staticmethod
    def print_time(rank, message, time):
        if rank == 0:
            print("{}: {}".format(message, time))
