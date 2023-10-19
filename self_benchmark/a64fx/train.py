import torch
import torch.distributed as dist
import torch.nn.functional as F
import argparse
import time
import yaml
from model import create_model_and_optimizer, set_random_seed
from communicator import Communicator
from data_manager import load_data
from assigner import Assigner

from torch.profiler import profile, record_function, ProfilerActivity

from time_recorder import TimeRecorder


def collect_acc(model, data):
    # check accuracy
    TimeRecorder.ctx.set_is_training(False)
    model.eval()
    predict_result = []
    out = model(data["graph"], data["nodes_features"])
    for mask in (data["nodes_train_masks"], data["nodes_valid_masks"], data["nodes_test_masks"]):
        num_correct_samples = (
            (out[mask].argmax(-1) == data["nodes_labels"][mask]).sum() if mask.size(0) != 0 else 0
        )
        num_samples = mask.size(0)
        predict_result.append(num_correct_samples)
        predict_result.append(num_samples)
    predict_result = torch.tensor(predict_result)
    if dist.get_world_size() > 1:
        dist.all_reduce(predict_result, op=dist.ReduceOp.SUM)

    train_acc = float(predict_result[0] / predict_result[1])
    val_acc = float(predict_result[2] / predict_result[3])
    test_acc = float(predict_result[4] / predict_result[5])
    TimeRecorder.ctx.set_is_training(True)
    return train_acc, val_acc, test_acc


def print_perf(total_forward_dur, total_backward_dur, total_update_weight_dur, total_training_dur):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    total_forward_dur = torch.tensor([total_forward_dur])
    total_backward_dur = torch.tensor([total_backward_dur])
    total_update_weight_dur = torch.tensor([total_update_weight_dur])
    ave_total_training_dur = torch.tensor([total_training_dur])
    max_total_training_dur = torch.tensor([total_training_dur])

    dist.reduce(total_forward_dur, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total_backward_dur, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total_update_weight_dur, 0, op=dist.ReduceOp.SUM)
    dist.reduce(ave_total_training_dur, 0, op=dist.ReduceOp.SUM)
    dist.reduce(max_total_training_dur, 0, op=dist.ReduceOp.MAX)

    if dist.get_rank() == 0:
        print("training end.")
        print("forward_time(ms): {}".format(total_forward_dur[0] / float(world_size) * 1000))
        print("backward_time(ms): {}".format(total_backward_dur[0] / float(world_size) * 1000))
        print("update_weight_time(ms): {}".format(total_update_weight_dur[0] / float(world_size) * 1000))
        print(
            "total_training_time(average)(ms): {}".format(
                ave_total_training_dur[0] / float(world_size) * 1000
            )
        )
        print("total_training_time(max)(ms): {}".format(max_total_training_dur[0] * 1000.0))


def train(model, data, optimizer, num_epochs, num_bits):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # start training
    total_forward_dur = 0
    total_backward_dur = 0
    total_update_weight_dur = 0
    total_training_dur = 0

    # with profile(activities=[ProfilerActivity.CPU]) as prof:
    for epoch in range(num_epochs):
        model.train()
        forward_start = time.perf_counter()
        Assigner.ctx.reassign_node_dataformat(epoch)
        optimizer.zero_grad()
        out = model(data["graph"], data["nodes_features"])
        backward_start = time.perf_counter()
        loss = F.nll_loss(out[data["nodes_train_masks"]], data["nodes_labels"][data["nodes_train_masks"]])
        loss.backward()

        update_weight_start = time.perf_counter()
        optimizer.step()
        update_weight_end = time.perf_counter()
        total_forward_dur += backward_start - forward_start
        total_backward_dur += update_weight_start - backward_start
        total_update_weight_dur += update_weight_end - update_weight_start
        total_training_dur += update_weight_end - forward_start

        train_acc, val_acc, test_acc = collect_acc(model, data)

        if rank == 0:
            print(
                f"Rank: {rank}, World_size: {world_size}, Epoch: {epoch}, Loss: {loss}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}, Time: {(update_weight_end - forward_start):.6f}"
            )
        TimeRecorder.ctx.next_epoch()

    print_perf(total_forward_dur, total_backward_dur, total_update_weight_dur, total_training_dur)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--num_bits", type=int, default=32)
    parser.add_argument("--is_pre_delay", type=str, default="false")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # config['is_fp16'] = True if args.is_fp16 == 'true' else False
    config["num_bits"] = args.num_bits
    config["is_pre_delay"] = True if args.is_pre_delay == "true" else False

    # print(config, flush=True)

    Communicator(config["num_bits"], config["is_async"])
    rank, world_size = Communicator.ctx.init_dist_group()
    if (
        config["graph_name"] != "arxiv"
        and config["graph_name"] != "products"
        and config["graph_name"] != "papers100M"
    ):
        config["input_dir"] += "{}_{}_part/".format(config["graph_name"], world_size)
    else:
        config["input_dir"] += "ogbn_{}_{}_part/".format(config["graph_name"], world_size)

    set_random_seed(config["random_seed"])
    model, optimizer = create_model_and_optimizer(config)
    data = load_data(config)

    Assigner(
        config["num_bits"],
        config["num_layers"],
        torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        config["assign_period"],
        data["graph"].comm_buf.send_buf.size(0),
        data["graph"].comm_buf.recv_buf.size(0),
    )

    TimeRecorder(config["num_layers"], config["num_epochs"])

    # print("finish data loading.", flush=True)
    train(model, data, optimizer, config["num_epochs"], config["num_bits"])

    # use mpi_reduce to get the average time of all mpi processes
    total_barrier_time = torch.tensor([TimeRecorder.ctx.get_total_barrier_time()])
    total_quantization_time = torch.tensor([TimeRecorder.ctx.get_total_quantization_time()])
    total_communication_time = torch.tensor([TimeRecorder.ctx.get_total_communication_time()])
    total_dequantization_time = torch.tensor([TimeRecorder.ctx.get_total_dequantization_time()])
    total_convolution_time = torch.tensor([TimeRecorder.ctx.get_total_convolution_time()])
    dist.reduce(total_barrier_time, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total_quantization_time, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total_communication_time, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total_dequantization_time, 0, op=dist.ReduceOp.SUM)
    dist.reduce(total_convolution_time, 0, op=dist.ReduceOp.SUM)

    if rank == 0:
        print("total_barrier_time(ms): {}".format(total_barrier_time[0] / float(world_size)))
        print("total_quantization_time(ms): {}".format(total_quantization_time[0] / float(world_size)))
        print("total_communication_time(ms): {}".format(total_communication_time[0] / float(world_size)))
        print("total_dequantization_time(ms): {}".format(total_dequantization_time[0] / float(world_size)))
        print("total_convolution_time(ms): {}".format(total_convolution_time[0] / float(world_size)))
