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


def train(model, data, optimizer, num_epochs, num_bits):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # start training
    start = time.perf_counter()
    total_forward_dur = 0
    total_backward_dur = 0
    total_update_weight_dur = 0

    node_dataformat_assign_period = 200

    if num_bits == 8:
        assign_weight = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    elif num_bits == 4:
        assign_weight = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    elif num_bits == 2:
        assign_weight = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    elif num_bits == -1:
        assign_weight = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    else:
        assign_weight = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    # with profile(activities=[ProfilerActivity.CPU]) as prof:
    model.train()
    for epoch in range(num_epochs):
        if num_bits != 32 and num_bits != 16 and epoch % node_dataformat_assign_period == 0:
            Assigner.ctx.assign_node_dataformat_randomly(assign_weight, num_bits)
        forward_start = time.perf_counter()
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
        if rank == 0:
            print(
                "rank: {}, epoch: {}, loss: {}, time: {}".format(
                    rank, epoch, loss.item(), (update_weight_end - forward_start)
                ),
                flush=True,
            )
        TimeRecorder.ctx.next_epoch()

    end = time.perf_counter()
    total_training_dur = end - start

    # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    # prof.export_chrome_trace("trace.json")

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

    if rank == 0:
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


def test(model, data):
    rank = dist.get_rank()
    # check accuracy
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
    dist.reduce(predict_result, 0, op=dist.ReduceOp.SUM)

    if rank == 0:
        train_acc = float(predict_result[0] / predict_result[1])
        val_acc = float(predict_result[2] / predict_result[3])
        test_acc = float(predict_result[4] / predict_result[5])
        print(f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")


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
    print(config, flush=True)

    rank, world_size = Communicator.init_dist_group()
    config["input_dir"] += "ogbn_{}_{}_part/".format(config["graph_name"], world_size)

    set_random_seed(config["random_seed"])
    model, optimizer = create_model_and_optimizer(config)
    data = load_data(config)

    Assigner(
        config["num_layers"], data["graph"].comm_buf.send_buf.size(0), data["graph"].comm_buf.recv_buf.size(0)
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

    test(model, data)
