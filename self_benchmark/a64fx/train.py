import torch
import torch.distributed as dist
import torch.nn.functional as F
import argparse
import time
import yaml
from model import create_model_and_optimizer, set_random_seed
from communicator import init_dist_group
from data_manager import load_data

def train(model, data, optimizer, num_epochs):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # start training
    start = time.perf_counter()
    total_forward_dur = 0
    total_backward_dur = 0
    total_update_weight_dur = 0

    model.train()
    for epoch in range(num_epochs):
        forward_start = time.perf_counter()
        optimizer.zero_grad()
        out = model(data['graph'], data['nodes_features'])
        backward_start = time.perf_counter()
        loss = F.nll_loss(out[data['nodes_train_masks']], data['nodes_labels'][data['nodes_train_masks']])
        loss.backward()

        update_weight_start = time.perf_counter()
        optimizer.step()
        update_weight_end = time.perf_counter()
        total_forward_dur += backward_start - forward_start
        total_backward_dur += update_weight_start - backward_start
        total_update_weight_dur += update_weight_end - update_weight_start
        if rank == 0:
            print("rank: {}, epoch: {}, loss: {}, time: {}".format(rank, epoch, loss.item(), (update_weight_end - forward_start)), flush=True)
    end = time.perf_counter()
    total_training_dur = (end - start)
    
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
        print("total_training_time(average)(ms): {}".format(ave_total_training_dur[0] / float(world_size) * 1000))
        print("total_training_time(max)(ms): {}".format(max_total_training_dur[0] * 1000.0))

def test(model, data):
    rank = dist.get_rank()
    # check accuracy
    model.eval()
    predict_result = []
    out = model(data['graph'], data['nodes_features'])
    for mask in (data['nodes_train_masks'], data['nodes_valid_masks'], data['nodes_test_masks']):
        num_correct_samples = (out[mask].argmax(-1) == data['nodes_labels'][mask]).sum() if mask.size(0) != 0 else 0
        num_samples = mask.size(0)
        predict_result.append(num_correct_samples) 
        predict_result.append(num_samples)
    predict_result = torch.tensor(predict_result)
    dist.reduce(predict_result, 0, op=dist.ReduceOp.SUM)

    if rank == 0:
        train_acc = float(predict_result[0] / predict_result[1])
        val_acc = float(predict_result[2] / predict_result[3])
        test_acc = float(predict_result[4] / predict_result[5])
        print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--is_fp16', type=str, default='false')
    parser.add_argument('--is_pre_delay', type=str, default='false')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)

    config['is_fp16'] = True if args.is_fp16 == 'true' else False
    config['is_pre_delay'] = True if args.is_pre_delay == 'true' else False
    print(config, flush=True)

    # rank, world_size = init_dist_group()

    # set_random_seed(config['random_seed'])
    # model, optimizer = create_model_and_optimizer(config)
    # data = load_data(config)

    # train(model, data, optimizer, config['num_epochs'])
    # test(model, data)
