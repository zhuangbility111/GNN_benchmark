from sqlite3 import Time
import sys
sys.path.append("../")
import torch
import pytest
from quantizer import QuantizerForMixedBits
from communicator import Communicator
from time_recorder import TimeRecorder

def test_get_splits_for_comm_quantized_data():
    world_size = 4
    recv_splits_tensor = torch.tensor([2, 0, 4, 1], dtype=torch.int32)
    send_splits_tensor = torch.tensor([0, 3, 0, 3], dtype=torch.int32)
    dequantized_nodes_feat_range = torch.tensor([0, 2, 4, 6, 8, 12, 13, 19], dtype=torch.int64)
    quantized_nodes_feat_range = torch.tensor([0, 1, 2, 5, 7, 11, 30], dtype=torch.int64)

    communicator = Communicator()
    communicator.world_size = world_size
    quantized_recv_splits, quantized_send_splits = communicator.get_splits_for_comm_quantized_data(
        recv_splits_tensor, send_splits_tensor, dequantized_nodes_feat_range, quantized_nodes_feat_range
    )

    expected_quantized_recv_splits = [4, 0, 9, 6]
    expected_quantized_send_splits = [0, 5, 0, 25]

    assert quantized_recv_splits == expected_quantized_recv_splits
    assert quantized_send_splits == expected_quantized_send_splits

def test_get_quantized_nodes_feat_range():
    num_nodes = 6
    feat_len = 4
    nodes_num_bits_tensor = torch.tensor([8, 4, 2, 8, 4, 2])

    quantized_nodes_feat_range = QuantizerForMixedBits.get_quantized_nodes_feat_range(num_nodes, feat_len, nodes_num_bits_tensor)

    expected_range = torch.tensor([0, 4, 6, 7, 11, 13, 14], dtype=torch.int64)
    assert torch.equal(quantized_nodes_feat_range, expected_range)

    num_nodes = 3
    feat_len = 1
    nodes_num_bits_tensor = torch.tensor([2, 2, 2])

    quantized_nodes_feat_range = QuantizerForMixedBits.get_quantized_nodes_feat_range(num_nodes, feat_len, nodes_num_bits_tensor)

    expected_range = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    assert torch.equal(quantized_nodes_feat_range, expected_range)

    num_nodes = 3
    feat_len = 3
    nodes_num_bits_tensor = torch.tensor([4, 4, 2])

    quantized_nodes_feat_range = QuantizerForMixedBits.get_quantized_nodes_feat_range(num_nodes, feat_len, nodes_num_bits_tensor)

    expected_range = torch.tensor([0, 2, 4, 5], dtype=torch.int64)
    assert torch.equal(quantized_nodes_feat_range, expected_range)

def test_quantize_fp32_to_intX():
    num_nodes = 5
    feat_len = 3
    data_fp32 = torch.tensor([[0.2, 0.1, 0.3],
                              [0.4, 0.5, 0.6],
                              [0.7, 0.8, 0.9],
                              [1.0, 1.1, 1.2],
                              [0.3, 0.1, 0.7]], dtype=torch.float32)
    
    nodes_num_bits_tensor = torch.tensor([8, 4, 2, 4, 4])
    quantized_params = torch.empty((num_nodes, 2 + 1), dtype=torch.float32)
    quantized_params[:, 0] = nodes_num_bits_tensor
    
    quantized_nodes_feat_range = QuantizerForMixedBits.get_quantized_nodes_feat_range(num_nodes, feat_len, nodes_num_bits_tensor)
    data_int8 = torch.empty(quantized_nodes_feat_range[-1].item(), dtype=torch.uint8)

    QuantizerForMixedBits.quantize_fp32_to_intX(data_fp32, data_int8, quantized_nodes_feat_range, quantized_params)

    # Add assertions to check the correctness of the quantization
    expected_range = torch.tensor([0, 3, 5, 6, 8, 10], dtype=torch.int64)
    expected_quantized_params = torch.empty((num_nodes, 2 + 1), dtype=torch.float32)
    expected_quantized_params[:, 0] = nodes_num_bits_tensor
    expected_quantized_params[:, 1] = data_fp32.min(dim=1)[0]
    expected_quantized_params[:, 2] = (data_fp32.max(dim=1)[0] - expected_quantized_params[:, 1]) / (2**expected_quantized_params[:, 0] - 1)
    expected_data_int8 = torch.tensor([127, 0, 255, 7, 240, 44, 8, 240, 80, 240], dtype=torch.uint8)
    assert torch.equal(quantized_nodes_feat_range, expected_range)
    assert torch.allclose(quantized_params, expected_quantized_params)
    assert torch.equal(data_int8, expected_data_int8)


def test_dequantize_intX_to_fp32():
    num_nodes = 5
    feat_len = 3
    data_int8 = torch.tensor([127, 0, 255, 7, 240, 44, 8, 240, 80, 240], dtype=torch.uint8)
    data_fp32 = torch.empty((num_nodes, feat_len), dtype=torch.float32)
    dequantized_params = torch.tensor([[8, 0.1, 7.8431e-04],
                                       [4, 0.4, 1.3333e-02],
                                       [2, 0.7, 6.6667e-02],
                                       [4, 1.0, 1.3333e-02],
                                       [4, 0.1, 4.0000e-02]], dtype=torch.float32)
    dequantized_nodes_feat_range = QuantizerForMixedBits.get_quantized_nodes_feat_range(
            num_nodes, feat_len, dequantized_params[:, 0]
        )

    time_recorder = TimeRecorder(3, 3)
    QuantizerForMixedBits.dequantize_intX_to_fp32(data_int8, data_fp32, dequantized_nodes_feat_range, dequantized_params)

    # Add assertions to check the correctness of the dequantization
    expected_range = torch.tensor([0, 3, 5, 6, 8, 10], dtype=torch.int64)
    expected_data_fp32 = torch.tensor([[0.2, 0.1, 0.3],
                                       [0.4, 0.5, 0.6],
                                       [0.7, 0.8, 0.9],
                                       [1.0, 1.1, 1.2],
                                       [0.3, 0.1, 0.7]], dtype=torch.float32)
    
    assert torch.equal(dequantized_nodes_feat_range, expected_range)
    print("data_fp32: ", data_fp32)
    assert torch.allclose(data_fp32, expected_data_fp32)
