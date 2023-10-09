import torch
import quantization_cpu
import math
import time
import numpy as np

# torch.set_num_threads(20)

num_nodes = 256
feat_len = 233


def diff(out, ref, num_of_out, num_of_ref, atol=1e-06, rtol=1e-05):
    torch.set_printoptions(precision=10)
    idx_of_diff = torch.where(torch.abs(out - ref) > (atol + rtol * torch.abs(ref)))
    print(f"idx_of_diff = {idx_of_diff}")
    print(f"{num_of_ref}[idx_of_diff] = {ref[idx_of_diff]}")
    print(f"{num_of_out}[idx_of_diff] = {out[idx_of_diff]}")

    return idx_of_diff


def run_torch_quantize_per_channel(data_fp32, scale, zero_point, bits):
    data_int8_ref = None
    if bits == 8:
        data_int8_ref = torch.quantize_per_channel(data_fp32, scale, zero_point, 0, torch.quint8)
    elif bits == 4:
        data_int8_ref = torch.quantize_per_channel(data_fp32, scale, zero_point, 0, torch.quint4x2)
    else:
        print("bits = {} is not supported on torch_quantize_per_channel".format(bits))
    return data_int8_ref


def run_torch_dequantize(data_int8_ref):
    return torch.dequantize(data_int8_ref)


def run_quantization_cpu_quantize_tensor(data_fp32, bits, local_num_nodes):
    data_int8 = torch.empty((math.ceil(local_num_nodes / float(8 / bits)) * feat_len), dtype=torch.uint8)
    quantized_params = torch.empty((local_num_nodes, 2), dtype=torch.float32)
    inner_quantization_begin = time.perf_counter()
    quantization_cpu.quantize_tensor(data_fp32, data_int8, quantized_params, bits)
    inner_quantization_end = time.perf_counter()
    print("inner_quantization_time on aggr on row (ms): ", (inner_quantization_end - inner_quantization_begin) * 1000.0)
    return data_int8, quantized_params


def run_quantization_cpu_dequantize_tensor(data_int8, dequantized_params, bits):
    data_fp32_dequant = torch.empty((num_nodes, feat_len), dtype=torch.float32)
    quantization_cpu.dequantize_tensor(data_int8, data_fp32_dequant, dequantized_params, bits)
    return data_fp32_dequant


def run_quantization_cpu_quantize_tensor_v1(data_fp32, nodes_num_bits_tensor):
    prepare_begin = time.perf_counter()
    # [num_bits, zero_point, scale]
    num_nodes = data_fp32.size(0)
    quantized_params = torch.empty((num_nodes, 3), dtype=torch.float32)
    quantized_params[:, 0] = nodes_num_bits_tensor

    # prefix sum of nodes_num_bits_tensor
    quantized_nodes_feat_range = torch.full((num_nodes + 1,), feat_len, dtype=torch.int64)
    quantized_nodes_feat_range[0] = 0
    quantized_nodes_feat_range[1:] = torch.ceil(quantized_nodes_feat_range[1:] * nodes_num_bits_tensor / 8.0)
    quantized_nodes_feat_range = torch.cumsum(quantized_nodes_feat_range, 0)
    # for i in range(0, num_nodes):
    #     quantized_nodes_feat_range[i+1] = quantized_nodes_feat_range[i] + \
    #                 torch.ceil((nodes_num_bits_tensor[i]) * feat_len / 8.0)
    data_int8 = torch.empty(quantized_nodes_feat_range[-1], dtype=torch.uint8)
    # zero_points = torch.empty((num_nodes), dtype=torch.float32)
    # scales = torch.empty((num_nodes), dtype=torch.float32)
    prepare_end = time.perf_counter()
    inner_quantization_begin = time.perf_counter()
    quantization_cpu.quantize_tensor_v1(data_fp32, data_int8, quantized_nodes_feat_range, quantized_params)
    inner_quantization_end = time.perf_counter()
    print("prepare_time (ms): ", (prepare_end - prepare_begin) * 1000.0)
    print("inner_quantization_time on aggr on col (ms): ", (inner_quantization_end - inner_quantization_begin) * 1000.0)
    return data_int8, quantized_nodes_feat_range, quantized_params


def run_quantization_cpu_dequantize_tensor_v1(data_int8, quantized_nodes_feat_range, quantized_params):
    num_nodes = quantized_nodes_feat_range.size(0) - 1
    data_fp32_dequant = torch.empty((num_nodes, feat_len), dtype=torch.float32)
    quantization_cpu.dequantize_tensor_v1(
        data_int8, data_fp32_dequant, quantized_nodes_feat_range, quantized_params
    )
    return data_fp32_dequant


# a version aligned with torch version based on v1
def run_quantization_cpu_quantize_tensor_v2(data_fp32, nodes_num_bits_tensor, zero_point, scale):
    # [num_bits, zero_point, scale]
    num_nodes = data_fp32.size(0)
    quantized_params = torch.empty((num_nodes, 3), dtype=torch.float32)
    quantized_params[:, 0].fill_(nodes_num_bits_tensor)

    # quantized_params[:, 1] = zero_point
    # quantized_params[:, 2] = scale

    # prefix sum of nodes_num_bits_tensor
    quantized_nodes_feat_range = torch.full((num_nodes + 1,), feat_len, dtype=torch.int64)
    quantized_nodes_feat_range[0] = 0
    quantized_nodes_feat_range[1:] = torch.ceil(quantized_nodes_feat_range[1:] * nodes_num_bits_tensor / 8.0)
    quantized_nodes_feat_range = torch.cumsum(quantized_nodes_feat_range, 0)
    # for i in range(0, num_nodes):
    #     quantized_nodes_feat_range[i+1] = quantized_nodes_feat_range[i] + \
    #                 torch.ceil((nodes_num_bits_tensor[i]) * feat_len / 8.0)
    data_int8 = torch.empty(quantized_nodes_feat_range[-1], dtype=torch.uint8)
    # zero_points = torch.empty((num_nodes), dtype=torch.float32)
    # scales = torch.empty((num_nodes), dtype=torch.float32)
    prepare_end = time.perf_counter()
    inner_quantization_begin = time.perf_counter()
    quantization_cpu.quantize_tensor_v2_torch(
        data_fp32, data_int8, quantized_nodes_feat_range, zero_point, scale, bits
    )
    return data_int8, quantized_nodes_feat_range


# a version aligned with torch version based on v1
def run_quantization_cpu_dequantize_tensor_v2(data_int8, quantized_nodes_feat_range, zero_point, scale, bits):
    num_nodes = quantized_nodes_feat_range.size(0) - 1
    data_fp32_dequant = torch.empty((num_nodes, feat_len), dtype=torch.float32)
    quantization_cpu.dequantize_tensor_v2_torch(
        data_int8, data_fp32_dequant, quantized_nodes_feat_range, zero_point, scale, bits
    )
    return data_fp32_dequant

def run_quantization_cpu_quantize_tensor_for_all_procs(data_fp32, bits):
    prepare_begin = time.perf_counter()
    num_procs = 256
    work_range_per_proc = torch.randint(0, num_nodes, (num_procs + 1,), dtype=torch.int32).sort()[0]
    work_range_per_proc[0] = 0
    work_range_per_proc[-1] = num_nodes
    data_int8_list = []
    for i in range(work_range_per_proc.size(0) - 1):
        data_int8_list.append(torch.empty((math.ceil((work_range_per_proc[i+1] - work_range_per_proc[i]) / float(8 / bits)) * feat_len), dtype=torch.uint8))
    data_int8 = torch.cat(data_int8_list, dim=0)
    prepare_end = time.perf_counter()
    print("prepare time on for_all_procs (ms): ", (prepare_end - prepare_begin) * 1000.0)
    quantized_params = torch.empty((num_nodes, 2), dtype=torch.float32)
    inner_quantization_begin = time.perf_counter()
    # quantization_cpu.quantize_tensor(data_fp32, data_int8, quantized_params, bits)
    quantization_cpu.quantize_tensor_for_all_procs(data_fp32, data_int8, quantized_params, work_range_per_proc, num_procs, bits)
    inner_quantization_end = time.perf_counter()
    print("inner_quantization_time for_all_procs (ms): ", (inner_quantization_end - inner_quantization_begin) * 1000.0)
    return data_int8, quantized_params, work_range_per_proc

def run_quantization_cpu_dequantize_tensor_for_all_procs(data_int8, dequantized_params, work_range_per_proc, bits):
    data_fp32_dequant = torch.empty((num_nodes, feat_len), dtype=torch.float32)
    num_procs = work_range_per_proc.size(0) - 1
    quantization_cpu.dequantize_tensor_for_all_procs(data_int8, data_fp32_dequant, dequantized_params, work_range_per_proc, num_procs, bits)
    return data_fp32_dequant


def test_correctness_for_quantize_tensor(data_fp32, min_val, zero_point, scale, bits):
    if bits == 8 or bits == 4:
        data_int8_ref = run_torch_quantize_per_channel(data_fp32, scale, zero_point, bits)
        data_fp32_dequant_ref = run_torch_dequantize(data_int8_ref)

    data_int8_aggr_on_row, quantized_params_0 = run_quantization_cpu_quantize_tensor(data_fp32, bits, num_nodes)
    data_fp32_dequant_aggr_on_row = run_quantization_cpu_dequantize_tensor(
        data_int8_aggr_on_row, quantized_params_0, bits
    )

    (
        data_int8_aggr_on_col,
        quantized_nodes_feat_range,
        quantized_params_1,
    ) = run_quantization_cpu_quantize_tensor_v1(data_fp32, bits)
    data_fp32_dequant_aggr_on_col = run_quantization_cpu_dequantize_tensor_v1(
        data_int8_aggr_on_col, quantized_nodes_feat_range, quantized_params_1
    )

    data_int8_for_all_procs, quantized_params_for_all_procs, work_range_per_proc = run_quantization_cpu_quantize_tensor_for_all_procs(data_fp32, bits)
    data_fp32_dequant_for_all_procs = run_quantization_cpu_dequantize_tensor_for_all_procs(data_int8_for_all_procs, quantized_params_for_all_procs, work_range_per_proc, bits)

    atol = 1e-06
    rtol = 1e-05
    torch.set_printoptions(precision=10)

    # assert(torch.allclose(quantized_params_0[:, 0], quantized_params_1[:, 1]))
    # assert(torch.allclose(quantized_params_0[:, 1], quantized_params_1[:, 2]))
    # diff(quantized_params_0[:, 0], quantized_params_1[:, 1], "quantized_params_0 zero_point", "quantized_params_1 zero_point")
    # diff(quantized_params_0[:, 1], quantized_params_1[:, 2], "quantized_params_0 scale", "quantized_params_1 scale")
    diff(quantized_params_for_all_procs[:, 0], quantized_params_1[:, 1], "quantized_params_for_all_procs zero_point", "quantized_params_1 zero_point")
    diff(quantized_params_for_all_procs[:, 1], quantized_params_1[:, 2], "quantized_params_for_all_procs scale", "quantized_params_1 scale")

    if bits == 8 or bits == 4:
        idx_of_diff = torch.where(
            torch.abs(data_fp32_dequant_aggr_on_row - data_fp32_dequant_ref)
            > (atol + rtol * torch.abs(data_fp32_dequant_ref))
        )
        print(f"ref_fp32[idx_of_diff] = {data_fp32_dequant_ref[idx_of_diff]}")
        print(f"our_fp32_on_row[idx_of_diff] = {data_fp32_dequant_aggr_on_row[idx_of_diff]}")

        idx_of_diff = torch.where(
            torch.abs(data_fp32_dequant_aggr_on_col - data_fp32_dequant_ref)
            > (atol + rtol * torch.abs(data_fp32_dequant_ref))
        )
        print(f"ref_fp32[idx_of_diff] = {data_fp32_dequant_ref[idx_of_diff]}")
        print(f"our_fp32_on_col[idx_of_diff] = {data_fp32_dequant_aggr_on_col[idx_of_diff]}")

    elif bits == 2:
        idx_of_diff = torch.where(
            torch.abs(data_fp32_dequant_aggr_on_col - data_fp32_dequant_aggr_on_row)
            > (atol + rtol * torch.abs(data_fp32_dequant_aggr_on_row))
        )
        print("---------------------------------")
        print(f"data_int8_aggr_on_row = {data_int8_aggr_on_row}")
        print(f"our_fp32_on_row idx_of_diff = {idx_of_diff}")
        print(f"our_fp32_on_row[idx_of_diff] = {data_fp32_dequant_aggr_on_row[idx_of_diff]}")
        print(f"our_fp32_on_col[idx_of_diff] = {data_fp32_dequant_aggr_on_col[idx_of_diff]}")

        idx_of_diff = torch.where(
            torch.abs(data_fp32_dequant_aggr_on_col - data_fp32_dequant_for_all_procs)
            > (atol + rtol * torch.abs(data_fp32_dequant_for_all_procs))
        )
        print("---------------------------------")
        print(f"our_fp32_for_all_procs idx_of_diff = {idx_of_diff}")
        print(f"our_fp32_for_all_procs[idx_of_diff] = {data_fp32_dequant_for_all_procs[idx_of_diff]}")
        print(f"our_fp32_on_col[idx_of_diff] = {data_fp32_dequant_aggr_on_col[idx_of_diff]}")

        data_int8_aggr_on_row_list = []
        begin = time.perf_counter()
        for i in range(work_range_per_proc.size(0) - 1):
            data_int8_aggr_on_row, quantized_params_0 = run_quantization_cpu_quantize_tensor(
                data_fp32[work_range_per_proc[i]: work_range_per_proc[i+1]], 
                bits, 
                work_range_per_proc[i+1] - work_range_per_proc[i])
            data_int8_aggr_on_row_list.append(data_int8_aggr_on_row)
        end = time.perf_counter()

        print("time of loop aggr on row (ms) = {}".format((end - begin) * 1000.0))

        
        data_int8_aggr_on_row_for_all_procs = torch.cat(data_int8_aggr_on_row_list, dim=0)
        idx_of_diff = torch.where(
            torch.abs(data_int8_aggr_on_row_for_all_procs - data_int8_for_all_procs) 
            > (atol + rtol * torch.abs(data_int8_for_all_procs))
        )
        print("---------------------------------")
        print(f"our_fp32_for_all_procs idx_of_diff = {idx_of_diff}")
        print(f"data_int8_for_all_procs[idx_of_diff] = {data_int8_for_all_procs[idx_of_diff]}")
        print(f"data_int8_aggr_on_row_for_all_procs[idx_of_diff] = {data_int8_aggr_on_row_for_all_procs[idx_of_diff]}")
        print("---------------------------------")


    # print("ref_int8[idx_of_diff] = {}".format(data_int8_ref.int_repr()[idx_of_diff]))
    # print("our_int8[idx_of_diff] = {}".format(data_int8[idx_of_diff]))
    # assert(torch.allclose(data_int8_ref.int_repr(), data_int8))


def test_perf_for_quantize_tensor(data_fp32, min_val, zero_point, scale, bits, warmup, repeat):
    # check performance
    repeat = warmup + repeat
    quantization_time_ref = np.zeros(repeat)
    dequantization_time_ref = np.zeros(repeat)

    quantization_time_ours_on_row = np.zeros(repeat)
    dequantization_time_ours_on_row = np.zeros(repeat)

    quantization_time_ours_on_col = np.zeros(repeat)
    dequantization_time_ours_on_col = np.zeros(repeat)

    quantization_time_ours_for_all_procs = np.zeros(repeat)
    dequantization_time_ours_for_all_procs = np.zeros(repeat)

    for i in range(repeat):
        start = time.perf_counter()
        data_int8, quantized_params = run_quantization_cpu_quantize_tensor(data_fp32, bits, num_nodes)
        end = time.perf_counter()
        quantization_time_ours_on_row[i] = (end - start) * 1000.0

        start = time.perf_counter()
        run_quantization_cpu_dequantize_tensor(data_int8, quantized_params, bits)
        end = time.perf_counter()
        dequantization_time_ours_on_row[i] = (end - start) * 1000.0

        start = time.perf_counter()
        (
            data_int8_aggr_on_col,
            quantized_nodes_feat_range,
            quantized_params,
        ) = run_quantization_cpu_quantize_tensor_v1(data_fp32, bits)
        end = time.perf_counter()
        quantization_time_ours_on_col[i] = (end - start) * 1000.0

        start = time.perf_counter()
        run_quantization_cpu_dequantize_tensor_v1(
            data_int8_aggr_on_col, quantized_nodes_feat_range, quantized_params
        )
        end = time.perf_counter()
        dequantization_time_ours_on_col[i] = (end - start) * 1000.0

        start = time.perf_counter()
        data_int8_for_all_procs, quantized_params_for_all_procs, work_range_per_proc = run_quantization_cpu_quantize_tensor_for_all_procs(data_fp32, bits)
        end = time.perf_counter()
        quantization_time_ours_for_all_procs[i] = (end - start) * 1000.0

        start = time.perf_counter()
        run_quantization_cpu_dequantize_tensor_for_all_procs(data_int8_for_all_procs, quantized_params_for_all_procs, work_range_per_proc, bits)
        end = time.perf_counter()
        dequantization_time_ours_for_all_procs[i] = (end - start) * 1000.0

        if bits == 8 or bits == 4:
            start = time.perf_counter()
            data_int8_ref = torch.quantize_per_channel(data_fp32, scale, zero_point, 0, torch.quint4x2)
            end = time.perf_counter()
            quantization_time_ref[i] = (end - start) * 1000.0

            start = time.perf_counter()
            torch.dequantize(data_int8_ref)
            end = time.perf_counter()
            dequantization_time_ref[i] = (end - start) * 1000.0

    print("quantization_time_ref (ms): ", np.mean(quantization_time_ref[warmup:]))
    print("dequantization_time_ref (ms): ", np.mean(dequantization_time_ref[warmup:]))
    print("quantization_time_ours on row (ms): ", np.mean(quantization_time_ours_on_row[warmup:]))
    print("dequantization_time_ours on row (ms): ", np.mean(dequantization_time_ours_on_row[warmup:]))
    print("quantization_time_ours on col (ms): ", np.mean(quantization_time_ours_on_col[warmup:]))
    print("dequantization_time_ours on col (ms): ", np.mean(dequantization_time_ours_on_col[warmup:]))
    print("quantization_time_ours for all procs (ms): ", np.mean(quantization_time_ours_for_all_procs[warmup:]))
    print("dequantization_time_ours for all procs (ms): ", np.mean(dequantization_time_ours_for_all_procs[warmup:]))


def test_correctness_for_random_bits(data_fp32, data_fp32_dequant, node_dataformat_tensor, bits):
    selected_data_fp32 = data_fp32[torch.nonzero(node_dataformat_tensor == bits).squeeze()]
    selected_data_scale = (selected_data_fp32.max(dim=1)[0] - selected_data_fp32.min(dim=1)[0] + 10e-20) / (
        2**bits - 1
    )
    selected_data_zero_point = selected_data_fp32.min(dim=1)[0] / selected_data_scale * (-1)
    selected_data_intX_ref = run_torch_quantize_per_channel(
        selected_data_fp32, selected_data_scale, selected_data_zero_point, bits
    )
    selected_data_fp32_dequant_ref = run_torch_dequantize(selected_data_intX_ref)

    print("selected_data_fp32_dequant_ref.shape = ", selected_data_fp32_dequant_ref.shape)
    print("selected_data_fp32_dequant_ref = ", selected_data_fp32_dequant_ref)

    print("data_fp32_dequant.shape = ", data_fp32_dequant.shape)
    print("data_fp32_dequant = ", data_fp32_dequant)

    diff(
        selected_data_fp32_dequant_ref,
        data_fp32_dequant,
        "selected_data_fp32_dequant_ref",
        "data_fp32_dequant",
    )


def test_quantize_tensor_on_random_bits(data_fp32):
    num_nodes = data_fp32.size(0)
    bits_idx = torch.tensor([8.0, 4.0, 2.0], dtype=torch.float32)
    weights = torch.tensor([1.0, 1.0, 1.0])
    node_dataformat_tensor = torch.multinomial(weights, num_nodes, replacement=True)
    node_dataformat_tensor = bits_idx[node_dataformat_tensor]
    print("node_dataformat_tensor = ", node_dataformat_tensor)
    data_int8, quantized_nodes_feat_range, quantized_params = run_quantization_cpu_quantize_tensor_v1(
        data_fp32, node_dataformat_tensor
    )
    data_fp32_dequant = run_quantization_cpu_dequantize_tensor_v1(
        data_int8, quantized_nodes_feat_range, quantized_params
    )

    test_correctness_for_random_bits(
        data_fp32,
        data_fp32_dequant[torch.nonzero(node_dataformat_tensor.to(torch.int32) == 8).squeeze()],
        node_dataformat_tensor,
        8.0,
    )

    test_correctness_for_random_bits(
        data_fp32,
        data_fp32_dequant[torch.nonzero(node_dataformat_tensor.to(torch.int32) == 4).squeeze()],
        node_dataformat_tensor,
        4.0,
    )


if __name__ == "__main__":
    '''
    data_fp32 = torch.Tensor(
        [
            [1.0, 2.0, 3.0],
            [0.0, 2.0, 1.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0],
            [7.0, 8.0, 9.0],
            [8.0, 9.0, 10.0],
            [9.0, 10.0, 11.0],
            [10.0, 11.0, 12.0],
            [11.0, 12.0, 13.0],
        ]
    )
    '''
    data_fp32 = torch.randn((num_nodes, feat_len), dtype=torch.float32)
    # data_fp32 = torch.empty((num_nodes, feat_len), dtype=torch.float32)
    # for i in range(num_nodes):
    #     data_fp32[i] = torch.arange(feat_len, dtype=torch.float32)
    bits = 2

    min_val = data_fp32.min(dim=1)[0]
    scale = (data_fp32.max(dim=1)[0] - data_fp32.min(dim=1)[0] + 10e-20) / (2**bits - 1)
    zero_point = data_fp32.min(dim=1)[0] / scale * (-1)

    test_correctness_for_quantize_tensor(data_fp32, min_val, zero_point, scale, bits)
    test_perf_for_quantize_tensor(data_fp32, min_val, zero_point, scale, bits, 2, 10)
    # test_quantize_tensor_on_random_bits(data_fp32)
    # scale = (data_fp32.max(dim=1)[0] - data_fp32.min(dim=1)[0]) / (2**bits - 1)
    # zero_point = data_fp32.min(dim=1)[0] / scale * (-1)
    # data_int8_ref = run_torch_quantize_per_channel(data_fp32, scale, zero_point, bits)

    # (
    #     data_int8_our_torch,
    #     quantized_nodes_feat_range,
    # ) = run_quantization_cpu_quantize_tensor_v2(data_fp32, bits, zero_point, scale)

    # data_int8_ours, quantized_nodes_feat_range, quantized_params = run_quantization_cpu_quantize_tensor_v1(
    #     data_fp32, bits
    # )

    # idx_of_diff = diff(
    #     data_int8_ref.int_repr().view(num_nodes, -1),
    #     data_int8_our_torch.view(num_nodes, -1),
    #     "data_int8_ref",
    #     "data_int8_our_torch",
    # )

    # print(f"data_fp32[idx_of_diff] = {data_fp32[idx_of_diff]}")
    # print(f"scales[idx_of_diff[0]] = {scale[idx_of_diff[0]]}")
    # print(f"zero_points[idx_of_diff[0]] = {zero_point[idx_of_diff[0]]}")

    # if idx_of_diff[0].size(0) != 0:
    #     print(
    #         f"value computed on python = {data_fp32[idx_of_diff] / scale[idx_of_diff[0]] + zero_point[idx_of_diff[0]]}"
    #     )

    # idx_of_diff = diff(
    #     data_int8_our_torch.view(num_nodes, -1),
    #     data_int8_ours.view(num_nodes, -1),
    #     "data_int8_our_torch",
    #     "data_int8_ours",
    # )

    # print(f"data_fp32[idx_of_diff] = {data_fp32[idx_of_diff]}")

    # data_fp32_ours = run_quantization_cpu_dequantize_tensor_v1(
    #     data_int8_ours, quantized_nodes_feat_range, quantized_params
    # )

    # data_fp32_our_torch = run_quantization_cpu_dequantize_tensor_v2(
    #     data_int8_our_torch, quantized_nodes_feat_range, zero_point, scale, bits
    # )

    # idx_of_diff = diff(
    #     data_fp32_ours.view(num_nodes, -1),
    #     data_fp32_our_torch.view(num_nodes, -1),
    #     "data_fp32_ours",
    #     "data_fp32_our_torch",
    # )

    # print(f"scales[idx_of_diff[0]] = {scale[idx_of_diff[0]]}")
    # print(f"zero_points[idx_of_diff[0]] = {zero_point[idx_of_diff[0]]}")

    # test_correctness_for_quantize_tensor_v1(data_fp32, zero_point, scale, bits)
    # test_perf_for_quantize_tensor_v1(data_fp32, zero_point, scale, bits, 2, 10)
