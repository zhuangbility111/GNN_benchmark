#include <torch/extension.h>

using torch::Tensor;

void quantize_tensor(Tensor input, Tensor output, Tensor quantized_params, int bits);
void dequantize_tensor(Tensor input, Tensor output, Tensor quantized_params, int bits);
void quantize_tensor_v1(Tensor input, Tensor output, Tensor quantized_nodes_feat_range, Tensor quantized_params);
void dequantize_tensor_v1(Tensor input, Tensor output, Tensor quantized_nodes_feat_range, Tensor quantized_params);
void quantize_tensor_for_all_procs(Tensor input, Tensor output, Tensor quantized_params, Tensor work_range_per_proc, int num_procs, int bits);
void dequantize_tensor_for_all_procs(Tensor input, Tensor output, Tensor dequantized_params, Tensor work_range_per_proc, int num_procs, int bits);
