#include <torch/extension.h>

using torch::Tensor;

void quantize_tensor(Tensor input, Tensor output, Tensor min, Tensor scale, int bits);
void dequantize_tensor(Tensor input, Tensor output, Tensor min, Tensor scale, int bits);
void quantize_tensor_v1(Tensor input, Tensor output, Tensor quantized_nodes_feat_range, Tensor quantized_params);
void dequantize_tensor_v1(Tensor input, Tensor output, Tensor quantized_nodes_feat_range, Tensor quantized_params);
void quantize_tensor_v2_torch(Tensor input, Tensor output,
                              Tensor quantized_nodes_feat_range, Tensor zero_points, Tensor scales, int num_bits);
void dequantize_tensor_v2_torch(Tensor input, Tensor output,
                                Tensor quantized_nodes_feat_range, Tensor zero_points, Tensor scales, int num_bits);