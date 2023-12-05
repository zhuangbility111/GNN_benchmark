#include <torch/extension.h>

std::tuple<torch::Tensor, torch::optional<torch::Tensor>> spmm_cpu_optimized_no_tile_v1(
    torch::Tensor rowptr, torch::Tensor col, torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
    torch::Tensor out, int64_t sparse_rows, torch::Tensor parallel_row_split,
    torch::Tensor parallel_col_split);