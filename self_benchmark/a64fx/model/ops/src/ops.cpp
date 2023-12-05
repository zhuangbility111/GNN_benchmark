
#include <torch/extension.h>
#include "quantization_x86.h"
#include "spmm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_tensor", &quantize_tensor);
    m.def("dequantize_tensor", &dequantize_tensor);
    m.def("quantize_tensor_v1", &quantize_tensor_v1);
    m.def("dequantize_tensor_v1", &dequantize_tensor_v1);
    m.def("quantize_tensor_v2_torch", &quantize_tensor_v2_torch);
    m.def("dequantize_tensor_v2_torch", &dequantize_tensor_v2_torch);
    m.def("spmm", &spmm_cpu_optimized_no_tile_v1);
}