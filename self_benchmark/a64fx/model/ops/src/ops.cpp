
#include <torch/extension.h>
#include "spmm.h"

#ifdef __ARM_FEATURE_SVE /* __ARM_FEATURE_SVE */
#include "quantization_arm.h"
#elif __AVX512F__ /* AVX512 */
#include "quantization_x86.h"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_tensor", &quantize_tensor);
    m.def("dequantize_tensor", &dequantize_tensor);
    m.def("quantize_tensor_v1", &quantize_tensor_v1);
    m.def("dequantize_tensor_v1", &dequantize_tensor_v1);
#ifdef __ARM_FEATURE_SVE /* __ARM_FEATURE_SVE */
    m.def("quantize_tensor_for_all_procs", &quantize_tensor_for_all_procs);
    m.def("dequantize_tensor_for_all_procs", &dequantize_tensor_for_all_procs);
#elif __AVX512F__ /* AVX512 */
    m.def("quantize_tensor_v2_torch", &quantize_tensor_v2_torch);
    m.def("dequantize_tensor_v2_torch", &dequantize_tensor_v2_torch);
#endif
    m.def("spmm", &spmm_cpu_optimized_no_tile_v1);
}