#include <torch/extension.h>
#include <omp.h>
#include <stdint.h>
#include <cmath>

using torch::Tensor;

// a function to round a float to the integer based on the round-to-nearest-even rule
inline int32_t round_to_nearest_even(float x) {
    return round(x / 2.0) * 2;
}

inline int32_t divup(int32_t x, int32_t y) {
	return (x + y - 1) / y;
}

void divide_work(int* work_range, int total_work, int num_threads) {
	int chunk_size;
	int remain_work = total_work;
	work_range[0] = 0;
	for (int i = 0; i < num_threads; i++) {
		chunk_size = divup(remain_work, num_threads - i);
		work_range[i+1] = work_range[i] + chunk_size;
		remain_work -= chunk_size;
	}
	work_range[num_threads] = total_work;
}

void quantize_tensor(Tensor input, Tensor output, Tensor min, Tensor scale, int bits) {
    int vertex_num = input.size(0);
    int feat_len = input.size(1);

    TORCH_CHECK(8 % bits == 0);

    float* input_ptr = input.data_ptr<float>();
    float* min_ptr = min.data_ptr<float>();
    float* scale_ptr = scale.data_ptr<float>();

    uint8_t *output_ptr = output.data_ptr<uint8_t>();

    int elems_per_byte = 8 / bits;

    int max_num_threads = omp_get_max_threads();
    int vertex_num_round = vertex_num + (elems_per_byte - vertex_num % elems_per_byte) % elems_per_byte;
    int* work_range = new int[max_num_threads+1];
    divide_work(work_range, vertex_num_round / elems_per_byte, max_num_threads);

    #pragma omp parallel 
	{
        int tid = omp_get_thread_num();
        int row_begin = work_range[tid] * elems_per_byte;
        int row_end = std::min(work_range[tid+1] * elems_per_byte, vertex_num);

		// printf("tid = %d, row_begin = %d, row_end = %d\n", tid, row_begin, row_end);

        int num_rows = row_end - row_begin;

        int remainder_num_rows = num_rows % elems_per_byte;
        int divisible_num_rows = num_rows - remainder_num_rows;

        for (int i = 0; i < divisible_num_rows; i += elems_per_byte) {
            int row_idx = row_begin + i;
            for (int j = 0; j < feat_len; j++) {
                uint8_t packed_val = 0;
                for (int k = 0; k < elems_per_byte; k++) {
					// printf("idx in input_dir = %d, idx in min_ptr = %d\n", (row_idx + k) * feat_len + j, row_idx + k);
                    const int32_t val = 
                        std::nearbyint((input_ptr[(row_idx + k) * feat_len + j] - min_ptr[row_idx + k]) / scale_ptr[row_idx + k]);
                    packed_val |= (val << ((elems_per_byte-k-1) * bits));
                }
				// printf("idx in out = %d\n", row_idx / elems_per_byte * feat_len + j);
                output_ptr[row_idx / elems_per_byte * feat_len + j] = packed_val;
            }
        }

		if (remainder_num_rows > 0) {
        	for (int j = 0; j < feat_len; j++) {
        	    const int row_idx = row_begin + divisible_num_rows;
        	    uint8_t packed_val = 0;
        	    for (int k = 0; k < remainder_num_rows; k++) {
					// printf("k = %d, remainder_num_rows = %d, remider.\n", k, remainder_num_rows);
        	        const int32_t val = 
        	            round((input_ptr[(row_idx + k) * feat_len + j] - min_ptr[row_idx + k]) / scale_ptr[row_idx + k]);
        	        packed_val |= (val << ((elems_per_byte-k-1) * bits));
        	    }
        	    output_ptr[row_idx / elems_per_byte * feat_len + j] = packed_val;
        	}
		}
    }

    delete [] work_range;
    // // help me write a function to quantize each row in the input, the quantized function is as follows:
    // // output[i][j] = round(input[i][j] - min[i]) / scale[i]
    // #pragma omp parallel for
    // for (int i = 0; i < vertex_num; i++) {
    //     for (int j = 0; j < feat_len; j += step) {
    //         uint8_t packed_val = 0;
    //         for (int k = 0; k < step; k++) {
    //             // const int32_t val = round((input_ptr[i*feat_len + j + k] - min_ptr[i]) / scale_ptr[i]);
    //             const int32_t val = std::nearbyint((input_ptr[i*feat_len + j + k] - min_ptr[i]) / scale_ptr[i]);
    //             // const int32_t val = std::nearbyint((input_ptr[i*feat_len + j + k] / scale_ptr[i]) + zero_ptr[i]);
    //             packed_val |= (val << (step-k-1) * bits);
    //         }
    //         output_ptr[i*feat_len/step + j/step] = packed_val;
    //     }
    // }
}

// void dequantize_tensor(Tensor input, Tensor output, Tensor min, Tensor scale, Tensor zero_point, int64_t vertex_num, int64_t feat_len, int bits) {
//     TORCH_CHECK(8 % bits == 0);

//     uint8_t* input_ptr = input.data_ptr<uint8_t>();
//     float* min_ptr = min.data_ptr<float>();
//     float* scale_ptr = scale.data_ptr<float>();

// 	float* zero_ptr = zero_point.data_ptr<float>();

//     float *output_ptr = output.data_ptr<float>();

//     int step = 8 / bits;
//     int mask = ((1 << bits) - 1);

//     int divisible_feat_len = feat_len - feat_len % step;
//     int remainder_feat_len = feat_len % step;

//     // help me write a function to dequantize each row in the input, the dequantized function is as follows:
//     // output[i][j] = input[i][j] * scale[i] + min[i]
//     #pragma omp parallel for
//     for (int i = 0; i < vertex_num; i++) {
//         // handle the divisible part
//         for (int j = 0; j < divisible_feat_len; j += step) {
//             uint8_t packed_val = input_ptr[i*feat_len/step + j/step];
//             for (int k = 0; k < step; k++) {
//                 const float val = static_cast<float>((packed_val >> (step-k-1) * bits) & mask);
//                 output_ptr[i*feat_len + j + k] = val * scale_ptr[i] + min_ptr[i];
//                 // output_ptr[i*feat_len + j + k] = (val - zero_ptr[i]) * scale_ptr[i];
//             }
//         }

//         // handle the remainder
//         uint8_t packed_val = input_ptr[i*feat_len/step + divisible_feat_len/step];
//         for (int k = 0; k < remainder_feat_len; k++) {
//             const float val = static_cast<float>((packed_val >> (step-k-1) * bits) & mask);
//             output_ptr[i*feat_len + divisible_feat_len + k] = val * scale_ptr[i] + min_ptr[i];
//         }
//     }
// }

void dequantize_tensor(Tensor input, Tensor output, Tensor min, Tensor scale, int bits) {
    TORCH_CHECK(8 % bits == 0);

    uint8_t* input_ptr = input.data_ptr<uint8_t>();
    float* min_ptr = min.data_ptr<float>();
    float* scale_ptr = scale.data_ptr<float>();
    float *output_ptr = output.data_ptr<float>();

    int vertex_num = output.size(0);
    int feat_len = output.size(1);

    int mask = ((1 << bits) - 1);

    int elems_per_byte = 8 / bits;
    int max_num_threads = omp_get_max_threads();
    int vertex_num_round = vertex_num + (elems_per_byte - vertex_num % elems_per_byte) % elems_per_byte;
    int* work_range = new int[max_num_threads+1];
    divide_work(work_range, vertex_num_round / elems_per_byte, max_num_threads);

    #pragma omp parallel 
	{
        int tid = omp_get_thread_num();
        int row_begin = work_range[tid] * elems_per_byte;
        int row_end = std::min(work_range[tid+1] * elems_per_byte, vertex_num);

        int num_rows = row_end - row_begin;

        int remainder_num_rows = num_rows % elems_per_byte;
        int divisible_num_rows = num_rows - remainder_num_rows;

        for (int i = 0; i < divisible_num_rows; i += elems_per_byte) {
            const int row_idx = row_begin + i;
            for (int j = 0; j < feat_len; j++) {
                const uint8_t packed_val = input_ptr[row_idx / elems_per_byte * feat_len + j];
                for (int k = 0; k < elems_per_byte; k++) {
                    const float val = static_cast<float>((packed_val >> ((elems_per_byte-k-1) * bits)) & mask);
                    output_ptr[(row_idx + k) * feat_len + j] = val * scale_ptr[row_idx + k] + min_ptr[row_idx + k];
                }
            }
        }

		if (remainder_num_rows > 0) {
        	for (int j = 0; j < feat_len; j++) {
        	    const int row_idx = row_begin + divisible_num_rows;
        	    const uint8_t packed_val = input_ptr[row_idx / elems_per_byte * feat_len + j];
        	    for (int k =0; k < remainder_num_rows; k++) {
        	        const float val = static_cast<float>((packed_val >> ((elems_per_byte-k-1) * bits)) & mask);
        	        output_ptr[(row_idx + k) * feat_len + j] = val * scale_ptr[row_idx + k] + min_ptr[row_idx + k];
        	    }
        	}
    	}


	}

    delete [] work_range;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_tensor", &quantize_tensor);
    m.def("dequantize_tensor", &dequantize_tensor);
}
