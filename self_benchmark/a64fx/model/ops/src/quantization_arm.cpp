#include <torch/extension.h>
#include <omp.h>
#include <stdint.h>
#include <cmath>
#include <vector>
#include <list>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#define VEC_LEN 16
#endif /* __ARM_FEATURE_SVE */

using torch::Tensor;

inline int32_t divup(int32_t x, int32_t y)
{
    return (x + y - 1) / y;
}

#define QUANTIZED_PARAMS_SIZE 2

void divide_work(int *work_range, int total_work, int num_threads)
{
    int chunk_size;
    int remain_work = total_work;
    work_range[0] = 0;
    for (int i = 0; i < num_threads; i++)
    {
        chunk_size = divup(remain_work, num_threads - i);
        work_range[i + 1] = work_range[i] + chunk_size;
        remain_work -= chunk_size;
    }
    work_range[num_threads] = total_work;
}

void divide_work_v1(int *work_range_per_proc, int num_threads, int num_procs)
{
    vector<list<int>> work_list(num_threads);
    int chunk_size;
    int cur_procs = 0;
    int j = 0;
    for (int i = 0; i < num_threads; i++)
    {
        chunk_size = divup(work_range_per_proc[num_procs + 1] - work_range_per_proc[j], num_threads - i);
        int remain_work = chunk_size;
        int while (remain_work > 0)
        {
            if (work_range_per_proc[j + 1] - work_range_per_proc[j] <= remain_work)
            {
                work_list[i].push_back(work_range_per_proc[j + 1]);
            }
        }
    }
}

void quantize_tensor(Tensor input, Tensor output, Tensor quantized_params, int bits)
{
    int vertex_num = input.size(0);
    int feat_len = input.size(1);

    TORCH_CHECK(8 % bits == 0);

    quantized_params = quantized_params.contiguous();
    float *quantized_params_ptr = quantized_params.data_ptr<float>();

    float *input_ptr = input.data_ptr<float>();
    uint8_t *output_ptr = output.data_ptr<uint8_t>();

    // int elems_per_byte = 8 / bits;
    const int ELEMS_PER_BYTE = 4;
    const int BITS = 2;

    int max_num_threads = omp_get_max_threads();
    int vertex_num_round = vertex_num + (ELEMS_PER_BYTE - vertex_num % ELEMS_PER_BYTE) % ELEMS_PER_BYTE;
    int *work_range = new int[max_num_threads + 1];
    divide_work(work_range, vertex_num_round / ELEMS_PER_BYTE, max_num_threads);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int row_begin = work_range[tid] * ELEMS_PER_BYTE;
        int row_end = std::min(work_range[tid + 1] * ELEMS_PER_BYTE, vertex_num);

        // printf("tid = %d, row_begin = %d, row_end = %d\n", tid, row_begin, row_end);

        int num_rows = row_end - row_begin;

        int remainder_num_rows = num_rows % ELEMS_PER_BYTE;
        int divisible_num_rows = num_rows - remainder_num_rows;
        for (int i = 0; i < divisible_num_rows; i += ELEMS_PER_BYTE)
        {
            for (int j = 0; j < ELEMS_PER_BYTE; j++)
            {
                int row_idx = row_begin + i + j;
                // get max and min of this row
                float max_val = input_ptr[row_idx * feat_len];
                float min_val = input_ptr[row_idx * feat_len];
                for (int k = 1; k < feat_len; k++)
                {
                    float val = input_ptr[row_idx * feat_len + k];
                    if (val > max_val)
                        max_val = val;
                    if (val < min_val)
                        min_val = val;
                    // max_val = std::max(max_val, val);
                    // min_val = std::min(min_val, val);
                }

                float zero_point = min_val;
                float scale = (max_val - zero_point) / ((1 << BITS) - 1);
                // float scale = (max_val - zero_point);
                if (scale == 0)
                    scale = 1.0f;

                quantized_params_ptr[row_idx * QUANTIZED_PARAMS_SIZE] = zero_point;
                quantized_params_ptr[row_idx * QUANTIZED_PARAMS_SIZE + 1] = scale;
            }

            int row_idx = row_begin + i;
            for (int j = 0; j < feat_len; j += VEC_LEN)
            {
                svbool_t pg_f32 = svwhilelt_b32(j, feat_len);
                svuint32_t v_res = svdup_n_u32(0);

                svfloat32_t v_input_0 = svld1(pg_f32, input_ptr + (row_idx)*feat_len + j);
                svfloat32_t v_input_1 = svld1(pg_f32, input_ptr + (row_idx + 1) * feat_len + j);
                svfloat32_t v_input_2 = svld1(pg_f32, input_ptr + (row_idx + 2) * feat_len + j);
                svfloat32_t v_input_3 = svld1(pg_f32, input_ptr + (row_idx + 3) * feat_len + j);

                int quantized_params_idx = row_idx * QUANTIZED_PARAMS_SIZE;

                __builtin_prefetch(&(input_ptr[(row_idx + 8) * feat_len]), 0, 3);
                __builtin_prefetch(&(input_ptr[(row_idx + 9) * feat_len]), 0, 3);
                __builtin_prefetch(&(input_ptr[(row_idx + 10) * feat_len]), 0, 3);
                __builtin_prefetch(&(input_ptr[(row_idx + 11) * feat_len]), 0, 3);

                svfloat32_t v_zero_point_0 = svdup_n_f32(quantized_params_ptr[quantized_params_idx]);
                svfloat32_t v_scale_0 = svdup_n_f32(quantized_params_ptr[quantized_params_idx + 1]);
                svfloat32_t v_zero_point_1 = svdup_n_f32(quantized_params_ptr[quantized_params_idx + 2]);
                svfloat32_t v_scale_1 = svdup_n_f32(quantized_params_ptr[quantized_params_idx + 3]);
                svfloat32_t v_zero_point_2 = svdup_n_f32(quantized_params_ptr[quantized_params_idx + 4]);
                svfloat32_t v_scale_2 = svdup_n_f32(quantized_params_ptr[quantized_params_idx + 5]);
                svfloat32_t v_zero_point_3 = svdup_n_f32(quantized_params_ptr[quantized_params_idx + 6]);
                svfloat32_t v_scale_3 = svdup_n_f32(quantized_params_ptr[quantized_params_idx + 7]);

                svfloat32_t v_quantized_0 = svdiv_f32_z(pg_f32, svsub_f32_z(pg_f32, v_input_0, v_zero_point_0), v_scale_0);
                svfloat32_t v_quantized_1 = svdiv_f32_z(pg_f32, svsub_f32_z(pg_f32, v_input_1, v_zero_point_1), v_scale_1);
                svfloat32_t v_quantized_2 = svdiv_f32_z(pg_f32, svsub_f32_z(pg_f32, v_input_2, v_zero_point_2), v_scale_2);
                svfloat32_t v_quantized_3 = svdiv_f32_z(pg_f32, svsub_f32_z(pg_f32, v_input_3, v_zero_point_3), v_scale_3);

                v_quantized_0 = svrintn_f32_z(pg_f32, v_quantized_0);
                v_quantized_1 = svrintn_f32_z(pg_f32, v_quantized_1);
                v_quantized_2 = svrintn_f32_z(pg_f32, v_quantized_2);
                v_quantized_3 = svrintn_f32_z(pg_f32, v_quantized_3);

                svuint32_t v_int_quantized_0 = svcvt_u32_f32_z(pg_f32, v_quantized_0);
                svuint32_t v_int_quantized_1 = svcvt_u32_f32_z(pg_f32, v_quantized_1);
                svuint32_t v_int_quantized_2 = svcvt_u32_f32_z(pg_f32, v_quantized_2);
                svuint32_t v_int_quantized_3 = svcvt_u32_f32_z(pg_f32, v_quantized_3);

                v_int_quantized_0 = svlsl_n_u32_z(pg_f32, v_int_quantized_0, (ELEMS_PER_BYTE - 1) * BITS);
                v_int_quantized_1 = svlsl_n_u32_z(pg_f32, v_int_quantized_1, (ELEMS_PER_BYTE - 2) * BITS);
                v_int_quantized_2 = svlsl_n_u32_z(pg_f32, v_int_quantized_2, (ELEMS_PER_BYTE - 3) * BITS);
                v_int_quantized_3 = svlsl_n_u32_z(pg_f32, v_int_quantized_3, (ELEMS_PER_BYTE - 4) * BITS);

                v_res = svorr_z(pg_f32, v_res, v_int_quantized_0);
                v_res = svorr_z(pg_f32, v_res, v_int_quantized_1);
                v_res = svorr_z(pg_f32, v_res, v_int_quantized_2);
                v_res = svorr_z(pg_f32, v_res, v_int_quantized_3);

                svst1b_u32(pg_f32, output_ptr + row_idx / ELEMS_PER_BYTE * feat_len + j, v_res);
            }
        }

        if (remainder_num_rows > 0)
        {
            for (int j = 0; j < remainder_num_rows; j++)
            {
                int row_idx = row_begin + divisible_num_rows + j;
                // get max and min of this row
                float max_val = input_ptr[row_idx * feat_len];
                float min_val = input_ptr[row_idx * feat_len];
                for (int k = 1; k < feat_len; k++)
                {
                    float val = input_ptr[row_idx * feat_len + k];
                    if (val > max_val)
                        max_val = val;
                    if (val < min_val)
                        min_val = val;
                }

                float zero_point = min_val;
                float scale = (max_val - zero_point) / ((1 << BITS) - 1);
                if (scale == 0)
                    scale = 1.0f;

                quantized_params_ptr[row_idx * QUANTIZED_PARAMS_SIZE] = zero_point;
                quantized_params_ptr[row_idx * QUANTIZED_PARAMS_SIZE + 1] = scale;
            }

            int row_idx = row_begin + divisible_num_rows;
            for (int j = 0; j < feat_len; j += VEC_LEN)
            {
                svbool_t pg_f32 = svwhilelt_b32(j, feat_len);
                svuint32_t v_res = svdup_n_u32(0);

                for (int k = 0; k < remainder_num_rows; k++)
                {
                    svfloat32_t v_input = svld1(pg_f32, input_ptr + (row_idx + k) * feat_len + j);
                    svfloat32_t v_zero_point = svdup_n_f32(quantized_params_ptr[(row_idx + k) * QUANTIZED_PARAMS_SIZE]);
                    svfloat32_t v_scale = svdup_n_f32(quantized_params_ptr[(row_idx + k) * QUANTIZED_PARAMS_SIZE + 1]);
                    svuint32_t v_quantized = svcvt_u32_f32_z(pg_f32, svrintn_f32_z(pg_f32, svdiv_f32_z(pg_f32, svsub_f32_z(pg_f32, v_input, v_zero_point), v_scale)));
                    v_quantized = svlsl_n_u32_z(pg_f32, v_quantized, (ELEMS_PER_BYTE - k - 1) * BITS);
                    v_res = svorr_z(pg_f32, v_res, v_quantized);
                }

                svst1b_u32(pg_f32, output_ptr + row_idx / ELEMS_PER_BYTE * feat_len + j, v_res);
            }
        }

        // int remainder_num_rows = num_rows % elems_per_byte;
        // int divisible_num_rows = num_rows - remainder_num_rows;
        // for (int i = 0; i < divisible_num_rows; i += elems_per_byte)
        // {
        //     // get zero point and scale for each row
        //     for (int j = i; j < i + elems_per_byte; j++)
        //     {
        //         int row_idx = row_begin + j;
        //         // get max and min of this row
        //         float max_val = input_ptr[row_idx * feat_len];
        //         float min_val = input_ptr[row_idx * feat_len];
        //         for (int k = 1; k < feat_len; k++)
        //         {
        //             float val = input_ptr[row_idx * feat_len + k];
        //             max_val = std::max(max_val, val);
        //             min_val = std::min(min_val, val);
        //         }

        //         float zero_point = min_val;
        //         float scale = (max_val - zero_point) / ((1 << bits) - 1);
        //         if (scale == 0)
        //             scale = 1.0f;

        //         quantized_params_ptr[row_idx * QUANTIZED_PARAMS_SIZE] = zero_point;
        //         quantized_params_ptr[row_idx * QUANTIZED_PARAMS_SIZE + 1] = scale;
        //     }

        //     // quantize each row
        //     int row_idx = row_begin + i;
        //     for (int j = 0; j < feat_len; j++)
        //     {
        //         uint8_t packed_val = 0;
        //         for (int k = 0; k < elems_per_byte; k++)
        //         {
        //             const int32_t val =
        //                 // std::nearbyint((input_ptr[(row_idx + k) * feat_len + j] - quantized_params_ptr[(row_idx + k) * QUANTIZED_PARAMS_SIZE]) / quantized_params_ptr[(row_idx + k) * QUANTIZED_PARAMS_SIZE + 1]);
        //                 lrintf((input_ptr[(row_idx + k) * feat_len + j] - quantized_params_ptr[(row_idx + k) * QUANTIZED_PARAMS_SIZE]) / quantized_params_ptr[(row_idx + k) * QUANTIZED_PARAMS_SIZE + 1]);
        //             packed_val |= (val << ((elems_per_byte - k - 1) * bits));
        //         }
        //         // printf("idx in out = %d, packed_val = %u\n", row_idx / elems_per_byte * feat_len + j, packed_val);
        //         output_ptr[row_idx / elems_per_byte * feat_len + j] = packed_val;
        //     }
        // }

        // if (remainder_num_rows > 0)
        // {
        //     // get zero point and scale for each row
        //     for (int j = divisible_num_rows; j < divisible_num_rows + remainder_num_rows; j++)
        //     {
        //         int row_idx = row_begin + j;
        //         // get max and min of this row
        //         float max_val = input_ptr[row_idx * feat_len];
        //         float min_val = input_ptr[row_idx * feat_len];
        //         for (int k = 1; k < feat_len; k++)
        //         {
        //             float val = input_ptr[row_idx * feat_len + k];
        //             max_val = std::max(max_val, val);
        //             min_val = std::min(min_val, val);
        //         }

        //         float zero_point = min_val;
        //         float scale = (max_val - zero_point) / ((1 << bits) - 1);
        //         if (scale == 0)
        //             scale = 1.0f;

        //         quantized_params_ptr[row_idx * QUANTIZED_PARAMS_SIZE] = zero_point;
        //         quantized_params_ptr[row_idx * QUANTIZED_PARAMS_SIZE + 1] = scale;
        //     }

        //     for (int j = 0; j < feat_len; j++)
        //     {
        //         const int row_idx = row_begin + divisible_num_rows;
        //         uint8_t packed_val = 0;
        //         for (int k = 0; k < remainder_num_rows; k++)
        //         {
        //             // printf("k = %d, remainder_num_rows = %d, remider.\n", k, remainder_num_rows);
        //             const int32_t val =
        //                 // std::nearbyint((input_ptr[(row_idx + k) * feat_len + j] - quantized_params_ptr[(row_idx + k) * QUANTIZED_PARAMS_SIZE]) / quantized_params_ptr[(row_idx + k) * QUANTIZED_PARAMS_SIZE + 1]);
        //                 lrintf((input_ptr[(row_idx + k) * feat_len + j] - quantized_params_ptr[(row_idx + k) * QUANTIZED_PARAMS_SIZE]) / quantized_params_ptr[(row_idx + k) * QUANTIZED_PARAMS_SIZE + 1]);
        //             packed_val |= (val << ((elems_per_byte - k - 1) * bits));
        //         }
        //         output_ptr[row_idx / elems_per_byte * feat_len + j] = packed_val;
        //     }
        // }
    }

    delete[] work_range;
}

void dequantize_tensor(Tensor input, Tensor output, Tensor dequantized_params, int bits)
{
    TORCH_CHECK(8 % bits == 0);

    uint8_t *input_ptr = input.data_ptr<uint8_t>();
    float *dequantized_params_ptr = dequantized_params.data_ptr<float>();
    float *output_ptr = output.data_ptr<float>();

    int vertex_num = output.size(0);
    int feat_len = output.size(1);

    const int BITS = 2;
    const int ELEMS_PER_BYTE = 4;
    int mask = ((1 << BITS) - 1);
    int max_num_threads = omp_get_max_threads();
    int vertex_num_round = vertex_num + (ELEMS_PER_BYTE - vertex_num % ELEMS_PER_BYTE) % ELEMS_PER_BYTE;
    int *work_range = new int[max_num_threads + 1];
    divide_work(work_range, vertex_num_round / ELEMS_PER_BYTE, max_num_threads);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int row_begin = work_range[tid] * ELEMS_PER_BYTE;
        int row_end = std::min(work_range[tid + 1] * ELEMS_PER_BYTE, vertex_num);

        int num_rows = row_end - row_begin;

        int remainder_num_rows = num_rows % ELEMS_PER_BYTE;
        int divisible_num_rows = num_rows - remainder_num_rows;

        svuint32_t v_mask = svdup_n_u32(mask);
        for (int i = 0; i < divisible_num_rows; i += ELEMS_PER_BYTE)
        {
            const int row_idx = row_begin + i;
            for (int j = 0; j < feat_len; j += VEC_LEN)
            {
                svbool_t pg_f32 = svwhilelt_b32(j, feat_len);
                svuint32_t v_packed_val = svld1ub_u32(pg_f32, input_ptr + row_idx / ELEMS_PER_BYTE * feat_len + j);

                svuint32_t v_input_0 = svlsr_n_u32_z(pg_f32, v_packed_val, (ELEMS_PER_BYTE - 1) * BITS);
                svuint32_t v_input_1 = svlsr_n_u32_z(pg_f32, v_packed_val, (ELEMS_PER_BYTE - 2) * BITS);
                svuint32_t v_input_2 = svlsr_n_u32_z(pg_f32, v_packed_val, (ELEMS_PER_BYTE - 3) * BITS);
                svuint32_t v_input_3 = svlsr_n_u32_z(pg_f32, v_packed_val, (ELEMS_PER_BYTE - 4) * BITS);

                v_input_0 = svand_z(pg_f32, v_input_0, v_mask);
                v_input_1 = svand_z(pg_f32, v_input_1, v_mask);
                v_input_2 = svand_z(pg_f32, v_input_2, v_mask);
                v_input_3 = svand_z(pg_f32, v_input_3, v_mask);

                int quantized_params_idx = row_idx * QUANTIZED_PARAMS_SIZE;

                svfloat32_t v_zero_point_0 = svdup_n_f32(dequantized_params_ptr[quantized_params_idx]);
                svfloat32_t v_scale_0 = svdup_n_f32(dequantized_params_ptr[quantized_params_idx + 1]);
                svfloat32_t v_zero_point_1 = svdup_n_f32(dequantized_params_ptr[quantized_params_idx + 2]);
                svfloat32_t v_scale_1 = svdup_n_f32(dequantized_params_ptr[quantized_params_idx + 3]);
                svfloat32_t v_zero_point_2 = svdup_n_f32(dequantized_params_ptr[quantized_params_idx + 4]);
                svfloat32_t v_scale_2 = svdup_n_f32(dequantized_params_ptr[quantized_params_idx + 5]);
                svfloat32_t v_zero_point_3 = svdup_n_f32(dequantized_params_ptr[quantized_params_idx + 6]);
                svfloat32_t v_scale_3 = svdup_n_f32(dequantized_params_ptr[quantized_params_idx + 7]);

                svfloat32_t v_dequantized_0 = svcvt_f32_u32_z(pg_f32, v_input_0);
                svfloat32_t v_dequantized_1 = svcvt_f32_u32_z(pg_f32, v_input_1);
                svfloat32_t v_dequantized_2 = svcvt_f32_u32_z(pg_f32, v_input_2);
                svfloat32_t v_dequantized_3 = svcvt_f32_u32_z(pg_f32, v_input_3);

                v_dequantized_0 = svmla_f32_z(pg_f32, v_zero_point_0, v_dequantized_0, v_scale_0);
                v_dequantized_1 = svmla_f32_z(pg_f32, v_zero_point_1, v_dequantized_1, v_scale_1);
                v_dequantized_2 = svmla_f32_z(pg_f32, v_zero_point_2, v_dequantized_2, v_scale_2);
                v_dequantized_3 = svmla_f32_z(pg_f32, v_zero_point_3, v_dequantized_3, v_scale_3);

                svst1_f32(pg_f32, output_ptr + (row_idx)*feat_len + j, v_dequantized_0);
                svst1_f32(pg_f32, output_ptr + (row_idx + 1) * feat_len + j, v_dequantized_1);
                svst1_f32(pg_f32, output_ptr + (row_idx + 2) * feat_len + j, v_dequantized_2);
                svst1_f32(pg_f32, output_ptr + (row_idx + 3) * feat_len + j, v_dequantized_3);
                // const uint8_t packed_val = input_ptr[row_idx / ELEMS_PER_BYTE * feat_len + j];
                // for (int k = 0; k < ELEMS_PER_BYTE; k++)
                // {
                //     const float val = static_cast<float>((packed_val >> ((ELEMS_PER_BYTE - k - 1) * bits)) & mask);
                //     output_ptr[(row_idx + k) * feat_len + j] = val * dequantized_params_ptr[(row_idx + k) * QUANTIZED_PARAMS_SIZE + 1] + dequantized_params_ptr[(row_idx + k) * QUANTIZED_PARAMS_SIZE];
                // }
            }
        }

        if (remainder_num_rows > 0)
        {
            const int row_idx = row_begin + divisible_num_rows;
            for (int j = 0; j < feat_len; j += VEC_LEN)
            {
                svbool_t pg_f32 = svwhilelt_b32(j, feat_len);
                svuint32_t v_packed_val = svld1ub_u32(pg_f32, input_ptr + row_idx / ELEMS_PER_BYTE * feat_len + j);

                for (int k = 0; k < remainder_num_rows; k++)
                {
                    svuint32_t v_input = svlsr_n_u32_z(pg_f32, v_packed_val, (ELEMS_PER_BYTE - k - 1) * BITS);
                    v_input = svand_z(pg_f32, v_input, v_mask);

                    int quantized_params_idx = (row_idx + k) * QUANTIZED_PARAMS_SIZE;

                    svfloat32_t v_zero_point = svdup_n_f32(dequantized_params_ptr[quantized_params_idx]);
                    svfloat32_t v_scale = svdup_n_f32(dequantized_params_ptr[quantized_params_idx + 1]);

                    svfloat32_t v_dequantized = svcvt_f32_u32_z(pg_f32, v_input);
                    v_dequantized = svmla_f32_z(pg_f32, v_zero_point, v_dequantized, v_scale);

                    svst1_f32(pg_f32, output_ptr + (row_idx + k) * feat_len + j, v_dequantized);
                }
                // const uint8_t packed_val = input_ptr[row_idx / ELEMS_PER_BYTE * feat_len + j];
                // for (int k = 0; k < remainder_num_rows; k++)
                // {
                //     const float val = static_cast<float>((packed_val >> ((ELEMS_PER_BYTE - k - 1) * bits)) & mask);
                //     output_ptr[(row_idx + k) * feat_len + j] = val * dequantized_params_ptr[(row_idx + k) * QUANTIZED_PARAMS_SIZE + 1] + dequantized_params_ptr[(row_idx + k) * QUANTIZED_PARAMS_SIZE];
                // }
            }
        }
    }

    delete[] work_range;
}

void quantize_tensor_v1(Tensor input, Tensor output,
                        Tensor quantized_nodes_feat_range, Tensor quantized_params)
{
    int vertex_num = input.size(0);
    int feat_len = input.size(1);

    quantized_nodes_feat_range = quantized_nodes_feat_range.contiguous();
    // nodes_num_bits_tensor = nodes_num_bits_tensor.contiguous();
    // zero_points = zero_points.contiguous();
    // scales = scales.contiguous();
    quantized_params = quantized_params.contiguous();

    float *input_ptr = input.data_ptr<float>();
    int64_t *quantized_nodes_feat_range_ptr = quantized_nodes_feat_range.data_ptr<int64_t>();
    // int *nodes_num_bits_ptr = nodes_num_bits_tensor.data_ptr<int>(); // [num_bits]
    // float *zero_points_ptr = zero_points.data_ptr<float>();
    // float *scales_ptr = scales.data_ptr<float>();
    float *quantized_params_ptr = quantized_params.data_ptr<float>(); // [num_bits, zero_point, scale]

    uint8_t *output_ptr = output.data_ptr<uint8_t>();

    // std::mt19937 rng{std::random_device{}()};
    // std::uniform_real_distribution<float> dist{0.0, 1.0};

#pragma omp parallel for
    for (int i = 0; i < vertex_num; i++)
    {
        // get max and min of this row
        float max_val = input_ptr[i * feat_len];
        float min_val = input_ptr[i * feat_len];
        for (int j = 1; j < feat_len; j++)
        {
            float val = input_ptr[i * feat_len + j];
            if (val > max_val)
                max_val = val;
            if (val < min_val)
                min_val = val;
        }

        // int num_bits = nodes_num_bits_ptr[i];
        // int num_bits = 8;
        int num_bits = static_cast<int>(quantized_params_ptr[i * 3]);
        float zero_point = min_val;
        float scale = (max_val - zero_point) / ((1 << num_bits) - 1);
        if (scale == 0)
            scale = 1.0f;

        quantized_params_ptr[i * 3 + 1] = zero_point;
        quantized_params_ptr[i * 3 + 2] = scale;

        const int elems_per_byte = 8 / num_bits;
        for (int j = 0; j < feat_len; j += elems_per_byte)
        {
            const int remain_feat_len = std::min(elems_per_byte, feat_len - j);
            uint8_t packed_val = 0;
            for (int k = 0; k < remain_feat_len; k++)
            {
                const int32_t val =
                    lrintf((input_ptr[i * feat_len + j + k] - zero_point) / scale);
                // float noise = dist(rng);
                // const int32_t val =
                //     static_cast<int>((input_ptr[i * feat_len + j + k] - zero_point) / scale + noise);
                packed_val |= (val << ((elems_per_byte - k - 1) * num_bits));
            }
            output_ptr[quantized_nodes_feat_range_ptr[i] + j / elems_per_byte] = packed_val;
        }
    }
}

void dequantize_tensor_v1(Tensor input, Tensor output,
                          Tensor quantized_nodes_feat_range, Tensor quantized_params)
{
    int vertex_num = output.size(0);
    int unpacked_feat_len = output.size(1);

    quantized_nodes_feat_range = quantized_nodes_feat_range.contiguous();
    quantized_params = quantized_params.contiguous();
    // zero_points = zero_points.contiguous();
    // scales = scales.contiguous();

    uint8_t *input_ptr = input.data_ptr<uint8_t>();
    int64_t *quantized_nodes_feat_range_ptr = quantized_nodes_feat_range.data_ptr<int64_t>();
    float *quantized_params_ptr = quantized_params.data_ptr<float>(); // [num_bits, zero_point, scale]
    // int *nodes_num_bits_ptr = nodes_num_bits_tensor.data_ptr<int>(); // [num_bits]
    // float *zero_points_ptr = zero_points.data_ptr<float>();
    // float *scales_ptr = scales.data_ptr<float>();

    float *output_ptr = output.data_ptr<float>();

#pragma omp parallel for
    for (int i = 0; i < vertex_num; i++)
    {
        int num_bits = static_cast<int>(quantized_params_ptr[i * 3]);
        float zero_point = quantized_params_ptr[i * 3 + 1];
        float scale = quantized_params_ptr[i * 3 + 2];
        // int num_bits = nodes_num_bits_ptr[i];
        // float scale = scales_ptr[i];
        // float zero_point = zero_points_ptr[i];

        int packed_feat_begin = quantized_nodes_feat_range_ptr[i];
        int packed_feat_end = quantized_nodes_feat_range_ptr[i + 1];
        int packed_feat_len = packed_feat_end - packed_feat_begin;

        const int elems_per_byte = 8 / num_bits;
        const int mask = ((1 << num_bits) - 1);

        for (int j = 0; j < packed_feat_len; j++)
        {
            const uint8_t packed_val = input_ptr[packed_feat_begin + j];
            for (int k = 0; k < elems_per_byte && j * elems_per_byte + k < unpacked_feat_len; k++)
            {
                const float val = static_cast<float>((packed_val >> ((elems_per_byte - k - 1) * num_bits)) & mask);
                output_ptr[i * unpacked_feat_len + j * elems_per_byte + k] = val * scale + zero_point;
            }
        }

        // dequantize_kernel_v1_for_4bits(input_ptr + packed_feat_begin, scale, zero_point, packed_feat_len,
        //                                 unpacked_feat_len, output_ptr + i * unpacked_feat_len);
        // else if (num_bits == 2)
        //     dequantize_kernel_v1_for_2bits(input_ptr + packed_feat_begin, scale, zero_point, packed_feat_len,
        //                                     unpacked_feat_len, output_ptr + i * unpacked_feat_len);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quantize_tensor", &quantize_tensor);
    m.def("dequantize_tensor", &dequantize_tensor);
    m.def("quantize_tensor_v1", &quantize_tensor_v1);
    m.def("dequantize_tensor_v1", &dequantize_tensor_v1);
}
