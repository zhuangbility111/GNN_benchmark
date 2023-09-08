#include <torch/extension.h>

using torch::Tensor;

// #ifdef __AVX512F__
    #include <immintrin.h>
    #include <emmintrin.h>
    #define VEC_LEN 16
// #endif // __AVX512F__

inline void quantize_kernel_v1_for_8bits(float* input_ptr, const float scale, const float zero_point, const int feat_len,
                                  uint8_t* output_ptr) {
    __m512 ones = _mm512_set1_ps(1.0);
    __m512 scale_val = _mm512_set1_ps(scale);
    __m512 zero_point_val = _mm512_set1_ps(zero_point);

    int i = 0;
    for (; i + VEC_LEN <= feat_len; i += VEC_LEN) {
        __mmask16 mask0 = 0xFFFF;
        __m512 input_val0 = _mm512_mask_loadu_ps(ones, mask0, &input_ptr[i]);
        __m512i val0 = _mm512_cvt_roundps_epi32(_mm512_div_ps(_mm512_sub_ps(input_val0, zero_point_val), scale_val),
                                                _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        
        // truncate the packed_val to uint8_t and store
        _mm512_mask_cvtepi32_storeu_epi8(&output_ptr[i], mask0, val0);
    }

    if (i < feat_len) {
        __mmask16 mask0 = (1 << (feat_len - i)) - 1;
        __m512 input_val0 = _mm512_mask_loadu_ps(ones, mask0, &input_ptr[i]);
        __m512i val0 = _mm512_cvt_roundps_epi32(_mm512_div_ps(_mm512_sub_ps(input_val0, zero_point_val), scale_val),
                                                _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        
        // truncate the packed_val to uint8_t and store
        _mm512_mask_cvtepi32_storeu_epi8(&output_ptr[i], mask0, val0);
    }
}

void dequantize_kernel_v1_for_8bits(uint8_t* input_ptr, const float scale, const float zero_point, 
                                    const int packed_feat_len, const int unpacked_feat_len, float* output_ptr) {

    __m512 scale_val = _mm512_set1_ps(scale);
    __m512 zero_point_val = _mm512_set1_ps(zero_point);

    int i = 0;
    for (; i + VEC_LEN <= packed_feat_len; i += VEC_LEN) {
        __m128i input_val0 = _mm_loadu_epi8(&input_ptr[i]);
        // expand to 32-bit integer
        __m512i input_val1 = _mm512_cvtepu8_epi32(input_val0);
        // convert to float
        __m512 input_val2 = _mm512_cvtepi32_ps(input_val1);
        // multiply by scale
        __m512 input_val3 = _mm512_mul_ps(input_val2, scale_val);
        // add zero point
        __m512 input_val4 = _mm512_add_ps(input_val3, zero_point_val);
        // store
        _mm512_storeu_ps(&output_ptr[i], input_val4);
    }

    if (i < packed_feat_len) {
        __mmask16 mask0 = (1 << (packed_feat_len - i)) - 1;
        __m128i input_val0 = _mm_mask_loadu_epi8(_mm_setzero_si128(), mask0, &input_ptr[i]);
        __m512i input_val1 = _mm512_cvtepu8_epi32(input_val0);
        __m512 input_val2 = _mm512_cvtepi32_ps(input_val1);
        __m512 input_val3 = _mm512_mul_ps(input_val2, scale_val);
        __m512 input_val4 = _mm512_add_ps(input_val3, zero_point_val);
        _mm512_mask_storeu_ps(&output_ptr[i], mask0, input_val4);
    }
}

// have bugs, need to fix
void quantize_kernel_v1_for_4bits(float* input_ptr, const float scale, const float zero_point, const int feat_len,
                                  uint8_t* output_ptr) {
    __m512 zeros = _mm512_setzero_ps();
    __m512 scale_val = _mm512_set1_ps(scale);
    __m512 zero_point_val = _mm512_set1_ps(zero_point);
    const int ELEMS_PER_BYTE = 2;

    for (int i = 0; i < feat_len; i += ELEMS_PER_BYTE * VEC_LEN) {
        __mmask16 mask0 = i + VEC_LEN < feat_len ? 0xFFFF : (1 << (feat_len - i)) - 1;
        // __mmask16 mask1 = i + ELEMS_PER_BYTE * VEC_LEN < feat_len ? 0xFFFF : std::max((1 << (feat_len - VEC_LEN - i)) - 1, 0);

        __m512 input_val0 = _mm512_mask_loadu_ps(zeros, mask0, &input_ptr[i]);
        // __m512 input_val1 = _mm512_mask_loadu_ps(zeros, mask1, &input_ptr[i+VEC_LEN]);

        __m512 v_sub0 = _mm512_sub_ps(input_val0, zero_point_val);
        // __m512 v_sub1 = _mm512_sub_ps(input_val1, zero_point_val);

        __m512 v_div0 = _mm512_div_ps(v_sub0, scale_val);
        // __m512 v_div1 = _mm512_div_ps(v_sub1, scale_val);

        __m512i v_round0 = _mm512_cvt_roundps_epi32(v_div0, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        // __m512i v_round1 = _mm512_cvt_roundps_epi32(v_div1, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);

        __m512i v_slli0 = _mm512_slli_epi32(v_round0, 4);

        // combine v_slli0 and v_round1
        // __m512i v_or0 = _mm512_or_epi32(v_slli0, v_round1);
        
        // truncate the packed_val to uint8_t and store
        _mm512_mask_cvtepi32_storeu_epi8(&output_ptr[i / ELEMS_PER_BYTE], mask0, v_slli0);
    }

    // if (i < feat_len) {
    //     __mmask16 mask0 = (1 << (feat_len - i)) - 1;
    //     __mmask16 mask1 = (feat_len - i - 1 >= 0 ? ((1 << (feat_len - i - 1)) - 1) : 0);

    //     __m512 input_val0 = _mm512_mask_loadu_ps(ones, mask0, &input_ptr[i]);
    //     __m512 input_val1 = _mm512_mask_loadu_ps(ones, mask1, &input_ptr[i+1]);

    //     __m512 v_sub0 = _mm512_sub_ps(input_val0, zero_point_val);
    //     __m512 v_sub1 = _mm512_sub_ps(input_val1, zero_point_val);

    //     __m512 v_div0 = _mm512_div_ps(v_sub0, scale_val);
    //     __m512 v_div1 = _mm512_div_ps(v_sub1, scale_val);

    //     __m512i v_round0 = _mm512_cvt_roundps_epi32(v_div0, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
    //     __m512i v_round1 = _mm512_cvt_roundps_epi32(v_div1, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);

    //     __m512i v_slli0 = _mm512_slli_epi32(v_round0, 4);

    //     // combine v_slli0 and v_round1
    //     __m512i v_or0 = _mm512_or_epi32(v_slli0, v_round1);
        
    //     // truncate the packed_val to uint8_t and store
    //     _mm512_mask_cvtepi32_storeu_epi8(&output_ptr[i / ELEMS_PER_BYTE], mask0, v_or0);
    // }
}

// have bugs, need to fix
void dequantize_kernel_v1_for_4bits(uint8_t* input_ptr, const float scale, const float zero_point, 
                                    const int packed_feat_len, const int unpacked_feat_len, float* output_ptr) {

    __m512 scale_val = _mm512_set1_ps(scale);
    __m512 zero_point_val = _mm512_set1_ps(zero_point);
    const int ELEMS_PER_BYTE = 2;
    __m512i mask = _mm512_set1_epi32(0x0F);

    for (int i = 0; i < packed_feat_len; i += VEC_LEN) {
        __mmask16 load_mask = i + VEC_LEN < packed_feat_len ? 0xFFFF : (1 << (packed_feat_len - i)) - 1;

        __mmask16 store_mask0 = i * ELEMS_PER_BYTE + VEC_LEN < unpacked_feat_len ? 
                                0xFFFF : (1 << (unpacked_feat_len - i * ELEMS_PER_BYTE)) - 1;
        __mmask16 store_mask1 = (i + VEC_LEN) * ELEMS_PER_BYTE < unpacked_feat_len ? 
                                0xFFFF : std::max((1 << (unpacked_feat_len - i * ELEMS_PER_BYTE - VEC_LEN)) - 1, 0);

        // mask load
        __m128i input_val0 = _mm_mask_loadu_epi8(_mm_setzero_si128(), load_mask, &input_ptr[i]);
        // extend to 32-bit integer
        __m512i input_val1 = _mm512_cvtepu8_epi32(input_val0);
        // shift right 4 bits to get the first 4-bit integer
        __m512i first_half_val = _mm512_srli_epi32(input_val1, 4);
        // and with mask to get the second 4-bit integer
        __m512i second_half_val = _mm512_and_epi32(input_val1, mask);

        // convert to float
        __m512 input_val4 = _mm512_cvtepi32_ps(first_half_val);
        __m512 input_val5 = _mm512_cvtepi32_ps(second_half_val);

        // multiply by scale
        input_val4 = _mm512_mul_ps(input_val4, scale_val);
        input_val5 = _mm512_mul_ps(input_val5, scale_val);

        // add zero point
        input_val4 = _mm512_add_ps(input_val4, zero_point_val);
        input_val5 = _mm512_add_ps(input_val5, zero_point_val);

        // mask store
        _mm512_mask_storeu_ps(&output_ptr[i*ELEMS_PER_BYTE], store_mask0, input_val4);
        _mm512_mask_storeu_ps(&output_ptr[i*ELEMS_PER_BYTE + VEC_LEN], store_mask1, input_val5);
    }
}

// have bugs, need to fix
void quantize_kernel_v1_for_2bits(float* input_ptr, const float scale, const float zero_point, const int feat_len,
                                  uint8_t* output_ptr) {
    __m512 ones = _mm512_set1_ps(1.0);
    __m512 scale_val = _mm512_set1_ps(scale);
    __m512 zero_point_val = _mm512_set1_ps(zero_point);

    const int ELEMS_PER_BYTE = 4;

    int i = 0;
    for (; i + VEC_LEN <= feat_len; i += VEC_LEN) {
        __mmask16 mask0 = 0xFFFF;
        __mmask16 mask1 = 0xFFFF;
        __mmask16 mask2 = 0xFFFF;
        __mmask16 mask3 = 0xFFFF;

        __m512 input_val0 = _mm512_mask_loadu_ps(ones, mask0, &input_ptr[i]);
        __m512 input_val1 = _mm512_mask_loadu_ps(ones, mask1, &input_ptr[i+1]);
        __m512 input_val2 = _mm512_mask_loadu_ps(ones, mask2, &input_ptr[i+2]);
        __m512 input_val3 = _mm512_mask_loadu_ps(ones, mask3, &input_ptr[i+3]);

        __m512 v_sub0 = _mm512_sub_ps(input_val0, zero_point_val);
        __m512 v_sub1 = _mm512_sub_ps(input_val1, zero_point_val);
        __m512 v_sub2 = _mm512_sub_ps(input_val2, zero_point_val);
        __m512 v_sub3 = _mm512_sub_ps(input_val3, zero_point_val);

        __m512 v_div0 = _mm512_div_ps(v_sub0, scale_val);
        __m512 v_div1 = _mm512_div_ps(v_sub1, scale_val);
        __m512 v_div2 = _mm512_div_ps(v_sub2, scale_val);
        __m512 v_div3 = _mm512_div_ps(v_sub3, scale_val);

        __m512i v_round0 = _mm512_cvt_roundps_epi32(v_div0, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        __m512i v_round1 = _mm512_cvt_roundps_epi32(v_div1, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        __m512i v_round2 = _mm512_cvt_roundps_epi32(v_div2, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        __m512i v_round3 = _mm512_cvt_roundps_epi32(v_div3, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);

        __m512i v_slli0 = _mm512_slli_epi32(v_round0, 6);
        __m512i v_slli1 = _mm512_slli_epi32(v_round1, 4);
        __m512i v_slli2 = _mm512_slli_epi32(v_round2, 2);

        // combine v_round3 and v_slli0, v_slli1, v_slli2
        __m512i v_or0 = _mm512_or_epi32(v_round3, v_slli0);
        v_or0 = _mm512_or_epi32(v_or0, v_slli1);
        v_or0 = _mm512_or_epi32(v_or0, v_slli2);
        
        // truncate the packed_val to uint8_t and store
        _mm512_mask_cvtepi32_storeu_epi8(&output_ptr[i / ELEMS_PER_BYTE], mask0, v_or0);
    }

    if (i < feat_len) {
        __mmask16 mask0 = (1 << (feat_len - i)) - 1;
        __mmask16 mask1 = (feat_len - i - 1 >= 0 ? (1 << (feat_len - i - 1)) - 1 : 0);
        __mmask16 mask2 = (feat_len - i - 2 >= 0 ? (1 << (feat_len - i - 2)) - 1 : 0);
        __mmask16 mask3 = (feat_len - i - 3 >= 0 ? (1 << (feat_len - i - 3)) - 1 : 0);

        __m512 input_val0 = _mm512_mask_loadu_ps(ones, mask0, &input_ptr[i]);
        __m512 input_val1 = _mm512_mask_loadu_ps(ones, mask1, &input_ptr[i+1]);
        __m512 input_val2 = _mm512_mask_loadu_ps(ones, mask2, &input_ptr[i+2]);
        __m512 input_val3 = _mm512_mask_loadu_ps(ones, mask3, &input_ptr[i+3]);

        __m512 v_sub0 = _mm512_sub_ps(input_val0, zero_point_val);
        __m512 v_sub1 = _mm512_sub_ps(input_val1, zero_point_val);
        __m512 v_sub2 = _mm512_sub_ps(input_val2, zero_point_val);
        __m512 v_sub3 = _mm512_sub_ps(input_val3, zero_point_val);

        __m512 v_div0 = _mm512_div_ps(v_sub0, scale_val);
        __m512 v_div1 = _mm512_div_ps(v_sub1, scale_val);
        __m512 v_div2 = _mm512_div_ps(v_sub2, scale_val);
        __m512 v_div3 = _mm512_div_ps(v_sub3, scale_val);

        __m512i v_round0 = _mm512_cvt_roundps_epi32(v_div0, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        __m512i v_round1 = _mm512_cvt_roundps_epi32(v_div1, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        __m512i v_round2 = _mm512_cvt_roundps_epi32(v_div2, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
        __m512i v_round3 = _mm512_cvt_roundps_epi32(v_div3, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);

        __m512i v_slli0 = _mm512_slli_epi32(v_round0, 6);
        __m512i v_slli1 = _mm512_slli_epi32(v_round1, 4);
        __m512i v_slli2 = _mm512_slli_epi32(v_round2, 2);

        // combine v_round3 and v_slli0, v_slli1, v_slli2
        __m512i v_or0 = _mm512_or_epi32(v_round3, v_slli0);
        v_or0 = _mm512_or_epi32(v_or0, v_slli1);
        v_or0 = _mm512_or_epi32(v_or0, v_slli2);
        
        // truncate the packed_val to uint8_t and store
        _mm512_mask_cvtepi32_storeu_epi8(&output_ptr[i / ELEMS_PER_BYTE], mask0, v_or0);
    }
}