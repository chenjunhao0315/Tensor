//
//  QuantizeX86.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/6.
//

#ifndef QuantizeX86_hpp
#define QuantizeX86_hpp

#if CPU_CAPABILITY_AVX2
#include "VecIntrinsic.hpp"
#include "Avx_Math.hpp"
#include "Tensor.hpp"

#if __SSE2__
#include "sse_mathfun.hpp"
static OTTER_ALWAYS_INLINE __m128 sigmoid_sse(__m128 inputs)
{
    const __m128 one = _mm_set1_ps(1.0f);
    return _mm_div_ps(one, _mm_add_ps(one, exp_ps(_mm_sub_ps(_mm_setzero_ps(), inputs))));
}

static OTTER_ALWAYS_INLINE __m128 tanh_sse(__m128 inputs)
{
    const __m128 one = _mm_set1_ps(1.0f);
    const __m128 two = _mm_set1_ps(2.0f);
    return _mm_sub_ps(_mm_mul_ps(sigmoid_sse(_mm_mul_ps(inputs, two)), two), one);
}

static OTTER_ALWAYS_INLINE __m128 mish_sse(__m128 inputs)
{
    return _mm_mul_ps(inputs, tanh_sse(log_ps(_mm_add_ps(exp_ps(inputs), _mm_set1_ps(1.f)))));
}

static OTTER_ALWAYS_INLINE __m128 swish_sse(__m128 inputs)
{
    return _mm_mul_ps(inputs, sigmoid_sse(inputs));
}

static OTTER_ALWAYS_INLINE __m128 hardswish_sse(__m128 inputs, __m128 a, __m128 b)
{
    const __m128 one = _mm_set1_ps(1.0f);
    b = _mm_add_ps(_mm_mul_ps(inputs, a), b);
    b = _mm_max_ps(b, _mm_setzero_ps());
    b = _mm_min_ps(b, one);
    return _mm_mul_ps(b, inputs);
}

static OTTER_ALWAYS_INLINE __m128 abs_sse(__m128 inputs)
{
    // Use negative zero as the sign bit mask.
    const __m128 magic_negative_zero = _mm_set_ps1(-0.0f);

    // return (!magic_negative_zero && x);
    return _mm_andnot_ps(magic_negative_zero, inputs);
}

static OTTER_ALWAYS_INLINE __m128 lrelu_sse(__m128 inputs, float slope)
{
    __m128 pos = _mm_max_ps(_mm_setzero_ps(), inputs);
    __m128 neg = _mm_min_ps(_mm_setzero_ps(), inputs);
    return _mm_add_ps(pos, _mm_mul_ps(_mm_set1_ps(slope), neg));
}

static OTTER_ALWAYS_INLINE __m128 prelu_sse(__m128 inputs, __m128 alphas)
{
    __m128 pos = _mm_max_ps(_mm_setzero_ps(), inputs);
    __m128 neg = _mm_min_ps(_mm_setzero_ps(), inputs);
    return _mm_add_ps(pos, _mm_mul_ps(alphas, neg));
}

static OTTER_ALWAYS_INLINE __m128 activation_sse(__m128 _v, int activation_type, const otter::Tensor& activation_params)
{
    // Process fused activations
    switch (activation_type)
    {
    case 1:
    {
        // Relu
        return _mm_max_ps(_v, _mm_setzero_ps());
    }
    case 2:
    {
        // Leaky relu
        return lrelu_sse(_v, activation_params.item().toFloat());
    }
    case 3:
    {
        // min max clip
        __m128 min = _mm_set1_ps(0);
        __m128 max = _mm_set1_ps(6);
        return _mm_min_ps(_mm_max_ps(_v, min), max);
    }
    case 4:
    {
        // Sigmoid
        return sigmoid_sse(_v);
    }
    case 5:
    {
        return mish_sse(_v);
    }
    }

    return _v;
}

#if __AVX__
#include <immintrin.h>

static OTTER_ALWAYS_INLINE __m256 sigmoid_avx(__m256 inputs)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), inputs))));
}

static OTTER_ALWAYS_INLINE __m256 tanh_avx(__m256 inputs)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
#if __FMA__
    return _mm256_fmsub_ps(sigmoid_avx(_mm256_mul_ps(inputs, two)), two, one);
#else
    return _mm256_sub_ps(_mm256_mul_ps(sigmoid_avx(_mm256_mul_ps(inputs, two)), two), one);
#endif
}

static OTTER_ALWAYS_INLINE __m256 mish_avx(__m256 inputs)
{
    return _mm256_mul_ps(inputs, tanh_avx(log256_ps(_mm256_add_ps(exp256_ps(inputs), _mm256_set1_ps(1.f)))));
}

static OTTER_ALWAYS_INLINE __m256 swish_avx(__m256 inputs)
{
    return _mm256_mul_ps(inputs, sigmoid_avx(inputs));
}

static OTTER_ALWAYS_INLINE __m256 hardswish_avx(__m256 inputs, __m256 a, __m256 b)
{
    const __m256 one = _mm256_set1_ps(1.0f);
    b = _mm256_comp_fmadd_ps(inputs, a, b);
    b = _mm256_max_ps(b, _mm256_setzero_ps());
    b = _mm256_min_ps(b, one);
    return _mm256_mul_ps(b, inputs);
}

static OTTER_ALWAYS_INLINE __m256 abs_avx(__m256 inputs)
{
    return _mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), inputs), inputs);
}

static OTTER_ALWAYS_INLINE __m256 lrelu_avx(__m256 inputs, float slope)
{
    __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), inputs);
    __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), inputs);
    return _mm256_add_ps(pos, _mm256_mul_ps(_mm256_set1_ps(slope), neg));
}

static OTTER_ALWAYS_INLINE __m256 prelu_avx(__m256 inputs, __m256 alphas)
{
    __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), inputs);
    __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), inputs);
    return _mm256_add_ps(pos, _mm256_mul_ps(alphas, neg));
}

static OTTER_ALWAYS_INLINE __m256 activation_avx(__m256 _v, int activation_type, const otter::Tensor& activation_params)
{
    // Process fused activations
    switch (activation_type)
    {
    case 1:
    {
        // Relu
        return _mm256_max_ps(_v, _mm256_setzero_ps());
    }
    case 2:
    {
        // Leaky relu
        return lrelu_avx(_v, activation_params.item().toFloat());
    }
    case 3:
    {
        // min max clip
        __m256 min = _mm256_set1_ps(0);
        __m256 max = _mm256_set1_ps(6);
        return _mm256_min_ps(_mm256_max_ps(_v, min), max);
    }
    case 4:
    {
        // Sigmoid
        return sigmoid_avx(_v);
    }
    case 5:
    {
        return mish_avx(_v);
    }
    }

    return _v;
}

#endif // __AVX__
#endif // __SSE2__

namespace otter {

Tensor requantize_from_int32_to_int8_x86(const Tensor& src, const Tensor& scale_in_data, const Tensor& scale_out_data, const Tensor& bias_data, int activation_type, const Tensor& activation_params);



}   // end namespace otter
#endif

#endif /* QuantizeX86_hpp */
