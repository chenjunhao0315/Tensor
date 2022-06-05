//
//  VecIntrinsic.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/16.
//

#ifndef VecIntrinsic_hpp
#define VecIntrinsic_hpp

#include "Macro.hpp"

// From PyTorch https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/Intrinsics.h

#if defined(__clang__) && (defined(__x86_64__) || defined(__i386__))
/* Clang-compatible compiler, targeting x86/x86-64 */
#include <x86intrin.h>
#elif defined(_MSC_VER)
/* Microsoft C/C++-compatible compiler */
#include <intrin.h>
#if _MSC_VER <= 1900
#define _mm256_extract_epi64(X, Y) (((uint64_t*)&X)[Y])
#endif
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* GCC-compatible compiler, targeting x86/x86-64 */
#include <x86intrin.h>
#elif defined(__GNUC__) && defined(__ARM_NEON__)
/* GCC-compatible compiler, targeting ARM with NEON */
#include <arm_neon.h>
#elif defined(__GNUC__) && defined(__IWMMXT__)
/* GCC-compatible compiler, targeting ARM with WMMX */
#include <mmintrin.h>
#elif (defined(__GNUC__) || defined(__xlC__)) && \
    (defined(__VEC__) || defined(__ALTIVEC__))
/* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
#include <altivec.h>
/* We need to undef those tokens defined by <altivec.h> to avoid conflicts
   with the C++ types. => Can still use __bool/__vector */
#undef bool
#undef vector
#undef pixel
#elif defined(__GNUC__) && defined(__SPE__)
/* GCC-compatible compiler, targeting PowerPC with SPE */
#include <spe.h>
#endif

#if __SSE2__
static OTTER_ALWAYS_INLINE float _mm_reduce_add_ps(__m128 x128)
{
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

static OTTER_ALWAYS_INLINE float _mm_reduce_max_ps(__m128 x128)
{
    const __m128 x64 = _mm_max_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_max_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

static OTTER_ALWAYS_INLINE int _mm_reduce_add_epi32(__m128i x)
{
    __m128i hi64 = _mm_unpackhi_epi64(x, x);
    __m128i sum64 = _mm_add_epi32(hi64, x);
    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i sum32 = _mm_add_epi32(sum64, hi32);
    return _mm_cvtsi128_si32(sum32);
}

static OTTER_ALWAYS_INLINE int32_t float2int8_sse(const __m128& _v0)
{
    // _MM_ROUND_NEAREST round to even
    // simulate round to nearest via +/-0.5 with round to zero
    __m128 _p5 = _mm_set1_ps(0.5f);
    __m128 _signmask = _mm_castsi128_ps(_mm_set1_epi32(1 << 31));
    __m128 _sign0 = _mm_and_ps(_v0, _signmask);
    __m128 _v0_p5 = _mm_or_ps(_p5, _sign0);
    __m128 _v0_adj = _mm_add_ps(_v0, _v0_p5);
    __m128i _v0_i = _mm_cvttps_epi32(_v0_adj);

    __m128i _v0_s16 = _mm_packs_epi32(_v0_i, _v0_i);

    _v0_s16 = _mm_min_epi16(_v0_s16, _mm_set1_epi16(127));
    _v0_s16 = _mm_max_epi16(_v0_s16, _mm_set1_epi16(-127));

    __m128i _v8 = _mm_packs_epi16(_v0_s16, _v0_s16);

#if defined(__x86_64__) || defined(_M_X64)
    return (int32_t)_mm_cvtsi128_si64(_v8);
#else
    return _mm_cvtsi128_si32(_v8);
#endif
}

static OTTER_ALWAYS_INLINE int64_t float2int8_sse(const __m128& _v0, const __m128& _v1)
{
    // _MM_ROUND_NEAREST round to even
    // simulate round to nearest via +/-0.5 with round to zero
    __m128 _p5 = _mm_set1_ps(0.5f);
    __m128 _signmask = _mm_castsi128_ps(_mm_set1_epi32(1 << 31));
    __m128 _sign0 = _mm_and_ps(_v0, _signmask);
    __m128 _sign1 = _mm_and_ps(_v1, _signmask);
    __m128 _v0_p5 = _mm_or_ps(_p5, _sign0);
    __m128 _v1_p5 = _mm_or_ps(_p5, _sign1);
    __m128 _v0_adj = _mm_add_ps(_v0, _v0_p5);
    __m128 _v1_adj = _mm_add_ps(_v1, _v1_p5);
    __m128i _v0_i = _mm_cvttps_epi32(_v0_adj);
    __m128i _v1_i = _mm_cvttps_epi32(_v1_adj);

    __m128i _v01_s16 = _mm_packs_epi32(_v0_i, _v1_i);

    _v01_s16 = _mm_min_epi16(_v01_s16, _mm_set1_epi16(127));
    _v01_s16 = _mm_max_epi16(_v01_s16, _mm_set1_epi16(-127));

    __m128i _v8 = _mm_packs_epi16(_v01_s16, _v01_s16);

#if defined(__x86_64__) || defined(_M_X64)
    return _mm_cvtsi128_si64(_v8);
#else
    int64_t v8[2];
    _mm_storeu_si128((__m128i*)v8, _v8);
    return v8[0];
#endif
}

static OTTER_ALWAYS_INLINE __m128i float2int8_sse(const __m128& _v0, const __m128& _v1, const __m128& _v2, const __m128& _v3)
{
    // _MM_ROUND_NEAREST round to even
    // simulate round to nearest via +/-0.5 with round to zero
    __m128 _p5 = _mm_set1_ps(0.5f);
    __m128 _signmask = _mm_castsi128_ps(_mm_set1_epi32(1 << 31));
    __m128 _sign0 = _mm_and_ps(_v0, _signmask);
    __m128 _sign1 = _mm_and_ps(_v1, _signmask);
    __m128 _sign2 = _mm_and_ps(_v2, _signmask);
    __m128 _sign3 = _mm_and_ps(_v3, _signmask);
    __m128 _v0_p5 = _mm_or_ps(_p5, _sign0);
    __m128 _v1_p5 = _mm_or_ps(_p5, _sign1);
    __m128 _v2_p5 = _mm_or_ps(_p5, _sign2);
    __m128 _v3_p5 = _mm_or_ps(_p5, _sign3);
    __m128 _v0_adj = _mm_add_ps(_v0, _v0_p5);
    __m128 _v1_adj = _mm_add_ps(_v1, _v1_p5);
    __m128 _v2_adj = _mm_add_ps(_v2, _v2_p5);
    __m128 _v3_adj = _mm_add_ps(_v3, _v3_p5);
    __m128i _v0_i = _mm_cvttps_epi32(_v0_adj);
    __m128i _v1_i = _mm_cvttps_epi32(_v1_adj);
    __m128i _v2_i = _mm_cvttps_epi32(_v2_adj);
    __m128i _v3_i = _mm_cvttps_epi32(_v3_adj);

    __m128i _v01_s16 = _mm_packs_epi32(_v0_i, _v1_i);
    __m128i _v23_s16 = _mm_packs_epi32(_v2_i, _v3_i);

    _v01_s16 = _mm_min_epi16(_v01_s16, _mm_set1_epi16(127));
    _v23_s16 = _mm_min_epi16(_v23_s16, _mm_set1_epi16(127));
    _v01_s16 = _mm_max_epi16(_v01_s16, _mm_set1_epi16(-127));
    _v23_s16 = _mm_max_epi16(_v23_s16, _mm_set1_epi16(-127));

    __m128i _v8 = _mm_packs_epi16(_v01_s16, _v23_s16);

    return _v8;
}

#ifndef __FMA__
static OTTER_ALWAYS_INLINE __m128 _mm_comp_fmadd_ps(__m128 _a, const __m128 _b, const __m128 _c) {
    return _mm_add_ps(_mm_mul_ps(_a, _b), _c);
}
static OTTER_ALWAYS_INLINE __m128 _mm_comp_fnmadd_ps(__m128 _a, const __m128 _b, const __m128 _c) {
    return _mm_sub_ps(_c, _mm_mul_ps(_a, _b));
}
#else
static OTTER_ALWAYS_INLINE __m128 _mm_comp_fmadd_ps(__m128 _a, const __m128 _b, const __m128 _c) {
    return _mm_fmadd_ps(_a, _b, _c);
}
static OTTER_ALWAYS_INLINE __m128 _mm_comp_fnmadd_ps(__m128 _a, const __m128 _b, const __m128 _c) {
    // return -a * b + c
    return _mm_fnmadd_ps(_a, _b, _c);
}
#endif // !__FMA__

#if __AVX__
#ifndef __FMA__
static OTTER_ALWAYS_INLINE __m256 _mm256_comp_fmadd_ps(__m256 _a, const __m256 _b, const __m256 _c) {
    return _mm256_add_ps(_mm256_mul_ps(_a, _b), _c);
}
static OTTER_ALWAYS_INLINE __m256 _mm256_comp_fnmadd_ps(__m256 _a, const __m256 _b, const __m256 _c) {
    return _mm256_sub_ps(_c, _mm256_mul_ps(_a, _b));
}
#else
static OTTER_ALWAYS_INLINE __m256 _mm256_comp_fmadd_ps(__m256 _a, const __m256 _b, const __m256 _c) {
    return _mm256_fmadd_ps(_a, _b, _c);
}
static OTTER_ALWAYS_INLINE __m256 _mm256_comp_fnmadd_ps(__m256 _a, const __m256 _b, const __m256 _c) {
    // return -a * b + c
    return _mm256_fnmadd_ps(_a, _b, _c);
}
#endif
#endif // __AVX__
#endif // __SSE2__

#endif /* VecIntrinsic_hpp */
