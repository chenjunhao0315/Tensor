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
