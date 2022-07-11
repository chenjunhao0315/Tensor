//
//  ReluLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/15.
//

#include "ReluLayer.hpp"
#include "Activation.hpp"
#include "Parallel.hpp"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace otter {

ReluLayer::ReluLayer() {
    one_blob_only = true;
    support_inplace = true;
    
#if __SSE2__
    support_packing = true;
#elif __ARM_NEON__
    support_packing = true;
#endif
}

int ReluLayer::forward_inplace(Tensor& bottom_blob, const NetOption& opt) const {
    if ((opt.use_non_lib_optimize || opt.use_packing_layout) && (bottom_blob.scalar_type() == otter::ScalarType::Float || bottom_blob.scalar_type() == otter::ScalarType::Float4 || bottom_blob.scalar_type() == otter::ScalarType::Float8)) {
        auto input_output = bottom_blob[0];
        int size = int(bottom_blob.size(2) * bottom_blob.size(3) * bottom_blob.elempack());
        auto input_output_ra = input_output.raw_accessor<float, 3>();
        
        otter::parallel_for(0, bottom_blob.size(1), 0, [&](int64_t begin, int64_t end) {
            for (const auto q : otter::irange(begin, end)) {
                float* ptr = (float*)input_output_ra[q].data();

                int i = 0;
    #if __ARM_NEON
                float32x4_t _zero = vdupq_n_f32(0.f);
                for (; i + 15 < size; i += 16)
                {
    #if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%0, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0] \n"
                        "fmax   v0.4s, v0.4s, %2.4s     \n"
                        "fmax   v1.4s, v1.4s, %2.4s     \n"
                        "fmax   v2.4s, v2.4s, %2.4s     \n"
                        "fmax   v3.4s, v3.4s, %2.4s     \n"
                        "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                        : "=r"(ptr) // %0
                        : "0"(ptr),
                        "w"(_zero) // %2
                        : "memory", "v0", "v1", "v2", "v3");
    #else  // __aarch64__
                    asm volatile(
                        "pld        [%0, #512]      \n"
                        "vldm       %0, {d0-d7}     \n"
                        "vmax.f32   q0, q0, %q2     \n"
                        "vmax.f32   q1, q1, %q2     \n"
                        "vmax.f32   q2, q2, %q2     \n"
                        "vmax.f32   q3, q3, %q2     \n"
                        "vstm       %0!, {d0-d7}    \n"
                        : "=r"(ptr) // %0
                        : "0"(ptr),
                        "w"(_zero) // %2
                        : "memory", "q0", "q1", "q2", "q3");
    #endif // __aarch64__
                }
                for (; i + 7 < size; i += 8)
                {
                    float32x4_t _p0 = vld1q_f32(ptr);
                    float32x4_t _p1 = vld1q_f32(ptr + 4);
                    _p0 = vmaxq_f32(_p0, _zero);
                    _p1 = vmaxq_f32(_p1, _zero);
                    vst1q_f32(ptr, _p0);
                    vst1q_f32(ptr + 4, _p1);
                    ptr += 8;
                }
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _ptr = vld1q_f32(ptr);
                    _ptr = vmaxq_f32(_ptr, _zero);
                    vst1q_f32(ptr, _ptr);
                    ptr += 4;
                }
    #elif __SSE2__
    #if __AVX__
                __m256 _zero_avx = _mm256_setzero_ps();
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _mm256_storeu_ps(ptr, _mm256_max_ps(_zero_avx, _p));
                    ptr += 8;
                }
    #endif  // __AVX__
                __m128 _zero = _mm_setzero_ps();
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = _mm_load_ps(ptr);
                    _mm_store_ps(ptr, _mm_max_ps(_zero, _p));
                    ptr += 4;
                }
    #endif  // __SSE2__
                for (; i < size; i++)
                {
                    *ptr = std::max(*ptr, 0.f);

                    ptr++;
                }
            }
        });
    } else {
        otter::relu_(bottom_blob);
    }
        
    return 0;
}

}   // end namespace otter
