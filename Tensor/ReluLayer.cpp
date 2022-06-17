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
}

int ReluLayer::forward_inplace(Tensor& bottom_blob, const NetOption& opt) const {
    if ((opt.use_non_lib_optimize || opt.use_packing_layout) && (bottom_blob.scalar_type() == otter::ScalarType::Float || bottom_blob.scalar_type() == otter::ScalarType::Float4 || bottom_blob.scalar_type() == otter::ScalarType::Float8)) {
        auto input_output = bottom_blob[0];
        int size = int(bottom_blob.size(2) * bottom_blob.size(3) * bottom_blob.elempack());
        
        otter::parallel_for(0, bottom_blob.size(1), 0, [&](int64_t begin, int64_t end) {
            for (const auto q : otter::irange(begin, end)) {
                float* ptr = (float*)input_output[q].raw_data();

    #if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn << 2);
    #else
                int remain = size;
    #endif // __ARM_NEON

    #if __ARM_NEON
    #if __aarch64__
                float32x4_t _zero = vdupq_n_f32(0.f);
                for (; nn > 0; nn--)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    _p = vmaxq_f32(_p, _zero);
                    vst1q_f32(ptr, _p);

                    ptr += 4;
                }
    #else
                if (nn > 0)
                {
                    asm volatile(
                        "veor       q1, q0, q0          \n"
                        "0:                             \n"
                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d0-d1}, [%1 :128]  \n"
                        "vmax.f32   q0, q0, q1          \n"
                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d1}, [%1 :128]! \n"
                        "bne        0b                  \n"
                        : "=r"(nn), // %0
                        "=r"(ptr) // %1
                        : "0"(nn),
                        "1"(ptr)
                        : "cc", "memory", "q0", "q1");
                }
    #endif // __aarch64__
    #endif // __ARM_NEON
                for (; remain > 0; remain--)
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
