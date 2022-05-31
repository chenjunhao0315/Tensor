//
//  Relu6Layer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/1.
//

#include "Relu6Layer.hpp"
#include "Activation.hpp"
#include "Parallel.hpp"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace otter {

Relu6Layer::Relu6Layer() {
    one_blob_only = true;
    support_inplace = true;
}

int Relu6Layer::forward_inplace(Tensor& bottom_blob, const NetOption& opt) const {
    if (opt.use_non_lib_optimize && bottom_blob.scalar_type() == ScalarType::Float) {
        auto input_output_a = bottom_blob.accessor<float, 4>()[0];
        int size = int(bottom_blob.size(2) * bottom_blob.size(3));
        
        otter::parallel_for(0, bottom_blob.size(1), 0, [&](int64_t begin, int64_t end) {
            for (const auto q : otter::irange(begin, end)) {
                float* ptr = input_output_a[q].data();

        #if __ARM_NEON
                int nn = size >> 2;
                int remain = size & 3;
        #else
                int remain = size;
        #endif

        #if __ARM_NEON
                float32x4_t _max = vdupq_n_f32(6);
                float32x4_t _min = vdupq_n_f32(0);
        #if __aarch64__
                for (; nn > 0; nn--)
                {
                    float32x4_t _ptr = vld1q_f32(ptr);
                    _ptr = vmaxq_f32(_ptr, _min);
                    _ptr = vminq_f32(_ptr, _max);
                    vst1q_f32(ptr, _ptr);
                    ptr += 4;
                }
        #else
                if (nn > 0)
                {
                    asm volatile(
                        "0:                             \n"
                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d0-d1}, [%1: 128]  \n"

                        "vmax.f32   q0, q0, %q4         \n"
                        "vmin.f32   q0, q0, %q5         \n"

                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d1}, [%1: 128]! \n"

                        "bne        0b                  \n"

                        : "=r"(nn), // %0
                        "=r"(ptr) // %1
                        : "0"(nn),
                        "1"(ptr),
                        "w"(_min), // %q4
                        "w"(_max)  // %q5
                        : "cc", "memory", "q0");
                }
        #endif // __aarch64__
        #endif // __ARM_NEON

                for (; remain > 0; remain--)
                {
                    if (*ptr < 0)
                        *ptr = 0;

                    if (*ptr > 6)
                        *ptr = 6;

                    ptr++;
                }
            }
        });
    } else {
        otter::relu6_(bottom_blob);
    }
        
    return 0;
}

}   // end namespace otter
