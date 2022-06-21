//
//  LReluLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/1.
//

#include "LReluLayer.hpp"
#include "TensorFunction.hpp"
#include "Parallel.hpp"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace otter {

LReluLayer::LReluLayer() {
    one_blob_only = true;
    support_inplace = true;
    
#if __SSE2__
    support_packing = true;
#elif __ARM_NEON__
    support_packing = true;
#endif
}

int LReluLayer::parse_param(LayerOption& option, ParamDict &pd) {
    pd.clear();
    
    float neg_slope = opt_find_float(option, "alpha", 0.1);
    
    pd.set((int)LReluParam::Neg_slope, neg_slope);
    
    return 0;
}

int LReluLayer::load_param(const ParamDict &pd) {
    neg_slope = pd.get((int)LReluParam::Neg_slope, 0.1f);
    
    return 0;
}

int LReluLayer::forward_inplace(Tensor &bottom_blob, const NetOption &opt) const {
    if ((opt.use_non_lib_optimize || opt.use_packing_layout) && (bottom_blob.scalar_type() == otter::ScalarType::Float || bottom_blob.scalar_type() == otter::ScalarType::Float4 || bottom_blob.scalar_type() == otter::ScalarType::Float8)) {
        auto input_output = bottom_blob[0];
        int64_t size = bottom_blob.size(2) * bottom_blob.size(3) * bottom_blob.elempack();
        auto input_output_ra = input_output.raw_accessor<float, 3>();
            
        otter::parallel_for(0, bottom_blob.size(1), 0, [&](int64_t begin, int64_t end) {
            for (const auto q : otter::irange(begin, end)) {
                float* ptr = (float*)input_output_ra[q].data();

            #if __ARM_NEON
                int nn = size >> 2;
                int remain = size - (nn << 2);
            #else
                int remain = size;
            #endif // __ARM_NEON

            #if __ARM_NEON
            #if __aarch64__
                float32x4_t _zero = vdupq_n_f32(0.f);
                float32x4_t _slope = vdupq_n_f32(neg_slope);
                for (; nn > 0; nn--)
                {
                    float32x4_t _p = vld1q_f32(ptr);
                    uint32x4_t _lemask = vcleq_f32(_p, _zero);
                    float32x4_t _ps = vmulq_f32(_p, _slope);
                    _p = vbslq_f32(_lemask, _ps, _p);
                    vst1q_f32(ptr, _p);

                    ptr += 4;
                }
            #else
                if (nn > 0)
                {
                    asm volatile(
                        "veor       q1, q0, q0          \n"
                        "vdup.f32   q2, %4              \n"
                        "0:                             \n"
                        "pld        [%1, #128]          \n"
                        "vld1.f32   {d0-d1}, [%1 :128]  \n"
                        "vcle.f32   q3, q0, q1          \n"
                        "vmul.f32   q4, q0, q2          \n"
                        "vbit.32    q0, q4, q3          \n"
                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d1}, [%1 :128]! \n"
                        "bne        0b                  \n"
                        : "=r"(nn), // %0
                        "=r"(ptr) // %1
                        : "0"(nn),
                        "1"(ptr),
                        "r"(neg_slope) // %4
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4");
                }
            #endif // __aarch64__
            #endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    if (*ptr < 0)
                        *ptr *= neg_slope;
                    
                    ptr++;
                }
            }
        });
    } else {
        otter::native::leaky_relu_(bottom_blob, neg_slope);
    }
    
    return 0;
}

}   // end namespace otter
