//
//  TensorPixel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#include "TensorPixel.hpp"
#include "TensorFactory.hpp"

#if __ARM_NEON__
#include <arm_neon.h>
#endif // __ARM_NEON__

namespace otter {

Tensor from_rgb(const unsigned char* rgb, int h, int w, int stride) {
    auto result = empty({1, 3, h, w}, otter::ScalarType::Float);
    OTTER_CHECK(result.defined(), "Tensor create failed!");
    
    const int wgap = stride - w * 3;
    if (wgap == 0) {
        w = w * h;
        h = 1;
    }
    
    auto accessor = result.accessor<float, 4>()[0];
    float* ptr0 = accessor[0].data();
    float* ptr1 = accessor[1].data();
    float* ptr2 = accessor[2].data();
    
    for (int y = 0; y < h; ++y) {
#if __ARM_NEON__
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON__
        
#if __ARM_NEON__
#if __aarch64__
        for (; nn > 0; nn--)
        {
            uint8x8x3_t _rgb = vld3_u8(rgb);
            uint16x8_t _r16 = vmovl_u8(_rgb.val[0]);
            uint16x8_t _g16 = vmovl_u8(_rgb.val[1]);
            uint16x8_t _b16 = vmovl_u8(_rgb.val[2]);

            float32x4_t _rlow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_r16)));
            float32x4_t _rhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_r16)));
            float32x4_t _glow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_g16)));
            float32x4_t _ghigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_g16)));
            float32x4_t _blow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_b16)));
            float32x4_t _bhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_b16)));

            vst1q_f32(ptr0, _rlow);
            vst1q_f32(ptr0 + 4, _rhigh);
            vst1q_f32(ptr1, _glow);
            vst1q_f32(ptr1 + 4, _ghigh);
            vst1q_f32(ptr2, _blow);
            vst1q_f32(ptr2 + 4, _bhigh);

            rgb += 3 * 8;
            ptr0 += 8;
            ptr1 += 8;
            ptr2 += 8;
        }
#else
        if (nn > 0)
        {
            asm volatile(
                "0:                             \n"
                "pld        [%1, #256]          \n"
                "vld3.u8    {d0-d2}, [%1]!      \n"
                "vmovl.u8   q8, d0              \n"
                "vmovl.u8   q9, d1              \n"
                "vmovl.u8   q10, d2             \n"
                "vmovl.u16  q0, d16             \n"
                "vmovl.u16  q1, d17             \n"
                "vmovl.u16  q2, d18             \n"
                "vmovl.u16  q3, d19             \n"
                "vmovl.u16  q8, d20             \n"
                "vmovl.u16  q9, d21             \n"
                "vcvt.f32.u32   q0, q0          \n"
                "vcvt.f32.u32   q1, q1          \n"
                "vcvt.f32.u32   q2, q2          \n"
                "vcvt.f32.u32   q3, q3          \n"
                "vcvt.f32.u32   q8, q8          \n"
                "subs       %0, #1              \n"
                "vst1.f32   {d0-d3}, [%2]!      \n"
                "vcvt.f32.u32   q9, q9          \n"
                "vst1.f32   {d4-d7}, [%3]!      \n"
                "vst1.f32   {d16-d19}, [%4]!    \n"
                "bne        0b                  \n"
                : "=r"(nn),   // %0
                "=r"(rgb),  // %1
                "=r"(ptr0), // %2
                "=r"(ptr1), // %3
                "=r"(ptr2)  // %4
                : "0"(nn),
                "1"(rgb),
                "2"(ptr0),
                "3"(ptr1),
                "4"(ptr2)
                : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10");
        }
#endif // __aarch64__
#endif // __ARM_NEON__
        for (; remain > 0; remain--)
        {
            *ptr0 = rgb[0];
            *ptr1 = rgb[1];
            *ptr2 = rgb[2];

            rgb += 3;
            ptr0++;
            ptr1++;
            ptr2++;
        }

        rgb += wgap;
        
    }
    
    return result;
}

}   // end namespace otter
