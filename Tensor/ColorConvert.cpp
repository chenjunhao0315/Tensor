//
//  ColorConvert.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/19.
//

#include "ColorConvert.hpp"

#include "Tensor.hpp"
#include "TensorFactory.hpp"
#include "TensorResize.hpp"
#include "Dispatch.hpp"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace otter {
namespace cv {

void convertCheck(const Tensor& self, int in_channels);

Tensor convertRGBtoGray(const Tensor& self);

Tensor convertColor(const Tensor& self, int mode) {
    switch (mode) {
        case RGB_TO_GRAY:
            return convertRGBtoGray(self);
            break;
            
        default:
            break;
    }
    OTTER_CHECK(false, "Invalid convert mode!");
    return Tensor();
}

void convertCheck(const Tensor& self, int in_channels) {
    OTTER_CHECK(self.dim() == 3, "Expect input tensor has 3 dimensions but get ", self.dim());
    OTTER_CHECK(self.size(2) == in_channels, "Expect input tensor has ", in_channels, " but get ", self.size(2));
}

Tensor convertRGBtoGray(const Tensor& self) {
    convertCheck(self, 3);
    
    auto out = otter::empty({self.size(0), self.size(1), 1}, otter::ScalarType::Float);
    
    const unsigned char Y_shift = 8; //14
    const unsigned char R2Y = 77;
    const unsigned char G2Y = 150;
    const unsigned char B2Y = 29;
    
    int w = (int)(self.size(0) * self.size(1));
    
    // Byte neon optimize
    if (self.scalar_type() == otter::ScalarType::Byte) {
        unsigned char *rgb = self.data_ptr<unsigned char>();
        float *ptr = out.data_ptr<float>();
        
#if __ARM_NEON
        int nn = w >> 3;
        int remain = w - (nn << 3);
#else
        int remain = w;
#endif // __ARM_NEON
        
        
#if __ARM_NEON
#if __aarch64__
        uint8x8_t _R2Y = vdup_n_u8(R2Y);
        uint8x8_t _G2Y = vdup_n_u8(G2Y);
        uint8x8_t _B2Y = vdup_n_u8(B2Y);
        for (; nn > 0; nn--) {
            uint8x8x3_t _rgb = vld3_u8(rgb);
            
            uint16x8_t _y16 = vmull_u8(_rgb.val[0], _R2Y);
            _y16 = vmlal_u8(_y16, _rgb.val[1], _G2Y);
            _y16 = vmlal_u8(_y16, _rgb.val[2], _B2Y);
            _y16 = vshrq_n_u16(_y16, Y_shift);
            
            float32x4_t _ylow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_y16)));
            float32x4_t _yhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_y16)));
            
            vst1q_f32(ptr, _ylow);
            vst1q_f32(ptr + 4, _yhigh);
            
            rgb += 3 * 8;
            ptr += 8;
        }
#else
        
#endif // __aarch64__
#endif // __ARM_NEON
        
        for (; remain > 0; remain--) {
            *ptr = static_cast<unsigned char>((rgb[0] * R2Y + rgb[1] * G2Y + rgb[2] * B2Y) >> Y_shift);
            
            rgb += 3;
            ptr++;
        }
    } else {
        int remain = w;
        
        OTTER_DISPATCH_ALL_TYPES(self.scalar_type(), "RGBtoGray", [&] {
            unsigned char *rgb = self.data_ptr<unsigned char>();
            scalar_t *ptr = out.data_ptr<scalar_t>();
            
            for (; remain > 0; remain--) {
                *ptr = static_cast<scalar_t>((rgb[0] * R2Y + rgb[1] * G2Y + rgb[2] * B2Y) >> Y_shift);
                
                rgb += 3;
                ptr++;
            }
        });
    }
    
    return out;
}

}   // end namespace cv
}   // end namespace otter
