//
//  ConvolutionMM2DTransposeNeon.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/5/13.
//

#ifndef ConvolutionMM2DTransposeNeon_hpp
#define ConvolutionMM2DTransposeNeon_hpp

#include "Tensor.hpp"

namespace otter {

Tensor& deconv2d_4x4s2_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding,
    IntArrayRef output_padding,
    Tensor& output);

Tensor deconv2d_4x4s2_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding,
    IntArrayRef output_padding);

}   // end namespace otter

#endif /* ConvolutionMM2DTransposeNeon_hpp */
