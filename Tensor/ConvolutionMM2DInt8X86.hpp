//
//  ConvolutionMM2DInt8X86.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/5.
//

#ifndef ConvolutionMM2DInt8X86_hpp
#define ConvolutionMM2DInt8X86_hpp

#include "ConvolutionUtils.hpp"

namespace otter {

Tensor& sgemm_conv2d_int8_x86_out(
    const Tensor& self,
    const Tensor& input_scale_data,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);
    
Tensor sgemm_conv2d_int8_x86(
    const Tensor& self,
    const Tensor& input_scale_data,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding);

}   // end namespace otter

#endif /* ConvolutionMM2DInt8X86_hpp */
