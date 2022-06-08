//
//  ConvolutionMM2D.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/15.
//

#ifndef ConvolutionMM2D_hpp
#define ConvolutionMM2D_hpp

#include "ConvolutionUtils.hpp"

namespace otter {

class Tensor;

Tensor& slow_conv2d_forward_out_cpu(
    const Tensor& self,
    const Tensor& weight_,
    const Tensor& bias_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);

Tensor& slow_conv2d_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);

Tensor slow_conv2d(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding);

Tensor& slide_win_conv2d_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);
    
Tensor slide_win_conv2d(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding);

Tensor& slide_win_conv2d_int8_out(
    const Tensor& self,
    const Tensor& input_scale_data,
    const Tensor& weight,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);
    
Tensor slide_win_conv2d_int8(
    const Tensor& self,
    const Tensor& input_scale_data,
    const Tensor& weight,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor& slide_win_conv2d_int8_fp32_out(
    const Tensor& self,
    const Tensor& input_scale_data,
    const Tensor& weight,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);
    
Tensor slide_win_conv2d_int8_fp32(
    const Tensor& self,
    const Tensor& input_scale_data,
    const Tensor& weight,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);


}   // end namespace otter

#endif /* ConvolutionMM2D_hpp */
