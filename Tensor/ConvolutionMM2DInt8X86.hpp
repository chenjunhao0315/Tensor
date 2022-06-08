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

void convolution_im2col_sgemm_transform_kernel_int8_sse(const Tensor& kernel_, Tensor& kernel_tf, int64_t input_channels, int64_t output_channels, int64_t kernel_width, int64_t kernel_height);

void im2col_sgemm_conv2d_int8_impl_x86(
    const Tensor& im2col_,
    const Tensor& kernel_tf_,
    const Tensor& bias_,
    int64_t input_channels,
    int64_t output_channels,
    Tensor& output);

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

Tensor& sgemm_conv2d_1x1s1_int8_x86_out(
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

Tensor sgemm_conv2d_1x1s1_int8_x86(
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
