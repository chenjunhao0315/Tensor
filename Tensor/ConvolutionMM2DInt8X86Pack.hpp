//
//  ConvolutionMM2DInt8X86Pack.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/25.
//

#ifndef ConvolutionMM2DInt8X86Pack_hpp
#define ConvolutionMM2DInt8X86Pack_hpp

#include "ConvolutionUtils.hpp"

namespace otter {

#if __SSE2__

void convolution_im2col_sgemm_transform_kernel_pack1to4_int8_x86(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);

void convolution_im2col_sgemm_transform_kernel_pack8to1_int8_x86(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);

void convolution_im2col_sgemm_transform_kernel_pack8to4_int8_x86(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);

void im2col_sgemm_conv2d_int8_pack1to4_impl_x86(
    const Tensor& im2col_,
    const Tensor& kernel_tf_,
    const Tensor& bias_,
    int64_t input_channels,
    int64_t output_channels,
    Tensor& output);

void im2col_sgemm_conv2d_int8_pack8to1_impl_x86(
    const Tensor& im2col_,
    const Tensor& kernel_tf_,
    const Tensor& bias_,
    int64_t input_channels,
    int64_t output_channels,
    Tensor& output);

void im2col_sgemm_conv2d_int8_pack8to4_impl_x86(
    const Tensor& im2col_,
    const Tensor& kernel_tf_,
    const Tensor& bias_,
    int64_t input_channels,
    int64_t output_channels,
    Tensor& output);

Tensor& sgemm_conv2d_int8_pack1to4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);
    
Tensor sgemm_conv2d_int8_pack1to4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor& sgemm_conv2d_int8_pack8to1_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);
    
Tensor sgemm_conv2d_int8_pack8to1_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor& sgemm_conv2d_int8_pack8to4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);
    
Tensor sgemm_conv2d_int8_pack8to4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor& sgemm_conv2d_1x1s1_int8_pack1to4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);
    
Tensor sgemm_conv2d_1x1s1_int8_pack1to4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding);

Tensor& sgemm_conv2d_1x1s1_int8_pack8to1_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);
    
Tensor sgemm_conv2d_1x1s1_int8_pack8to1_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding);

Tensor& sgemm_conv2d_1x1s1_int8_pack8to4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);
    
Tensor sgemm_conv2d_1x1s1_int8_pack8to4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding);

#endif  // __SSE2__

}   // end namespace otter

#endif /* ConvolutionMM2DInt8X86Pack_hpp */
