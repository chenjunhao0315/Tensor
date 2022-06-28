//
//  ConvolutionMM2DInt8NeonPack.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/28.
//

#ifndef ConvolutionMM2DInt8NeonPack_hpp
#define ConvolutionMM2DInt8NeonPack_hpp

#include "ConvolutionUtils.hpp"

namespace otter {

#if __ARM_NEON__

void convolution_im2col_sgemm_transform_kernel_pack1to4_int8_neon(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);

void convolution_im2col_sgemm_transform_kernel_pack8to1_int8_neon(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);

void convolution_im2col_sgemm_transform_kernel_pack8to4_int8_neon(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);

void im2col_sgemm_conv2d_int8_pack1to4_impl_neon(
    const Tensor& im2col_,
    const Tensor& kernel_tf_,
    const Tensor& bias_,
    int64_t input_channels,
    int64_t output_channels,
    Tensor& output);

void im2col_sgemm_conv2d_int8_pack8to1_impl_neon(
    const Tensor& im2col_,
    const Tensor& kernel_tf_,
    const Tensor& bias_,
    int64_t input_channels,
    int64_t output_channels,
    Tensor& output);

void im2col_sgemm_conv2d_int8_pack8to4_impl_neon(
    const Tensor& im2col_,
    const Tensor& kernel_tf_,
    const Tensor& bias_,
    int64_t input_channels,
    int64_t output_channels,
    Tensor& output);

Tensor& sgemm_conv2d_int8_pack1to4_neon_out(
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
    
Tensor sgemm_conv2d_int8_pack1to4_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor& sgemm_conv2d_int8_pack8to1_neon_out(
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
    
Tensor sgemm_conv2d_int8_pack8to1_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor& sgemm_conv2d_int8_pack8to4_neon_out(
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
    
Tensor sgemm_conv2d_int8_pack8to4_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor& sgemm_conv2d_1x1s1_int8_pack1to4_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);
    
Tensor sgemm_conv2d_1x1s1_int8_pack1to4_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding);

Tensor& sgemm_conv2d_1x1s1_int8_pack8to1_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);
    
Tensor sgemm_conv2d_1x1s1_int8_pack8to1_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding);

Tensor& sgemm_conv2d_1x1s1_int8_pack8to4_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);
    
Tensor sgemm_conv2d_1x1s1_int8_pack8to4_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding);

#endif  // __ARM_NEON__

}   // end namespace otter

#endif /* ConvolutionMM2DInt8NeonPack_hpp */
