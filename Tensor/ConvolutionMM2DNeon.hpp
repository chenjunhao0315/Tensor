//
//  ConvolutionMM2DNeon.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/26.
//

#ifndef ConvolutionMM2DNeon_hpp
#define ConvolutionMM2DNeon_hpp

#include "ConvolutionUtils.hpp"

namespace otter {

void convolution_im2col_sgemm_transform_kernel_neon(
    const Tensor& kernel_,
    Tensor& kernel_tf,
    int64_t input_channels,
    int64_t output_channels,
    int64_t kernel_width,
    int64_t kernel_height);

void conv3x3s1_winograd64_transform_kernel_neon5(
    const Tensor& kernel_,
    Tensor& kernel_tf,
    int64_t input_channels,
    int64_t output_channels);

void conv3x3s2_transform_kernel_neon(
    const Tensor& kernel_,
    Tensor& kernel_tf,
    int64_t input_channels,
    int64_t output_channels);

Tensor& sgemm_conv2d_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);
    
Tensor sgemm_conv2d_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding);

Tensor& sgemm_conv2d_1x1s1_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);

Tensor sgemm_conv2d_1x1s1_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding);

Tensor& conv2d_1x1s1_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_1x1s1_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding);

Tensor& sgemm_conv2d_1x1s2_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);

Tensor sgemm_conv2d_1x1s2_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding);

Tensor& conv2d_3x3s1_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_3x3s1_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding);

Tensor& conv2d_3x3s1_winograd64_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_3x3s1_winograd64_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding);

Tensor& conv2d_3x3s2_packed_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_3x3s2_packed_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding);

}

#endif /* ConvolutionMM2DNeon_hpp */
