//
//  DepthwiseConvKernelInt8NeonPack.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/28.
//

#ifndef DepthwiseConvKernelInt8NeonPack_hpp
#define DepthwiseConvKernelInt8NeonPack_hpp

#include "ConvolutionUtils.hpp"

namespace otter {

#if __ARM_NEON__

Tensor& depthwise_conv2d_int8_neon_pack8_out(
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

Tensor depthwise_conv2d_int8_neon_pack8(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor& depthwise_conv2d_int8_neon_pack1_out(
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

Tensor depthwise_conv2d_int8_neon_pack1(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor& depthwise_conv2d_3x3s1_int8_neon_pack8_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor depthwise_conv2d_3x3s1_int8_neon_pack8(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding);

Tensor& depthwise_conv2d_3x3s2_int8_neon_pack8_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor depthwise_conv2d_3x3s2_int8_neon_pack8(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding);

#endif

}   // end namespace otter

#endif /* DepthwiseConvKernelInt8NeonPack_hpp */
