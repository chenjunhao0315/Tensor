//
//  DepthwiseConvKernelX86Pack.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/17.
//

#ifndef DepthwiseConvKernelX86Pack_hpp
#define DepthwiseConvKernelX86Pack_hpp

#include "ConvolutionUtils.hpp"

namespace otter {

#if __SSE2__

Tensor& depthwise_conv2d_x86_pack4_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);

Tensor depthwise_conv2d_x86_pack4(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor& depthwise_conv2d_3x3s1_x86_pack4_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor depthwise_conv2d_3x3s1_x86_pack4(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

Tensor& depthwise_conv2d_3x3s2_x86_pack4_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor depthwise_conv2d_3x3s2_x86_pack4(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

Tensor& depthwise_conv2d_5x5s1_x86_pack4_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor depthwise_conv2d_5x5s1_x86_pack4(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

Tensor& depthwise_conv2d_5x5s2_x86_pack4_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor depthwise_conv2d_5x5s2_x86_pack4(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

#endif // __SSE2__

}   // end namespace otter

#endif /* DepthwiseConvKernelX86Pack_hpp */
