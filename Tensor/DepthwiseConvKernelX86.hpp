//
//  DepthwiseConvKernelX86.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/2.
//

#ifndef DepthwiseConvKernelX86_hpp
#define DepthwiseConvKernelX86_hpp

#include "ConvolutionUtils.hpp"

namespace otter {

Tensor& depthwise_conv2d_3x3s1_x86_sse_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);

Tensor depthwise_conv2d_3x3s1_x86_sse(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding);

Tensor& depthwise_conv2d_3x3s2_x86_sse_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);

Tensor depthwise_conv2d_3x3s2_x86_sse(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding);

}   // end namespace otter

#endif /* DepthwiseConvKernelX86_hpp */
