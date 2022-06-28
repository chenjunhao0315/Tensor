//
//  DepthwiseConvKernelInt8X86Pack.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/25.
//

#ifndef DepthwiseConvKernelInt8X86Pack_hpp
#define DepthwiseConvKernelInt8X86Pack_hpp

#include "ConvolutionUtils.hpp"

namespace otter {

#if __SSE2__

Tensor& depthwise_conv2d_int8_x86_pack8_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);

Tensor depthwise_conv2d_int8_x86_pack8(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor& depthwise_conv2d_int8_x86_pack1_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);

Tensor depthwise_conv2d_int8_x86_pack1(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

#endif

}   // end namespace otter

#endif /* DepthwiseConvKernelInt8X86Pack_hpp */
