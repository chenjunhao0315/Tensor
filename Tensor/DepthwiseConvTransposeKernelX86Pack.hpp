//
//  DepthwiseConvTransposeKernelX86Pack.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/29.
//

#ifndef DepthwiseConvTransposeKernelX86Pack_hpp
#define DepthwiseConvTransposeKernelX86Pack_hpp

#include "ConvolutionUtils.hpp"

namespace otter {

#if __SSE2__

void depthwise_deconv2d_kernel_transform_pack_x86(const Tensor& weight, Tensor& kernel_tf);

Tensor& depthwise_deconv2d_pack4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    Tensor& output);

Tensor depthwise_deconv2d_pack4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation);

Tensor& depthwise_deconv2d_pack1_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    Tensor& output);

Tensor depthwise_deconv2d_pack1_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation);

#endif  // __SSE2__

}   // end namespace otter

#endif /* DepthwiseConvTransposeKernelX86Pack_hpp */
