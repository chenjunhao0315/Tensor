//
//  DepthwiseConvTransposeKernelNeonPack.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/29.
//

#ifndef DepthwiseConvTransposeKernelNeonPack_hpp
#define DepthwiseConvTransposeKernelNeonPack_hpp

#include "ConvolutionUtils.hpp"

namespace otter {

#if __ARM_NEON__

void depthwise_deconv2d_kernel_transform_pack_neon(const Tensor& weight, Tensor& kernel_tf);

Tensor& depthwise_deconv2d_pack4_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    Tensor& output);

Tensor depthwise_deconv2d_pack4_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation);

Tensor& depthwise_deconv2d_pack1_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    Tensor& output);

Tensor depthwise_deconv2d_pack1_neon(
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

#endif /* DepthwiseConvTransposeKernelNeonPack_hpp */
