//
//  DepthwiseConvTransposeKernelNeon.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/5/13.
//

#ifndef DepthwiseConvTransposeKernelNeon_hpp
#define DepthwiseConvTransposeKernelNeon_hpp

#include "Tensor.hpp"

namespace otter {

void depthwise_deconv2d_kernel_transform(const Tensor& weight, Tensor& kernel_tf);

Tensor& depthwise_deconv2d_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    Tensor& output);

Tensor depthwise_deconv2d_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation);

}   // end namespace otter

#endif /* DepthwiseConvTransposeKernelNeon_hpp */
