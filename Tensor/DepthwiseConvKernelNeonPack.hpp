//
//  DepthwiseConvKernelNeonPack.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/17.
//

#ifndef DepthwiseConvKernelNeonPack_hpp
#define DepthwiseConvKernelNeonPack_hpp

#include "ConvolutionUtils.hpp"

namespace otter {

#if __ARM_NEON__

Tensor& depthwise_conv2d_neon_pack4_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);

Tensor depthwise_conv2d_neon_pack4(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

#endif // __ARM_NEON__

}   // end namespace otter

#endif /* DepthwiseConvKernelNeonPack_hpp */
