//
//  DepthwiseConvKernel.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#ifndef DepthwiseConvKernel_hpp
#define DepthwiseConvKernel_hpp

#include "ConvolutionUtils.hpp"
#include "DispatchStub.hpp"

namespace otter {

class Tensor;

using convolution_depthwise3x3_winograd_fn = Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t);
DECLARE_DISPATCH(convolution_depthwise3x3_winograd_fn, convolution_depthwise3x3_winograd_stub);

Tensor& depthwise_conv2d_3x3s1_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);

Tensor depthwise_conv2d_3x3s1_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding);

Tensor& depthwise_conv2d_3x3s2_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);

Tensor depthwise_conv2d_3x3s2_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding);

Tensor& depthwise_conv2d_5x5s1_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);

Tensor depthwise_conv2d_5x5s1_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding);

Tensor& depthwise_conv2d_5x5s2_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);

Tensor depthwise_conv2d_5x5s2_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding);

}

#endif /* DepthwiseConvKernel_hpp */
