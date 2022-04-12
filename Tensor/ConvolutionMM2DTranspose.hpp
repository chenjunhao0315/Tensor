//
//  ConvolutionMM2DTranspose.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/3.
//

#ifndef ConvolutionMM2DTranspose_hpp
#define ConvolutionMM2DTranspose_hpp

#include "Tensor.hpp"

namespace otter {

Tensor slow_conv_transpose2d(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation);

Tensor& slow_conv_transpose2d_out(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    Tensor& output);

Tensor slide_win_conv_transpose2d(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation);

Tensor& slide_win_conv_transpose2d_out(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    Tensor& output);

}

#endif /* ConvolutionMM2DTranspose_hpp */
