//
//  ConvolutionMM2DX86.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/2.
//

#ifndef ConvolutionMM2DX86_hpp
#define ConvolutionMM2DX86_hpp

#include "ConvolutionUtils.hpp"

namespace otter {

void convolution_im2col_sgemm_transform_kernel_x86(
    const Tensor& kernel_,
    Tensor& kernel_tf,
    int64_t input_channels,
    int64_t output_channels,
    int64_t kernel_width,
    int64_t kernel_height);

Tensor& sgemm_conv2d_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);
    
Tensor sgemm_conv2d_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding);

}   // end namespace otter

#endif /* ConvolutionMM2DX86_hpp */
