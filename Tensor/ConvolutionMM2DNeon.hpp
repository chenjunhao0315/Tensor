//
//  ConvolutionMM2D.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/15.
//

#ifndef ConvolutionMM2D_hpp
#define ConvolutionMM2D_hpp

#include "ArrayRef.hpp"

namespace otter {

class Tensor;

Tensor& slow_conv2d_forward_out_cpu(
    const Tensor& self,
    const Tensor& weight_,
    const Tensor& bias_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);

Tensor& slow_conv2d_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);

Tensor slow_conv2d(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding);


}   // end namespace otter

#endif /* ConvolutionMM2D_hpp */
