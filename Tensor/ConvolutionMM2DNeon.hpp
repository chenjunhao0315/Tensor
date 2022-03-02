//
//  ConvolutionMM2DNeon.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/26.
//

#ifndef ConvolutionMM2DNeon_hpp
#define ConvolutionMM2DNeon_hpp

#include "ArrayRef.hpp"

namespace otter {

class Tensor;

Tensor& slow_conv2d_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output);
    
Tensor slow_conv2d_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding);

}

#endif /* ConvolutionMM2DNeon_hpp */
