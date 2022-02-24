//
//  DilatedConvolution.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/23.
//

#ifndef DilatedConvolution_hpp
#define DilatedConvolution_hpp

#include "ArrayRef.hpp"

namespace otter {

class Tensor;

Tensor slow_conv_dilated2d_forward_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias, IntArrayRef kernel_size, IntArrayRef stride_size, IntArrayRef pad_size, IntArrayRef dilation_size);

Tensor slow_conv_dilated2d(const Tensor& input, const Tensor& weight, const Tensor& bias, IntArrayRef kernel_size, IntArrayRef stride_size, IntArrayRef pad_size, IntArrayRef dilation_size);

}   // end namespace otter

#endif /* DilatedConvolution_hpp */
