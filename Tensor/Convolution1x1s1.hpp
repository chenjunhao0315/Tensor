//
//  Convolution1x1s1.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/24.
//

#ifndef Convolution1x1s1_hpp
#define Convolution1x1s1_hpp

#include "Tensor.hpp"

namespace otter {

Tensor conv_gemm_1x1s1(const Tensor& input, const Tensor& weight, const Tensor& bias, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding);

}

#endif /* Convolution1x1s1_hpp */
