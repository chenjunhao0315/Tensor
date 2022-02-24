//
//  Convolution1x1s1.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/24.
//

#include "ConvolutionUtils.hpp"
#include "Convolution1x1s1.hpp"
#include "TensorFactory.hpp"

namespace otter {

#if __ARM_NEON__
void conv_gemm_1x1s1_impl(Tensor& output, const Tensor& input, const Tensor& weight, const Tensor& bias, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding) {
    
}
#else
void conv_gemm_1x1s1_impl(Tensor& output, const Tensor& input, const Tensor& weight, const Tensor& bias, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding) {
    
}
#endif

Tensor conv_gemm_1x1s1(const Tensor& input, const Tensor& weight, const Tensor& bias, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding) {
    
    Tensor output = otter::empty(calculate_conv_output_size(input.sizes(), kernel_size, stride, padding), input.options());
    
    conv_gemm_1x1s1_impl(output, input, weight, bias, kernel_size, stride, padding);
    
    return output;
}

}
