//
//  ConvolutionMM2DInt8Neon.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/9.
//

#ifndef ConvolutionMM2DInt8Neon_hpp
#define ConvolutionMM2DInt8Neon_hpp

#include "Tensor.hpp"

namespace otter {

void convolution_im2col_sgemm_transform_kernel_int8_neon(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);

Tensor& sgemm_conv2d_int8_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);
    
Tensor sgemm_conv2d_int8_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

}   // end namespace otter

#endif /* ConvolutionMM2DInt8Neon_hpp */
