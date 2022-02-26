//
//  ConvolutionMM2DNeon.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/26.
//

#include "ConvolutionMM2DNeon.hpp"
#include "Tensor.hpp"
#include "TensorFactory.hpp"
#include "im2col.hpp"

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

namespace otter {

#ifdef __ARM_NEON__
void im2col_sgemm_conv2d_impl(const Tensor& im2col, const Tensor& weight, const Tensor& bias, IntArrayRef stride, IntArrayRef padding, Tensor& output) {
    
    const int n_input_planes  = input.size(1);
    const int n_output_planes = output.size(1);
    
}

#else
void im2col_sgemm_conv2d_impl(const Tensor& im2col, const Tensor& weight, const Tensor& bias, IntArrayRef stride, IntArrayRef padding, Tensor& output) {}
#endif

Tensor& slow_conv2d_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    const int64_t kernel_height = kernel_size[0];
    const int64_t kernel_width  = kernel_size[1];
    const int64_t pad_height    = padding[0];
    const int64_t pad_width     = padding[1];
    const int64_t stride_height = stride[0];
    const int64_t stride_width  = stride[1];
    
    const int64_t dim_batch = 0;
    const int64_t dim_planes = 1;
    const int64_t dim_height = 2;
    const int64_t dim_width  = 3;
    
    const Tensor input = self.contiguous();
    const int64_t n_input_planes  = input.size(dim_planes);
    const int64_t input_height    = input.size(dim_height);
    const int64_t input_width     = input.size(dim_width);
    const int64_t n_output_planes = weight.size(0);
    const int64_t output_height   = (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
    const int64_t output_width    = (input_width  + 2 * pad_width  - kernel_width ) / stride_width  + 1;
    
    const int64_t batch_size      = input.size(dim_batch);
    
    Tensor im2col = otter::im2col_cpu(input, kernel_size, stride, padding, {1, 1});
    output.resize_({batch_size, n_output_planes, output_height, output_width});
    
    im2col_sgemm_conv2d_impl(im2col, weight, bias, stride, padding, output);
    
    
    return output;
}

Tensor slow_conv2d_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {
    
    auto out = otter::empty({}, self.options());
    slow_conv2d_neon_out(self, weight, bias, kernel_size, stride, padding, out);
    
    return out;
}

}
