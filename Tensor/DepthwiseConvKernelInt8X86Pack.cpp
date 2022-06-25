//
//  DepthwiseConvKernelInt8X86Pack.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/25.
//

#include "DepthwiseConvKernelInt8X86Pack.hpp"
#include "Tensor.hpp"
#include "TensorFactory.hpp"
#include "Padding.hpp"
#include "Parallel.hpp"

namespace otter {

#if __SSE2__

Tensor& depthwise_conv2d_int8_x86_pack8_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_({output_size[0], output_size[1] / 8, output_size[2], output_size[3]});
    
    const int kernel_h = kernel_size[0];
    const int kernel_w = kernel_size[1];
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];
    
    int channels = int(input.size(0));
    int w = int(input.size(2));

    int outw = int(output.size(3));
    int outh = int(output.size(2));

    const int group = int(self.size(1) * self.elempack());
    
    const int maxk = kernel_w * kernel_h;
    
    Tensor weight_data_packed;
    if (weight_o.defined())
        weight_data_packed = weight_o;
    else
        weight_data_packed = weight.view({group, maxk}).packing(4);
    
    bool bias_term = (bias_.defined());
    const float* bias_data = (bias_.defined()) ? bias_.data_ptr<float>() : nullptr;

    
    return output;
}

Tensor depthwise_conv2d_int8_x86_pack8(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, otter::ScalarType::Int8);
    
    return depthwise_conv2d_int8_x86_pack8_out(self, weight, weight_o, weight_int8_scales, bias, kernel_size, stride, padding, dilation, output);
}

#endif

}   // end namespace otter
