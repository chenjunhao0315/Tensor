//
//  DepthwiseConvKernelInt8NeonPack.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/28.
//

#include "DepthwiseConvKernelInt8NeonPack.hpp"
#include "Tensor.hpp"
#include "TensorFactory.hpp"
#include "Padding.hpp"
#include "Parallel.hpp"
#include "Quantize.hpp"

namespace otter {

#if __ARM_NEON__

Tensor& depthwise_conv2d_int8_neon_pack8_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0);
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_({output_size[0], output_size[1] / 8, output_size[2], output_size[3]});
    
    const int kernel_h = kernel_size[0];
    const int kernel_w = kernel_size[1];
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];
    
    int channels = int(input.size(1));
    int w = int(input.size(3));

    int outw = int(output.size(3));
    int outh = int(output.size(2));

    const int group = int(self.size(1) * self.elempack());
    
    const int maxk = kernel_w * kernel_h;
    
    Tensor weight_data_packed;
    if (weight_o.defined())
        weight_data_packed = weight_o;
    else
        weight_data_packed = weight.view({group, maxk}).packing(8);

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }
    
    auto input_a = input.accessor<signed char, 4, 8>()[0];
    auto output_ra = output.raw_accessor<int, 4>()[0];
    const signed char* weight_data_tm_ptr = (const signed char*)weight_data_packed.data_ptr();
    
    otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
        for (const auto g : otter::irange(begin, end)) {
            int* outptr_s8 = output_ra[g].data();
            const signed char* kptr = (const signed char*)weight_data_tm_ptr + maxk * g * 8;
            const auto m = input_a[g];

            for (int i = 0; i < outh; i++) {
                for (int j = 0; j < outw; j++) {
                    int32x4_t _sum0 = vdupq_n_s32(0);
                    int32x4_t _sum1 = vdupq_n_s32(0);

                    const signed char* sptr = m[i * stride_h].data() + j * stride_w * 8;

                    for (int k = 0; k < maxk; k++)
                    {
                        int8x8_t _val = vld1_s8(sptr + space_ofs[k] * 8);
                        int8x8_t _w = vld1_s8(kptr + k * 8);
                        int16x8_t _s0 = vmull_s8(_val, _w);
                        _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                        _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                    }

                    vst1q_s32(outptr_s8, _sum0);
                    vst1q_s32(outptr_s8 + 4, _sum1);
                    outptr_s8 += 8;
                }
            }
        }
    });
    
    return output;
}

Tensor depthwise_conv2d_int8_neon_pack8(
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
    
    return depthwise_conv2d_int8_neon_pack8_out(self, weight, weight_o, weight_int8_scales, bias, kernel_size, stride, padding, dilation, output);
}

Tensor& depthwise_conv2d_int8_neon_pack1_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0);
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_({output_size[0], output_size[1] / 8, output_size[2], output_size[3]});
    
    const int kernel_h = kernel_size[0];
    const int kernel_w = kernel_size[1];
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];
    
    int w = int(input.size(3));

    int outw = int(output.size(3));
    int outh = int(output.size(2));

    const int group = int(self.size(1) * self.elempack());
    
    const int maxk = kernel_w * kernel_h;
    
    Tensor weight_data_packed;
    if (weight_o.defined())
        weight_data_packed = weight_o;
    else
        weight_data_packed = weight;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }
    
    auto input_a = input.accessor<signed char, 4>()[0];
    auto output_ra = output.raw_accessor<int, 4>()[0];
    const signed char* weight_data_tm_ptr = (const signed char*)weight_data_packed.data_ptr();

    otter::parallel_for(0, group, 0, [&](int64_t begin, int64_t end) {
        for (const auto g : otter::irange(begin, end)) {
            int* outptr_s8 = output_ra[g].data();
            const signed char* kptr = (const signed char*)weight_data_tm_ptr + maxk * g;
            const auto m = input_a[g];

            for (int i = 0; i < outh; i++) {
                for (int j = 0; j < outw; j++) {
                    int sum = 0;

                    const signed char* sptr = m[i * stride_h].data() + j * stride_w;

                    for (int k = 0; k < maxk; k++) {
                        signed char val = sptr[space_ofs[k]];
                        signed char w = kptr[k];
                        sum += val * w;
                    }
                    
                    outptr_s8[0] = sum;
                    outptr_s8 += 1;
                }
            }
        }
    });
    
    return output;
}

Tensor depthwise_conv2d_int8_neon_pack1(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, otter::ScalarType::Int);
    
    return depthwise_conv2d_int8_neon_pack1_out(self, weight, weight_o, weight_int8_scales, bias, kernel_size, stride, padding, dilation, output);
}

#endif

}   // end namespace otter
