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
#include "TensorPacking.hpp"

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

Tensor& depthwise_conv2d_3x3s1_int8_neon_pack8_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output) {
    
    int out_elempack = weight.size(0) % 8 == 0 ? 8 : 1;
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0);
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
    output = otter::empty({output_size[0], output_size[1] / out_elempack, output_size[2], output_size[3]}, otter::get_update_scalarType(otter::ScalarType::Int, out_elempack));
    
    const int w = input.size(3);
    const int outw = output.size(3);
    const int outh = output.size(2);
    const int group = int(self.size(1) * self.elempack());
    const int maxk = 3 * 3;
    
    Tensor weight_data_packed;
    if (weight_o.defined())
        weight_data_packed = weight_o;
    else
        weight_data_packed = weight.view({group, maxk}).packing(8);
    
    auto input_a = input.accessor<signed char, 4, 8>()[0];
    auto kernel_a = weight_data_packed.accessor<signed char, 2, 8>();
    auto output_ra = output.raw_accessor<int, 4>()[0];

    otter::parallel_for(0, input.size(1), 0, [&](int64_t begin, int64_t end) {
        for (const auto g : otter::irange(begin, end)) {
            auto out = output_ra[g];

            const signed char* k0 = kernel_a[g].data();

            int* outptr0 = out[0].data();
            int* outptr1 = out[1].data();

            const auto img0 = input_a[g];

            const signed char* r0 = img0[0].data();
            const signed char* r1 = img0[1].data();
            const signed char* r2 = img0[2].data();
            const signed char* r3 = img0[3].data();

            int8x8_t _k00 = vld1_s8(k0);
            int8x8_t _k01 = vld1_s8(k0 + 8);
            int8x8_t _k02 = vld1_s8(k0 + 16);
            int8x8_t _k10 = vld1_s8(k0 + 24);
            int8x8_t _k11 = vld1_s8(k0 + 32);
            int8x8_t _k12 = vld1_s8(k0 + 40);
            int8x8_t _k20 = vld1_s8(k0 + 48);
            int8x8_t _k21 = vld1_s8(k0 + 56);
            int8x8_t _k22 = vld1_s8(k0 + 64);

            int i = 0;
            for (; i + 1 < outh; i += 2)
            {
                int j = 0;
                for (; j + 1 < outw; j += 2)
                {
                    int8x16_t _r0001 = vld1q_s8(r0);
                    int8x16_t _r0203 = vld1q_s8(r0 + 16);
                    int8x16_t _r1011 = vld1q_s8(r1);
                    int8x16_t _r1213 = vld1q_s8(r1 + 16);
                    int8x16_t _r2021 = vld1q_s8(r2);
                    int8x16_t _r2223 = vld1q_s8(r2 + 16);
                    int8x16_t _r3031 = vld1q_s8(r3);
                    int8x16_t _r3233 = vld1q_s8(r3 + 16);

                    int16x8_t _s00 = vmull_s8(vget_low_s8(_r0001), _k00);
                    int16x8_t _s01 = vmull_s8(vget_high_s8(_r0001), _k01);
                    int16x8_t _s02 = vmull_s8(vget_low_s8(_r0203), _k02);
                    int16x8_t _s03 = vmull_s8(vget_low_s8(_r1011), _k10);
                    int16x8_t _s10 = vmull_s8(vget_high_s8(_r0001), _k00);
                    int16x8_t _s11 = vmull_s8(vget_low_s8(_r0203), _k01);
                    int16x8_t _s12 = vmull_s8(vget_high_s8(_r0203), _k02);
                    int16x8_t _s13 = vmull_s8(vget_high_s8(_r1011), _k10);

                    int16x8_t _s20 = vmull_s8(vget_low_s8(_r1011), _k00);
                    int16x8_t _s21 = vmull_s8(vget_high_s8(_r1011), _k01);
                    int16x8_t _s22 = vmull_s8(vget_low_s8(_r1213), _k02);
                    int16x8_t _s23 = vmull_s8(vget_low_s8(_r2021), _k10);
                    int16x8_t _s30 = vmull_s8(vget_high_s8(_r1011), _k00);
                    int16x8_t _s31 = vmull_s8(vget_low_s8(_r1213), _k01);
                    int16x8_t _s32 = vmull_s8(vget_high_s8(_r1213), _k02);
                    int16x8_t _s33 = vmull_s8(vget_high_s8(_r2021), _k10);

                    _s00 = vmlal_s8(_s00, vget_high_s8(_r1011), _k11);
                    _s01 = vmlal_s8(_s01, vget_low_s8(_r1213), _k12);
                    _s02 = vmlal_s8(_s02, vget_low_s8(_r2021), _k20);
                    _s03 = vmlal_s8(_s03, vget_high_s8(_r2021), _k21);
                    _s10 = vmlal_s8(_s10, vget_low_s8(_r1213), _k11);
                    _s11 = vmlal_s8(_s11, vget_high_s8(_r1213), _k12);
                    _s12 = vmlal_s8(_s12, vget_high_s8(_r2021), _k20);
                    _s13 = vmlal_s8(_s13, vget_low_s8(_r2223), _k21);

                    _s20 = vmlal_s8(_s20, vget_high_s8(_r2021), _k11);
                    _s21 = vmlal_s8(_s21, vget_low_s8(_r2223), _k12);
                    _s22 = vmlal_s8(_s22, vget_low_s8(_r3031), _k20);
                    _s23 = vmlal_s8(_s23, vget_high_s8(_r3031), _k21);
                    _s30 = vmlal_s8(_s30, vget_low_s8(_r2223), _k11);
                    _s31 = vmlal_s8(_s31, vget_high_s8(_r2223), _k12);
                    _s32 = vmlal_s8(_s32, vget_high_s8(_r3031), _k20);
                    _s33 = vmlal_s8(_s33, vget_low_s8(_r3233), _k21);

                    int16x8_t _s08 = vmull_s8(vget_low_s8(_r2223), _k22);
                    int16x8_t _s18 = vmull_s8(vget_high_s8(_r2223), _k22);
                    int16x8_t _s28 = vmull_s8(vget_low_s8(_r3233), _k22);
                    int16x8_t _s38 = vmull_s8(vget_high_s8(_r3233), _k22);

                    int32x4_t _sum00 = vaddl_s16(vget_low_s16(_s00), vget_low_s16(_s01));
                    int32x4_t _sum01 = vaddl_s16(vget_high_s16(_s00), vget_high_s16(_s01));
                    int32x4_t _sum02 = vaddl_s16(vget_low_s16(_s02), vget_low_s16(_s03));
                    int32x4_t _sum03 = vaddl_s16(vget_high_s16(_s02), vget_high_s16(_s03));
                    int32x4_t _sum10 = vaddl_s16(vget_low_s16(_s10), vget_low_s16(_s11));
                    int32x4_t _sum11 = vaddl_s16(vget_high_s16(_s10), vget_high_s16(_s11));
                    int32x4_t _sum12 = vaddl_s16(vget_low_s16(_s12), vget_low_s16(_s13));
                    int32x4_t _sum13 = vaddl_s16(vget_high_s16(_s12), vget_high_s16(_s13));
                    int32x4_t _sum20 = vaddl_s16(vget_low_s16(_s20), vget_low_s16(_s21));
                    int32x4_t _sum21 = vaddl_s16(vget_high_s16(_s20), vget_high_s16(_s21));
                    int32x4_t _sum22 = vaddl_s16(vget_low_s16(_s22), vget_low_s16(_s23));
                    int32x4_t _sum23 = vaddl_s16(vget_high_s16(_s22), vget_high_s16(_s23));
                    int32x4_t _sum30 = vaddl_s16(vget_low_s16(_s30), vget_low_s16(_s31));
                    int32x4_t _sum31 = vaddl_s16(vget_high_s16(_s30), vget_high_s16(_s31));
                    int32x4_t _sum32 = vaddl_s16(vget_low_s16(_s32), vget_low_s16(_s33));
                    int32x4_t _sum33 = vaddl_s16(vget_high_s16(_s32), vget_high_s16(_s33));
                    _sum00 = vaddw_s16(_sum00, vget_low_s16(_s08));
                    _sum01 = vaddw_s16(_sum01, vget_high_s16(_s08));
                    _sum10 = vaddw_s16(_sum10, vget_low_s16(_s18));
                    _sum11 = vaddw_s16(_sum11, vget_high_s16(_s18));
                    _sum20 = vaddw_s16(_sum20, vget_low_s16(_s28));
                    _sum21 = vaddw_s16(_sum21, vget_high_s16(_s28));
                    _sum30 = vaddw_s16(_sum30, vget_low_s16(_s38));
                    _sum31 = vaddw_s16(_sum31, vget_high_s16(_s38));
                    _sum00 = vaddq_s32(_sum00, _sum02);
                    _sum01 = vaddq_s32(_sum01, _sum03);
                    _sum10 = vaddq_s32(_sum10, _sum12);
                    _sum11 = vaddq_s32(_sum11, _sum13);
                    _sum20 = vaddq_s32(_sum20, _sum22);
                    _sum21 = vaddq_s32(_sum21, _sum23);
                    _sum30 = vaddq_s32(_sum30, _sum32);
                    _sum31 = vaddq_s32(_sum31, _sum33);

                    vst1q_s32(outptr0, _sum00);
                    vst1q_s32(outptr0 + 4, _sum01);
                    vst1q_s32(outptr0 + 8, _sum10);
                    vst1q_s32(outptr0 + 12, _sum11);
                    vst1q_s32(outptr1, _sum20);
                    vst1q_s32(outptr1 + 4, _sum21);
                    vst1q_s32(outptr1 + 8, _sum30);
                    vst1q_s32(outptr1 + 12, _sum31);

                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                    r3 += 16;
                    outptr0 += 16;
                    outptr1 += 16;
                }
                for (; j < outw; j++)
                {
                    int8x8_t _r00 = vld1_s8(r0);
                    int8x8_t _r01 = vld1_s8(r0 + 8);
                    int8x8_t _r02 = vld1_s8(r0 + 16);
                    int8x8_t _r10 = vld1_s8(r1);
                    int8x8_t _r11 = vld1_s8(r1 + 8);
                    int8x8_t _r12 = vld1_s8(r1 + 16);
                    int8x8_t _r20 = vld1_s8(r2);
                    int8x8_t _r21 = vld1_s8(r2 + 8);
                    int8x8_t _r22 = vld1_s8(r2 + 16);
                    int8x8_t _r30 = vld1_s8(r3);
                    int8x8_t _r31 = vld1_s8(r3 + 8);
                    int8x8_t _r32 = vld1_s8(r3 + 16);

                    int16x8_t _s00 = vmull_s8(_r00, _k00);
                    int16x8_t _s01 = vmull_s8(_r01, _k01);
                    int16x8_t _s02 = vmull_s8(_r02, _k02);
                    int16x8_t _s03 = vmull_s8(_r10, _k10);
                    int16x8_t _s10 = vmull_s8(_r10, _k00);
                    int16x8_t _s11 = vmull_s8(_r11, _k01);
                    int16x8_t _s12 = vmull_s8(_r12, _k02);
                    int16x8_t _s13 = vmull_s8(_r20, _k10);
                    _s00 = vmlal_s8(_s00, _r11, _k11);
                    _s01 = vmlal_s8(_s01, _r12, _k12);
                    _s02 = vmlal_s8(_s02, _r20, _k20);
                    _s03 = vmlal_s8(_s03, _r21, _k21);
                    _s10 = vmlal_s8(_s10, _r21, _k11);
                    _s11 = vmlal_s8(_s11, _r22, _k12);
                    _s12 = vmlal_s8(_s12, _r30, _k20);
                    _s13 = vmlal_s8(_s13, _r31, _k21);
                    int16x8_t _s08 = vmull_s8(_r22, _k22);
                    int16x8_t _s18 = vmull_s8(_r32, _k22);

                    int32x4_t _sum00 = vaddl_s16(vget_low_s16(_s00), vget_low_s16(_s01));
                    int32x4_t _sum01 = vaddl_s16(vget_high_s16(_s00), vget_high_s16(_s01));
                    int32x4_t _sum02 = vaddl_s16(vget_low_s16(_s02), vget_low_s16(_s03));
                    int32x4_t _sum03 = vaddl_s16(vget_high_s16(_s02), vget_high_s16(_s03));
                    int32x4_t _sum10 = vaddl_s16(vget_low_s16(_s10), vget_low_s16(_s11));
                    int32x4_t _sum11 = vaddl_s16(vget_high_s16(_s10), vget_high_s16(_s11));
                    int32x4_t _sum12 = vaddl_s16(vget_low_s16(_s12), vget_low_s16(_s13));
                    int32x4_t _sum13 = vaddl_s16(vget_high_s16(_s12), vget_high_s16(_s13));
                    _sum00 = vaddw_s16(_sum00, vget_low_s16(_s08));
                    _sum01 = vaddw_s16(_sum01, vget_high_s16(_s08));
                    _sum10 = vaddw_s16(_sum10, vget_low_s16(_s18));
                    _sum11 = vaddw_s16(_sum11, vget_high_s16(_s18));
                    _sum00 = vaddq_s32(_sum00, _sum02);
                    _sum01 = vaddq_s32(_sum01, _sum03);
                    _sum10 = vaddq_s32(_sum10, _sum12);
                    _sum11 = vaddq_s32(_sum11, _sum13);

                    vst1q_s32(outptr0, _sum00);
                    vst1q_s32(outptr0 + 4, _sum01);
                    vst1q_s32(outptr1, _sum10);
                    vst1q_s32(outptr1 + 4, _sum11);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                }

                r0 += 2 * 8 + w * 8;
                r1 += 2 * 8 + w * 8;
                r2 += 2 * 8 + w * 8;
                r3 += 2 * 8 + w * 8;

                outptr0 += outw * 8;
                outptr1 += outw * 8;
            }
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 1 < outw; j += 2)
                {
                    int8x16_t _r0001 = vld1q_s8(r0);
                    int8x16_t _r0203 = vld1q_s8(r0 + 16);
                    int8x16_t _r1011 = vld1q_s8(r1);
                    int8x16_t _r1213 = vld1q_s8(r1 + 16);
                    int8x16_t _r2021 = vld1q_s8(r2);
                    int8x16_t _r2223 = vld1q_s8(r2 + 16);

                    int16x8_t _s00 = vmull_s8(vget_low_s8(_r0001), _k00);
                    int16x8_t _s01 = vmull_s8(vget_high_s8(_r0001), _k01);
                    int16x8_t _s02 = vmull_s8(vget_low_s8(_r0203), _k02);
                    int16x8_t _s03 = vmull_s8(vget_low_s8(_r1011), _k10);
                    int16x8_t _s10 = vmull_s8(vget_high_s8(_r0001), _k00);
                    int16x8_t _s11 = vmull_s8(vget_low_s8(_r0203), _k01);
                    int16x8_t _s12 = vmull_s8(vget_high_s8(_r0203), _k02);
                    int16x8_t _s13 = vmull_s8(vget_high_s8(_r1011), _k10);
                    _s00 = vmlal_s8(_s00, vget_high_s8(_r1011), _k11);
                    _s01 = vmlal_s8(_s01, vget_low_s8(_r1213), _k12);
                    _s02 = vmlal_s8(_s02, vget_low_s8(_r2021), _k20);
                    _s03 = vmlal_s8(_s03, vget_high_s8(_r2021), _k21);
                    _s10 = vmlal_s8(_s10, vget_low_s8(_r1213), _k11);
                    _s11 = vmlal_s8(_s11, vget_high_s8(_r1213), _k12);
                    _s12 = vmlal_s8(_s12, vget_high_s8(_r2021), _k20);
                    _s13 = vmlal_s8(_s13, vget_low_s8(_r2223), _k21);
                    int16x8_t _s08 = vmull_s8(vget_low_s8(_r2223), _k22);
                    int16x8_t _s18 = vmull_s8(vget_high_s8(_r2223), _k22);

                    int32x4_t _sum00 = vaddl_s16(vget_low_s16(_s00), vget_low_s16(_s01));
                    int32x4_t _sum01 = vaddl_s16(vget_high_s16(_s00), vget_high_s16(_s01));
                    int32x4_t _sum02 = vaddl_s16(vget_low_s16(_s02), vget_low_s16(_s03));
                    int32x4_t _sum03 = vaddl_s16(vget_high_s16(_s02), vget_high_s16(_s03));
                    int32x4_t _sum10 = vaddl_s16(vget_low_s16(_s10), vget_low_s16(_s11));
                    int32x4_t _sum11 = vaddl_s16(vget_high_s16(_s10), vget_high_s16(_s11));
                    int32x4_t _sum12 = vaddl_s16(vget_low_s16(_s12), vget_low_s16(_s13));
                    int32x4_t _sum13 = vaddl_s16(vget_high_s16(_s12), vget_high_s16(_s13));
                    _sum00 = vaddw_s16(_sum00, vget_low_s16(_s08));
                    _sum01 = vaddw_s16(_sum01, vget_high_s16(_s08));
                    _sum10 = vaddw_s16(_sum10, vget_low_s16(_s18));
                    _sum11 = vaddw_s16(_sum11, vget_high_s16(_s18));
                    _sum00 = vaddq_s32(_sum00, _sum02);
                    _sum01 = vaddq_s32(_sum01, _sum03);
                    _sum10 = vaddq_s32(_sum10, _sum12);
                    _sum11 = vaddq_s32(_sum11, _sum13);

                    vst1q_s32(outptr0, _sum00);
                    vst1q_s32(outptr0 + 4, _sum01);
                    vst1q_s32(outptr0 + 8, _sum10);
                    vst1q_s32(outptr0 + 12, _sum11);

                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                    outptr0 += 16;
                }
                for (; j < outw; j++)
                {
                    int8x8_t _r00 = vld1_s8(r0);
                    int8x8_t _r01 = vld1_s8(r0 + 8);
                    int8x8_t _r02 = vld1_s8(r0 + 16);
                    int8x8_t _r10 = vld1_s8(r1);
                    int8x8_t _r11 = vld1_s8(r1 + 8);
                    int8x8_t _r12 = vld1_s8(r1 + 16);
                    int8x8_t _r20 = vld1_s8(r2);
                    int8x8_t _r21 = vld1_s8(r2 + 8);
                    int8x8_t _r22 = vld1_s8(r2 + 16);

                    int16x8_t _s0 = vmull_s8(_r00, _k00);
                    int16x8_t _s1 = vmull_s8(_r01, _k01);
                    int16x8_t _s2 = vmull_s8(_r02, _k02);
                    int16x8_t _s3 = vmull_s8(_r10, _k10);
                    _s0 = vmlal_s8(_s0, _r11, _k11);
                    _s1 = vmlal_s8(_s1, _r12, _k12);
                    _s2 = vmlal_s8(_s2, _r20, _k20);
                    _s3 = vmlal_s8(_s3, _r21, _k21);
                    int16x8_t _s4 = vmull_s8(_r22, _k22);

                    int32x4_t _sum0 = vaddl_s16(vget_low_s16(_s0), vget_low_s16(_s1));
                    int32x4_t _sum1 = vaddl_s16(vget_high_s16(_s0), vget_high_s16(_s1));
                    int32x4_t _sum2 = vaddl_s16(vget_low_s16(_s2), vget_low_s16(_s3));
                    int32x4_t _sum3 = vaddl_s16(vget_high_s16(_s2), vget_high_s16(_s3));
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s4));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s4));
                    _sum0 = vaddq_s32(_sum0, _sum2);
                    _sum1 = vaddq_s32(_sum1, _sum3);

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    outptr0 += 8;
                }

                r0 += 2 * 8;
                r1 += 2 * 8;
                r2 += 2 * 8;
            }
        }
    });
    
    return output;
}

Tensor depthwise_conv2d_3x3s1_int8_neon_pack8(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding) {
    
    Tensor output;
    
    return depthwise_conv2d_3x3s1_int8_neon_pack8_out(self, weight, weight_o, weight_int8_scales, bias, padding, output);
}

Tensor& depthwise_conv2d_3x3s2_int8_neon_pack8_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output) {
    
    int out_elempack = weight.size(0) % 8 == 0 ? 8 : 1;
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0);
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {2, 2}, padding);
    output = otter::empty({output_size[0], output_size[1] / out_elempack, output_size[2], output_size[3]}, otter::get_update_scalarType(otter::ScalarType::Int, out_elempack));
    
    const int w = input.size(3);
    const int outw = output.size(3);
    const int outh = output.size(2);
    const int group = int(self.size(1) * self.elempack());
    const int maxk = 3 * 3;
    
    Tensor weight_data_packed;
    if (weight_o.defined())
        weight_data_packed = weight_o;
    else
        weight_data_packed = weight.view({group, maxk}).packing(8);
    
    const int tailstep = (w - 2 * outw + w) * 8;
    
    auto input_a = input.accessor<signed char, 4, 8>()[0];
    auto kernel_a = weight_data_packed.accessor<signed char, 2, 8>();
    auto output_ra = output.raw_accessor<int, 4>()[0];

    otter::parallel_for(0, input.size(1), 0, [&](int64_t begin, int64_t end) {
        for (const auto g : otter::irange(begin, end)) {
            auto out = output_ra[g];

            const signed char* k0 = kernel_a[g].data();

            int* outptr0 = out.data();

            const auto img0 = input_a[g];

            const signed char* r0 = img0[0].data();
            const signed char* r1 = img0[1].data();
            const signed char* r2 = img0[2].data();

            int8x8_t _k00 = vld1_s8(k0);
            int8x8_t _k01 = vld1_s8(k0 + 8);
            int8x8_t _k02 = vld1_s8(k0 + 16);
            int8x8_t _k10 = vld1_s8(k0 + 24);
            int8x8_t _k11 = vld1_s8(k0 + 32);
            int8x8_t _k12 = vld1_s8(k0 + 40);
            int8x8_t _k20 = vld1_s8(k0 + 48);
            int8x8_t _k21 = vld1_s8(k0 + 56);
            int8x8_t _k22 = vld1_s8(k0 + 64);

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 1 < outw; j += 2)
                {
                    int8x8_t _r00 = vld1_s8(r0);
                    int8x8_t _r01 = vld1_s8(r0 + 8);
                    int8x8_t _r02 = vld1_s8(r0 + 16);
                    int8x8_t _r03 = vld1_s8(r0 + 24);
                    int8x8_t _r04 = vld1_s8(r0 + 32);
                    int8x8_t _r10 = vld1_s8(r1);
                    int8x8_t _r11 = vld1_s8(r1 + 8);
                    int8x8_t _r12 = vld1_s8(r1 + 16);
                    int8x8_t _r13 = vld1_s8(r1 + 24);
                    int8x8_t _r14 = vld1_s8(r1 + 32);
                    int8x8_t _r20 = vld1_s8(r2);
                    int8x8_t _r21 = vld1_s8(r2 + 8);
                    int8x8_t _r22 = vld1_s8(r2 + 16);
                    int8x8_t _r23 = vld1_s8(r2 + 24);
                    int8x8_t _r24 = vld1_s8(r2 + 32);

                    int16x8_t _s00 = vmull_s8(_r00, _k00);
                    int16x8_t _s01 = vmull_s8(_r01, _k01);
                    int16x8_t _s02 = vmull_s8(_r02, _k02);
                    int16x8_t _s03 = vmull_s8(_r10, _k10);
                    int16x8_t _s10 = vmull_s8(_r02, _k00);
                    int16x8_t _s11 = vmull_s8(_r03, _k01);
                    int16x8_t _s12 = vmull_s8(_r04, _k02);
                    int16x8_t _s13 = vmull_s8(_r12, _k10);
                    _s00 = vmlal_s8(_s00, _r11, _k11);
                    _s01 = vmlal_s8(_s01, _r12, _k12);
                    _s02 = vmlal_s8(_s02, _r20, _k20);
                    _s03 = vmlal_s8(_s03, _r21, _k21);
                    _s10 = vmlal_s8(_s10, _r13, _k11);
                    _s11 = vmlal_s8(_s11, _r14, _k12);
                    _s12 = vmlal_s8(_s12, _r22, _k20);
                    _s13 = vmlal_s8(_s13, _r23, _k21);
                    int16x8_t _s08 = vmull_s8(_r22, _k22);
                    int16x8_t _s18 = vmull_s8(_r24, _k22);

                    int32x4_t _sum00 = vaddl_s16(vget_low_s16(_s00), vget_low_s16(_s01));
                    int32x4_t _sum01 = vaddl_s16(vget_high_s16(_s00), vget_high_s16(_s01));
                    int32x4_t _sum02 = vaddl_s16(vget_low_s16(_s02), vget_low_s16(_s03));
                    int32x4_t _sum03 = vaddl_s16(vget_high_s16(_s02), vget_high_s16(_s03));
                    int32x4_t _sum10 = vaddl_s16(vget_low_s16(_s10), vget_low_s16(_s11));
                    int32x4_t _sum11 = vaddl_s16(vget_high_s16(_s10), vget_high_s16(_s11));
                    int32x4_t _sum12 = vaddl_s16(vget_low_s16(_s12), vget_low_s16(_s13));
                    int32x4_t _sum13 = vaddl_s16(vget_high_s16(_s12), vget_high_s16(_s13));
                    _sum00 = vaddw_s16(_sum00, vget_low_s16(_s08));
                    _sum01 = vaddw_s16(_sum01, vget_high_s16(_s08));
                    _sum10 = vaddw_s16(_sum10, vget_low_s16(_s18));
                    _sum11 = vaddw_s16(_sum11, vget_high_s16(_s18));
                    _sum00 = vaddq_s32(_sum00, _sum02);
                    _sum01 = vaddq_s32(_sum01, _sum03);
                    _sum10 = vaddq_s32(_sum10, _sum12);
                    _sum11 = vaddq_s32(_sum11, _sum13);

                    vst1q_s32(outptr0, _sum00);
                    vst1q_s32(outptr0 + 4, _sum01);
                    vst1q_s32(outptr0 + 8, _sum10);
                    vst1q_s32(outptr0 + 12, _sum11);

                    r0 += 32;
                    r1 += 32;
                    r2 += 32;
                    outptr0 += 16;
                }
                for (; j < outw; j++)
                {
                    int8x8_t _r00 = vld1_s8(r0);
                    int8x8_t _r01 = vld1_s8(r0 + 8);
                    int8x8_t _r02 = vld1_s8(r0 + 16);
                    int8x8_t _r10 = vld1_s8(r1);
                    int8x8_t _r11 = vld1_s8(r1 + 8);
                    int8x8_t _r12 = vld1_s8(r1 + 16);
                    int8x8_t _r20 = vld1_s8(r2);
                    int8x8_t _r21 = vld1_s8(r2 + 8);
                    int8x8_t _r22 = vld1_s8(r2 + 16);

                    int16x8_t _s0 = vmull_s8(_r00, _k00);
                    int16x8_t _s1 = vmull_s8(_r01, _k01);
                    int16x8_t _s2 = vmull_s8(_r02, _k02);
                    int16x8_t _s3 = vmull_s8(_r10, _k10);
                    _s0 = vmlal_s8(_s0, _r11, _k11);
                    _s1 = vmlal_s8(_s1, _r12, _k12);
                    _s2 = vmlal_s8(_s2, _r20, _k20);
                    _s3 = vmlal_s8(_s3, _r21, _k21);
                    int16x8_t _s4 = vmull_s8(_r22, _k22);

                    int32x4_t _sum0 = vaddl_s16(vget_low_s16(_s0), vget_low_s16(_s1));
                    int32x4_t _sum1 = vaddl_s16(vget_high_s16(_s0), vget_high_s16(_s1));
                    int32x4_t _sum2 = vaddl_s16(vget_low_s16(_s2), vget_low_s16(_s3));
                    int32x4_t _sum3 = vaddl_s16(vget_high_s16(_s2), vget_high_s16(_s3));
                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s4));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s4));
                    _sum0 = vaddq_s32(_sum0, _sum2);
                    _sum1 = vaddq_s32(_sum1, _sum3);

                    vst1q_s32(outptr0, _sum0);
                    vst1q_s32(outptr0 + 4, _sum1);

                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                    outptr0 += 8;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }
        }
    });
    
    return output;
}

Tensor depthwise_conv2d_3x3s2_int8_neon_pack8(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding) {
    
    Tensor output;
    
    return depthwise_conv2d_3x3s2_int8_neon_pack8_out(self, weight, weight_o, weight_int8_scales, bias, padding, output);
}

#endif

}   // end namespace otter
