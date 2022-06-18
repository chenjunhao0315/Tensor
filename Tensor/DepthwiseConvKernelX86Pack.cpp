//
//  DepthwiseConvKernelX86Pack.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/17.
//

#include "DepthwiseConvKernelX86Pack.hpp"
#include "Tensor.hpp"
#include "TensorFactory.hpp"
#include "Parallel.hpp"
#include "Padding.hpp"

#include "VecIntrinsic.hpp"

namespace otter {

#if __SSE2__

Tensor& depthwise_conv2d_x86_pack4_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_({output_size[0], output_size[1] / 4, output_size[2], output_size[3]});
    
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

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++) {
            for (int j = 0; j < kernel_w; j++) {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }
    
    auto input_a = input.accessor<float, 3, 4>();
    auto output_a = output.accessor<float, 4, 4>()[0];

    otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
        for (const auto g : otter::irange(begin, end)) {
            float* outptr = (float*)output_a[g].data();
            const float* kptr = (const float*)weight_data_packed.raw_data() + maxk * g * 4;
            const auto m = input_a[g];

            for (int i = 0; i < outh; i++) {
                for (int j = 0; j < outw; j++) {
                    __m128 _sum = _mm_set1_ps(0.f);

                    if (bias_term) {
                        _sum = _mm_loadu_ps(((const float*)bias_data) + g * 4);
                    }

                    const float* sptr = (const float*)m[i * stride_h].data() + j * stride_w * 4;

                    for (int k = 0; k < maxk; k++) {
                        __m128 _val = _mm_loadu_ps(sptr + space_ofs[k] * 4);
                        __m128 _w = _mm_loadu_ps(kptr + k * 4);
                        _sum = _mm_add_ps(_mm_mul_ps(_val, _w), _sum);
                    }

                    _mm_storeu_ps(outptr + j * 4, _sum);
                }

                outptr += outw * 4;
            }
        }
    });
    
    return output;
}

Tensor depthwise_conv2d_x86_pack4(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, otter::ScalarType::Float4);
    
    return depthwise_conv2d_x86_pack4_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
}

Tensor& depthwise_conv2d_3x3s1_x86_pack4_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_({output_size[0], output_size[1] / 4, output_size[2], output_size[3]});
    
    int w = int(input.size(2));

    int outw = int(output.size(3));
    int outh = int(output.size(2));

    const int group = int(self.size(1));
    
    const int maxk = 3 * 3;
    
    Tensor weight_data_packed;
    if (weight_o.defined())
        weight_data_packed = weight_o;
    else
        weight_data_packed = weight.view({group * 4, maxk}).packing(4);
    
    const float* bias = (bias_.defined()) ? bias_.data_ptr<float>() : nullptr;
    
    auto input_a = input.accessor<float, 3, 4>();
    auto output_a = output.accessor<float, 4, 4>()[0];
    auto kernel_a = weight_data_packed.accessor<float, 2, 4>();
    
    otter::parallel_for(0, group, 0, [&](int64_t begin, int64_t end) {
        for (const auto g : otter::irange(begin, end)) {
            auto out = output_a[g];

            __m128 _bias0 = bias ? _mm_loadu_ps((const float*)bias + g * 4) : _mm_set1_ps(0.f);

            const float* k0 = kernel_a[g].data();

            float* outptr0 = out[0].data();

            const auto img0 = input_a[g];

            const float* r0 = img0[0].data();
            const float* r1 = img0[1].data();
            const float* r2 = img0[2].data();

            __m128 _k00 = _mm_load_ps(k0);
            __m128 _k01 = _mm_load_ps(k0 + 4);
            __m128 _k02 = _mm_load_ps(k0 + 8);
            __m128 _k10 = _mm_load_ps(k0 + 12);
            __m128 _k11 = _mm_load_ps(k0 + 16);
            __m128 _k12 = _mm_load_ps(k0 + 20);
            __m128 _k20 = _mm_load_ps(k0 + 24);
            __m128 _k21 = _mm_load_ps(k0 + 28);
            __m128 _k22 = _mm_load_ps(k0 + 32);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 7 < outw; j += 8)
                {
                    __m128 _sum0 = _bias0;

                    __m128 _r00 = _mm_load_ps(r0);
                    __m128 _r01 = _mm_load_ps(r0 + 4);
                    __m128 _r02 = _mm_load_ps(r0 + 8);
                    __m128 _r10 = _mm_load_ps(r1);
                    __m128 _r11 = _mm_load_ps(r1 + 4);
                    __m128 _r12 = _mm_load_ps(r1 + 8);
                    __m128 _r20 = _mm_load_ps(r2);
                    __m128 _r21 = _mm_load_ps(r2 + 4);
                    __m128 _r22 = _mm_load_ps(r2 + 8);

                    _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);

                    __m128 _sum1 = _bias0;
                    __m128 _r03 = _mm_load_ps(r0 + 12);
                    __m128 _r13 = _mm_load_ps(r1 + 12);
                    __m128 _r23 = _mm_load_ps(r2 + 12);
                    _mm_store_ps(outptr0, _sum0);

                    _sum1 = _mm_comp_fmadd_ps(_k00, _r01, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k01, _r02, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k02, _r03, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k10, _r11, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k11, _r12, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k12, _r13, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k20, _r21, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k21, _r22, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k22, _r23, _sum1);

                    __m128 _sum2 = _bias0;
                    __m128 _r04 = _mm_load_ps(r0 + 16);
                    __m128 _r14 = _mm_load_ps(r1 + 16);
                    __m128 _r24 = _mm_load_ps(r2 + 16);
                    _mm_store_ps(outptr0 + 4, _sum1);

                    _sum2 = _mm_comp_fmadd_ps(_k00, _r02, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k01, _r03, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k02, _r04, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k10, _r12, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k11, _r13, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k12, _r14, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k20, _r22, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k21, _r23, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k22, _r24, _sum2);

                    __m128 _sum3 = _bias0;
                    __m128 _r05 = _mm_load_ps(r0 + 20);
                    __m128 _r15 = _mm_load_ps(r1 + 20);
                    __m128 _r25 = _mm_load_ps(r2 + 20);
                    _mm_store_ps(outptr0 + 8, _sum2);

                    _sum3 = _mm_comp_fmadd_ps(_k00, _r03, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k01, _r04, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k02, _r05, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k10, _r13, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k11, _r14, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k12, _r15, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k20, _r23, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k21, _r24, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k22, _r25, _sum3);

                    __m128 _sum4 = _bias0;
                    __m128 _r06 = _mm_load_ps(r0 + 24);
                    __m128 _r16 = _mm_load_ps(r1 + 24);
                    __m128 _r26 = _mm_load_ps(r2 + 24);
                    _mm_store_ps(outptr0 + 12, _sum3);

                    _sum4 = _mm_comp_fmadd_ps(_k00, _r04, _sum4);
                    _sum4 = _mm_comp_fmadd_ps(_k01, _r05, _sum4);
                    _sum4 = _mm_comp_fmadd_ps(_k02, _r06, _sum4);
                    _sum4 = _mm_comp_fmadd_ps(_k10, _r14, _sum4);
                    _sum4 = _mm_comp_fmadd_ps(_k11, _r15, _sum4);
                    _sum4 = _mm_comp_fmadd_ps(_k12, _r16, _sum4);
                    _sum4 = _mm_comp_fmadd_ps(_k20, _r24, _sum4);
                    _sum4 = _mm_comp_fmadd_ps(_k21, _r25, _sum4);
                    _sum4 = _mm_comp_fmadd_ps(_k22, _r26, _sum4);

                    __m128 _sum5 = _bias0;
                    __m128 _r07 = _mm_load_ps(r0 + 28);
                    __m128 _r17 = _mm_load_ps(r1 + 28);
                    __m128 _r27 = _mm_load_ps(r2 + 28);
                    _mm_store_ps(outptr0 + 16, _sum4);

                    _sum5 = _mm_comp_fmadd_ps(_k00, _r05, _sum5);
                    _sum5 = _mm_comp_fmadd_ps(_k01, _r06, _sum5);
                    _sum5 = _mm_comp_fmadd_ps(_k02, _r07, _sum5);
                    _sum5 = _mm_comp_fmadd_ps(_k10, _r15, _sum5);
                    _sum5 = _mm_comp_fmadd_ps(_k11, _r16, _sum5);
                    _sum5 = _mm_comp_fmadd_ps(_k12, _r17, _sum5);
                    _sum5 = _mm_comp_fmadd_ps(_k20, _r25, _sum5);
                    _sum5 = _mm_comp_fmadd_ps(_k21, _r26, _sum5);
                    _sum5 = _mm_comp_fmadd_ps(_k22, _r27, _sum5);

                    __m128 _sum6 = _bias0;
                    __m128 _r08 = _mm_load_ps(r0 + 32);
                    __m128 _r18 = _mm_load_ps(r1 + 32);
                    __m128 _r28 = _mm_load_ps(r2 + 32);
                    _mm_store_ps(outptr0 + 20, _sum5);

                    _sum6 = _mm_comp_fmadd_ps(_k00, _r06, _sum6);
                    _sum6 = _mm_comp_fmadd_ps(_k01, _r07, _sum6);
                    _sum6 = _mm_comp_fmadd_ps(_k02, _r08, _sum6);
                    _sum6 = _mm_comp_fmadd_ps(_k10, _r16, _sum6);
                    _sum6 = _mm_comp_fmadd_ps(_k11, _r17, _sum6);
                    _sum6 = _mm_comp_fmadd_ps(_k12, _r18, _sum6);
                    _sum6 = _mm_comp_fmadd_ps(_k20, _r26, _sum6);
                    _sum6 = _mm_comp_fmadd_ps(_k21, _r27, _sum6);
                    _sum6 = _mm_comp_fmadd_ps(_k22, _r28, _sum6);

                    __m128 _sum7 = _bias0;
                    __m128 _r09 = _mm_load_ps(r0 + 36);
                    __m128 _r19 = _mm_load_ps(r1 + 36);
                    __m128 _r29 = _mm_load_ps(r2 + 36);
                    _mm_store_ps(outptr0 + 24, _sum6);

                    _sum7 = _mm_comp_fmadd_ps(_k00, _r07, _sum7);
                    _sum7 = _mm_comp_fmadd_ps(_k01, _r08, _sum7);
                    _sum7 = _mm_comp_fmadd_ps(_k02, _r09, _sum7);
                    _sum7 = _mm_comp_fmadd_ps(_k10, _r17, _sum7);
                    _sum7 = _mm_comp_fmadd_ps(_k11, _r18, _sum7);
                    _sum7 = _mm_comp_fmadd_ps(_k12, _r19, _sum7);
                    _sum7 = _mm_comp_fmadd_ps(_k20, _r27, _sum7);
                    _sum7 = _mm_comp_fmadd_ps(_k21, _r28, _sum7);
                    _sum7 = _mm_comp_fmadd_ps(_k22, _r29, _sum7);
                    _mm_store_ps(outptr0 + 28, _sum7);

                    r0 += 32;
                    r1 += 32;
                    r2 += 32;
                    outptr0 += 32;
                }
                for (; j + 3 < outw; j += 4)
                {
                    __m128 _sum0 = _bias0;

                    __m128 _r00 = _mm_load_ps(r0);
                    __m128 _r01 = _mm_load_ps(r0 + 4);
                    __m128 _r02 = _mm_load_ps(r0 + 8);
                    __m128 _r10 = _mm_load_ps(r1);
                    __m128 _r11 = _mm_load_ps(r1 + 4);
                    __m128 _r12 = _mm_load_ps(r1 + 8);
                    __m128 _r20 = _mm_load_ps(r2);
                    __m128 _r21 = _mm_load_ps(r2 + 4);
                    __m128 _r22 = _mm_load_ps(r2 + 8);

                    _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);

                    __m128 _sum1 = _bias0;
                    __m128 _r03 = _mm_load_ps(r0 + 12);
                    __m128 _r13 = _mm_load_ps(r1 + 12);
                    __m128 _r23 = _mm_load_ps(r2 + 12);
                    _mm_store_ps(outptr0, _sum0);

                    _sum1 = _mm_comp_fmadd_ps(_k00, _r01, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k01, _r02, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k02, _r03, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k10, _r11, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k11, _r12, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k12, _r13, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k20, _r21, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k21, _r22, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k22, _r23, _sum1);

                    __m128 _sum2 = _bias0;
                    __m128 _r04 = _mm_load_ps(r0 + 16);
                    __m128 _r14 = _mm_load_ps(r1 + 16);
                    __m128 _r24 = _mm_load_ps(r2 + 16);
                    _mm_store_ps(outptr0 + 4, _sum1);

                    _sum2 = _mm_comp_fmadd_ps(_k00, _r02, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k01, _r03, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k02, _r04, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k10, _r12, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k11, _r13, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k12, _r14, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k20, _r22, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k21, _r23, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k22, _r24, _sum2);

                    __m128 _sum3 = _bias0;
                    __m128 _r05 = _mm_load_ps(r0 + 20);
                    __m128 _r15 = _mm_load_ps(r1 + 20);
                    __m128 _r25 = _mm_load_ps(r2 + 20);
                    _mm_store_ps(outptr0 + 8, _sum2);

                    _sum3 = _mm_comp_fmadd_ps(_k00, _r03, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k01, _r04, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k02, _r05, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k10, _r13, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k11, _r14, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k12, _r15, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k20, _r23, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k21, _r24, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k22, _r25, _sum3);

                    _mm_store_ps(outptr0 + 12, _sum3);

                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                    outptr0 += 16;
                }
                for (; j + 1 < outw; j += 2)
                {
                    __m128 _sum0 = _bias0;

                    __m128 _r00 = _mm_load_ps(r0);
                    __m128 _r01 = _mm_load_ps(r0 + 4);
                    __m128 _r02 = _mm_load_ps(r0 + 8);
                    __m128 _r10 = _mm_load_ps(r1);
                    __m128 _r11 = _mm_load_ps(r1 + 4);
                    __m128 _r12 = _mm_load_ps(r1 + 8);
                    __m128 _r20 = _mm_load_ps(r2);
                    __m128 _r21 = _mm_load_ps(r2 + 4);
                    __m128 _r22 = _mm_load_ps(r2 + 8);

                    _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);

                    __m128 _sum1 = _bias0;
                    __m128 _r03 = _mm_load_ps(r0 + 12);
                    __m128 _r13 = _mm_load_ps(r1 + 12);
                    __m128 _r23 = _mm_load_ps(r2 + 12);
                    _mm_store_ps(outptr0, _sum0);

                    _sum1 = _mm_comp_fmadd_ps(_k00, _r01, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k01, _r02, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k02, _r03, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k10, _r11, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k11, _r12, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k12, _r13, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k20, _r21, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k21, _r22, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k22, _r23, _sum1);

                    _mm_store_ps(outptr0 + 4, _sum1);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    outptr0 += 8;
                }
                for (; j < outw; j++)
                {
                    __m128 _sum0 = _bias0;

                    __m128 _r00 = _mm_load_ps(r0);
                    __m128 _r01 = _mm_load_ps(r0 + 4);
                    __m128 _r02 = _mm_load_ps(r0 + 8);
                    __m128 _r10 = _mm_load_ps(r1);
                    __m128 _r11 = _mm_load_ps(r1 + 4);
                    __m128 _r12 = _mm_load_ps(r1 + 8);
                    __m128 _r20 = _mm_load_ps(r2);
                    __m128 _r21 = _mm_load_ps(r2 + 4);
                    __m128 _r22 = _mm_load_ps(r2 + 8);

                    _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);

                    _mm_store_ps(outptr0, _sum0);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    outptr0 += 4;
                }

                r0 += 2 * 4;
                r1 += 2 * 4;
                r2 += 2 * 4;
            }
        }
    });
    
    return output;
}

Tensor depthwise_conv2d_3x3s1_x86_pack4(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, otter::ScalarType::Float4);
    
    return depthwise_conv2d_3x3s1_x86_pack4_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
}

Tensor& depthwise_conv2d_3x3s2_x86_pack4_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_({output_size[0], output_size[1] / 4, output_size[2], output_size[3]});
    
    int w = int(input.size(2));

    int outw = int(output.size(3));
    int outh = int(output.size(2));

    const int group = int(self.size(1));
    
    const int maxk = 3 * 3;
    
    Tensor weight_data_packed;
    if (weight_o.defined())
        weight_data_packed = weight_o;
    else
        weight_data_packed = weight.view({group * 4, maxk}).packing(4);
    
    const float* bias = (bias_.defined()) ? bias_.data_ptr<float>() : nullptr;
    
    const int tailstep = (w - 2 * outw + w) * 4;
    
    auto input_a = input.accessor<float, 3, 4>();
    auto output_a = output.accessor<float, 4, 4>()[0];
    auto kernel_a = weight_data_packed.accessor<float, 2, 4>();
    
    otter::parallel_for(0, group, 0, [&](int64_t begin, int64_t end) {
        for (const auto g : otter::irange(begin, end)) {
            auto out = output_a[g];

            __m128 _bias0 = bias ? _mm_loadu_ps((const float*)bias + g * 4) : _mm_set1_ps(0.f);

            const float* k0 = kernel_a[g].data();

            float* outptr0 = out[0].data();

            const auto img0 = input_a[g];

            const float* r0 = img0[0].data();
            const float* r1 = img0[1].data();
            const float* r2 = img0[2].data();

            __m128 _k00 = _mm_load_ps(k0);
            __m128 _k01 = _mm_load_ps(k0 + 4);
            __m128 _k02 = _mm_load_ps(k0 + 8);
            __m128 _k10 = _mm_load_ps(k0 + 12);
            __m128 _k11 = _mm_load_ps(k0 + 16);
            __m128 _k12 = _mm_load_ps(k0 + 20);
            __m128 _k20 = _mm_load_ps(k0 + 24);
            __m128 _k21 = _mm_load_ps(k0 + 28);
            __m128 _k22 = _mm_load_ps(k0 + 32);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;
                for (; j + 3 < outw; j += 4)
                {
                    __m128 _sum0 = _bias0;

                    __m128 _r00 = _mm_load_ps(r0);
                    __m128 _r01 = _mm_load_ps(r0 + 4);
                    __m128 _r02 = _mm_load_ps(r0 + 8);
                    __m128 _r10 = _mm_load_ps(r1);
                    __m128 _r11 = _mm_load_ps(r1 + 4);
                    __m128 _r12 = _mm_load_ps(r1 + 8);
                    __m128 _r20 = _mm_load_ps(r2);
                    __m128 _r21 = _mm_load_ps(r2 + 4);
                    __m128 _r22 = _mm_load_ps(r2 + 8);

                    _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);

                    __m128 _sum1 = _bias0;
                    __m128 _r03 = _mm_load_ps(r0 + 12);
                    __m128 _r13 = _mm_load_ps(r1 + 12);
                    __m128 _r23 = _mm_load_ps(r2 + 12);
                    __m128 _r04 = _mm_load_ps(r0 + 16);
                    __m128 _r14 = _mm_load_ps(r1 + 16);
                    __m128 _r24 = _mm_load_ps(r2 + 16);
                    _mm_store_ps(outptr0, _sum0);

                    _sum1 = _mm_comp_fmadd_ps(_k00, _r02, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k01, _r03, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k02, _r04, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k10, _r12, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k11, _r13, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k12, _r14, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k20, _r22, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k21, _r23, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k22, _r24, _sum1);

                    __m128 _sum2 = _bias0;
                    __m128 _r05 = _mm_load_ps(r0 + 20);
                    __m128 _r15 = _mm_load_ps(r1 + 20);
                    __m128 _r25 = _mm_load_ps(r2 + 20);
                    __m128 _r06 = _mm_load_ps(r0 + 24);
                    __m128 _r16 = _mm_load_ps(r1 + 24);
                    __m128 _r26 = _mm_load_ps(r2 + 24);
                    _mm_store_ps(outptr0 + 4, _sum1);

                    _sum2 = _mm_comp_fmadd_ps(_k00, _r04, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k01, _r05, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k02, _r06, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k10, _r14, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k11, _r15, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k12, _r16, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k20, _r24, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k21, _r25, _sum2);
                    _sum2 = _mm_comp_fmadd_ps(_k22, _r26, _sum2);

                    __m128 _sum3 = _bias0;
                    __m128 _r07 = _mm_load_ps(r0 + 28);
                    __m128 _r17 = _mm_load_ps(r1 + 28);
                    __m128 _r27 = _mm_load_ps(r2 + 28);
                    __m128 _r08 = _mm_load_ps(r0 + 32);
                    __m128 _r18 = _mm_load_ps(r1 + 32);
                    __m128 _r28 = _mm_load_ps(r2 + 32);
                    _mm_store_ps(outptr0 + 8, _sum2);

                    _sum3 = _mm_comp_fmadd_ps(_k00, _r06, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k01, _r07, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k02, _r08, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k10, _r16, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k11, _r17, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k12, _r18, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k20, _r26, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k21, _r27, _sum3);
                    _sum3 = _mm_comp_fmadd_ps(_k22, _r28, _sum3);

                    _mm_store_ps(outptr0 + 12, _sum3);

                    r0 += 2 * 16;
                    r1 += 2 * 16;
                    r2 += 2 * 16;
                    outptr0 += 16;
                }
                for (; j + 1 < outw; j += 2)
                {
                    __m128 _sum0 = _bias0;

                    __m128 _r00 = _mm_load_ps(r0);
                    __m128 _r01 = _mm_load_ps(r0 + 4);
                    __m128 _r02 = _mm_load_ps(r0 + 8);
                    __m128 _r10 = _mm_load_ps(r1);
                    __m128 _r11 = _mm_load_ps(r1 + 4);
                    __m128 _r12 = _mm_load_ps(r1 + 8);
                    __m128 _r20 = _mm_load_ps(r2);
                    __m128 _r21 = _mm_load_ps(r2 + 4);
                    __m128 _r22 = _mm_load_ps(r2 + 8);

                    _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);

                    __m128 _sum1 = _bias0;
                    __m128 _r03 = _mm_load_ps(r0 + 12);
                    __m128 _r13 = _mm_load_ps(r1 + 12);
                    __m128 _r23 = _mm_load_ps(r2 + 12);
                    __m128 _r04 = _mm_load_ps(r0 + 16);
                    __m128 _r14 = _mm_load_ps(r1 + 16);
                    __m128 _r24 = _mm_load_ps(r2 + 16);
                    _mm_store_ps(outptr0, _sum0);

                    _sum1 = _mm_comp_fmadd_ps(_k00, _r02, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k01, _r03, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k02, _r04, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k10, _r12, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k11, _r13, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k12, _r14, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k20, _r22, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k21, _r23, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k22, _r24, _sum1);

                    _mm_store_ps(outptr0 + 4, _sum1);

                    r0 += 2 * 8;
                    r1 += 2 * 8;
                    r2 += 2 * 8;
                    outptr0 += 8;
                }
                for (; j < outw; j++)
                {
                    __m128 _sum0 = _bias0;

                    __m128 _r00 = _mm_load_ps(r0);
                    __m128 _r01 = _mm_load_ps(r0 + 4);
                    __m128 _r02 = _mm_load_ps(r0 + 8);
                    __m128 _r10 = _mm_load_ps(r1);
                    __m128 _r11 = _mm_load_ps(r1 + 4);
                    __m128 _r12 = _mm_load_ps(r1 + 8);
                    __m128 _r20 = _mm_load_ps(r2);
                    __m128 _r21 = _mm_load_ps(r2 + 4);
                    __m128 _r22 = _mm_load_ps(r2 + 8);

                    _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);
                    _mm_store_ps(outptr0, _sum0);
                    r0 += 2 * 4;
                    r1 += 2 * 4;
                    r2 += 2 * 4;
                    outptr0 += 4;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }
        }
    });
    
    return output;
}

Tensor depthwise_conv2d_3x3s2_x86_pack4(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, otter::ScalarType::Float4);
    
    return depthwise_conv2d_3x3s2_x86_pack4_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
}

Tensor& depthwise_conv2d_5x5s1_x86_pack4_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_({output_size[0], output_size[1] / 4, output_size[2], output_size[3]});
    
    int w = int(input.size(2));

    int outw = int(output.size(3));
    int outh = int(output.size(2));

    const int group = int(self.size(1));
    
    const int maxk = 5 * 5;
    
    Tensor weight_data_packed;
    if (weight_o.defined())
        weight_data_packed = weight_o;
    else
        weight_data_packed = weight.view({group * 4, maxk}).packing(4);
    
    const float* bias = (bias_.defined()) ? bias_.data_ptr<float>() : nullptr;
    
    auto input_a = input.accessor<float, 3, 4>();
    auto output_a = output.accessor<float, 4, 4>()[0];
    auto kernel_a = weight_data_packed.accessor<float, 2, 4>();
    
    otter::parallel_for(0, group, 0, [&](int64_t begin, int64_t end) {
        for (const auto g : otter::irange(begin, end)) {
            auto out = output_a[g];

            __m128 _bias0 = bias ? _mm_loadu_ps(bias + g * 4) : _mm_setzero_ps();

            const float* k0 = kernel_a[g].data();

            float* outptr0 = out[0].data();
            float* outptr1 = out[1].data();

            const auto img0 = input_a[g];

            const float* r0 = img0[0].data();
            const float* r1 = img0[1].data();
            const float* r2 = img0[2].data();
            const float* r3 = img0[3].data();
            const float* r4 = img0[4].data();
            const float* r5 = img0[5].data();

            int i = 0;
            for (; i + 1 < outh; i += 2)
            {
                int j = 0;
                for (; j < outw; j++)
                {
                    __m128 _sum0 = _bias0;
                    __m128 _sum1 = _bias0;

                    __m128 _r00 = _mm_load_ps(r0);
                    __m128 _r01 = _mm_load_ps(r0 + 4);
                    __m128 _r02 = _mm_load_ps(r0 + 4 * 2);
                    __m128 _r03 = _mm_load_ps(r0 + 4 * 3);
                    __m128 _r04 = _mm_load_ps(r0 + 4 * 4);

                    __m128 _k00 = _mm_load_ps(k0);
                    __m128 _k01 = _mm_load_ps(k0 + 4);
                    __m128 _k02 = _mm_load_ps(k0 + 4 * 2);
                    __m128 _k03 = _mm_load_ps(k0 + 4 * 3);
                    __m128 _k04 = _mm_load_ps(k0 + 4 * 4);
                    k0 += 4 * 5;

                    _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k03, _r03, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k04, _r04, _sum0);

                    __m128 _r10 = _mm_load_ps(r1);
                    __m128 _r11 = _mm_load_ps(r1 + 4);
                    __m128 _r12 = _mm_load_ps(r1 + 4 * 2);
                    __m128 _r13 = _mm_load_ps(r1 + 4 * 3);
                    __m128 _r14 = _mm_load_ps(r1 + 4 * 4);

                    _sum1 = _mm_comp_fmadd_ps(_k00, _r10, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k01, _r11, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k02, _r12, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k03, _r13, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k04, _r14, _sum1);

                    __m128 _k10 = _mm_load_ps(k0);
                    __m128 _k11 = _mm_load_ps(k0 + 4);
                    __m128 _k12 = _mm_load_ps(k0 + 4 * 2);
                    __m128 _k13 = _mm_load_ps(k0 + 4 * 3);
                    __m128 _k14 = _mm_load_ps(k0 + 4 * 4);
                    k0 += 4 * 5;

                    _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k13, _r13, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k14, _r14, _sum0);

                    __m128 _r20 = _mm_load_ps(r2);
                    __m128 _r21 = _mm_load_ps(r2 + 4);
                    __m128 _r22 = _mm_load_ps(r2 + 4 * 2);
                    __m128 _r23 = _mm_load_ps(r2 + 4 * 3);
                    __m128 _r24 = _mm_load_ps(r2 + 4 * 4);

                    _sum1 = _mm_comp_fmadd_ps(_k10, _r20, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k11, _r21, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k12, _r22, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k13, _r23, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k14, _r24, _sum1);

                    __m128 _k20 = _mm_load_ps(k0);
                    __m128 _k21 = _mm_load_ps(k0 + 4);
                    __m128 _k22 = _mm_load_ps(k0 + 4 * 2);
                    __m128 _k23 = _mm_load_ps(k0 + 4 * 3);
                    __m128 _k24 = _mm_load_ps(k0 + 4 * 4);
                    k0 += 4 * 5;

                    _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k23, _r23, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k24, _r24, _sum0);

                    __m128 _r30 = _mm_load_ps(r3);
                    __m128 _r31 = _mm_load_ps(r3 + 4);
                    __m128 _r32 = _mm_load_ps(r3 + 4 * 2);
                    __m128 _r33 = _mm_load_ps(r3 + 4 * 3);
                    __m128 _r34 = _mm_load_ps(r3 + 4 * 4);

                    _sum1 = _mm_comp_fmadd_ps(_k20, _r30, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k21, _r31, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k22, _r32, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k23, _r33, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k24, _r34, _sum1);

                    __m128 _k30 = _mm_load_ps(k0);
                    __m128 _k31 = _mm_load_ps(k0 + 4);
                    __m128 _k32 = _mm_load_ps(k0 + 4 * 2);
                    __m128 _k33 = _mm_load_ps(k0 + 4 * 3);
                    __m128 _k34 = _mm_load_ps(k0 + 4 * 4);
                    k0 += 4 * 5;

                    _sum0 = _mm_comp_fmadd_ps(_k30, _r30, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k31, _r31, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k32, _r32, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k33, _r33, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k34, _r34, _sum0);

                    __m128 _r40 = _mm_load_ps(r4);
                    __m128 _r41 = _mm_load_ps(r4 + 4);
                    __m128 _r42 = _mm_load_ps(r4 + 4 * 2);
                    __m128 _r43 = _mm_load_ps(r4 + 4 * 3);
                    __m128 _r44 = _mm_load_ps(r4 + 4 * 4);

                    _sum1 = _mm_comp_fmadd_ps(_k30, _r40, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k31, _r41, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k32, _r42, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k33, _r43, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k34, _r44, _sum1);

                    __m128 _k40 = _mm_load_ps(k0);
                    __m128 _k41 = _mm_load_ps(k0 + 4);
                    __m128 _k42 = _mm_load_ps(k0 + 4 * 2);
                    __m128 _k43 = _mm_load_ps(k0 + 4 * 3);
                    __m128 _k44 = _mm_load_ps(k0 + 4 * 4);
                    k0 -= 4 * 20;

                    _sum0 = _mm_comp_fmadd_ps(_k40, _r40, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k41, _r41, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k42, _r42, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k43, _r43, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k44, _r44, _sum0);

                    __m128 _r50 = _mm_load_ps(r5);
                    __m128 _r51 = _mm_load_ps(r5 + 4);
                    __m128 _r52 = _mm_load_ps(r5 + 4 * 2);
                    __m128 _r53 = _mm_load_ps(r5 + 4 * 3);
                    __m128 _r54 = _mm_load_ps(r5 + 4 * 4);

                    _sum1 = _mm_comp_fmadd_ps(_k40, _r50, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k41, _r51, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k42, _r52, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k43, _r53, _sum1);
                    _sum1 = _mm_comp_fmadd_ps(_k44, _r54, _sum1);

                    _mm_store_ps(outptr0, _sum0);
                    _mm_store_ps(outptr1, _sum1);

                    outptr0 += 4;
                    outptr1 += 4;

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    r4 += 4;
                    r5 += 4;
                }

                r0 += 4 * 4 + w * 4;
                r1 += 4 * 4 + w * 4;
                r2 += 4 * 4 + w * 4;
                r3 += 4 * 4 + w * 4;
                r4 += 4 * 4 + w * 4;
                r5 += 4 * 4 + w * 4;

                outptr0 += outw * 4;
                outptr1 += outw * 4;
            }
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j < outw; j++)
                {
                    __m128 _sum0 = _bias0;

                    __m128 _r00 = _mm_load_ps(r0);
                    __m128 _r01 = _mm_load_ps(r0 + 4);
                    __m128 _r02 = _mm_load_ps(r0 + 4 * 2);
                    __m128 _r03 = _mm_load_ps(r0 + 4 * 3);
                    __m128 _r04 = _mm_load_ps(r0 + 4 * 4);

                    __m128 _k00 = _mm_load_ps(k0);
                    __m128 _k01 = _mm_load_ps(k0 + 4);
                    __m128 _k02 = _mm_load_ps(k0 + 4 * 2);
                    __m128 _k03 = _mm_load_ps(k0 + 4 * 3);
                    __m128 _k04 = _mm_load_ps(k0 + 4 * 4);
                    k0 += 4 * 5;

                    _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k03, _r03, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k04, _r04, _sum0);

                    __m128 _r10 = _mm_load_ps(r1);
                    __m128 _r11 = _mm_load_ps(r1 + 4);
                    __m128 _r12 = _mm_load_ps(r1 + 4 * 2);
                    __m128 _r13 = _mm_load_ps(r1 + 4 * 3);
                    __m128 _r14 = _mm_load_ps(r1 + 4 * 4);

                    __m128 _k10 = _mm_load_ps(k0);
                    __m128 _k11 = _mm_load_ps(k0 + 4);
                    __m128 _k12 = _mm_load_ps(k0 + 4 * 2);
                    __m128 _k13 = _mm_load_ps(k0 + 4 * 3);
                    __m128 _k14 = _mm_load_ps(k0 + 4 * 4);
                    k0 += 4 * 5;

                    _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k13, _r13, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k14, _r14, _sum0);

                    __m128 _r20 = _mm_load_ps(r2);
                    __m128 _r21 = _mm_load_ps(r2 + 4);
                    __m128 _r22 = _mm_load_ps(r2 + 4 * 2);
                    __m128 _r23 = _mm_load_ps(r2 + 4 * 3);
                    __m128 _r24 = _mm_load_ps(r2 + 4 * 4);

                    __m128 _k20 = _mm_load_ps(k0);
                    __m128 _k21 = _mm_load_ps(k0 + 4);
                    __m128 _k22 = _mm_load_ps(k0 + 4 * 2);
                    __m128 _k23 = _mm_load_ps(k0 + 4 * 3);
                    __m128 _k24 = _mm_load_ps(k0 + 4 * 4);
                    k0 += 4 * 5;

                    _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k23, _r23, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k24, _r24, _sum0);

                    __m128 _r30 = _mm_load_ps(r3);
                    __m128 _r31 = _mm_load_ps(r3 + 4);
                    __m128 _r32 = _mm_load_ps(r3 + 4 * 2);
                    __m128 _r33 = _mm_load_ps(r3 + 4 * 3);
                    __m128 _r34 = _mm_load_ps(r3 + 4 * 4);

                    __m128 _k30 = _mm_load_ps(k0);
                    __m128 _k31 = _mm_load_ps(k0 + 4);
                    __m128 _k32 = _mm_load_ps(k0 + 4 * 2);
                    __m128 _k33 = _mm_load_ps(k0 + 4 * 3);
                    __m128 _k34 = _mm_load_ps(k0 + 4 * 4);
                    k0 += 4 * 5;

                    _sum0 = _mm_comp_fmadd_ps(_k30, _r30, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k31, _r31, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k32, _r32, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k33, _r33, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k34, _r34, _sum0);

                    __m128 _r40 = _mm_load_ps(r4);
                    __m128 _r41 = _mm_load_ps(r4 + 4);
                    __m128 _r42 = _mm_load_ps(r4 + 4 * 2);
                    __m128 _r43 = _mm_load_ps(r4 + 4 * 3);
                    __m128 _r44 = _mm_load_ps(r4 + 4 * 4);

                    __m128 _k40 = _mm_load_ps(k0);
                    __m128 _k41 = _mm_load_ps(k0 + 4);
                    __m128 _k42 = _mm_load_ps(k0 + 4 * 2);
                    __m128 _k43 = _mm_load_ps(k0 + 4 * 3);
                    __m128 _k44 = _mm_load_ps(k0 + 4 * 4);
                    k0 -= 4 * 20;

                    _sum0 = _mm_comp_fmadd_ps(_k40, _r40, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k41, _r41, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k42, _r42, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k43, _r43, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k44, _r44, _sum0);

                    _mm_store_ps(outptr0, _sum0);

                    outptr0 += 4;

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    r4 += 4;
                }

                r0 += 4 * 4;
                r1 += 4 * 4;
                r2 += 4 * 4;
                r3 += 4 * 4;
                r4 += 4 * 4;
            }
        }
    });
    
    return output;
}

Tensor depthwise_conv2d_5x5s1_x86_pack4(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, otter::ScalarType::Float4);
    
    return depthwise_conv2d_5x5s1_x86_pack4_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
}

Tensor& depthwise_conv2d_5x5s2_x86_pack4_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_({output_size[0], output_size[1] / 4, output_size[2], output_size[3]});
    
    int w = int(input.size(2));

    int outw = int(output.size(3));
    int outh = int(output.size(2));

    const int group = int(self.size(1));
    
    const int maxk = 5 * 5;
    
    Tensor weight_data_packed;
    if (weight_o.defined())
        weight_data_packed = weight_o;
    else
        weight_data_packed = weight.view({group * 4, maxk}).packing(4);
    
    const int tailstep = (w - 2 * outw + w) * 4;
    
    const float* bias = (bias_.defined()) ? bias_.data_ptr<float>() : nullptr;
    
    auto input_a = input.accessor<float, 3, 4>();
    auto output_a = output.accessor<float, 4, 4>()[0];
    auto kernel_a = weight_data_packed.accessor<float, 2, 4>();
    
    otter::parallel_for(0, group, 0, [&](int64_t begin, int64_t end) {
        for (const auto g : otter::irange(begin, end)) {
            auto out = output_a[g];

            __m128 _bias0 = bias ? _mm_loadu_ps(bias + g * 4) : _mm_setzero_ps();

            const float* k0 = kernel_a[g].data();

            float* outptr0 = out.data();

            const auto img0 = input_a[g];

            const float* r0 = img0[0].data();
            const float* r1 = img0[1].data();
            const float* r2 = img0[2].data();
            const float* r3 = img0[3].data();
            const float* r4 = img0[4].data();

            int i = 0;
            for (; i < outh; i++)
            {
                int j = 0;
                for (; j < outw; j++)
                {
                    __m128 _sum0 = _bias0;

                    __m128 _r00 = _mm_load_ps(r0);
                    __m128 _r01 = _mm_load_ps(r0 + 4);
                    __m128 _r02 = _mm_load_ps(r0 + 4 * 2);
                    __m128 _r03 = _mm_load_ps(r0 + 4 * 3);
                    __m128 _r04 = _mm_load_ps(r0 + 4 * 4);

                    __m128 _k00 = _mm_load_ps(k0);
                    __m128 _k01 = _mm_load_ps(k0 + 4);
                    __m128 _k02 = _mm_load_ps(k0 + 4 * 2);
                    __m128 _k03 = _mm_load_ps(k0 + 4 * 3);
                    __m128 _k04 = _mm_load_ps(k0 + 4 * 4);
                    k0 += 4 * 5;

                    _sum0 = _mm_comp_fmadd_ps(_k00, _r00, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k01, _r01, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k02, _r02, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k03, _r03, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k04, _r04, _sum0);

                    __m128 _r10 = _mm_load_ps(r1);
                    __m128 _r11 = _mm_load_ps(r1 + 4);
                    __m128 _r12 = _mm_load_ps(r1 + 4 * 2);
                    __m128 _r13 = _mm_load_ps(r1 + 4 * 3);
                    __m128 _r14 = _mm_load_ps(r1 + 4 * 4);

                    __m128 _k10 = _mm_load_ps(k0);
                    __m128 _k11 = _mm_load_ps(k0 + 4);
                    __m128 _k12 = _mm_load_ps(k0 + 4 * 2);
                    __m128 _k13 = _mm_load_ps(k0 + 4 * 3);
                    __m128 _k14 = _mm_load_ps(k0 + 4 * 4);
                    k0 += 4 * 5;

                    _sum0 = _mm_comp_fmadd_ps(_k10, _r10, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k11, _r11, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k12, _r12, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k13, _r13, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k14, _r14, _sum0);

                    __m128 _r20 = _mm_load_ps(r2);
                    __m128 _r21 = _mm_load_ps(r2 + 4);
                    __m128 _r22 = _mm_load_ps(r2 + 4 * 2);
                    __m128 _r23 = _mm_load_ps(r2 + 4 * 3);
                    __m128 _r24 = _mm_load_ps(r2 + 4 * 4);

                    __m128 _k20 = _mm_load_ps(k0);
                    __m128 _k21 = _mm_load_ps(k0 + 4);
                    __m128 _k22 = _mm_load_ps(k0 + 4 * 2);
                    __m128 _k23 = _mm_load_ps(k0 + 4 * 3);
                    __m128 _k24 = _mm_load_ps(k0 + 4 * 4);
                    k0 += 4 * 5;

                    _sum0 = _mm_comp_fmadd_ps(_k20, _r20, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k21, _r21, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k22, _r22, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k23, _r23, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k24, _r24, _sum0);

                    __m128 _r30 = _mm_load_ps(r3);
                    __m128 _r31 = _mm_load_ps(r3 + 4);
                    __m128 _r32 = _mm_load_ps(r3 + 4 * 2);
                    __m128 _r33 = _mm_load_ps(r3 + 4 * 3);
                    __m128 _r34 = _mm_load_ps(r3 + 4 * 4);

                    __m128 _k30 = _mm_load_ps(k0);
                    __m128 _k31 = _mm_load_ps(k0 + 4);
                    __m128 _k32 = _mm_load_ps(k0 + 4 * 2);
                    __m128 _k33 = _mm_load_ps(k0 + 4 * 3);
                    __m128 _k34 = _mm_load_ps(k0 + 4 * 4);
                    k0 += 4 * 5;

                    _sum0 = _mm_comp_fmadd_ps(_k30, _r30, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k31, _r31, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k32, _r32, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k33, _r33, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k34, _r34, _sum0);

                    __m128 _r40 = _mm_load_ps(r4);
                    __m128 _r41 = _mm_load_ps(r4 + 4);
                    __m128 _r42 = _mm_load_ps(r4 + 4 * 2);
                    __m128 _r43 = _mm_load_ps(r4 + 4 * 3);
                    __m128 _r44 = _mm_load_ps(r4 + 4 * 4);

                    __m128 _k40 = _mm_load_ps(k0);
                    __m128 _k41 = _mm_load_ps(k0 + 4);
                    __m128 _k42 = _mm_load_ps(k0 + 4 * 2);
                    __m128 _k43 = _mm_load_ps(k0 + 4 * 3);
                    __m128 _k44 = _mm_load_ps(k0 + 4 * 4);
                    k0 -= 4 * 20;

                    _sum0 = _mm_comp_fmadd_ps(_k40, _r40, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k41, _r41, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k42, _r42, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k43, _r43, _sum0);
                    _sum0 = _mm_comp_fmadd_ps(_k44, _r44, _sum0);

                    _mm_store_ps(outptr0, _sum0);

                    outptr0 += 4;

                    r0 += 4 * 2;
                    r1 += 4 * 2;
                    r2 += 4 * 2;
                    r3 += 4 * 2;
                    r4 += 4 * 2;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
                r3 += tailstep;
                r4 += tailstep;
            }
        }
    });
    
    return output;
}

Tensor depthwise_conv2d_5x5s2_x86_pack4(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, otter::ScalarType::Float4);
    
    return depthwise_conv2d_5x5s2_x86_pack4_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
}

#endif // __SSE2__

}   // end namespace otter
