//
//  DepthwiseConvKernelNeonPack.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/17.
//

#include "DepthwiseConvKernelNeonPack.hpp"
#include "Tensor.hpp"
#include "TensorFactory.hpp"
#include "Parallel.hpp"
#include "Padding.hpp"

#include "VecIntrinsic.hpp"

namespace otter {

#if __ARM_NEON__

Tensor& depthwise_conv2d_neon_pack4_out(
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
            float* outptr = output_a[g].data();
            const float* kptr = (const float*)weight_data_packed.raw_data() + maxk * g * 4;
            const auto m = input_a[g];

            for (int i = 0; i < outh; i++) {
                for (int j = 0; j < outw; j++) {
                    float32x4_t _sum = vdupq_n_f32(0.f);

                    if (bias_term) {
                        _sum = vld1q_f32(((const float*)bias_data) + g * 4);
                    }

                    const float* sptr = m[i * stride_h].data() + j * stride_w * 4;

                    for (int k = 0; k < maxk; k++) {
                        float32x4_t _val = vld1q_f32(sptr + space_ofs[k] * 4);
                        float32x4_t _w = vld1q_f32(kptr + k * 4);
                        _sum = vmlaq_f32(_sum, _val, _w);
                    }

                    vst1q_f32(outptr + j * 4, _sum);
                }

                outptr += outw * 4;
            }
        }
    });
    
    return output;
}

Tensor depthwise_conv2d_neon_pack4(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, otter::ScalarType::Float4);
    
    return depthwise_conv2d_neon_pack4_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
}

Tensor& depthwise_conv2d_3x3s1_neon_pack4_out(
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
    
#if __aarch64__
    int w = int(input.size(2));
#endif

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

            float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + g * 4) : vdupq_n_f32(0.f);

            const float* k0 = kernel_a[g].data();

            float* outptr0 = out[0].data();

            const auto img0 = input_a[g];

            const float* r0 = img0[0].data();
            const float* r1 = img0[1].data();
            const float* r2 = img0[2].data();

            float32x4_t _k00 = vld1q_f32(k0);
            float32x4_t _k01 = vld1q_f32(k0 + 4);
            float32x4_t _k02 = vld1q_f32(k0 + 8);
            float32x4_t _k10 = vld1q_f32(k0 + 12);
            float32x4_t _k11 = vld1q_f32(k0 + 16);
            float32x4_t _k12 = vld1q_f32(k0 + 20);
            float32x4_t _k20 = vld1q_f32(k0 + 24);
            float32x4_t _k21 = vld1q_f32(k0 + 28);
            float32x4_t _k22 = vld1q_f32(k0 + 32);

            int i = 0;

    #if __aarch64__
            float* outptr1 = out[1].data();
            const float* r3 = img0[3].data();

            for (; i + 1 < outh; i += 2)
            {
                int j = 0;

                for (; j + 3 < outw; j += 4)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v10.4s, v11.4s}, [%3], #32 \n" // r10 r11

                        "mov    v16.16b, %21.16b            \n" // sum00
                        "mov    v17.16b, %21.16b            \n" // sum01
                        "mov    v18.16b, %21.16b            \n" // sum02
                        "mov    v19.16b, %21.16b            \n" // sum03

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%3] \n" // r12 r13 r14 r15

                        "mov    v20.16b, %21.16b            \n" // sum10
                        "mov    v21.16b, %21.16b            \n" // sum11
                        "mov    v22.16b, %21.16b            \n" // sum12
                        "mov    v23.16b, %21.16b            \n" // sum13

                        "fmla   v16.4s, %15.4s, v10.4s      \n"
                        "fmla   v17.4s, %15.4s, v11.4s      \n"
                        "fmla   v18.4s, %15.4s, v12.4s      \n"
                        "fmla   v19.4s, %15.4s, v13.4s      \n"
                        "fmla   v20.4s, %12.4s, v10.4s      \n"
                        "fmla   v21.4s, %12.4s, v11.4s      \n"
                        "fmla   v22.4s, %12.4s, v12.4s      \n"
                        "fmla   v23.4s, %12.4s, v13.4s      \n"

                        "add    %3, %3, #32                 \n"

                        "fmla   v16.4s, %16.4s, v11.4s      \n"
                        "fmla   v17.4s, %16.4s, v12.4s      \n"
                        "fmla   v18.4s, %16.4s, v13.4s      \n"
                        "fmla   v19.4s, %16.4s, v14.4s      \n"
                        "fmla   v20.4s, %13.4s, v11.4s      \n"
                        "fmla   v21.4s, %13.4s, v12.4s      \n"
                        "fmla   v22.4s, %13.4s, v13.4s      \n"
                        "fmla   v23.4s, %13.4s, v14.4s      \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v10.4s, v11.4s}, [%4], #32 \n" // r20 r21

                        "fmla   v16.4s, %17.4s, v12.4s      \n"
                        "fmla   v17.4s, %17.4s, v13.4s      \n"
                        "fmla   v18.4s, %17.4s, v14.4s      \n"
                        "fmla   v19.4s, %17.4s, v15.4s      \n"
                        "fmla   v20.4s, %14.4s, v12.4s      \n"
                        "fmla   v21.4s, %14.4s, v13.4s      \n"
                        "fmla   v22.4s, %14.4s, v14.4s      \n"
                        "fmla   v23.4s, %14.4s, v15.4s      \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%4] \n" // r22 r23 r24 r25

                        "fmla   v16.4s, %18.4s, v10.4s      \n"
                        "fmla   v17.4s, %18.4s, v11.4s      \n"
                        "fmla   v18.4s, %18.4s, v12.4s      \n"
                        "fmla   v19.4s, %18.4s, v13.4s      \n"
                        "fmla   v20.4s, %15.4s, v10.4s      \n"
                        "fmla   v21.4s, %15.4s, v11.4s      \n"
                        "fmla   v22.4s, %15.4s, v12.4s      \n"
                        "fmla   v23.4s, %15.4s, v13.4s      \n"

                        "add    %4, %4, #32                 \n"

                        "fmla   v16.4s, %19.4s, v11.4s      \n"
                        "fmla   v17.4s, %19.4s, v12.4s      \n"
                        "fmla   v18.4s, %19.4s, v13.4s      \n"
                        "fmla   v19.4s, %19.4s, v14.4s      \n"
                        "fmla   v20.4s, %16.4s, v11.4s      \n"
                        "fmla   v21.4s, %16.4s, v12.4s      \n"
                        "fmla   v22.4s, %16.4s, v13.4s      \n"
                        "fmla   v23.4s, %16.4s, v14.4s      \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v10.4s, v11.4s}, [%2], #32 \n" // r00 r01

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v24.4s, v25.4s}, [%5], #32 \n" // r30 r31

                        "fmla   v16.4s, %20.4s, v12.4s      \n"
                        "fmla   v17.4s, %20.4s, v13.4s      \n"
                        "fmla   v18.4s, %20.4s, v14.4s      \n"
                        "fmla   v19.4s, %20.4s, v15.4s      \n"
                        "fmla   v20.4s, %17.4s, v12.4s      \n"
                        "fmla   v21.4s, %17.4s, v13.4s      \n"
                        "fmla   v22.4s, %17.4s, v14.4s      \n"
                        "fmla   v23.4s, %17.4s, v15.4s      \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%2] \n" // r02 r03 r04 r05

                        "prfm   pldl1keep, [%5, #512]       \n"
                        "ld1    {v26.4s, v27.4s, v28.4s, v29.4s}, [%5] \n" // r32 r33 r34 r35

                        "fmla   v16.4s, %12.4s, v10.4s      \n"
                        "fmla   v17.4s, %12.4s, v11.4s      \n"
                        "fmla   v18.4s, %12.4s, v12.4s      \n"
                        "fmla   v19.4s, %12.4s, v13.4s      \n"
                        "fmla   v20.4s, %18.4s, v24.4s      \n"
                        "fmla   v21.4s, %18.4s, v25.4s      \n"
                        "fmla   v22.4s, %18.4s, v26.4s      \n"
                        "fmla   v23.4s, %18.4s, v27.4s      \n"

                        "add    %2, %2, #32                 \n"

                        "fmla   v16.4s, %13.4s, v11.4s      \n"
                        "fmla   v17.4s, %13.4s, v12.4s      \n"
                        "fmla   v18.4s, %13.4s, v13.4s      \n"
                        "fmla   v19.4s, %13.4s, v14.4s      \n"
                        "fmla   v20.4s, %19.4s, v25.4s      \n"
                        "fmla   v21.4s, %19.4s, v26.4s      \n"
                        "fmla   v22.4s, %19.4s, v27.4s      \n"
                        "fmla   v23.4s, %19.4s, v28.4s      \n"

                        "add    %5, %5, #32                 \n"

                        "fmla   v16.4s, %14.4s, v12.4s      \n"
                        "fmla   v17.4s, %14.4s, v13.4s      \n"
                        "fmla   v18.4s, %14.4s, v14.4s      \n"
                        "fmla   v19.4s, %14.4s, v15.4s      \n"
                        "fmla   v20.4s, %20.4s, v26.4s      \n"
                        "fmla   v21.4s, %20.4s, v27.4s      \n"
                        "fmla   v22.4s, %20.4s, v28.4s      \n"
                        "fmla   v23.4s, %20.4s, v29.4s      \n"

                        "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                        "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%1], #64 \n"

                        : "=r"(outptr0), // %0
                        "=r"(outptr1), // %1
                        "=r"(r0),      // %2
                        "=r"(r1),      // %3
                        "=r"(r2),      // %4
                        "=r"(r3)       // %5
                        : "0"(outptr0),
                        "1"(outptr1),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "w"(_k00),  // %12
                        "w"(_k01),  // %13
                        "w"(_k02),  // %14
                        "w"(_k10),  // %15
                        "w"(_k11),  // %16
                        "w"(_k12),  // %17
                        "w"(_k20),  // %18
                        "w"(_k21),  // %19
                        "w"(_k22),  // %20
                        "w"(_bias0) // %21
                        : "memory", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29");
                }
                for (; j + 1 < outw; j += 2)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v10.4s, v11.4s, v12.4s, v13.4s}, [%3] \n" // r10 r11 r12 r13

                        "mov    v16.16b, %21.16b            \n" // sum00
                        "mov    v17.16b, %21.16b            \n" // sum01
                        "mov    v18.16b, %21.16b            \n" // sum10
                        "mov    v19.16b, %21.16b            \n" // sum11

                        "fmla   v16.4s, %15.4s, v10.4s      \n"
                        "fmla   v17.4s, %15.4s, v11.4s      \n"
                        "fmla   v18.4s, %12.4s, v10.4s      \n"
                        "fmla   v19.4s, %12.4s, v11.4s      \n"

                        "add    %3, %3, #32                 \n"

                        "fmla   v16.4s, %16.4s, v11.4s      \n"
                        "fmla   v17.4s, %16.4s, v12.4s      \n"
                        "fmla   v18.4s, %13.4s, v11.4s      \n"
                        "fmla   v19.4s, %13.4s, v12.4s      \n"

                        "prfm   pldl1keep, [%4, #512]       \n"
                        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%4] \n" // r20 r21 r22 r23

                        "fmla   v16.4s, %17.4s, v12.4s      \n"
                        "fmla   v17.4s, %17.4s, v13.4s      \n"
                        "fmla   v18.4s, %14.4s, v12.4s      \n"
                        "fmla   v19.4s, %14.4s, v13.4s      \n"

                        "add    %4, %4, #32                 \n"

                        "fmla   v16.4s, %18.4s, v20.4s      \n"
                        "fmla   v17.4s, %18.4s, v21.4s      \n"
                        "fmla   v18.4s, %15.4s, v20.4s      \n"
                        "fmla   v19.4s, %15.4s, v21.4s      \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v10.4s, v11.4s, v12.4s, v13.4s}, [%2] \n" // r00 r01 r02 r03

                        "fmla   v16.4s, %19.4s, v21.4s      \n"
                        "fmla   v17.4s, %19.4s, v22.4s      \n"
                        "fmla   v18.4s, %16.4s, v21.4s      \n"
                        "fmla   v19.4s, %16.4s, v22.4s      \n"

                        "prfm   pldl1keep, [%5, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%5] \n" // r30 r31 r32 r33

                        "fmla   v16.4s, %20.4s, v22.4s      \n"
                        "fmla   v17.4s, %20.4s, v23.4s      \n"
                        "fmla   v18.4s, %17.4s, v22.4s      \n"
                        "fmla   v19.4s, %17.4s, v23.4s      \n"

                        "add    %2, %2, #32                 \n"

                        "fmla   v16.4s, %12.4s, v10.4s      \n"
                        "fmla   v17.4s, %12.4s, v11.4s      \n"
                        "fmla   v18.4s, %18.4s, v24.4s      \n"
                        "fmla   v19.4s, %18.4s, v25.4s      \n"

                        "add    %5, %5, #32                 \n"

                        "fmla   v16.4s, %13.4s, v11.4s      \n"
                        "fmla   v17.4s, %13.4s, v12.4s      \n"
                        "fmla   v18.4s, %19.4s, v25.4s      \n"
                        "fmla   v19.4s, %19.4s, v26.4s      \n"

                        "fmla   v16.4s, %14.4s, v12.4s      \n"
                        "fmla   v17.4s, %14.4s, v13.4s      \n"
                        "fmla   v18.4s, %20.4s, v26.4s      \n"
                        "fmla   v19.4s, %20.4s, v27.4s      \n"

                        "st1    {v16.4s, v17.4s}, [%0], #32 \n"
                        "st1    {v18.4s, v19.4s}, [%1], #32 \n"

                        : "=r"(outptr0), // %0
                        "=r"(outptr1), // %1
                        "=r"(r0),      // %2
                        "=r"(r1),      // %3
                        "=r"(r2),      // %4
                        "=r"(r3)       // %5
                        : "0"(outptr0),
                        "1"(outptr1),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "w"(_k00),  // %12
                        "w"(_k01),  // %13
                        "w"(_k02),  // %14
                        "w"(_k10),  // %15
                        "w"(_k11),  // %16
                        "w"(_k12),  // %17
                        "w"(_k20),  // %18
                        "w"(_k21),  // %19
                        "w"(_k22),  // %20
                        "w"(_bias0) // %21
                        : "memory", "v10", "v11", "v12", "v13", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
                }
                for (; j < outw; j++)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%3, #384]       \n"
                        "ld1    {v10.4s, v11.4s, v12.4s}, [%3] \n" // r10 r11 r12

                        "mov    v16.16b, %21.16b            \n" // sum0
                        "mov    v17.16b, %21.16b            \n" // sum1

                        "fmla   v16.4s, %15.4s, v10.4s      \n"
                        "fmla   v17.4s, %12.4s, v10.4s      \n"

                        "add    %3, %3, #16                 \n"

                        "fmla   v16.4s, %16.4s, v11.4s      \n"
                        "fmla   v17.4s, %13.4s, v11.4s      \n"

                        "prfm   pldl1keep, [%4, #384]       \n"
                        "ld1    {v20.4s, v21.4s, v22.4s}, [%4] \n" // r20 r21 r22

                        "fmla   v16.4s, %17.4s, v12.4s      \n"
                        "fmla   v17.4s, %14.4s, v12.4s      \n"

                        "add    %4, %4, #16                 \n"

                        "fmla   v16.4s, %18.4s, v20.4s      \n"
                        "fmla   v17.4s, %15.4s, v20.4s      \n"

                        "prfm   pldl1keep, [%2, #384]       \n"
                        "ld1    {v10.4s, v11.4s, v12.4s}, [%2] \n" // r00 r01 r02

                        "fmla   v16.4s, %19.4s, v21.4s      \n"
                        "fmla   v17.4s, %16.4s, v21.4s      \n"

                        "prfm   pldl1keep, [%5, #384]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s}, [%5] \n" // r30 r31 r32

                        "fmla   v16.4s, %20.4s, v22.4s      \n"
                        "fmla   v17.4s, %17.4s, v22.4s      \n"

                        "add    %2, %2, #16                 \n"

                        "fmla   v16.4s, %12.4s, v10.4s      \n"
                        "fmla   v17.4s, %18.4s, v24.4s      \n"

                        "add    %5, %5, #16                 \n"

                        "fmla   v16.4s, %13.4s, v11.4s      \n"
                        "fmla   v17.4s, %19.4s, v25.4s      \n"

                        "fmla   v16.4s, %14.4s, v12.4s      \n"
                        "fmla   v17.4s, %20.4s, v26.4s      \n"

                        "st1    {v16.4s}, [%0], #16         \n"
                        "st1    {v17.4s}, [%1], #16         \n"

                        : "=r"(outptr0), // %0
                        "=r"(outptr1), // %1
                        "=r"(r0),      // %2
                        "=r"(r1),      // %3
                        "=r"(r2),      // %4
                        "=r"(r3)       // %5
                        : "0"(outptr0),
                        "1"(outptr1),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "w"(_k00),  // %12
                        "w"(_k01),  // %13
                        "w"(_k02),  // %14
                        "w"(_k10),  // %15
                        "w"(_k11),  // %16
                        "w"(_k12),  // %17
                        "w"(_k20),  // %18
                        "w"(_k21),  // %19
                        "w"(_k22),  // %20
                        "w"(_bias0) // %21
                        : "memory", "v10", "v11", "v12", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v24", "v25", "v26");
                }

                r0 += 2 * 4 + w * 4;
                r1 += 2 * 4 + w * 4;
                r2 += 2 * 4 + w * 4;
                r3 += 2 * 4 + w * 4;

                outptr0 += outw * 4;
                outptr1 += outw * 4;
            }
    #endif // __aarch64__
            for (; i < outh; i++)
            {
                int j = 0;

                for (; j + 3 < outw; j += 4)
                {
    #if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v10.4s, v11.4s}, [%1], #32 \n" // r00 r01

                        "mov    v16.16b, %17.16b            \n" // sum00
                        "mov    v17.16b, %17.16b            \n" // sum01
                        "mov    v18.16b, %17.16b            \n" // sum02
                        "mov    v19.16b, %17.16b            \n" // sum03

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%1] \n" // r02 r03 r04 r05

                        "fmla   v16.4s, %8.4s, v10.4s       \n"
                        "fmla   v17.4s, %8.4s, v11.4s       \n"
                        "fmla   v18.4s, %8.4s, v12.4s       \n"
                        "fmla   v19.4s, %8.4s, v13.4s       \n"

                        "add    %1, %1, #32                 \n"

                        "fmla   v16.4s, %9.4s, v11.4s       \n"
                        "fmla   v17.4s, %9.4s, v12.4s       \n"
                        "fmla   v18.4s, %9.4s, v13.4s       \n"
                        "fmla   v19.4s, %9.4s, v14.4s       \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v10.4s, v11.4s}, [%2], #32 \n" // r10 r11

                        "fmla   v16.4s, %10.4s, v12.4s      \n"
                        "fmla   v17.4s, %10.4s, v13.4s      \n"
                        "fmla   v18.4s, %10.4s, v14.4s      \n"
                        "fmla   v19.4s, %10.4s, v15.4s      \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%2] \n" // r12 r13 r14 r15

                        "fmla   v16.4s, %11.4s, v10.4s      \n"
                        "fmla   v17.4s, %11.4s, v11.4s      \n"
                        "fmla   v18.4s, %11.4s, v12.4s      \n"
                        "fmla   v19.4s, %11.4s, v13.4s      \n"

                        "add    %2, %2, #32                 \n"

                        "fmla   v16.4s, %12.4s, v11.4s      \n"
                        "fmla   v17.4s, %12.4s, v12.4s      \n"
                        "fmla   v18.4s, %12.4s, v13.4s      \n"
                        "fmla   v19.4s, %12.4s, v14.4s      \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v10.4s, v11.4s}, [%3], #32 \n" // r20 r21

                        "fmla   v16.4s, %13.4s, v12.4s      \n"
                        "fmla   v17.4s, %13.4s, v13.4s      \n"
                        "fmla   v18.4s, %13.4s, v14.4s      \n"
                        "fmla   v19.4s, %13.4s, v15.4s      \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%3] \n" // r22 r23 r24 r25

                        "fmla   v16.4s, %14.4s, v10.4s      \n"
                        "fmla   v17.4s, %14.4s, v11.4s      \n"
                        "fmla   v18.4s, %14.4s, v12.4s      \n"
                        "fmla   v19.4s, %14.4s, v13.4s      \n"

                        "add    %3, %3, #32                 \n"

                        "fmla   v16.4s, %15.4s, v11.4s      \n"
                        "fmla   v17.4s, %15.4s, v12.4s      \n"
                        "fmla   v18.4s, %15.4s, v13.4s      \n"
                        "fmla   v19.4s, %15.4s, v14.4s      \n"

                        "fmla   v16.4s, %16.4s, v12.4s      \n"
                        "fmla   v17.4s, %16.4s, v13.4s      \n"
                        "fmla   v18.4s, %16.4s, v14.4s      \n"
                        "fmla   v19.4s, %16.4s, v15.4s      \n"

                        "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00),  // %8
                        "w"(_k01),  // %9
                        "w"(_k02),  // %10
                        "w"(_k10),  // %11
                        "w"(_k11),  // %12
                        "w"(_k12),  // %13
                        "w"(_k20),  // %14
                        "w"(_k21),  // %15
                        "w"(_k22),  // %16
                        "w"(_bias0) // %17
                        : "memory", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
    #else
                    asm volatile(
                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d28-d31}, [%1 :128]! \n" // r00 r01

                        "vmov       q10, %q17       \n" // sum00
                        "vmov       q11, %q17       \n" // sum01

                        "vmla.f32   q10, %q8, q14   \n"
                        "vmla.f32   q11, %q8, q15   \n"
                        "vmla.f32   q10, %q9, q15   \n"

                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d28-d31}, [%1 :128]! \n" // r02 r03

                        "vmov       q12, %q17       \n" // sum02
                        "vmov       q13, %q17       \n" // sum03

                        "vmla.f32   q12, %q8, q14   \n"
                        "vmla.f32   q11, %q9, q14   \n"
                        "vmla.f32   q13, %q8, q15   \n"
                        "vmla.f32   q10, %q10, q14  \n"
                        "vmla.f32   q12, %q9, q15   \n"
                        "vmla.f32   q11, %q10, q15  \n"

                        //                     "pld        [%1, #256]      \n"
                        "vld1.f32   {d28-d31}, [%1 :128] \n" // r04 r05

                        "vmla.f32   q13, %q9, q14   \n"
                        "vmla.f32   q12, %q10, q14  \n"
                        "vmla.f32   q13, %q10, q15  \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d28-d31}, [%2 :128]! \n" // r10 r11

                        "vmla.f32   q10, %q11, q14  \n"
                        "vmla.f32   q11, %q11, q15  \n"
                        "vmla.f32   q10, %q12, q15  \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d28-d31}, [%2 :128]! \n" // r12 r13

                        "vmla.f32   q12, %q11, q14  \n"
                        "vmla.f32   q11, %q12, q14  \n"
                        "vmla.f32   q13, %q11, q15  \n"
                        "vmla.f32   q10, %q13, q14  \n"
                        "vmla.f32   q12, %q12, q15  \n"
                        "vmla.f32   q11, %q13, q15  \n"

                        //                     "pld        [%2, #256]      \n"
                        "vld1.f32   {d28-d31}, [%2 :128] \n" // r14 r15

                        "vmla.f32   q13, %q12, q14  \n"
                        "vmla.f32   q12, %q13, q14  \n"
                        "vmla.f32   q13, %q13, q15  \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d28-d31}, [%3 :128]! \n" // r20 r21

                        "vmla.f32   q10, %q14, q14  \n"
                        "vmla.f32   q11, %q14, q15  \n"
                        "vmla.f32   q10, %q15, q15  \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d28-d31}, [%3 :128]! \n" // r22 r23

                        "vmla.f32   q12, %q14, q14  \n"
                        "vmla.f32   q11, %q15, q14  \n"
                        "vmla.f32   q13, %q14, q15  \n"
                        "vmla.f32   q10, %q16, q14  \n"
                        "vmla.f32   q12, %q15, q15  \n"
                        "vmla.f32   q11, %q16, q15  \n"

                        //                     "pld        [%3, #256]      \n"
                        "vld1.f32   {d28-d31}, [%3 :128] \n" // r24 r25

                        "vmla.f32   q13, %q15, q14  \n"
                        "vmla.f32   q12, %q16, q14  \n"
                        "vmla.f32   q13, %q16, q15  \n"

                        "vstm       %0!, {d20-d27}  \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00),  // %8
                        "w"(_k01),  // %9
                        "w"(_k02),  // %10
                        "w"(_k10),  // %11
                        "w"(_k11),  // %12
                        "w"(_k12),  // %13
                        "w"(_k20),  // %14
                        "w"(_k21),  // %15
                        "w"(_k22),  // %16
                        "w"(_bias0) // %17
                        : "memory", "q10", "q11", "q12", "q13", "q14", "q15");
    #endif
                }
                for (; j + 1 < outw; j += 2)
                {
    #if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%1] \n" // r00 r01 r02 r03

                        "mov    v16.16b, %17.16b            \n" // sum00
                        "mov    v17.16b, %17.16b            \n" // sum01

                        "eor    v18.16b, v18.16b, v18.16b   \n"
                        "eor    v19.16b, v19.16b, v19.16b   \n"

                        "fmla   v16.4s, %8.4s, v12.4s       \n"
                        "fmla   v17.4s, %8.4s, v13.4s       \n"

                        "add    %1, %1, #32                 \n"

                        "fmla   v18.4s, %9.4s, v13.4s       \n"
                        "fmla   v19.4s, %9.4s, v14.4s       \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%2] \n" // r10 r11 r12 r13

                        "fmla   v16.4s, %10.4s, v14.4s      \n"
                        "fmla   v17.4s, %10.4s, v15.4s      \n"

                        "add    %2, %2, #32                 \n"

                        "fmla   v18.4s, %11.4s, v20.4s      \n"
                        "fmla   v19.4s, %11.4s, v21.4s      \n"

                        "fmla   v16.4s, %12.4s, v21.4s      \n"
                        "fmla   v17.4s, %12.4s, v22.4s      \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%3] \n" // r20 r21 r22 r23

                        "fmla   v18.4s, %13.4s, v22.4s      \n"
                        "fmla   v19.4s, %13.4s, v23.4s      \n"

                        "fmla   v16.4s, %14.4s, v12.4s      \n"
                        "fmla   v17.4s, %14.4s, v13.4s      \n"

                        "fmla   v18.4s, %15.4s, v13.4s      \n"
                        "fmla   v19.4s, %15.4s, v14.4s      \n"

                        "fmla   v16.4s, %16.4s, v14.4s      \n"
                        "fmla   v17.4s, %16.4s, v15.4s      \n"

                        "add    %3, %3, #32                 \n"

                        "fadd   v16.4s, v16.4s, v18.4s      \n"
                        "fadd   v17.4s, v17.4s, v19.4s      \n"

                        "st1    {v16.4s, v17.4s}, [%0], #32 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00),  // %8
                        "w"(_k01),  // %9
                        "w"(_k02),  // %10
                        "w"(_k10),  // %11
                        "w"(_k11),  // %12
                        "w"(_k12),  // %13
                        "w"(_k20),  // %14
                        "w"(_k21),  // %15
                        "w"(_k22),  // %16
                        "w"(_bias0) // %17
                        : "memory", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
    #else
                    asm volatile(
                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d24-d27}, [%1 :128]! \n" // r00 r01

                        "vmov       q10, %q17       \n" // sum00
                        "vmov       q11, %q17       \n" // sum01

                        "vmla.f32   q10, %q8, q12   \n"
                        "vmla.f32   q11, %q8, q13   \n"

                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d28-d31}, [%1 :128] \n" // r02 r03

                        "vmla.f32   q10, %q9, q13   \n"

                        "vmla.f32   q11, %q9, q14   \n"
                        "vmla.f32   q10, %q10, q14  \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d24-d27}, [%2 :128]! \n" // r10 r11

                        "vmla.f32   q11, %q10, q15  \n"

                        "vmla.f32   q10, %q11, q12  \n"
                        "vmla.f32   q11, %q11, q13  \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d28-d31}, [%2 :128] \n" // r12 r13

                        "vmla.f32   q10, %q12, q13  \n"

                        "vmla.f32   q11, %q12, q14  \n"
                        "vmla.f32   q10, %q13, q14  \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d24-d27}, [%3 :128]! \n" // r20 r21

                        "vmla.f32   q11, %q13, q15  \n"

                        "vmla.f32   q10, %q14, q12  \n"
                        "vmla.f32   q11, %q14, q13  \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d28-d31}, [%3 :128] \n" // r22 r23

                        "vmla.f32   q10, %q15, q13  \n"

                        "vmla.f32   q11, %q15, q14  \n"
                        "vmla.f32   q10, %q16, q14  \n"
                        "vmla.f32   q11, %q16, q15  \n"

                        "vst1.f32   {d20-d23}, [%0 :128]! \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00),  // %8
                        "w"(_k01),  // %9
                        "w"(_k02),  // %10
                        "w"(_k10),  // %11
                        "w"(_k11),  // %12
                        "w"(_k12),  // %13
                        "w"(_k20),  // %14
                        "w"(_k21),  // %15
                        "w"(_k22),  // %16
                        "w"(_bias0) // %17
                        : "memory", "q10", "q11", "q12", "q13", "q14", "q15");
    #endif
                }
                for (; j < outw; j++)
                {
                    float32x4_t _sum0 = _bias0;

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r01 = vld1q_f32(r0 + 4);
                    float32x4_t _r02 = vld1q_f32(r0 + 8);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r11 = vld1q_f32(r1 + 4);
                    float32x4_t _r12 = vld1q_f32(r1 + 8);
                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r21 = vld1q_f32(r2 + 4);
                    float32x4_t _r22 = vld1q_f32(r2 + 8);

                    _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                    _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                    _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                    _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                    _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                    _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                    _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                    _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                    _sum0 = vmlaq_f32(_sum0, _k22, _r22);

                    vst1q_f32(outptr0, _sum0);

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

Tensor depthwise_conv2d_3x3s1_neon_pack4(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, otter::ScalarType::Float4);
    
    return depthwise_conv2d_3x3s1_neon_pack4_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
}

Tensor& depthwise_conv2d_3x3s2_neon_pack4_out(
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

            float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + g * 4) : vdupq_n_f32(0.f);

            const float* k0 = kernel_a[g].data();

            float* outptr0 = out.data();

            const Mat img0 = input_a[g].data();

            const float* r0 = img0[0].data();
            const float* r1 = img0[1].data();
            const float* r2 = img0[2].data();

            float32x4_t _k00 = vld1q_f32(k0);
            float32x4_t _k01 = vld1q_f32(k0 + 4);
            float32x4_t _k02 = vld1q_f32(k0 + 8);
            float32x4_t _k10 = vld1q_f32(k0 + 12);
            float32x4_t _k11 = vld1q_f32(k0 + 16);
            float32x4_t _k12 = vld1q_f32(k0 + 20);
            float32x4_t _k20 = vld1q_f32(k0 + 24);
            float32x4_t _k21 = vld1q_f32(k0 + 28);
            float32x4_t _k22 = vld1q_f32(k0 + 32);

            int i = 0;

            for (; i < outh; i++)
            {
                int j = 0;

                for (; j + 3 < outw; j += 4)
                {
    #if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v10.4s, v11.4s, v12.4s, v13.4s}, [%1], #64 \n" // r00 r01 r02 r03

                        "mov    v28.16b, %17.16b            \n" // sum00
                        "mov    v29.16b, %17.16b            \n" // sum01
                        "mov    v30.16b, %17.16b            \n" // sum02
                        "mov    v31.16b, %17.16b            \n" // sum03

                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v14.4s, v15.4s, v16.4s, v17.4s}, [%1], #64 \n" // r04 r05 r06 r07

                        "fmla   v28.4s, %8.4s, v10.4s       \n"
                        "fmla   v29.4s, %8.4s, v12.4s       \n"
                        "fmla   v30.4s, %8.4s, v14.4s       \n"
                        "fmla   v31.4s, %8.4s, v16.4s       \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v18.4s}, [%1]              \n" // r08

                        "fmla   v28.4s, %9.4s, v11.4s       \n"
                        "fmla   v29.4s, %9.4s, v13.4s       \n"
                        "fmla   v30.4s, %9.4s, v15.4s       \n"
                        "fmla   v31.4s, %9.4s, v17.4s       \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%2], #64 \n" // r10 r11 r12 r13

                        "fmla   v28.4s, %10.4s, v12.4s      \n"
                        "fmla   v29.4s, %10.4s, v14.4s      \n"
                        "fmla   v30.4s, %10.4s, v16.4s      \n"
                        "fmla   v31.4s, %10.4s, v18.4s      \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%2], #64 \n" // r14 r15 r16 r17

                        "fmla   v28.4s, %11.4s, v20.4s      \n"
                        "fmla   v29.4s, %11.4s, v22.4s      \n"
                        "fmla   v30.4s, %11.4s, v24.4s      \n"
                        "fmla   v31.4s, %11.4s, v26.4s      \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v19.4s}, [%2]              \n" // r18

                        "fmla   v28.4s, %12.4s, v21.4s      \n"
                        "fmla   v29.4s, %12.4s, v23.4s      \n"
                        "fmla   v30.4s, %12.4s, v25.4s      \n"
                        "fmla   v31.4s, %12.4s, v27.4s      \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v10.4s, v11.4s, v12.4s, v13.4s}, [%3], #64 \n" // r20 r21 r22 r23

                        "fmla   v28.4s, %13.4s, v22.4s      \n"
                        "fmla   v29.4s, %13.4s, v24.4s      \n"
                        "fmla   v30.4s, %13.4s, v26.4s      \n"
                        "fmla   v31.4s, %13.4s, v19.4s      \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v14.4s, v15.4s, v16.4s, v17.4s}, [%3], #64 \n" // r24 r25 r26 r27

                        "fmla   v28.4s, %14.4s, v10.4s      \n"
                        "fmla   v29.4s, %14.4s, v12.4s      \n"
                        "fmla   v30.4s, %14.4s, v14.4s      \n"
                        "fmla   v31.4s, %14.4s, v16.4s      \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v18.4s}, [%3]              \n" // r28

                        "fmla   v28.4s, %15.4s, v11.4s      \n"
                        "fmla   v29.4s, %15.4s, v13.4s      \n"
                        "fmla   v30.4s, %15.4s, v15.4s      \n"
                        "fmla   v31.4s, %15.4s, v17.4s      \n"

                        "fmla   v28.4s, %16.4s, v12.4s      \n"
                        "fmla   v29.4s, %16.4s, v14.4s      \n"
                        "fmla   v30.4s, %16.4s, v16.4s      \n"
                        "fmla   v31.4s, %16.4s, v18.4s      \n"

                        "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00),  // %8
                        "w"(_k01),  // %9
                        "w"(_k02),  // %10
                        "w"(_k10),  // %11
                        "w"(_k11),  // %12
                        "w"(_k12),  // %13
                        "w"(_k20),  // %14
                        "w"(_k21),  // %15
                        "w"(_k22),  // %16
                        "w"(_bias0) // %17
                        : "memory", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    #else
                    asm volatile(
                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d28-d31}, [%1 :128]! \n" // r00 r01

                        "vmov       q10, %q17       \n" // sum00

                        "vmla.f32   q10, %q8, q14   \n"

                        "vmov       q11, %q17       \n" // sum01

                        "vmla.f32   q10, %q9, q15   \n"

                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d28-d31}, [%1 :128]! \n" // r02 r03

                        "vmla.f32   q11, %q8, q14   \n"
                        "vmla.f32   q10, %q10, q14  \n"

                        "vmov       q12, %q17       \n" // sum02

                        "vmla.f32   q11, %q9, q15   \n"

                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d28-d31}, [%1 :128]! \n" // r04 r05

                        "vmla.f32   q12, %q8, q14   \n"
                        "vmla.f32   q11, %q10, q14  \n"

                        "vmla.f32   q12, %q9, q15   \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d28-d31}, [%2 :128]! \n" // r10 r11

                        "vmla.f32   q10, %q11, q14  \n"

                        "vmov       q13, %q17       \n" // sum03

                        "vmla.f32   q10, %q12, q15  \n"

                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d28-d31}, [%1 :128]! \n" // r06 r07

                        "vmla.f32   q13, %q8, q14   \n"
                        "vmla.f32   q12, %q10, q14  \n"

                        "vmla.f32   q13, %q9, q15   \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d28-d31}, [%2 :128]! \n" // r12 r13

                        "vmla.f32   q11, %q11, q14  \n"
                        "vmla.f32   q10, %q13, q14  \n"

                        "vmla.f32   q11, %q12, q15  \n"

                        "vld1.f32   {d28-d29}, [%1 :128] \n" // r08

                        "vmla.f32   q13, %q10, q14  \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d28-d31}, [%2 :128]! \n" // r14 r15

                        "vmla.f32   q12, %q11, q14  \n"
                        "vmla.f32   q11, %q13, q14  \n"

                        "vmla.f32   q12, %q12, q15  \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d28-d31}, [%3 :128]! \n" // r20 r21

                        "vmla.f32   q10, %q14, q14  \n"
                        "vmla.f32   q10, %q15, q15  \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d28-d31}, [%2 :128]! \n" // r16 r17

                        "vmla.f32   q13, %q11, q14  \n"
                        "vmla.f32   q12, %q13, q14  \n"

                        "vmla.f32   q13, %q12, q15  \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d28-d31}, [%3 :128]! \n" // r22 r23

                        "vmla.f32   q11, %q14, q14  \n"
                        "vmla.f32   q10, %q16, q14  \n"

                        "vmla.f32   q11, %q15, q15  \n"

                        "vld1.f32   {d28-d29}, [%2 :128] \n" // r18

                        "vmla.f32   q13, %q13, q14  \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d28-d31}, [%3 :128]! \n" // r24 r25

                        "vmla.f32   q12, %q14, q14  \n"
                        "vmla.f32   q11, %q16, q14  \n"

                        "vmla.f32   q12, %q15, q15  \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d28-d31}, [%3 :128]! \n" // r26 r27

                        "vmla.f32   q13, %q14, q14  \n"
                        "vmla.f32   q12, %q16, q14  \n"

                        "vmla.f32   q13, %q15, q15  \n"

                        "vld1.f32   {d28-d29}, [%3 :128] \n" // r28

                        "vmla.f32   q13, %q16, q14  \n"

                        "vstm       %0!, {d20-d27}  \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00),  // %8
                        "w"(_k01),  // %9
                        "w"(_k02),  // %10
                        "w"(_k10),  // %11
                        "w"(_k11),  // %12
                        "w"(_k12),  // %13
                        "w"(_k20),  // %14
                        "w"(_k21),  // %15
                        "w"(_k22),  // %16
                        "w"(_bias0) // %17
                        : "memory", "q10", "q11", "q12", "q13", "q14", "q15");
    #endif
                }
                for (; j + 1 < outw; j += 2)
                {
    #if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #512]       \n"
                        "ld1    {v10.4s, v11.4s, v12.4s, v13.4s}, [%1], #64 \n" // r00 r01 r02 r03

                        "mov    v20.16b, %17.16b            \n" // sum00
                        "mov    v21.16b, %17.16b            \n" // sum01

                        "eor    v22.16b, v22.16b, v22.16b   \n"
                        "eor    v23.16b, v23.16b, v23.16b   \n"

                        "fmla   v20.4s, %8.4s, v10.4s       \n"
                        "fmla   v21.4s, %8.4s, v12.4s       \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v14.4s}, [%1]              \n" // r04

                        "fmla   v22.4s, %9.4s, v11.4s       \n"
                        "fmla   v23.4s, %9.4s, v13.4s       \n"

                        "prfm   pldl1keep, [%2, #512]       \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%2], #64 \n" // r10 r11 r12 r13

                        "fmla   v20.4s, %10.4s, v12.4s      \n"
                        "fmla   v21.4s, %10.4s, v14.4s      \n"

                        "fmla   v22.4s, %11.4s, v16.4s      \n"
                        "fmla   v23.4s, %11.4s, v18.4s      \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v15.4s}, [%2]              \n" // r14

                        "fmla   v20.4s, %12.4s, v17.4s      \n"
                        "fmla   v21.4s, %12.4s, v19.4s      \n"

                        "prfm   pldl1keep, [%3, #512]       \n"
                        "ld1    {v10.4s, v11.4s, v12.4s, v13.4s}, [%3], #64 \n" // r20 r21 r22 r23

                        "fmla   v22.4s, %13.4s, v18.4s      \n"
                        "fmla   v23.4s, %13.4s, v15.4s      \n"

                        "fmla   v20.4s, %14.4s, v10.4s      \n"
                        "fmla   v21.4s, %14.4s, v12.4s      \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v14.4s}, [%3]              \n" // r24

                        "fmla   v22.4s, %15.4s, v11.4s      \n"
                        "fmla   v23.4s, %15.4s, v13.4s      \n"

                        "fmla   v20.4s, %16.4s, v12.4s      \n"
                        "fmla   v21.4s, %16.4s, v14.4s      \n"

                        "fadd   v20.4s, v20.4s, v22.4s      \n"
                        "fadd   v21.4s, v21.4s, v23.4s      \n"

                        "st1    {v20.4s, v21.4s}, [%0], #32 \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00),  // %8
                        "w"(_k01),  // %9
                        "w"(_k02),  // %10
                        "w"(_k10),  // %11
                        "w"(_k11),  // %12
                        "w"(_k12),  // %13
                        "w"(_k20),  // %14
                        "w"(_k21),  // %15
                        "w"(_k22),  // %16
                        "w"(_bias0) // %17
                        : "memory", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
    #else
                    asm volatile(
                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d24-d27}, [%1 :128]! \n" // r00 r01

                        "vmov       q10, %q17       \n" // sum00
                        "vmov       q11, %q17       \n" // sum01

                        "vmla.f32   q10, %q8, q12   \n"

                        "pld        [%1, #256]      \n"
                        "vld1.f32   {d28-d31}, [%1 :128]! \n" // r02 r03

                        "vmla.f32   q10, %q9, q13   \n"

                        "vmla.f32   q11, %q8, q14   \n"
                        "vmla.f32   q10, %q10, q14  \n"

                        "vld1.f32   {d24-d25}, [%1 :128] \n" // r04

                        "vmla.f32   q11, %q9, q15   \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d28-d31}, [%2 :128]! \n" // r10 r11

                        "vmla.f32   q11, %q10, q12  \n"

                        "vmla.f32   q10, %q11, q14  \n"

                        "pld        [%2, #256]      \n"
                        "vld1.f32   {d24-d27}, [%2 :128]! \n" // r12 r13

                        "vmla.f32   q10, %q12, q15  \n"

                        "vmla.f32   q11, %q11, q12  \n"
                        "vmla.f32   q10, %q13, q12  \n"

                        "vld1.f32   {d28-d29}, [%2 :128] \n" // r14

                        "vmla.f32   q11, %q12, q13  \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d24-d27}, [%3 :128]! \n" // r20 r21

                        "vmla.f32   q11, %q13, q14  \n"

                        "vmla.f32   q10, %q14, q12  \n"

                        "pld        [%3, #256]      \n"
                        "vld1.f32   {d28-d31}, [%3 :128]! \n" // r22 r23

                        "vmla.f32   q10, %q15, q13  \n"

                        "vmla.f32   q11, %q14, q14  \n"
                        "vmla.f32   q10, %q16, q14  \n"

                        "vld1.f32   {d24-d25}, [%3 :128] \n" // r24

                        "vmla.f32   q11, %q15, q15  \n"

                        "vmla.f32   q11, %q16, q12  \n"

                        "vst1.f32   {d20-d23}, [%0 :128]! \n"

                        : "=r"(outptr0), // %0
                        "=r"(r0),      // %1
                        "=r"(r1),      // %2
                        "=r"(r2)       // %3
                        : "0"(outptr0),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "w"(_k00),  // %8
                        "w"(_k01),  // %9
                        "w"(_k02),  // %10
                        "w"(_k10),  // %11
                        "w"(_k11),  // %12
                        "w"(_k12),  // %13
                        "w"(_k20),  // %14
                        "w"(_k21),  // %15
                        "w"(_k22),  // %16
                        "w"(_bias0) // %17
                        : "memory", "q10", "q11", "q12", "q13", "q14", "q15");
    #endif
                }
                for (; j < outw; j++)
                {
                    float32x4_t _sum0 = _bias0;

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r01 = vld1q_f32(r0 + 4);
                    float32x4_t _r02 = vld1q_f32(r0 + 8);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r11 = vld1q_f32(r1 + 4);
                    float32x4_t _r12 = vld1q_f32(r1 + 8);
                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r21 = vld1q_f32(r2 + 4);
                    float32x4_t _r22 = vld1q_f32(r2 + 8);

                    _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                    _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                    _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                    _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                    _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                    _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                    _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                    _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                    _sum0 = vmlaq_f32(_sum0, _k22, _r22);

                    vst1q_f32(outptr0, _sum0);

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

Tensor depthwise_conv2d_3x3s2_neon_pack4(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, otter::ScalarType::Float4);
    
    return depthwise_conv2d_3x3s2_neon_pack4_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
}

Tensor& depthwise_conv2d_5x5s1_neon_pack4_out(
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
    
#if __aarch64__
    int w = int(input.size(2));
#endif

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

            float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + g * 4) : vdupq_n_f32(0.f);

            const float* k0 = kernel_a[g].data();

            float* outptr0 = out[0].data();

            const auto img0 = input_a[g];

            const float* r0 = img0[0].data();
            const float* r1 = img0[1].data();
            const float* r2 = img0[2].data();
            const float* r3 = img0[3].data();
            const float* r4 = img0[4].data();

            int i = 0;

    #if __aarch64__
            float* outptr1 = out[1].data();
            const float* r5 = img0[5].data(0);

            for (; i + 1 < outh; i += 2)
            {
                int j = 0;

                for (; j + 3 < outw; j += 4)
                {
                    float32x4_t _sum00 = _bias0;
                    float32x4_t _sum01 = _bias0;
                    float32x4_t _sum02 = _bias0;
                    float32x4_t _sum03 = _bias0;
                    float32x4_t _sum10 = _bias0;
                    float32x4_t _sum11 = _bias0;
                    float32x4_t _sum12 = _bias0;
                    float32x4_t _sum13 = _bias0;

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r01 = vld1q_f32(r0 + 4);
                    float32x4_t _r02 = vld1q_f32(r0 + 8);
                    float32x4_t _r03 = vld1q_f32(r0 + 12);
                    float32x4_t _r04 = vld1q_f32(r0 + 16);
                    float32x4_t _r05 = vld1q_f32(r0 + 20);
                    float32x4_t _r06 = vld1q_f32(r0 + 24);
                    float32x4_t _r07 = vld1q_f32(r0 + 28);

                    float32x4_t _k00 = vld1q_f32(k0);
                    float32x4_t _k01 = vld1q_f32(k0 + 4);
                    float32x4_t _k02 = vld1q_f32(k0 + 8);
                    float32x4_t _k03 = vld1q_f32(k0 + 12);
                    float32x4_t _k04 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum00 = vmlaq_f32(_sum00, _k00, _r00);
                    _sum00 = vmlaq_f32(_sum00, _k01, _r01);
                    _sum00 = vmlaq_f32(_sum00, _k02, _r02);
                    _sum00 = vmlaq_f32(_sum00, _k03, _r03);
                    _sum00 = vmlaq_f32(_sum00, _k04, _r04);
                    _sum01 = vmlaq_f32(_sum01, _k00, _r01);
                    _sum01 = vmlaq_f32(_sum01, _k01, _r02);
                    _sum01 = vmlaq_f32(_sum01, _k02, _r03);
                    _sum01 = vmlaq_f32(_sum01, _k03, _r04);
                    _sum01 = vmlaq_f32(_sum01, _k04, _r05);
                    _sum02 = vmlaq_f32(_sum02, _k00, _r02);
                    _sum02 = vmlaq_f32(_sum02, _k01, _r03);
                    _sum02 = vmlaq_f32(_sum02, _k02, _r04);
                    _sum02 = vmlaq_f32(_sum02, _k03, _r05);
                    _sum02 = vmlaq_f32(_sum02, _k04, _r06);
                    _sum03 = vmlaq_f32(_sum03, _k00, _r03);
                    _sum03 = vmlaq_f32(_sum03, _k01, _r04);
                    _sum03 = vmlaq_f32(_sum03, _k02, _r05);
                    _sum03 = vmlaq_f32(_sum03, _k03, _r06);
                    _sum03 = vmlaq_f32(_sum03, _k04, _r07);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r11 = vld1q_f32(r1 + 4);
                    float32x4_t _r12 = vld1q_f32(r1 + 8);
                    float32x4_t _r13 = vld1q_f32(r1 + 12);
                    float32x4_t _r14 = vld1q_f32(r1 + 16);
                    float32x4_t _r15 = vld1q_f32(r1 + 20);
                    float32x4_t _r16 = vld1q_f32(r1 + 24);
                    float32x4_t _r17 = vld1q_f32(r1 + 28);

                    float32x4_t _k10 = vld1q_f32(k0);
                    float32x4_t _k11 = vld1q_f32(k0 + 4);
                    float32x4_t _k12 = vld1q_f32(k0 + 8);
                    float32x4_t _k13 = vld1q_f32(k0 + 12);
                    float32x4_t _k14 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum10 = vmlaq_f32(_sum10, _k00, _r10);
                    _sum10 = vmlaq_f32(_sum10, _k01, _r11);
                    _sum10 = vmlaq_f32(_sum10, _k02, _r12);
                    _sum10 = vmlaq_f32(_sum10, _k03, _r13);
                    _sum10 = vmlaq_f32(_sum10, _k04, _r14);
                    _sum11 = vmlaq_f32(_sum11, _k00, _r11);
                    _sum11 = vmlaq_f32(_sum11, _k01, _r12);
                    _sum11 = vmlaq_f32(_sum11, _k02, _r13);
                    _sum11 = vmlaq_f32(_sum11, _k03, _r14);
                    _sum11 = vmlaq_f32(_sum11, _k04, _r15);
                    _sum12 = vmlaq_f32(_sum12, _k00, _r12);
                    _sum12 = vmlaq_f32(_sum12, _k01, _r13);
                    _sum12 = vmlaq_f32(_sum12, _k02, _r14);
                    _sum12 = vmlaq_f32(_sum12, _k03, _r15);
                    _sum12 = vmlaq_f32(_sum12, _k04, _r16);
                    _sum13 = vmlaq_f32(_sum13, _k00, _r13);
                    _sum13 = vmlaq_f32(_sum13, _k01, _r14);
                    _sum13 = vmlaq_f32(_sum13, _k02, _r15);
                    _sum13 = vmlaq_f32(_sum13, _k03, _r16);
                    _sum13 = vmlaq_f32(_sum13, _k04, _r17);

                    _sum00 = vmlaq_f32(_sum00, _k10, _r10);
                    _sum00 = vmlaq_f32(_sum00, _k11, _r11);
                    _sum00 = vmlaq_f32(_sum00, _k12, _r12);
                    _sum00 = vmlaq_f32(_sum00, _k13, _r13);
                    _sum00 = vmlaq_f32(_sum00, _k14, _r14);
                    _sum01 = vmlaq_f32(_sum01, _k10, _r11);
                    _sum01 = vmlaq_f32(_sum01, _k11, _r12);
                    _sum01 = vmlaq_f32(_sum01, _k12, _r13);
                    _sum01 = vmlaq_f32(_sum01, _k13, _r14);
                    _sum01 = vmlaq_f32(_sum01, _k14, _r15);
                    _sum02 = vmlaq_f32(_sum02, _k10, _r12);
                    _sum02 = vmlaq_f32(_sum02, _k11, _r13);
                    _sum02 = vmlaq_f32(_sum02, _k12, _r14);
                    _sum02 = vmlaq_f32(_sum02, _k13, _r15);
                    _sum02 = vmlaq_f32(_sum02, _k14, _r16);
                    _sum03 = vmlaq_f32(_sum03, _k10, _r13);
                    _sum03 = vmlaq_f32(_sum03, _k11, _r14);
                    _sum03 = vmlaq_f32(_sum03, _k12, _r15);
                    _sum03 = vmlaq_f32(_sum03, _k13, _r16);
                    _sum03 = vmlaq_f32(_sum03, _k14, _r17);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r21 = vld1q_f32(r2 + 4);
                    float32x4_t _r22 = vld1q_f32(r2 + 8);
                    float32x4_t _r23 = vld1q_f32(r2 + 12);
                    float32x4_t _r24 = vld1q_f32(r2 + 16);
                    float32x4_t _r25 = vld1q_f32(r2 + 20);
                    float32x4_t _r26 = vld1q_f32(r2 + 24);
                    float32x4_t _r27 = vld1q_f32(r2 + 28);

                    float32x4_t _k20 = vld1q_f32(k0);
                    float32x4_t _k21 = vld1q_f32(k0 + 4);
                    float32x4_t _k22 = vld1q_f32(k0 + 8);
                    float32x4_t _k23 = vld1q_f32(k0 + 12);
                    float32x4_t _k24 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum10 = vmlaq_f32(_sum10, _k10, _r20);
                    _sum10 = vmlaq_f32(_sum10, _k11, _r21);
                    _sum10 = vmlaq_f32(_sum10, _k12, _r22);
                    _sum10 = vmlaq_f32(_sum10, _k13, _r23);
                    _sum10 = vmlaq_f32(_sum10, _k14, _r24);
                    _sum11 = vmlaq_f32(_sum11, _k10, _r21);
                    _sum11 = vmlaq_f32(_sum11, _k11, _r22);
                    _sum11 = vmlaq_f32(_sum11, _k12, _r23);
                    _sum11 = vmlaq_f32(_sum11, _k13, _r24);
                    _sum11 = vmlaq_f32(_sum11, _k14, _r25);
                    _sum12 = vmlaq_f32(_sum12, _k10, _r22);
                    _sum12 = vmlaq_f32(_sum12, _k11, _r23);
                    _sum12 = vmlaq_f32(_sum12, _k12, _r24);
                    _sum12 = vmlaq_f32(_sum12, _k13, _r25);
                    _sum12 = vmlaq_f32(_sum12, _k14, _r26);
                    _sum13 = vmlaq_f32(_sum13, _k10, _r23);
                    _sum13 = vmlaq_f32(_sum13, _k11, _r24);
                    _sum13 = vmlaq_f32(_sum13, _k12, _r25);
                    _sum13 = vmlaq_f32(_sum13, _k13, _r26);
                    _sum13 = vmlaq_f32(_sum13, _k14, _r27);

                    _sum00 = vmlaq_f32(_sum00, _k20, _r20);
                    _sum00 = vmlaq_f32(_sum00, _k21, _r21);
                    _sum00 = vmlaq_f32(_sum00, _k22, _r22);
                    _sum00 = vmlaq_f32(_sum00, _k23, _r23);
                    _sum00 = vmlaq_f32(_sum00, _k24, _r24);
                    _sum01 = vmlaq_f32(_sum01, _k20, _r21);
                    _sum01 = vmlaq_f32(_sum01, _k21, _r22);
                    _sum01 = vmlaq_f32(_sum01, _k22, _r23);
                    _sum01 = vmlaq_f32(_sum01, _k23, _r24);
                    _sum01 = vmlaq_f32(_sum01, _k24, _r25);
                    _sum02 = vmlaq_f32(_sum02, _k20, _r22);
                    _sum02 = vmlaq_f32(_sum02, _k21, _r23);
                    _sum02 = vmlaq_f32(_sum02, _k22, _r24);
                    _sum02 = vmlaq_f32(_sum02, _k23, _r25);
                    _sum02 = vmlaq_f32(_sum02, _k24, _r26);
                    _sum03 = vmlaq_f32(_sum03, _k20, _r23);
                    _sum03 = vmlaq_f32(_sum03, _k21, _r24);
                    _sum03 = vmlaq_f32(_sum03, _k22, _r25);
                    _sum03 = vmlaq_f32(_sum03, _k23, _r26);
                    _sum03 = vmlaq_f32(_sum03, _k24, _r27);

                    float32x4_t _r30 = vld1q_f32(r3);
                    float32x4_t _r31 = vld1q_f32(r3 + 4);
                    float32x4_t _r32 = vld1q_f32(r3 + 8);
                    float32x4_t _r33 = vld1q_f32(r3 + 12);
                    float32x4_t _r34 = vld1q_f32(r3 + 16);
                    float32x4_t _r35 = vld1q_f32(r3 + 20);
                    float32x4_t _r36 = vld1q_f32(r3 + 24);
                    float32x4_t _r37 = vld1q_f32(r3 + 28);

                    float32x4_t _k30 = vld1q_f32(k0);
                    float32x4_t _k31 = vld1q_f32(k0 + 4);
                    float32x4_t _k32 = vld1q_f32(k0 + 8);
                    float32x4_t _k33 = vld1q_f32(k0 + 12);
                    float32x4_t _k34 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum10 = vmlaq_f32(_sum10, _k20, _r30);
                    _sum10 = vmlaq_f32(_sum10, _k21, _r31);
                    _sum10 = vmlaq_f32(_sum10, _k22, _r32);
                    _sum10 = vmlaq_f32(_sum10, _k23, _r33);
                    _sum10 = vmlaq_f32(_sum10, _k24, _r34);
                    _sum11 = vmlaq_f32(_sum11, _k20, _r31);
                    _sum11 = vmlaq_f32(_sum11, _k21, _r32);
                    _sum11 = vmlaq_f32(_sum11, _k22, _r33);
                    _sum11 = vmlaq_f32(_sum11, _k23, _r34);
                    _sum11 = vmlaq_f32(_sum11, _k24, _r35);
                    _sum12 = vmlaq_f32(_sum12, _k20, _r32);
                    _sum12 = vmlaq_f32(_sum12, _k21, _r33);
                    _sum12 = vmlaq_f32(_sum12, _k22, _r34);
                    _sum12 = vmlaq_f32(_sum12, _k23, _r35);
                    _sum12 = vmlaq_f32(_sum12, _k24, _r36);
                    _sum13 = vmlaq_f32(_sum13, _k20, _r33);
                    _sum13 = vmlaq_f32(_sum13, _k21, _r34);
                    _sum13 = vmlaq_f32(_sum13, _k22, _r35);
                    _sum13 = vmlaq_f32(_sum13, _k23, _r36);
                    _sum13 = vmlaq_f32(_sum13, _k24, _r37);

                    _sum00 = vmlaq_f32(_sum00, _k30, _r30);
                    _sum00 = vmlaq_f32(_sum00, _k31, _r31);
                    _sum00 = vmlaq_f32(_sum00, _k32, _r32);
                    _sum00 = vmlaq_f32(_sum00, _k33, _r33);
                    _sum00 = vmlaq_f32(_sum00, _k34, _r34);
                    _sum01 = vmlaq_f32(_sum01, _k30, _r31);
                    _sum01 = vmlaq_f32(_sum01, _k31, _r32);
                    _sum01 = vmlaq_f32(_sum01, _k32, _r33);
                    _sum01 = vmlaq_f32(_sum01, _k33, _r34);
                    _sum01 = vmlaq_f32(_sum01, _k34, _r35);
                    _sum02 = vmlaq_f32(_sum02, _k30, _r32);
                    _sum02 = vmlaq_f32(_sum02, _k31, _r33);
                    _sum02 = vmlaq_f32(_sum02, _k32, _r34);
                    _sum02 = vmlaq_f32(_sum02, _k33, _r35);
                    _sum02 = vmlaq_f32(_sum02, _k34, _r36);
                    _sum03 = vmlaq_f32(_sum03, _k30, _r33);
                    _sum03 = vmlaq_f32(_sum03, _k31, _r34);
                    _sum03 = vmlaq_f32(_sum03, _k32, _r35);
                    _sum03 = vmlaq_f32(_sum03, _k33, _r36);
                    _sum03 = vmlaq_f32(_sum03, _k34, _r37);

                    float32x4_t _r40 = vld1q_f32(r4);
                    float32x4_t _r41 = vld1q_f32(r4 + 4);
                    float32x4_t _r42 = vld1q_f32(r4 + 8);
                    float32x4_t _r43 = vld1q_f32(r4 + 12);
                    float32x4_t _r44 = vld1q_f32(r4 + 16);
                    float32x4_t _r45 = vld1q_f32(r4 + 20);
                    float32x4_t _r46 = vld1q_f32(r4 + 24);
                    float32x4_t _r47 = vld1q_f32(r4 + 28);

                    float32x4_t _k40 = vld1q_f32(k0);
                    float32x4_t _k41 = vld1q_f32(k0 + 4);
                    float32x4_t _k42 = vld1q_f32(k0 + 8);
                    float32x4_t _k43 = vld1q_f32(k0 + 12);
                    float32x4_t _k44 = vld1q_f32(k0 + 16);
                    k0 -= 80;

                    _sum10 = vmlaq_f32(_sum10, _k30, _r40);
                    _sum10 = vmlaq_f32(_sum10, _k31, _r41);
                    _sum10 = vmlaq_f32(_sum10, _k32, _r42);
                    _sum10 = vmlaq_f32(_sum10, _k33, _r43);
                    _sum10 = vmlaq_f32(_sum10, _k34, _r44);
                    _sum11 = vmlaq_f32(_sum11, _k30, _r41);
                    _sum11 = vmlaq_f32(_sum11, _k31, _r42);
                    _sum11 = vmlaq_f32(_sum11, _k32, _r43);
                    _sum11 = vmlaq_f32(_sum11, _k33, _r44);
                    _sum11 = vmlaq_f32(_sum11, _k34, _r45);
                    _sum12 = vmlaq_f32(_sum12, _k30, _r42);
                    _sum12 = vmlaq_f32(_sum12, _k31, _r43);
                    _sum12 = vmlaq_f32(_sum12, _k32, _r44);
                    _sum12 = vmlaq_f32(_sum12, _k33, _r45);
                    _sum12 = vmlaq_f32(_sum12, _k34, _r46);
                    _sum13 = vmlaq_f32(_sum13, _k30, _r43);
                    _sum13 = vmlaq_f32(_sum13, _k31, _r44);
                    _sum13 = vmlaq_f32(_sum13, _k32, _r45);
                    _sum13 = vmlaq_f32(_sum13, _k33, _r46);
                    _sum13 = vmlaq_f32(_sum13, _k34, _r47);

                    _sum00 = vmlaq_f32(_sum00, _k40, _r40);
                    _sum00 = vmlaq_f32(_sum00, _k41, _r41);
                    _sum00 = vmlaq_f32(_sum00, _k42, _r42);
                    _sum00 = vmlaq_f32(_sum00, _k43, _r43);
                    _sum00 = vmlaq_f32(_sum00, _k44, _r44);
                    _sum01 = vmlaq_f32(_sum01, _k40, _r41);
                    _sum01 = vmlaq_f32(_sum01, _k41, _r42);
                    _sum01 = vmlaq_f32(_sum01, _k42, _r43);
                    _sum01 = vmlaq_f32(_sum01, _k43, _r44);
                    _sum01 = vmlaq_f32(_sum01, _k44, _r45);
                    _sum02 = vmlaq_f32(_sum02, _k40, _r42);
                    _sum02 = vmlaq_f32(_sum02, _k41, _r43);
                    _sum02 = vmlaq_f32(_sum02, _k42, _r44);
                    _sum02 = vmlaq_f32(_sum02, _k43, _r45);
                    _sum02 = vmlaq_f32(_sum02, _k44, _r46);
                    _sum03 = vmlaq_f32(_sum03, _k40, _r43);
                    _sum03 = vmlaq_f32(_sum03, _k41, _r44);
                    _sum03 = vmlaq_f32(_sum03, _k42, _r45);
                    _sum03 = vmlaq_f32(_sum03, _k43, _r46);
                    _sum03 = vmlaq_f32(_sum03, _k44, _r47);

                    float32x4_t _r50 = vld1q_f32(r5);
                    float32x4_t _r51 = vld1q_f32(r5 + 4);
                    float32x4_t _r52 = vld1q_f32(r5 + 8);
                    float32x4_t _r53 = vld1q_f32(r5 + 12);
                    float32x4_t _r54 = vld1q_f32(r5 + 16);
                    float32x4_t _r55 = vld1q_f32(r5 + 20);
                    float32x4_t _r56 = vld1q_f32(r5 + 24);
                    float32x4_t _r57 = vld1q_f32(r5 + 28);

                    _sum10 = vmlaq_f32(_sum10, _k40, _r50);
                    _sum10 = vmlaq_f32(_sum10, _k41, _r51);
                    _sum10 = vmlaq_f32(_sum10, _k42, _r52);
                    _sum10 = vmlaq_f32(_sum10, _k43, _r53);
                    _sum10 = vmlaq_f32(_sum10, _k44, _r54);
                    _sum11 = vmlaq_f32(_sum11, _k40, _r51);
                    _sum11 = vmlaq_f32(_sum11, _k41, _r52);
                    _sum11 = vmlaq_f32(_sum11, _k42, _r53);
                    _sum11 = vmlaq_f32(_sum11, _k43, _r54);
                    _sum11 = vmlaq_f32(_sum11, _k44, _r55);
                    _sum12 = vmlaq_f32(_sum12, _k40, _r52);
                    _sum12 = vmlaq_f32(_sum12, _k41, _r53);
                    _sum12 = vmlaq_f32(_sum12, _k42, _r54);
                    _sum12 = vmlaq_f32(_sum12, _k43, _r55);
                    _sum12 = vmlaq_f32(_sum12, _k44, _r56);
                    _sum13 = vmlaq_f32(_sum13, _k40, _r53);
                    _sum13 = vmlaq_f32(_sum13, _k41, _r54);
                    _sum13 = vmlaq_f32(_sum13, _k42, _r55);
                    _sum13 = vmlaq_f32(_sum13, _k43, _r56);
                    _sum13 = vmlaq_f32(_sum13, _k44, _r57);

                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + 4, _sum01);
                    vst1q_f32(outptr0 + 8, _sum02);
                    vst1q_f32(outptr0 + 12, _sum03);
                    vst1q_f32(outptr1, _sum10);
                    vst1q_f32(outptr1 + 4, _sum11);
                    vst1q_f32(outptr1 + 8, _sum12);
                    vst1q_f32(outptr1 + 12, _sum13);

                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                    r3 += 16;
                    r4 += 16;
                    r5 += 16;
                    outptr0 += 16;
                    outptr1 += 16;
                }
                for (; j + 1 < outw; j += 2)
                {
                    float32x4_t _sum00 = _bias0;
                    float32x4_t _sum01 = _bias0;
                    float32x4_t _sum10 = _bias0;
                    float32x4_t _sum11 = _bias0;

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r01 = vld1q_f32(r0 + 4);
                    float32x4_t _r02 = vld1q_f32(r0 + 8);
                    float32x4_t _r03 = vld1q_f32(r0 + 12);
                    float32x4_t _r04 = vld1q_f32(r0 + 16);
                    float32x4_t _r05 = vld1q_f32(r0 + 20);

                    float32x4_t _k00 = vld1q_f32(k0);
                    float32x4_t _k01 = vld1q_f32(k0 + 4);
                    float32x4_t _k02 = vld1q_f32(k0 + 8);
                    float32x4_t _k03 = vld1q_f32(k0 + 12);
                    float32x4_t _k04 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum00 = vmlaq_f32(_sum00, _k00, _r00);
                    _sum00 = vmlaq_f32(_sum00, _k01, _r01);
                    _sum00 = vmlaq_f32(_sum00, _k02, _r02);
                    _sum00 = vmlaq_f32(_sum00, _k03, _r03);
                    _sum00 = vmlaq_f32(_sum00, _k04, _r04);
                    _sum01 = vmlaq_f32(_sum01, _k00, _r01);
                    _sum01 = vmlaq_f32(_sum01, _k01, _r02);
                    _sum01 = vmlaq_f32(_sum01, _k02, _r03);
                    _sum01 = vmlaq_f32(_sum01, _k03, _r04);
                    _sum01 = vmlaq_f32(_sum01, _k04, _r05);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r11 = vld1q_f32(r1 + 4);
                    float32x4_t _r12 = vld1q_f32(r1 + 8);
                    float32x4_t _r13 = vld1q_f32(r1 + 12);
                    float32x4_t _r14 = vld1q_f32(r1 + 16);
                    float32x4_t _r15 = vld1q_f32(r1 + 20);

                    float32x4_t _k10 = vld1q_f32(k0);
                    float32x4_t _k11 = vld1q_f32(k0 + 4);
                    float32x4_t _k12 = vld1q_f32(k0 + 8);
                    float32x4_t _k13 = vld1q_f32(k0 + 12);
                    float32x4_t _k14 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum10 = vmlaq_f32(_sum10, _k00, _r10);
                    _sum10 = vmlaq_f32(_sum10, _k01, _r11);
                    _sum10 = vmlaq_f32(_sum10, _k02, _r12);
                    _sum10 = vmlaq_f32(_sum10, _k03, _r13);
                    _sum10 = vmlaq_f32(_sum10, _k04, _r14);
                    _sum11 = vmlaq_f32(_sum11, _k00, _r11);
                    _sum11 = vmlaq_f32(_sum11, _k01, _r12);
                    _sum11 = vmlaq_f32(_sum11, _k02, _r13);
                    _sum11 = vmlaq_f32(_sum11, _k03, _r14);
                    _sum11 = vmlaq_f32(_sum11, _k04, _r15);

                    _sum00 = vmlaq_f32(_sum00, _k10, _r10);
                    _sum00 = vmlaq_f32(_sum00, _k11, _r11);
                    _sum00 = vmlaq_f32(_sum00, _k12, _r12);
                    _sum00 = vmlaq_f32(_sum00, _k13, _r13);
                    _sum00 = vmlaq_f32(_sum00, _k14, _r14);
                    _sum01 = vmlaq_f32(_sum01, _k10, _r11);
                    _sum01 = vmlaq_f32(_sum01, _k11, _r12);
                    _sum01 = vmlaq_f32(_sum01, _k12, _r13);
                    _sum01 = vmlaq_f32(_sum01, _k13, _r14);
                    _sum01 = vmlaq_f32(_sum01, _k14, _r15);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r21 = vld1q_f32(r2 + 4);
                    float32x4_t _r22 = vld1q_f32(r2 + 8);
                    float32x4_t _r23 = vld1q_f32(r2 + 12);
                    float32x4_t _r24 = vld1q_f32(r2 + 16);
                    float32x4_t _r25 = vld1q_f32(r2 + 20);

                    float32x4_t _k20 = vld1q_f32(k0);
                    float32x4_t _k21 = vld1q_f32(k0 + 4);
                    float32x4_t _k22 = vld1q_f32(k0 + 8);
                    float32x4_t _k23 = vld1q_f32(k0 + 12);
                    float32x4_t _k24 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum10 = vmlaq_f32(_sum10, _k10, _r20);
                    _sum10 = vmlaq_f32(_sum10, _k11, _r21);
                    _sum10 = vmlaq_f32(_sum10, _k12, _r22);
                    _sum10 = vmlaq_f32(_sum10, _k13, _r23);
                    _sum10 = vmlaq_f32(_sum10, _k14, _r24);
                    _sum11 = vmlaq_f32(_sum11, _k10, _r21);
                    _sum11 = vmlaq_f32(_sum11, _k11, _r22);
                    _sum11 = vmlaq_f32(_sum11, _k12, _r23);
                    _sum11 = vmlaq_f32(_sum11, _k13, _r24);
                    _sum11 = vmlaq_f32(_sum11, _k14, _r25);

                    _sum00 = vmlaq_f32(_sum00, _k20, _r20);
                    _sum00 = vmlaq_f32(_sum00, _k21, _r21);
                    _sum00 = vmlaq_f32(_sum00, _k22, _r22);
                    _sum00 = vmlaq_f32(_sum00, _k23, _r23);
                    _sum00 = vmlaq_f32(_sum00, _k24, _r24);
                    _sum01 = vmlaq_f32(_sum01, _k20, _r21);
                    _sum01 = vmlaq_f32(_sum01, _k21, _r22);
                    _sum01 = vmlaq_f32(_sum01, _k22, _r23);
                    _sum01 = vmlaq_f32(_sum01, _k23, _r24);
                    _sum01 = vmlaq_f32(_sum01, _k24, _r25);

                    float32x4_t _r30 = vld1q_f32(r3);
                    float32x4_t _r31 = vld1q_f32(r3 + 4);
                    float32x4_t _r32 = vld1q_f32(r3 + 8);
                    float32x4_t _r33 = vld1q_f32(r3 + 12);
                    float32x4_t _r34 = vld1q_f32(r3 + 16);
                    float32x4_t _r35 = vld1q_f32(r3 + 20);

                    float32x4_t _k30 = vld1q_f32(k0);
                    float32x4_t _k31 = vld1q_f32(k0 + 4);
                    float32x4_t _k32 = vld1q_f32(k0 + 8);
                    float32x4_t _k33 = vld1q_f32(k0 + 12);
                    float32x4_t _k34 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum10 = vmlaq_f32(_sum10, _k20, _r30);
                    _sum10 = vmlaq_f32(_sum10, _k21, _r31);
                    _sum10 = vmlaq_f32(_sum10, _k22, _r32);
                    _sum10 = vmlaq_f32(_sum10, _k23, _r33);
                    _sum10 = vmlaq_f32(_sum10, _k24, _r34);
                    _sum11 = vmlaq_f32(_sum11, _k20, _r31);
                    _sum11 = vmlaq_f32(_sum11, _k21, _r32);
                    _sum11 = vmlaq_f32(_sum11, _k22, _r33);
                    _sum11 = vmlaq_f32(_sum11, _k23, _r34);
                    _sum11 = vmlaq_f32(_sum11, _k24, _r35);

                    _sum00 = vmlaq_f32(_sum00, _k30, _r30);
                    _sum00 = vmlaq_f32(_sum00, _k31, _r31);
                    _sum00 = vmlaq_f32(_sum00, _k32, _r32);
                    _sum00 = vmlaq_f32(_sum00, _k33, _r33);
                    _sum00 = vmlaq_f32(_sum00, _k34, _r34);
                    _sum01 = vmlaq_f32(_sum01, _k30, _r31);
                    _sum01 = vmlaq_f32(_sum01, _k31, _r32);
                    _sum01 = vmlaq_f32(_sum01, _k32, _r33);
                    _sum01 = vmlaq_f32(_sum01, _k33, _r34);
                    _sum01 = vmlaq_f32(_sum01, _k34, _r35);

                    float32x4_t _r40 = vld1q_f32(r4);
                    float32x4_t _r41 = vld1q_f32(r4 + 4);
                    float32x4_t _r42 = vld1q_f32(r4 + 8);
                    float32x4_t _r43 = vld1q_f32(r4 + 12);
                    float32x4_t _r44 = vld1q_f32(r4 + 16);
                    float32x4_t _r45 = vld1q_f32(r4 + 20);

                    float32x4_t _k40 = vld1q_f32(k0);
                    float32x4_t _k41 = vld1q_f32(k0 + 4);
                    float32x4_t _k42 = vld1q_f32(k0 + 8);
                    float32x4_t _k43 = vld1q_f32(k0 + 12);
                    float32x4_t _k44 = vld1q_f32(k0 + 16);
                    k0 -= 80;

                    _sum10 = vmlaq_f32(_sum10, _k30, _r40);
                    _sum10 = vmlaq_f32(_sum10, _k31, _r41);
                    _sum10 = vmlaq_f32(_sum10, _k32, _r42);
                    _sum10 = vmlaq_f32(_sum10, _k33, _r43);
                    _sum10 = vmlaq_f32(_sum10, _k34, _r44);
                    _sum11 = vmlaq_f32(_sum11, _k30, _r41);
                    _sum11 = vmlaq_f32(_sum11, _k31, _r42);
                    _sum11 = vmlaq_f32(_sum11, _k32, _r43);
                    _sum11 = vmlaq_f32(_sum11, _k33, _r44);
                    _sum11 = vmlaq_f32(_sum11, _k34, _r45);

                    _sum00 = vmlaq_f32(_sum00, _k40, _r40);
                    _sum00 = vmlaq_f32(_sum00, _k41, _r41);
                    _sum00 = vmlaq_f32(_sum00, _k42, _r42);
                    _sum00 = vmlaq_f32(_sum00, _k43, _r43);
                    _sum00 = vmlaq_f32(_sum00, _k44, _r44);
                    _sum01 = vmlaq_f32(_sum01, _k40, _r41);
                    _sum01 = vmlaq_f32(_sum01, _k41, _r42);
                    _sum01 = vmlaq_f32(_sum01, _k42, _r43);
                    _sum01 = vmlaq_f32(_sum01, _k43, _r44);
                    _sum01 = vmlaq_f32(_sum01, _k44, _r45);

                    float32x4_t _r50 = vld1q_f32(r5);
                    float32x4_t _r51 = vld1q_f32(r5 + 4);
                    float32x4_t _r52 = vld1q_f32(r5 + 8);
                    float32x4_t _r53 = vld1q_f32(r5 + 12);
                    float32x4_t _r54 = vld1q_f32(r5 + 16);
                    float32x4_t _r55 = vld1q_f32(r5 + 20);

                    _sum10 = vmlaq_f32(_sum10, _k40, _r50);
                    _sum10 = vmlaq_f32(_sum10, _k41, _r51);
                    _sum10 = vmlaq_f32(_sum10, _k42, _r52);
                    _sum10 = vmlaq_f32(_sum10, _k43, _r53);
                    _sum10 = vmlaq_f32(_sum10, _k44, _r54);
                    _sum11 = vmlaq_f32(_sum11, _k40, _r51);
                    _sum11 = vmlaq_f32(_sum11, _k41, _r52);
                    _sum11 = vmlaq_f32(_sum11, _k42, _r53);
                    _sum11 = vmlaq_f32(_sum11, _k43, _r54);
                    _sum11 = vmlaq_f32(_sum11, _k44, _r55);

                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + 4, _sum01);
                    vst1q_f32(outptr1, _sum10);
                    vst1q_f32(outptr1 + 4, _sum11);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    r4 += 8;
                    r5 += 8;
                    outptr0 += 8;
                    outptr1 += 8;
                }
                for (; j < outw; j++)
                {
                    float32x4_t _sum0 = _bias0;
                    float32x4_t _sum1 = _bias0;

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r01 = vld1q_f32(r0 + 4);
                    float32x4_t _r02 = vld1q_f32(r0 + 8);
                    float32x4_t _r03 = vld1q_f32(r0 + 12);
                    float32x4_t _r04 = vld1q_f32(r0 + 16);

                    float32x4_t _k00 = vld1q_f32(k0);
                    float32x4_t _k01 = vld1q_f32(k0 + 4);
                    float32x4_t _k02 = vld1q_f32(k0 + 8);
                    float32x4_t _k03 = vld1q_f32(k0 + 12);
                    float32x4_t _k04 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                    _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                    _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                    _sum0 = vmlaq_f32(_sum0, _k03, _r03);
                    _sum0 = vmlaq_f32(_sum0, _k04, _r04);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r11 = vld1q_f32(r1 + 4);
                    float32x4_t _r12 = vld1q_f32(r1 + 8);
                    float32x4_t _r13 = vld1q_f32(r1 + 12);
                    float32x4_t _r14 = vld1q_f32(r1 + 16);

                    float32x4_t _k10 = vld1q_f32(k0);
                    float32x4_t _k11 = vld1q_f32(k0 + 4);
                    float32x4_t _k12 = vld1q_f32(k0 + 8);
                    float32x4_t _k13 = vld1q_f32(k0 + 12);
                    float32x4_t _k14 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum1 = vmlaq_f32(_sum1, _k00, _r10);
                    _sum1 = vmlaq_f32(_sum1, _k01, _r11);
                    _sum1 = vmlaq_f32(_sum1, _k02, _r12);
                    _sum1 = vmlaq_f32(_sum1, _k03, _r13);
                    _sum1 = vmlaq_f32(_sum1, _k04, _r14);

                    _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                    _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                    _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                    _sum0 = vmlaq_f32(_sum0, _k13, _r13);
                    _sum0 = vmlaq_f32(_sum0, _k14, _r14);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r21 = vld1q_f32(r2 + 4);
                    float32x4_t _r22 = vld1q_f32(r2 + 8);
                    float32x4_t _r23 = vld1q_f32(r2 + 12);
                    float32x4_t _r24 = vld1q_f32(r2 + 16);

                    float32x4_t _k20 = vld1q_f32(k0);
                    float32x4_t _k21 = vld1q_f32(k0 + 4);
                    float32x4_t _k22 = vld1q_f32(k0 + 8);
                    float32x4_t _k23 = vld1q_f32(k0 + 12);
                    float32x4_t _k24 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum1 = vmlaq_f32(_sum1, _k10, _r20);
                    _sum1 = vmlaq_f32(_sum1, _k11, _r21);
                    _sum1 = vmlaq_f32(_sum1, _k12, _r22);
                    _sum1 = vmlaq_f32(_sum1, _k13, _r23);
                    _sum1 = vmlaq_f32(_sum1, _k14, _r24);

                    _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                    _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                    _sum0 = vmlaq_f32(_sum0, _k22, _r22);
                    _sum0 = vmlaq_f32(_sum0, _k23, _r23);
                    _sum0 = vmlaq_f32(_sum0, _k24, _r24);

                    float32x4_t _r30 = vld1q_f32(r3);
                    float32x4_t _r31 = vld1q_f32(r3 + 4);
                    float32x4_t _r32 = vld1q_f32(r3 + 8);
                    float32x4_t _r33 = vld1q_f32(r3 + 12);
                    float32x4_t _r34 = vld1q_f32(r3 + 16);

                    float32x4_t _k30 = vld1q_f32(k0);
                    float32x4_t _k31 = vld1q_f32(k0 + 4);
                    float32x4_t _k32 = vld1q_f32(k0 + 8);
                    float32x4_t _k33 = vld1q_f32(k0 + 12);
                    float32x4_t _k34 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum1 = vmlaq_f32(_sum1, _k20, _r30);
                    _sum1 = vmlaq_f32(_sum1, _k21, _r31);
                    _sum1 = vmlaq_f32(_sum1, _k22, _r32);
                    _sum1 = vmlaq_f32(_sum1, _k23, _r33);
                    _sum1 = vmlaq_f32(_sum1, _k24, _r34);

                    _sum0 = vmlaq_f32(_sum0, _k30, _r30);
                    _sum0 = vmlaq_f32(_sum0, _k31, _r31);
                    _sum0 = vmlaq_f32(_sum0, _k32, _r32);
                    _sum0 = vmlaq_f32(_sum0, _k33, _r33);
                    _sum0 = vmlaq_f32(_sum0, _k34, _r34);

                    float32x4_t _r40 = vld1q_f32(r4);
                    float32x4_t _r41 = vld1q_f32(r4 + 4);
                    float32x4_t _r42 = vld1q_f32(r4 + 8);
                    float32x4_t _r43 = vld1q_f32(r4 + 12);
                    float32x4_t _r44 = vld1q_f32(r4 + 16);

                    float32x4_t _k40 = vld1q_f32(k0);
                    float32x4_t _k41 = vld1q_f32(k0 + 4);
                    float32x4_t _k42 = vld1q_f32(k0 + 8);
                    float32x4_t _k43 = vld1q_f32(k0 + 12);
                    float32x4_t _k44 = vld1q_f32(k0 + 16);
                    k0 -= 80;

                    _sum1 = vmlaq_f32(_sum1, _k30, _r40);
                    _sum1 = vmlaq_f32(_sum1, _k31, _r41);
                    _sum1 = vmlaq_f32(_sum1, _k32, _r42);
                    _sum1 = vmlaq_f32(_sum1, _k33, _r43);
                    _sum1 = vmlaq_f32(_sum1, _k34, _r44);

                    _sum0 = vmlaq_f32(_sum0, _k40, _r40);
                    _sum0 = vmlaq_f32(_sum0, _k41, _r41);
                    _sum0 = vmlaq_f32(_sum0, _k42, _r42);
                    _sum0 = vmlaq_f32(_sum0, _k43, _r43);
                    _sum0 = vmlaq_f32(_sum0, _k44, _r44);

                    float32x4_t _r50 = vld1q_f32(r5);
                    float32x4_t _r51 = vld1q_f32(r5 + 4);
                    float32x4_t _r52 = vld1q_f32(r5 + 8);
                    float32x4_t _r53 = vld1q_f32(r5 + 12);
                    float32x4_t _r54 = vld1q_f32(r5 + 16);

                    _sum1 = vmlaq_f32(_sum1, _k40, _r50);
                    _sum1 = vmlaq_f32(_sum1, _k41, _r51);
                    _sum1 = vmlaq_f32(_sum1, _k42, _r52);
                    _sum1 = vmlaq_f32(_sum1, _k43, _r53);
                    _sum1 = vmlaq_f32(_sum1, _k44, _r54);

                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr1, _sum1);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    r4 += 4;
                    r5 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
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
    #endif // __aarch64__
            for (; i < outh; i++)
            {
                int j = 0;

                for (; j + 3 < outw; j += 4)
                {
                    float32x4_t _sum0 = _bias0;
                    float32x4_t _sum1 = _bias0;
                    float32x4_t _sum2 = _bias0;
                    float32x4_t _sum3 = _bias0;

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r01 = vld1q_f32(r0 + 4);
                    float32x4_t _r02 = vld1q_f32(r0 + 8);
                    float32x4_t _r03 = vld1q_f32(r0 + 12);
                    float32x4_t _r04 = vld1q_f32(r0 + 16);
                    float32x4_t _r05 = vld1q_f32(r0 + 20);
                    float32x4_t _r06 = vld1q_f32(r0 + 24);
                    float32x4_t _r07 = vld1q_f32(r0 + 28);

                    float32x4_t _k00 = vld1q_f32(k0);
                    float32x4_t _k01 = vld1q_f32(k0 + 4);
                    float32x4_t _k02 = vld1q_f32(k0 + 8);
                    float32x4_t _k03 = vld1q_f32(k0 + 12);
                    float32x4_t _k04 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                    _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                    _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                    _sum0 = vmlaq_f32(_sum0, _k03, _r03);
                    _sum0 = vmlaq_f32(_sum0, _k04, _r04);
                    _sum1 = vmlaq_f32(_sum1, _k00, _r01);
                    _sum1 = vmlaq_f32(_sum1, _k01, _r02);
                    _sum1 = vmlaq_f32(_sum1, _k02, _r03);
                    _sum1 = vmlaq_f32(_sum1, _k03, _r04);
                    _sum1 = vmlaq_f32(_sum1, _k04, _r05);
                    _sum2 = vmlaq_f32(_sum2, _k00, _r02);
                    _sum2 = vmlaq_f32(_sum2, _k01, _r03);
                    _sum2 = vmlaq_f32(_sum2, _k02, _r04);
                    _sum2 = vmlaq_f32(_sum2, _k03, _r05);
                    _sum2 = vmlaq_f32(_sum2, _k04, _r06);
                    _sum3 = vmlaq_f32(_sum3, _k00, _r03);
                    _sum3 = vmlaq_f32(_sum3, _k01, _r04);
                    _sum3 = vmlaq_f32(_sum3, _k02, _r05);
                    _sum3 = vmlaq_f32(_sum3, _k03, _r06);
                    _sum3 = vmlaq_f32(_sum3, _k04, _r07);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r11 = vld1q_f32(r1 + 4);
                    float32x4_t _r12 = vld1q_f32(r1 + 8);
                    float32x4_t _r13 = vld1q_f32(r1 + 12);
                    float32x4_t _r14 = vld1q_f32(r1 + 16);
                    float32x4_t _r15 = vld1q_f32(r1 + 20);
                    float32x4_t _r16 = vld1q_f32(r1 + 24);
                    float32x4_t _r17 = vld1q_f32(r1 + 28);

                    float32x4_t _k10 = vld1q_f32(k0);
                    float32x4_t _k11 = vld1q_f32(k0 + 4);
                    float32x4_t _k12 = vld1q_f32(k0 + 8);
                    float32x4_t _k13 = vld1q_f32(k0 + 12);
                    float32x4_t _k14 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                    _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                    _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                    _sum0 = vmlaq_f32(_sum0, _k13, _r13);
                    _sum0 = vmlaq_f32(_sum0, _k14, _r14);
                    _sum1 = vmlaq_f32(_sum1, _k10, _r11);
                    _sum1 = vmlaq_f32(_sum1, _k11, _r12);
                    _sum1 = vmlaq_f32(_sum1, _k12, _r13);
                    _sum1 = vmlaq_f32(_sum1, _k13, _r14);
                    _sum1 = vmlaq_f32(_sum1, _k14, _r15);
                    _sum2 = vmlaq_f32(_sum2, _k10, _r12);
                    _sum2 = vmlaq_f32(_sum2, _k11, _r13);
                    _sum2 = vmlaq_f32(_sum2, _k12, _r14);
                    _sum2 = vmlaq_f32(_sum2, _k13, _r15);
                    _sum2 = vmlaq_f32(_sum2, _k14, _r16);
                    _sum3 = vmlaq_f32(_sum3, _k10, _r13);
                    _sum3 = vmlaq_f32(_sum3, _k11, _r14);
                    _sum3 = vmlaq_f32(_sum3, _k12, _r15);
                    _sum3 = vmlaq_f32(_sum3, _k13, _r16);
                    _sum3 = vmlaq_f32(_sum3, _k14, _r17);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r21 = vld1q_f32(r2 + 4);
                    float32x4_t _r22 = vld1q_f32(r2 + 8);
                    float32x4_t _r23 = vld1q_f32(r2 + 12);
                    float32x4_t _r24 = vld1q_f32(r2 + 16);
                    float32x4_t _r25 = vld1q_f32(r2 + 20);
                    float32x4_t _r26 = vld1q_f32(r2 + 24);
                    float32x4_t _r27 = vld1q_f32(r2 + 28);

                    float32x4_t _k20 = vld1q_f32(k0);
                    float32x4_t _k21 = vld1q_f32(k0 + 4);
                    float32x4_t _k22 = vld1q_f32(k0 + 8);
                    float32x4_t _k23 = vld1q_f32(k0 + 12);
                    float32x4_t _k24 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                    _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                    _sum0 = vmlaq_f32(_sum0, _k22, _r22);
                    _sum0 = vmlaq_f32(_sum0, _k23, _r23);
                    _sum0 = vmlaq_f32(_sum0, _k24, _r24);
                    _sum1 = vmlaq_f32(_sum1, _k20, _r21);
                    _sum1 = vmlaq_f32(_sum1, _k21, _r22);
                    _sum1 = vmlaq_f32(_sum1, _k22, _r23);
                    _sum1 = vmlaq_f32(_sum1, _k23, _r24);
                    _sum1 = vmlaq_f32(_sum1, _k24, _r25);
                    _sum2 = vmlaq_f32(_sum2, _k20, _r22);
                    _sum2 = vmlaq_f32(_sum2, _k21, _r23);
                    _sum2 = vmlaq_f32(_sum2, _k22, _r24);
                    _sum2 = vmlaq_f32(_sum2, _k23, _r25);
                    _sum2 = vmlaq_f32(_sum2, _k24, _r26);
                    _sum3 = vmlaq_f32(_sum3, _k20, _r23);
                    _sum3 = vmlaq_f32(_sum3, _k21, _r24);
                    _sum3 = vmlaq_f32(_sum3, _k22, _r25);
                    _sum3 = vmlaq_f32(_sum3, _k23, _r26);
                    _sum3 = vmlaq_f32(_sum3, _k24, _r27);

                    float32x4_t _r30 = vld1q_f32(r3);
                    float32x4_t _r31 = vld1q_f32(r3 + 4);
                    float32x4_t _r32 = vld1q_f32(r3 + 8);
                    float32x4_t _r33 = vld1q_f32(r3 + 12);
                    float32x4_t _r34 = vld1q_f32(r3 + 16);
                    float32x4_t _r35 = vld1q_f32(r3 + 20);
                    float32x4_t _r36 = vld1q_f32(r3 + 24);
                    float32x4_t _r37 = vld1q_f32(r3 + 28);

                    float32x4_t _k30 = vld1q_f32(k0);
                    float32x4_t _k31 = vld1q_f32(k0 + 4);
                    float32x4_t _k32 = vld1q_f32(k0 + 8);
                    float32x4_t _k33 = vld1q_f32(k0 + 12);
                    float32x4_t _k34 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k30, _r30);
                    _sum0 = vmlaq_f32(_sum0, _k31, _r31);
                    _sum0 = vmlaq_f32(_sum0, _k32, _r32);
                    _sum0 = vmlaq_f32(_sum0, _k33, _r33);
                    _sum0 = vmlaq_f32(_sum0, _k34, _r34);
                    _sum1 = vmlaq_f32(_sum1, _k30, _r31);
                    _sum1 = vmlaq_f32(_sum1, _k31, _r32);
                    _sum1 = vmlaq_f32(_sum1, _k32, _r33);
                    _sum1 = vmlaq_f32(_sum1, _k33, _r34);
                    _sum1 = vmlaq_f32(_sum1, _k34, _r35);
                    _sum2 = vmlaq_f32(_sum2, _k30, _r32);
                    _sum2 = vmlaq_f32(_sum2, _k31, _r33);
                    _sum2 = vmlaq_f32(_sum2, _k32, _r34);
                    _sum2 = vmlaq_f32(_sum2, _k33, _r35);
                    _sum2 = vmlaq_f32(_sum2, _k34, _r36);
                    _sum3 = vmlaq_f32(_sum3, _k30, _r33);
                    _sum3 = vmlaq_f32(_sum3, _k31, _r34);
                    _sum3 = vmlaq_f32(_sum3, _k32, _r35);
                    _sum3 = vmlaq_f32(_sum3, _k33, _r36);
                    _sum3 = vmlaq_f32(_sum3, _k34, _r37);

                    float32x4_t _r40 = vld1q_f32(r4);
                    float32x4_t _r41 = vld1q_f32(r4 + 4);
                    float32x4_t _r42 = vld1q_f32(r4 + 8);
                    float32x4_t _r43 = vld1q_f32(r4 + 12);
                    float32x4_t _r44 = vld1q_f32(r4 + 16);
                    float32x4_t _r45 = vld1q_f32(r4 + 20);
                    float32x4_t _r46 = vld1q_f32(r4 + 24);
                    float32x4_t _r47 = vld1q_f32(r4 + 28);

                    float32x4_t _k40 = vld1q_f32(k0);
                    float32x4_t _k41 = vld1q_f32(k0 + 4);
                    float32x4_t _k42 = vld1q_f32(k0 + 8);
                    float32x4_t _k43 = vld1q_f32(k0 + 12);
                    float32x4_t _k44 = vld1q_f32(k0 + 16);
                    k0 -= 80;

                    _sum0 = vmlaq_f32(_sum0, _k40, _r40);
                    _sum0 = vmlaq_f32(_sum0, _k41, _r41);
                    _sum0 = vmlaq_f32(_sum0, _k42, _r42);
                    _sum0 = vmlaq_f32(_sum0, _k43, _r43);
                    _sum0 = vmlaq_f32(_sum0, _k44, _r44);
                    _sum1 = vmlaq_f32(_sum1, _k40, _r41);
                    _sum1 = vmlaq_f32(_sum1, _k41, _r42);
                    _sum1 = vmlaq_f32(_sum1, _k42, _r43);
                    _sum1 = vmlaq_f32(_sum1, _k43, _r44);
                    _sum1 = vmlaq_f32(_sum1, _k44, _r45);
                    _sum2 = vmlaq_f32(_sum2, _k40, _r42);
                    _sum2 = vmlaq_f32(_sum2, _k41, _r43);
                    _sum2 = vmlaq_f32(_sum2, _k42, _r44);
                    _sum2 = vmlaq_f32(_sum2, _k43, _r45);
                    _sum2 = vmlaq_f32(_sum2, _k44, _r46);
                    _sum3 = vmlaq_f32(_sum3, _k40, _r43);
                    _sum3 = vmlaq_f32(_sum3, _k41, _r44);
                    _sum3 = vmlaq_f32(_sum3, _k42, _r45);
                    _sum3 = vmlaq_f32(_sum3, _k43, _r46);
                    _sum3 = vmlaq_f32(_sum3, _k44, _r47);

                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);
                    vst1q_f32(outptr0 + 8, _sum2);
                    vst1q_f32(outptr0 + 12, _sum3);

                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                    r3 += 16;
                    r4 += 16;
                    outptr0 += 16;
                }
                for (; j + 1 < outw; j += 2)
                {
                    float32x4_t _sum0 = _bias0;
                    float32x4_t _sum1 = _bias0;

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r01 = vld1q_f32(r0 + 4);
                    float32x4_t _r02 = vld1q_f32(r0 + 8);
                    float32x4_t _r03 = vld1q_f32(r0 + 12);
                    float32x4_t _r04 = vld1q_f32(r0 + 16);
                    float32x4_t _r05 = vld1q_f32(r0 + 20);

                    float32x4_t _k00 = vld1q_f32(k0);
                    float32x4_t _k01 = vld1q_f32(k0 + 4);
                    float32x4_t _k02 = vld1q_f32(k0 + 8);
                    float32x4_t _k03 = vld1q_f32(k0 + 12);
                    float32x4_t _k04 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                    _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                    _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                    _sum0 = vmlaq_f32(_sum0, _k03, _r03);
                    _sum0 = vmlaq_f32(_sum0, _k04, _r04);
                    _sum1 = vmlaq_f32(_sum1, _k00, _r01);
                    _sum1 = vmlaq_f32(_sum1, _k01, _r02);
                    _sum1 = vmlaq_f32(_sum1, _k02, _r03);
                    _sum1 = vmlaq_f32(_sum1, _k03, _r04);
                    _sum1 = vmlaq_f32(_sum1, _k04, _r05);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r11 = vld1q_f32(r1 + 4);
                    float32x4_t _r12 = vld1q_f32(r1 + 8);
                    float32x4_t _r13 = vld1q_f32(r1 + 12);
                    float32x4_t _r14 = vld1q_f32(r1 + 16);
                    float32x4_t _r15 = vld1q_f32(r1 + 20);

                    float32x4_t _k10 = vld1q_f32(k0);
                    float32x4_t _k11 = vld1q_f32(k0 + 4);
                    float32x4_t _k12 = vld1q_f32(k0 + 8);
                    float32x4_t _k13 = vld1q_f32(k0 + 12);
                    float32x4_t _k14 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                    _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                    _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                    _sum0 = vmlaq_f32(_sum0, _k13, _r13);
                    _sum0 = vmlaq_f32(_sum0, _k14, _r14);
                    _sum1 = vmlaq_f32(_sum1, _k10, _r11);
                    _sum1 = vmlaq_f32(_sum1, _k11, _r12);
                    _sum1 = vmlaq_f32(_sum1, _k12, _r13);
                    _sum1 = vmlaq_f32(_sum1, _k13, _r14);
                    _sum1 = vmlaq_f32(_sum1, _k14, _r15);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r21 = vld1q_f32(r2 + 4);
                    float32x4_t _r22 = vld1q_f32(r2 + 8);
                    float32x4_t _r23 = vld1q_f32(r2 + 12);
                    float32x4_t _r24 = vld1q_f32(r2 + 16);
                    float32x4_t _r25 = vld1q_f32(r2 + 20);

                    float32x4_t _k20 = vld1q_f32(k0);
                    float32x4_t _k21 = vld1q_f32(k0 + 4);
                    float32x4_t _k22 = vld1q_f32(k0 + 8);
                    float32x4_t _k23 = vld1q_f32(k0 + 12);
                    float32x4_t _k24 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                    _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                    _sum0 = vmlaq_f32(_sum0, _k22, _r22);
                    _sum0 = vmlaq_f32(_sum0, _k23, _r23);
                    _sum0 = vmlaq_f32(_sum0, _k24, _r24);
                    _sum1 = vmlaq_f32(_sum1, _k20, _r21);
                    _sum1 = vmlaq_f32(_sum1, _k21, _r22);
                    _sum1 = vmlaq_f32(_sum1, _k22, _r23);
                    _sum1 = vmlaq_f32(_sum1, _k23, _r24);
                    _sum1 = vmlaq_f32(_sum1, _k24, _r25);

                    float32x4_t _r30 = vld1q_f32(r3);
                    float32x4_t _r31 = vld1q_f32(r3 + 4);
                    float32x4_t _r32 = vld1q_f32(r3 + 8);
                    float32x4_t _r33 = vld1q_f32(r3 + 12);
                    float32x4_t _r34 = vld1q_f32(r3 + 16);
                    float32x4_t _r35 = vld1q_f32(r3 + 20);

                    float32x4_t _k30 = vld1q_f32(k0);
                    float32x4_t _k31 = vld1q_f32(k0 + 4);
                    float32x4_t _k32 = vld1q_f32(k0 + 8);
                    float32x4_t _k33 = vld1q_f32(k0 + 12);
                    float32x4_t _k34 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k30, _r30);
                    _sum0 = vmlaq_f32(_sum0, _k31, _r31);
                    _sum0 = vmlaq_f32(_sum0, _k32, _r32);
                    _sum0 = vmlaq_f32(_sum0, _k33, _r33);
                    _sum0 = vmlaq_f32(_sum0, _k34, _r34);
                    _sum1 = vmlaq_f32(_sum1, _k30, _r31);
                    _sum1 = vmlaq_f32(_sum1, _k31, _r32);
                    _sum1 = vmlaq_f32(_sum1, _k32, _r33);
                    _sum1 = vmlaq_f32(_sum1, _k33, _r34);
                    _sum1 = vmlaq_f32(_sum1, _k34, _r35);

                    float32x4_t _r40 = vld1q_f32(r4);
                    float32x4_t _r41 = vld1q_f32(r4 + 4);
                    float32x4_t _r42 = vld1q_f32(r4 + 8);
                    float32x4_t _r43 = vld1q_f32(r4 + 12);
                    float32x4_t _r44 = vld1q_f32(r4 + 16);
                    float32x4_t _r45 = vld1q_f32(r4 + 20);

                    float32x4_t _k40 = vld1q_f32(k0);
                    float32x4_t _k41 = vld1q_f32(k0 + 4);
                    float32x4_t _k42 = vld1q_f32(k0 + 8);
                    float32x4_t _k43 = vld1q_f32(k0 + 12);
                    float32x4_t _k44 = vld1q_f32(k0 + 16);
                    k0 -= 80;

                    _sum0 = vmlaq_f32(_sum0, _k40, _r40);
                    _sum0 = vmlaq_f32(_sum0, _k41, _r41);
                    _sum0 = vmlaq_f32(_sum0, _k42, _r42);
                    _sum0 = vmlaq_f32(_sum0, _k43, _r43);
                    _sum0 = vmlaq_f32(_sum0, _k44, _r44);
                    _sum1 = vmlaq_f32(_sum1, _k40, _r41);
                    _sum1 = vmlaq_f32(_sum1, _k41, _r42);
                    _sum1 = vmlaq_f32(_sum1, _k42, _r43);
                    _sum1 = vmlaq_f32(_sum1, _k43, _r44);
                    _sum1 = vmlaq_f32(_sum1, _k44, _r45);

                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    r3 += 8;
                    r4 += 8;
                    outptr0 += 8;
                }
                for (; j < outw; j++)
                {
                    float32x4_t _sum0 = _bias0;

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r01 = vld1q_f32(r0 + 4);
                    float32x4_t _r02 = vld1q_f32(r0 + 8);
                    float32x4_t _r03 = vld1q_f32(r0 + 12);
                    float32x4_t _r04 = vld1q_f32(r0 + 16);

                    float32x4_t _k00 = vld1q_f32(k0);
                    float32x4_t _k01 = vld1q_f32(k0 + 4);
                    float32x4_t _k02 = vld1q_f32(k0 + 8);
                    float32x4_t _k03 = vld1q_f32(k0 + 12);
                    float32x4_t _k04 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                    _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                    _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                    _sum0 = vmlaq_f32(_sum0, _k03, _r03);
                    _sum0 = vmlaq_f32(_sum0, _k04, _r04);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r11 = vld1q_f32(r1 + 4);
                    float32x4_t _r12 = vld1q_f32(r1 + 8);
                    float32x4_t _r13 = vld1q_f32(r1 + 12);
                    float32x4_t _r14 = vld1q_f32(r1 + 16);

                    float32x4_t _k10 = vld1q_f32(k0);
                    float32x4_t _k11 = vld1q_f32(k0 + 4);
                    float32x4_t _k12 = vld1q_f32(k0 + 8);
                    float32x4_t _k13 = vld1q_f32(k0 + 12);
                    float32x4_t _k14 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                    _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                    _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                    _sum0 = vmlaq_f32(_sum0, _k13, _r13);
                    _sum0 = vmlaq_f32(_sum0, _k14, _r14);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r21 = vld1q_f32(r2 + 4);
                    float32x4_t _r22 = vld1q_f32(r2 + 8);
                    float32x4_t _r23 = vld1q_f32(r2 + 12);
                    float32x4_t _r24 = vld1q_f32(r2 + 16);

                    float32x4_t _k20 = vld1q_f32(k0);
                    float32x4_t _k21 = vld1q_f32(k0 + 4);
                    float32x4_t _k22 = vld1q_f32(k0 + 8);
                    float32x4_t _k23 = vld1q_f32(k0 + 12);
                    float32x4_t _k24 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                    _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                    _sum0 = vmlaq_f32(_sum0, _k22, _r22);
                    _sum0 = vmlaq_f32(_sum0, _k23, _r23);
                    _sum0 = vmlaq_f32(_sum0, _k24, _r24);

                    float32x4_t _r30 = vld1q_f32(r3);
                    float32x4_t _r31 = vld1q_f32(r3 + 4);
                    float32x4_t _r32 = vld1q_f32(r3 + 8);
                    float32x4_t _r33 = vld1q_f32(r3 + 12);
                    float32x4_t _r34 = vld1q_f32(r3 + 16);

                    float32x4_t _k30 = vld1q_f32(k0);
                    float32x4_t _k31 = vld1q_f32(k0 + 4);
                    float32x4_t _k32 = vld1q_f32(k0 + 8);
                    float32x4_t _k33 = vld1q_f32(k0 + 12);
                    float32x4_t _k34 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k30, _r30);
                    _sum0 = vmlaq_f32(_sum0, _k31, _r31);
                    _sum0 = vmlaq_f32(_sum0, _k32, _r32);
                    _sum0 = vmlaq_f32(_sum0, _k33, _r33);
                    _sum0 = vmlaq_f32(_sum0, _k34, _r34);

                    float32x4_t _r40 = vld1q_f32(r4);
                    float32x4_t _r41 = vld1q_f32(r4 + 4);
                    float32x4_t _r42 = vld1q_f32(r4 + 8);
                    float32x4_t _r43 = vld1q_f32(r4 + 12);
                    float32x4_t _r44 = vld1q_f32(r4 + 16);

                    float32x4_t _k40 = vld1q_f32(k0);
                    float32x4_t _k41 = vld1q_f32(k0 + 4);
                    float32x4_t _k42 = vld1q_f32(k0 + 8);
                    float32x4_t _k43 = vld1q_f32(k0 + 12);
                    float32x4_t _k44 = vld1q_f32(k0 + 16);
                    k0 -= 80;

                    _sum0 = vmlaq_f32(_sum0, _k40, _r40);
                    _sum0 = vmlaq_f32(_sum0, _k41, _r41);
                    _sum0 = vmlaq_f32(_sum0, _k42, _r42);
                    _sum0 = vmlaq_f32(_sum0, _k43, _r43);
                    _sum0 = vmlaq_f32(_sum0, _k44, _r44);

                    vst1q_f32(outptr0, _sum0);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    r4 += 4;
                    outptr0 += 4;
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

Tensor depthwise_conv2d_5x5s1_neon_pack4(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, otter::ScalarType::Float4);
    
    return depthwise_conv2d_5x5s1_neon_pack4_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
}

Tensor& depthwise_conv2d_5x5s2_neon_pack4_out(
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

            float32x4_t _bias0 = bias ? vld1q_f32((const float*)bias + g * 4) : vdupq_n_f32(0.f);

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

                for (; j + 3 < outw; j += 4)
                {
                    float32x4_t _sum0 = _bias0;
                    float32x4_t _sum1 = _bias0;
                    float32x4_t _sum2 = _bias0;
                    float32x4_t _sum3 = _bias0;

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r01 = vld1q_f32(r0 + 4);
                    float32x4_t _r02 = vld1q_f32(r0 + 8);
                    float32x4_t _r03 = vld1q_f32(r0 + 12);
                    float32x4_t _r04 = vld1q_f32(r0 + 16);
                    float32x4_t _r05 = vld1q_f32(r0 + 20);
                    float32x4_t _r06 = vld1q_f32(r0 + 24);
                    float32x4_t _r07 = vld1q_f32(r0 + 28);
                    float32x4_t _r08 = vld1q_f32(r0 + 32);
                    float32x4_t _r09 = vld1q_f32(r0 + 36);
                    float32x4_t _r010 = vld1q_f32(r0 + 40);

                    float32x4_t _k00 = vld1q_f32(k0);
                    float32x4_t _k01 = vld1q_f32(k0 + 4);
                    float32x4_t _k02 = vld1q_f32(k0 + 8);
                    float32x4_t _k03 = vld1q_f32(k0 + 12);
                    float32x4_t _k04 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                    _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                    _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                    _sum0 = vmlaq_f32(_sum0, _k03, _r03);
                    _sum0 = vmlaq_f32(_sum0, _k04, _r04);
                    _sum1 = vmlaq_f32(_sum1, _k00, _r02);
                    _sum1 = vmlaq_f32(_sum1, _k01, _r03);
                    _sum1 = vmlaq_f32(_sum1, _k02, _r04);
                    _sum1 = vmlaq_f32(_sum1, _k03, _r05);
                    _sum1 = vmlaq_f32(_sum1, _k04, _r06);
                    _sum2 = vmlaq_f32(_sum2, _k00, _r04);
                    _sum2 = vmlaq_f32(_sum2, _k01, _r05);
                    _sum2 = vmlaq_f32(_sum2, _k02, _r06);
                    _sum2 = vmlaq_f32(_sum2, _k03, _r07);
                    _sum2 = vmlaq_f32(_sum2, _k04, _r08);
                    _sum3 = vmlaq_f32(_sum3, _k00, _r06);
                    _sum3 = vmlaq_f32(_sum3, _k01, _r07);
                    _sum3 = vmlaq_f32(_sum3, _k02, _r08);
                    _sum3 = vmlaq_f32(_sum3, _k03, _r09);
                    _sum3 = vmlaq_f32(_sum3, _k04, _r010);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r11 = vld1q_f32(r1 + 4);
                    float32x4_t _r12 = vld1q_f32(r1 + 8);
                    float32x4_t _r13 = vld1q_f32(r1 + 12);
                    float32x4_t _r14 = vld1q_f32(r1 + 16);
                    float32x4_t _r15 = vld1q_f32(r1 + 20);
                    float32x4_t _r16 = vld1q_f32(r1 + 24);
                    float32x4_t _r17 = vld1q_f32(r1 + 28);
                    float32x4_t _r18 = vld1q_f32(r1 + 32);
                    float32x4_t _r19 = vld1q_f32(r1 + 36);
                    float32x4_t _r110 = vld1q_f32(r1 + 40);

                    float32x4_t _k10 = vld1q_f32(k0);
                    float32x4_t _k11 = vld1q_f32(k0 + 4);
                    float32x4_t _k12 = vld1q_f32(k0 + 8);
                    float32x4_t _k13 = vld1q_f32(k0 + 12);
                    float32x4_t _k14 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                    _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                    _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                    _sum0 = vmlaq_f32(_sum0, _k13, _r13);
                    _sum0 = vmlaq_f32(_sum0, _k14, _r14);
                    _sum1 = vmlaq_f32(_sum1, _k10, _r12);
                    _sum1 = vmlaq_f32(_sum1, _k11, _r13);
                    _sum1 = vmlaq_f32(_sum1, _k12, _r14);
                    _sum1 = vmlaq_f32(_sum1, _k13, _r15);
                    _sum1 = vmlaq_f32(_sum1, _k14, _r16);
                    _sum2 = vmlaq_f32(_sum2, _k10, _r14);
                    _sum2 = vmlaq_f32(_sum2, _k11, _r15);
                    _sum2 = vmlaq_f32(_sum2, _k12, _r16);
                    _sum2 = vmlaq_f32(_sum2, _k13, _r17);
                    _sum2 = vmlaq_f32(_sum2, _k14, _r18);
                    _sum3 = vmlaq_f32(_sum3, _k10, _r16);
                    _sum3 = vmlaq_f32(_sum3, _k11, _r17);
                    _sum3 = vmlaq_f32(_sum3, _k12, _r18);
                    _sum3 = vmlaq_f32(_sum3, _k13, _r19);
                    _sum3 = vmlaq_f32(_sum3, _k14, _r110);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r21 = vld1q_f32(r2 + 4);
                    float32x4_t _r22 = vld1q_f32(r2 + 8);
                    float32x4_t _r23 = vld1q_f32(r2 + 12);
                    float32x4_t _r24 = vld1q_f32(r2 + 16);
                    float32x4_t _r25 = vld1q_f32(r2 + 20);
                    float32x4_t _r26 = vld1q_f32(r2 + 24);
                    float32x4_t _r27 = vld1q_f32(r2 + 28);
                    float32x4_t _r28 = vld1q_f32(r2 + 32);
                    float32x4_t _r29 = vld1q_f32(r2 + 36);
                    float32x4_t _r210 = vld1q_f32(r2 + 40);

                    float32x4_t _k20 = vld1q_f32(k0);
                    float32x4_t _k21 = vld1q_f32(k0 + 4);
                    float32x4_t _k22 = vld1q_f32(k0 + 8);
                    float32x4_t _k23 = vld1q_f32(k0 + 12);
                    float32x4_t _k24 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                    _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                    _sum0 = vmlaq_f32(_sum0, _k22, _r22);
                    _sum0 = vmlaq_f32(_sum0, _k23, _r23);
                    _sum0 = vmlaq_f32(_sum0, _k24, _r24);
                    _sum1 = vmlaq_f32(_sum1, _k20, _r22);
                    _sum1 = vmlaq_f32(_sum1, _k21, _r23);
                    _sum1 = vmlaq_f32(_sum1, _k22, _r24);
                    _sum1 = vmlaq_f32(_sum1, _k23, _r25);
                    _sum1 = vmlaq_f32(_sum1, _k24, _r26);
                    _sum2 = vmlaq_f32(_sum2, _k20, _r24);
                    _sum2 = vmlaq_f32(_sum2, _k21, _r25);
                    _sum2 = vmlaq_f32(_sum2, _k22, _r26);
                    _sum2 = vmlaq_f32(_sum2, _k23, _r27);
                    _sum2 = vmlaq_f32(_sum2, _k24, _r28);
                    _sum3 = vmlaq_f32(_sum3, _k20, _r26);
                    _sum3 = vmlaq_f32(_sum3, _k21, _r27);
                    _sum3 = vmlaq_f32(_sum3, _k22, _r28);
                    _sum3 = vmlaq_f32(_sum3, _k23, _r29);
                    _sum3 = vmlaq_f32(_sum3, _k24, _r210);

                    float32x4_t _r30 = vld1q_f32(r3);
                    float32x4_t _r31 = vld1q_f32(r3 + 4);
                    float32x4_t _r32 = vld1q_f32(r3 + 8);
                    float32x4_t _r33 = vld1q_f32(r3 + 12);
                    float32x4_t _r34 = vld1q_f32(r3 + 16);
                    float32x4_t _r35 = vld1q_f32(r3 + 20);
                    float32x4_t _r36 = vld1q_f32(r3 + 24);
                    float32x4_t _r37 = vld1q_f32(r3 + 28);
                    float32x4_t _r38 = vld1q_f32(r3 + 32);
                    float32x4_t _r39 = vld1q_f32(r3 + 36);
                    float32x4_t _r310 = vld1q_f32(r3 + 40);

                    float32x4_t _k30 = vld1q_f32(k0);
                    float32x4_t _k31 = vld1q_f32(k0 + 4);
                    float32x4_t _k32 = vld1q_f32(k0 + 8);
                    float32x4_t _k33 = vld1q_f32(k0 + 12);
                    float32x4_t _k34 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k30, _r30);
                    _sum0 = vmlaq_f32(_sum0, _k31, _r31);
                    _sum0 = vmlaq_f32(_sum0, _k32, _r32);
                    _sum0 = vmlaq_f32(_sum0, _k33, _r33);
                    _sum0 = vmlaq_f32(_sum0, _k34, _r34);
                    _sum1 = vmlaq_f32(_sum1, _k30, _r32);
                    _sum1 = vmlaq_f32(_sum1, _k31, _r33);
                    _sum1 = vmlaq_f32(_sum1, _k32, _r34);
                    _sum1 = vmlaq_f32(_sum1, _k33, _r35);
                    _sum1 = vmlaq_f32(_sum1, _k34, _r36);
                    _sum2 = vmlaq_f32(_sum2, _k30, _r34);
                    _sum2 = vmlaq_f32(_sum2, _k31, _r35);
                    _sum2 = vmlaq_f32(_sum2, _k32, _r36);
                    _sum2 = vmlaq_f32(_sum2, _k33, _r37);
                    _sum2 = vmlaq_f32(_sum2, _k34, _r38);
                    _sum3 = vmlaq_f32(_sum3, _k30, _r36);
                    _sum3 = vmlaq_f32(_sum3, _k31, _r37);
                    _sum3 = vmlaq_f32(_sum3, _k32, _r38);
                    _sum3 = vmlaq_f32(_sum3, _k33, _r39);
                    _sum3 = vmlaq_f32(_sum3, _k34, _r310);

                    float32x4_t _r40 = vld1q_f32(r4);
                    float32x4_t _r41 = vld1q_f32(r4 + 4);
                    float32x4_t _r42 = vld1q_f32(r4 + 8);
                    float32x4_t _r43 = vld1q_f32(r4 + 12);
                    float32x4_t _r44 = vld1q_f32(r4 + 16);
                    float32x4_t _r45 = vld1q_f32(r4 + 20);
                    float32x4_t _r46 = vld1q_f32(r4 + 24);
                    float32x4_t _r47 = vld1q_f32(r4 + 28);
                    float32x4_t _r48 = vld1q_f32(r4 + 32);
                    float32x4_t _r49 = vld1q_f32(r4 + 36);
                    float32x4_t _r410 = vld1q_f32(r4 + 40);

                    float32x4_t _k40 = vld1q_f32(k0);
                    float32x4_t _k41 = vld1q_f32(k0 + 4);
                    float32x4_t _k42 = vld1q_f32(k0 + 8);
                    float32x4_t _k43 = vld1q_f32(k0 + 12);
                    float32x4_t _k44 = vld1q_f32(k0 + 16);
                    k0 -= 80;

                    _sum0 = vmlaq_f32(_sum0, _k40, _r40);
                    _sum0 = vmlaq_f32(_sum0, _k41, _r41);
                    _sum0 = vmlaq_f32(_sum0, _k42, _r42);
                    _sum0 = vmlaq_f32(_sum0, _k43, _r43);
                    _sum0 = vmlaq_f32(_sum0, _k44, _r44);
                    _sum1 = vmlaq_f32(_sum1, _k40, _r42);
                    _sum1 = vmlaq_f32(_sum1, _k41, _r43);
                    _sum1 = vmlaq_f32(_sum1, _k42, _r44);
                    _sum1 = vmlaq_f32(_sum1, _k43, _r45);
                    _sum1 = vmlaq_f32(_sum1, _k44, _r46);
                    _sum2 = vmlaq_f32(_sum2, _k40, _r44);
                    _sum2 = vmlaq_f32(_sum2, _k41, _r45);
                    _sum2 = vmlaq_f32(_sum2, _k42, _r46);
                    _sum2 = vmlaq_f32(_sum2, _k43, _r47);
                    _sum2 = vmlaq_f32(_sum2, _k44, _r48);
                    _sum3 = vmlaq_f32(_sum3, _k40, _r46);
                    _sum3 = vmlaq_f32(_sum3, _k41, _r47);
                    _sum3 = vmlaq_f32(_sum3, _k42, _r48);
                    _sum3 = vmlaq_f32(_sum3, _k43, _r49);
                    _sum3 = vmlaq_f32(_sum3, _k44, _r410);

                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);
                    vst1q_f32(outptr0 + 8, _sum2);
                    vst1q_f32(outptr0 + 12, _sum3);

                    r0 += 8 * 4;
                    r1 += 8 * 4;
                    r2 += 8 * 4;
                    r3 += 8 * 4;
                    r4 += 8 * 4;
                    outptr0 += 16;
                }
                for (; j + 1 < outw; j += 2)
                {
                    float32x4_t _sum0 = _bias0;
                    float32x4_t _sum1 = _bias0;

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r01 = vld1q_f32(r0 + 4);
                    float32x4_t _r02 = vld1q_f32(r0 + 8);
                    float32x4_t _r03 = vld1q_f32(r0 + 12);
                    float32x4_t _r04 = vld1q_f32(r0 + 16);
                    float32x4_t _r05 = vld1q_f32(r0 + 20);
                    float32x4_t _r06 = vld1q_f32(r0 + 24);

                    float32x4_t _k00 = vld1q_f32(k0);
                    float32x4_t _k01 = vld1q_f32(k0 + 4);
                    float32x4_t _k02 = vld1q_f32(k0 + 8);
                    float32x4_t _k03 = vld1q_f32(k0 + 12);
                    float32x4_t _k04 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                    _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                    _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                    _sum0 = vmlaq_f32(_sum0, _k03, _r03);
                    _sum0 = vmlaq_f32(_sum0, _k04, _r04);
                    _sum1 = vmlaq_f32(_sum1, _k00, _r02);
                    _sum1 = vmlaq_f32(_sum1, _k01, _r03);
                    _sum1 = vmlaq_f32(_sum1, _k02, _r04);
                    _sum1 = vmlaq_f32(_sum1, _k03, _r05);
                    _sum1 = vmlaq_f32(_sum1, _k04, _r06);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r11 = vld1q_f32(r1 + 4);
                    float32x4_t _r12 = vld1q_f32(r1 + 8);
                    float32x4_t _r13 = vld1q_f32(r1 + 12);
                    float32x4_t _r14 = vld1q_f32(r1 + 16);
                    float32x4_t _r15 = vld1q_f32(r1 + 20);
                    float32x4_t _r16 = vld1q_f32(r1 + 24);

                    float32x4_t _k10 = vld1q_f32(k0);
                    float32x4_t _k11 = vld1q_f32(k0 + 4);
                    float32x4_t _k12 = vld1q_f32(k0 + 8);
                    float32x4_t _k13 = vld1q_f32(k0 + 12);
                    float32x4_t _k14 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                    _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                    _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                    _sum0 = vmlaq_f32(_sum0, _k13, _r13);
                    _sum0 = vmlaq_f32(_sum0, _k14, _r14);
                    _sum1 = vmlaq_f32(_sum1, _k10, _r12);
                    _sum1 = vmlaq_f32(_sum1, _k11, _r13);
                    _sum1 = vmlaq_f32(_sum1, _k12, _r14);
                    _sum1 = vmlaq_f32(_sum1, _k13, _r15);
                    _sum1 = vmlaq_f32(_sum1, _k14, _r16);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r21 = vld1q_f32(r2 + 4);
                    float32x4_t _r22 = vld1q_f32(r2 + 8);
                    float32x4_t _r23 = vld1q_f32(r2 + 12);
                    float32x4_t _r24 = vld1q_f32(r2 + 16);
                    float32x4_t _r25 = vld1q_f32(r2 + 20);
                    float32x4_t _r26 = vld1q_f32(r2 + 24);

                    float32x4_t _k20 = vld1q_f32(k0);
                    float32x4_t _k21 = vld1q_f32(k0 + 4);
                    float32x4_t _k22 = vld1q_f32(k0 + 8);
                    float32x4_t _k23 = vld1q_f32(k0 + 12);
                    float32x4_t _k24 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                    _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                    _sum0 = vmlaq_f32(_sum0, _k22, _r22);
                    _sum0 = vmlaq_f32(_sum0, _k23, _r23);
                    _sum0 = vmlaq_f32(_sum0, _k24, _r24);
                    _sum1 = vmlaq_f32(_sum1, _k20, _r22);
                    _sum1 = vmlaq_f32(_sum1, _k21, _r23);
                    _sum1 = vmlaq_f32(_sum1, _k22, _r24);
                    _sum1 = vmlaq_f32(_sum1, _k23, _r25);
                    _sum1 = vmlaq_f32(_sum1, _k24, _r26);

                    float32x4_t _r30 = vld1q_f32(r3);
                    float32x4_t _r31 = vld1q_f32(r3 + 4);
                    float32x4_t _r32 = vld1q_f32(r3 + 8);
                    float32x4_t _r33 = vld1q_f32(r3 + 12);
                    float32x4_t _r34 = vld1q_f32(r3 + 16);
                    float32x4_t _r35 = vld1q_f32(r3 + 20);
                    float32x4_t _r36 = vld1q_f32(r3 + 24);

                    float32x4_t _k30 = vld1q_f32(k0);
                    float32x4_t _k31 = vld1q_f32(k0 + 4);
                    float32x4_t _k32 = vld1q_f32(k0 + 8);
                    float32x4_t _k33 = vld1q_f32(k0 + 12);
                    float32x4_t _k34 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k30, _r30);
                    _sum0 = vmlaq_f32(_sum0, _k31, _r31);
                    _sum0 = vmlaq_f32(_sum0, _k32, _r32);
                    _sum0 = vmlaq_f32(_sum0, _k33, _r33);
                    _sum0 = vmlaq_f32(_sum0, _k34, _r34);
                    _sum1 = vmlaq_f32(_sum1, _k30, _r32);
                    _sum1 = vmlaq_f32(_sum1, _k31, _r33);
                    _sum1 = vmlaq_f32(_sum1, _k32, _r34);
                    _sum1 = vmlaq_f32(_sum1, _k33, _r35);
                    _sum1 = vmlaq_f32(_sum1, _k34, _r36);

                    float32x4_t _r40 = vld1q_f32(r4);
                    float32x4_t _r41 = vld1q_f32(r4 + 4);
                    float32x4_t _r42 = vld1q_f32(r4 + 8);
                    float32x4_t _r43 = vld1q_f32(r4 + 12);
                    float32x4_t _r44 = vld1q_f32(r4 + 16);
                    float32x4_t _r45 = vld1q_f32(r4 + 20);
                    float32x4_t _r46 = vld1q_f32(r4 + 24);

                    float32x4_t _k40 = vld1q_f32(k0);
                    float32x4_t _k41 = vld1q_f32(k0 + 4);
                    float32x4_t _k42 = vld1q_f32(k0 + 8);
                    float32x4_t _k43 = vld1q_f32(k0 + 12);
                    float32x4_t _k44 = vld1q_f32(k0 + 16);
                    k0 -= 80;

                    _sum0 = vmlaq_f32(_sum0, _k40, _r40);
                    _sum0 = vmlaq_f32(_sum0, _k41, _r41);
                    _sum0 = vmlaq_f32(_sum0, _k42, _r42);
                    _sum0 = vmlaq_f32(_sum0, _k43, _r43);
                    _sum0 = vmlaq_f32(_sum0, _k44, _r44);
                    _sum1 = vmlaq_f32(_sum1, _k40, _r42);
                    _sum1 = vmlaq_f32(_sum1, _k41, _r43);
                    _sum1 = vmlaq_f32(_sum1, _k42, _r44);
                    _sum1 = vmlaq_f32(_sum1, _k43, _r45);
                    _sum1 = vmlaq_f32(_sum1, _k44, _r46);

                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);

                    r0 += 4 * 4;
                    r1 += 4 * 4;
                    r2 += 4 * 4;
                    r3 += 4 * 4;
                    r4 += 4 * 4;
                    outptr0 += 8;
                }
                for (; j < outw; j++)
                {
                    float32x4_t _sum0 = _bias0;

                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r01 = vld1q_f32(r0 + 4);
                    float32x4_t _r02 = vld1q_f32(r0 + 8);
                    float32x4_t _r03 = vld1q_f32(r0 + 12);
                    float32x4_t _r04 = vld1q_f32(r0 + 16);

                    float32x4_t _k00 = vld1q_f32(k0);
                    float32x4_t _k01 = vld1q_f32(k0 + 4);
                    float32x4_t _k02 = vld1q_f32(k0 + 8);
                    float32x4_t _k03 = vld1q_f32(k0 + 12);
                    float32x4_t _k04 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k00, _r00);
                    _sum0 = vmlaq_f32(_sum0, _k01, _r01);
                    _sum0 = vmlaq_f32(_sum0, _k02, _r02);
                    _sum0 = vmlaq_f32(_sum0, _k03, _r03);
                    _sum0 = vmlaq_f32(_sum0, _k04, _r04);

                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r11 = vld1q_f32(r1 + 4);
                    float32x4_t _r12 = vld1q_f32(r1 + 8);
                    float32x4_t _r13 = vld1q_f32(r1 + 12);
                    float32x4_t _r14 = vld1q_f32(r1 + 16);

                    float32x4_t _k10 = vld1q_f32(k0);
                    float32x4_t _k11 = vld1q_f32(k0 + 4);
                    float32x4_t _k12 = vld1q_f32(k0 + 8);
                    float32x4_t _k13 = vld1q_f32(k0 + 12);
                    float32x4_t _k14 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k10, _r10);
                    _sum0 = vmlaq_f32(_sum0, _k11, _r11);
                    _sum0 = vmlaq_f32(_sum0, _k12, _r12);
                    _sum0 = vmlaq_f32(_sum0, _k13, _r13);
                    _sum0 = vmlaq_f32(_sum0, _k14, _r14);

                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r21 = vld1q_f32(r2 + 4);
                    float32x4_t _r22 = vld1q_f32(r2 + 8);
                    float32x4_t _r23 = vld1q_f32(r2 + 12);
                    float32x4_t _r24 = vld1q_f32(r2 + 16);

                    float32x4_t _k20 = vld1q_f32(k0);
                    float32x4_t _k21 = vld1q_f32(k0 + 4);
                    float32x4_t _k22 = vld1q_f32(k0 + 8);
                    float32x4_t _k23 = vld1q_f32(k0 + 12);
                    float32x4_t _k24 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k20, _r20);
                    _sum0 = vmlaq_f32(_sum0, _k21, _r21);
                    _sum0 = vmlaq_f32(_sum0, _k22, _r22);
                    _sum0 = vmlaq_f32(_sum0, _k23, _r23);
                    _sum0 = vmlaq_f32(_sum0, _k24, _r24);

                    float32x4_t _r30 = vld1q_f32(r3);
                    float32x4_t _r31 = vld1q_f32(r3 + 4);
                    float32x4_t _r32 = vld1q_f32(r3 + 8);
                    float32x4_t _r33 = vld1q_f32(r3 + 12);
                    float32x4_t _r34 = vld1q_f32(r3 + 16);

                    float32x4_t _k30 = vld1q_f32(k0);
                    float32x4_t _k31 = vld1q_f32(k0 + 4);
                    float32x4_t _k32 = vld1q_f32(k0 + 8);
                    float32x4_t _k33 = vld1q_f32(k0 + 12);
                    float32x4_t _k34 = vld1q_f32(k0 + 16);
                    k0 += 20;

                    _sum0 = vmlaq_f32(_sum0, _k30, _r30);
                    _sum0 = vmlaq_f32(_sum0, _k31, _r31);
                    _sum0 = vmlaq_f32(_sum0, _k32, _r32);
                    _sum0 = vmlaq_f32(_sum0, _k33, _r33);
                    _sum0 = vmlaq_f32(_sum0, _k34, _r34);

                    float32x4_t _r40 = vld1q_f32(r4);
                    float32x4_t _r41 = vld1q_f32(r4 + 4);
                    float32x4_t _r42 = vld1q_f32(r4 + 8);
                    float32x4_t _r43 = vld1q_f32(r4 + 12);
                    float32x4_t _r44 = vld1q_f32(r4 + 16);

                    float32x4_t _k40 = vld1q_f32(k0);
                    float32x4_t _k41 = vld1q_f32(k0 + 4);
                    float32x4_t _k42 = vld1q_f32(k0 + 8);
                    float32x4_t _k43 = vld1q_f32(k0 + 12);
                    float32x4_t _k44 = vld1q_f32(k0 + 16);
                    k0 -= 80;

                    _sum0 = vmlaq_f32(_sum0, _k40, _r40);
                    _sum0 = vmlaq_f32(_sum0, _k41, _r41);
                    _sum0 = vmlaq_f32(_sum0, _k42, _r42);
                    _sum0 = vmlaq_f32(_sum0, _k43, _r43);
                    _sum0 = vmlaq_f32(_sum0, _k44, _r44);

                    vst1q_f32(outptr0, _sum0);

                    r0 += 2 * 4;
                    r1 += 2 * 4;
                    r2 += 2 * 4;
                    r3 += 2 * 4;
                    r4 += 2 * 4;
                    outptr0 += 4;
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

Tensor depthwise_conv2d_5x5s2_neon_pack4(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, otter::ScalarType::Float4);
    
    return depthwise_conv2d_5x5s2_neon_pack4_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
}

#endif // __ARM_NEON__

}   // end namespace otter
