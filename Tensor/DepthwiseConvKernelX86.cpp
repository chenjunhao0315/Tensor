//
//  DepthwiseConvKernelX86.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/2.
//

#include "DepthwiseConvKernelX86.hpp"
#include "Tensor.hpp"
#include "Utils.hpp"
#include "Parallel.hpp"
#include "TensorFactory.hpp"
#include "Padding.hpp"

namespace otter {

Tensor& depthwise_conv2d_3x3s1_x86_sse_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias_,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0);
    auto output_shape = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_(output_shape);
    
    int w = int(input.size(3));

    int outw = int(output.size(3));
    int outh = int(output.size(2));

    const int group = int(self.size(1));

//    const int tailstep = w - 2 * outw + w;

    const float* kernel = weight.data_ptr<float>();
    const float* bias = (bias_.defined()) ? bias_.data_ptr<float>() : nullptr;
    
    auto input_a = input.accessor<float, 4>()[0];
    auto output_a = output.accessor<float, 4>()[0];

    otter::parallel_for(0, group, 0, [&](int64_t begin, int64_t end) {
        for (const auto g : otter::irange(begin, end))
        {
            auto out = output_a[g];

            const float bias0 = bias ? bias[g] : 0.f;

            const float* kernel0 = kernel + g * 9;

            float* outptr = out.data();
            float* outptr2 = outptr + outw;

            const float* img0 = input_a[g].data();

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;
            const float* r3 = img0 + w * 3;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            int i = 0;

            for (; i + 1 < outh; i += 2)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    float sum = bias0;
                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    float sum2 = bias0;
                    sum2 += r1[0] * k0[0];
                    sum2 += r1[1] * k0[1];
                    sum2 += r1[2] * k0[2];
                    sum2 += r2[0] * k1[0];
                    sum2 += r2[1] * k1[1];
                    sum2 += r2[2] * k1[2];
                    sum2 += r3[0] * k2[0];
                    sum2 += r3[1] * k2[1];
                    sum2 += r3[2] * k2[2];

                    *outptr = sum;
                    *outptr2 = sum2;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    outptr++;
                    outptr2++;
                }

                r0 += 2 + w;
                r1 += 2 + w;
                r2 += 2 + w;
                r3 += 2 + w;

                outptr += outw;
                outptr2 += outw;
            }

            for (; i < outh; i++)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    float sum = bias0;
                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    *outptr = sum;

                    r0++;
                    r1++;
                    r2++;
                    outptr++;
                }

                r0 += 2;
                r1 += 2;
                r2 += 2;
            }
        }
    });
        
    return output;
}

Tensor depthwise_conv2d_3x3s1_x86_sse(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, self.options());

    return depthwise_conv2d_3x3s1_x86_sse_out(self, weight, bias, stride, padding, output);
}

Tensor& depthwise_conv2d_3x3s2_x86_sse_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias_,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0);
    auto output_shape = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_(output_shape);
    
    int w = int(input.size(3));

    int outw = int(output.size(3));
    int outh = int(output.size(2));

    const int group = int(self.size(1));

    const int tailstep = w - 2 * outw + w;

    const float* kernel = weight.data_ptr<float>();
    const float* bias = (bias_.defined()) ? bias_.data_ptr<float>() : nullptr;
    
    auto input_a = input.accessor<float, 4>()[0];
    auto output_a = output.accessor<float, 4>()[0];

    otter::parallel_for(0, group, 0, [&](int64_t begin, int64_t end) {
        for (const auto g : otter::irange(begin, end))
        {
            auto out = output_a[g];

            const float bias0 = bias ? bias[g] : 0.f;

            const float* kernel0 = kernel + g * 9;

            float* outptr = out.data();

            const float* img0 = input_a[g].data();

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;

            int i = 0;

            for (; i < outh; i++)
            {
                int remain = outw;

                for (; remain > 0; remain--)
                {
                    float sum = bias0;
                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    *outptr = sum;

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }
        }
    });
        
    return output;
}

Tensor depthwise_conv2d_3x3s2_x86_sse(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, self.options());

    return depthwise_conv2d_3x3s2_x86_sse_out(self, weight, bias, stride, padding, output);
}

}   // end namespace otter
