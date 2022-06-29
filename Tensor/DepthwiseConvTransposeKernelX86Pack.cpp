//
//  DepthwiseConvTransposeKernelX86Pack.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/29.
//

#include "DepthwiseConvTransposeKernelX86Pack.hpp"

#include "Tensor.hpp"
#include "TensorFactory.hpp"
#include "Parallel.hpp"
#include "TensorTransform.hpp"
#include "VecIntrinsic.hpp"

namespace otter {

#if __SSE2__

void depthwise_deconv2d_kernel_transform_pack_x86(const Tensor& weight, Tensor& kernel_tf) {
    kernel_tf = otter::empty_like(weight);
    
    int64_t channels = weight.size(0);
    int64_t group = channels;
    int64_t num_output = weight.size(1) * group;
    
    int64_t maxk = weight.size(2) * weight.size(3);
    
    {
        float* pt = kernel_tf.data_ptr<float>();
        const float* p = weight.data_ptr<float>();

        for (int i = 0; i < (channels / group) * (num_output / group) * group; i++) {
            for (int k = 0; k < maxk; k++) {
                pt[maxk - 1 - k] = p[k];
            }

            p += maxk;
            pt += maxk;
        }
    }
}

Tensor& depthwise_deconv2d_pack4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    const int64_t w = self.size(3);
    const int64_t h = self.size(2);
    const int64_t inch = self.size(1);
    
    const int64_t kernel_w = weight.size(3);
    const int64_t kernel_h = weight.size(2);
    
    const int64_t stride_w = stride[1];
    const int64_t stride_h = stride[0];
    
    const int64_t dilation_w = dilation[1];
    const int64_t dilation_h = dilation[0];
    
    const int64_t kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int64_t kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    
    const int64_t outw = (w - 1) * stride_w + kernel_extent_w + output_padding[1];
    const int64_t outh = (h - 1) * stride_h + kernel_extent_h + output_padding[0];
    const int64_t outch = weight.size(0);
    
    auto output_pad = otter::empty({1, outch / 4, outh, outw}, otter::ScalarType::Float4);
    
    const int64_t group = inch * 4;
    const int64_t maxk = kernel_w * kernel_h;
    
    Tensor kernel_tf;
    if (weight_o.defined()) {
        kernel_tf = weight_o;
    } else {
        Tensor kernel_tf_unpack;
        depthwise_deconv2d_kernel_transform_pack_x86(weight, kernel_tf_unpack);
        kernel_tf = kernel_tf_unpack.view({group, maxk}).packing(4);
    }
    
    auto input_a = self.accessor<float, 4, 4>()[0];
    auto output_pad_a = output_pad.accessor<float, 4, 4>()[0];
    const float* weight_data_tm = (const float*)kernel_tf.data_ptr();
    bool bias_term = bias.defined();
    const float* bias_data = (const float*)bias.data_ptr();
    
    otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
        for (const auto g : otter::irange(begin, end)) {
            float* outptr = output_pad_a[g].data();
            const float* kptr = (const float*)weight_data_tm + maxk * g * 4;
            const auto m = input_a[g];

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    __m128 _sum = _mm_setzero_ps();

                    if (bias_term)
                    {
                        _sum = _mm_loadu_ps((const float*)bias_data + g * 4);
                    }

                    for (int y = 0; y < kernel_h; y++)
                    {
                        int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                        if (sys < 0 || sys % stride_h != 0)
                            continue;

                        int sy = sys / stride_h;
                        if (sy >= h)
                            continue;

                        for (int x = 0; x < kernel_w; x++)
                        {
                            int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                            if (sxs < 0 || sxs % stride_w != 0)
                                continue;

                            int sx = sxs / stride_w;
                            if (sx >= w)
                                continue;

                            const float* sptr = m[sy].data() + sx * 4;

                            int k = y * kernel_w + x;

                            __m128 _val = _mm_loadu_ps(sptr);
                            __m128 _w = _mm_loadu_ps(kptr + k * 4);
                            _sum = _mm_comp_fmadd_ps(_val, _w, _sum);
                        }
                    }
                    
                    _mm_storeu_ps(outptr, _sum);
                    outptr += 4;
                }
            }
        }
    });
    
    if (padding[0] > 0 || padding[1] > 0) {
        output = otter::crop(output_pad, {padding[1], padding[1], padding[0], padding[0]});
    } else {
        output = output_pad;
    }
    
    return output;
}

Tensor depthwise_deconv2d_pack4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
    
    Tensor output;
    
    return depthwise_deconv2d_pack4_x86_out(self, weight, weight_o, bias, stride, padding, output_padding, dilation, output);
}

Tensor& depthwise_deconv2d_pack1_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    const int64_t w = self.size(3);
    const int64_t h = self.size(2);
    const int64_t inch = self.size(1);
    
    const int64_t kernel_w = weight.size(3);
    const int64_t kernel_h = weight.size(2);
    
    const int64_t stride_w = stride[1];
    const int64_t stride_h = stride[0];
    
    const int64_t dilation_w = dilation[1];
    const int64_t dilation_h = dilation[0];
    
    const int64_t kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int64_t kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    
    const int64_t outw = (w - 1) * stride_w + kernel_extent_w + output_padding[1];
    const int64_t outh = (h - 1) * stride_h + kernel_extent_h + output_padding[0];
    const int64_t outch = weight.size(0);
    
    auto output_pad = otter::empty({1, outch, outh, outw}, otter::ScalarType::Float);
    
    const int64_t maxk = kernel_w * kernel_h;
    
    Tensor kernel_tf;
    if (weight_o.defined()) {
        kernel_tf = weight_o;
    } else {
        depthwise_deconv2d_kernel_transform_pack_x86(weight, kernel_tf);
    }
    
    auto input_a = self.accessor<float, 4>()[0];
    auto output_pad_a = output_pad.accessor<float, 4>()[0];
    const float* weight_data_tm = (const float*)kernel_tf.data_ptr();
    bool bias_term = bias.defined();
    const float* bias_data = (const float*)bias.data_ptr();
    
    otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
        for (const auto g : otter::irange(begin, end)) {
            float* outptr = output_pad_a[g].data();
            const float* kptr = (const float*)weight_data_tm + maxk * g;
            const auto m = input_a[g];

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                    {
                        sum = bias_data[g];
                    }

                    for (int y = 0; y < kernel_h; y++)
                    {
                        int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                        if (sys < 0 || sys % stride_h != 0)
                            continue;

                        int sy = sys / stride_h;
                        if (sy >= h)
                            continue;

                        const float* sptr = m[sy].data();

                        for (int x = 0; x < kernel_w; x++)
                        {
                            int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                            if (sxs < 0 || sxs % stride_w != 0)
                                continue;

                            int sx = sxs / stride_w;
                            if (sx >= w)
                                continue;

                            float val = sptr[sx];

                            int k = y * kernel_w + x;

                            float w = kptr[k];

                            sum += val * w;
                        }
                    }

                    outptr[0] = sum;
                    outptr++;
                }
            }
        }
    });
    
    if (padding[0] > 0 || padding[1] > 0) {
        output = otter::crop(output_pad, {padding[1], padding[1], padding[0], padding[0]});
    } else {
        output = output_pad;
    }
    
    return output;
}

Tensor depthwise_deconv2d_pack1_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
    
    Tensor output;
    
    return depthwise_deconv2d_pack1_x86_out(self, weight, weight_o, bias, stride, padding, output_padding, dilation, output);
}

#endif

}   // end namespace otter
