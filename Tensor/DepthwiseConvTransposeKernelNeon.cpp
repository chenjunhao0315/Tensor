//
//  DepthwiseConvTransposeKernelNeon.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/5/13.
//

#include "DepthwiseConvTransposeKernelNeon.hpp"
#include "TensorFactory.hpp"
#include "TensorTransform.hpp"
#include "Parallel.hpp"

namespace otter {

void depthwise_deconv2d_kernel_transform(const Tensor& weight, Tensor& kernel_tf) {
    kernel_tf = otter::empty_like(weight);
    
    int64_t channels = weight.size(0);
    int64_t num_output = weight.size(1);
    int64_t group = channels;
    
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

Tensor& depthwise_deconv2d_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    int64_t w = self.size(3);
    int64_t h = self.size(2);
    int64_t channels = self.size(1);
    
    int64_t kernel_w = weight.size(3);
    int64_t kernel_h = weight.size(2);
    
    int64_t dilation_w = dilation[1];
    int64_t dilation_h = dilation[0];
    
    int64_t stride_w = stride[1];
    int64_t stride_h = stride[0];

    const int64_t kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int64_t kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
    
    int64_t outw = (w - 1) * stride_w + kernel_extent_w + output_padding[1];
    int64_t outh = (h - 1) * stride_h + kernel_extent_h + output_padding[0];
    int64_t outch = weight.size(1);
    
    auto output_pad = otter::empty({1, outch, outh, outw}, self.options());
    
    auto input_a = self.accessor<float, 4>()[0];
    
    const int64_t maxk = kernel_w * kernel_h;
    
    const int bias_term = (bias.defined()) ? 1 : 0;
    
    float* bias_data = (bias_term) ? bias.data_ptr<float>() : nullptr;
    
    Tensor kernel_tf;
    otter::depthwise_deconv2d_kernel_transform(weight, kernel_tf);
    
    
    otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
        for (const auto g : otter::irange(begin, end)) {
            float* outptr = output_pad[0][g].data_ptr<float>();
            const float* kptr = (const float*)kernel_tf.data_ptr<float>() + maxk * g;
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

                    outptr[j] = sum;
                }

                outptr += outw;
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

Tensor depthwise_deconv2d_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, self.options());
    
    return depthwise_deconv2d_neon_out(self, weight, bias, stride, padding, output_padding, dilation, output);
}

}   // end namespace otter
