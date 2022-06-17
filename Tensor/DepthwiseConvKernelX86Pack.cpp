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

    otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
        for (const auto g : otter::irange(begin, end)) {
            float* outptr = (float*)output[0][g].raw_data();
            const float* kptr = (const float*)weight_data_packed.raw_data() + maxk * g * 4;
            const Tensor m = input[g];

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    __m128 _sum = _mm_set1_ps(0.f);

                    if (bias_term)
                    {
                        _sum = _mm_loadu_ps(((const float*)bias_data) + g * 4);
                    }

                    const float* sptr = (const float*)m[i * stride_h].raw_data() + j * stride_w * 4;

                    for (int k = 0; k < maxk; k++)
                    {
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

#endif // __SSE2__

}   // end namespace otter
