//
//  ConvolutionMM2DTransposeNeon.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/5/13.
//

#include "ConvolutionMM2DTransposeNeon.hpp"
#include "TensorFactory.hpp"
#include "ConvolutionUtils.hpp"
#include "TensorTransform.hpp"
#include "Parallel.hpp"

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

namespace otter {

void deconv2d_kernel_transpose(const Tensor& weight, Tensor& kernel_tf) {
    kernel_tf = otter::empty_like(weight);
    
    const int64_t maxk = weight.size(2) * weight.size(3);
    int64_t num_output = weight.size(1);
    int64_t num_input = weight.numel() / maxk / num_output;
    
    {
        float* pt = kernel_tf.data_ptr<float>();
        const float* p = weight.data_ptr<float>();

        for (int i = 0; i < num_input * num_output; i++) {
            for (int k = 0; k < maxk; k++) {
                pt[maxk - 1 - k] = p[k];
            }

            p += maxk;
            pt += maxk;
        }
    }
}

Tensor& deconv2d_4x4s2_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias_,
    IntArrayRef padding,
    IntArrayRef output_padding,
    Tensor& output) {
    
    auto output_pad_size = otter::calculate_deconv_output_size_without_padding(self.sizes(), weight.sizes(), {2, 2}, {1, 1}, padding, output_padding);
    auto output_pad = otter::empty(output_pad_size, otter::ScalarType::Float);
    
    int64_t w = self.size(3);
    int64_t h = self.size(2);
    int64_t inch = self.size(1);
    
    int64_t outw = output_pad.size(3);
    int64_t outch = output_pad.size(1);
    
    const float* kernel = weight.data_ptr<float>();
    const float* bias = (bias_.defined()) ? bias_.data_ptr<float>() : nullptr;
    
    auto input_a = self.accessor<float, 4>()[0];
    auto output_t = output_pad[0];
    
    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            auto out = output_t[p];

            const float bias0 = bias ? bias[p] : 0.f;

            out.fill_(bias0);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = input_a[q].data();

                const float* kernel0 = kernel + p * inch * 16 + q * 16;

                const float* r0 = img0;

                const float* k0 = kernel0;
                const float* k1 = kernel0 + 4;
                const float* k2 = kernel0 + 8;
                const float* k3 = kernel0 + 12;

    #if __ARM_NEON
                float32x4_t _k0 = vld1q_f32(k0);
                float32x4_t _k1 = vld1q_f32(k1);
                float32x4_t _k2 = vld1q_f32(k2);
                float32x4_t _k3 = vld1q_f32(k3);
    #endif // __ARM_NEON

                for (int i = 0; i < h; i++)
                {
                    float* outptr = out[i * 2].data_ptr<float>();

                    float* outptr0 = outptr;
                    float* outptr1 = outptr0 + outw;
                    float* outptr2 = outptr1 + outw;
                    float* outptr3 = outptr2 + outw;

                    int j = 0;
    #if __ARM_NEON
                    for (; j + 3 < w; j += 4)
                    {
                        float32x4_t _v = vld1q_f32(r0);

                        // row 0
                        float32x4x2_t _out0 = vld2q_f32(outptr0);
                        // 0,2,4,6
                        _out0.val[0] = vmlaq_lane_f32(_out0.val[0], _v, vget_low_f32(_k0), 0);
                        // 1,3,5,7
                        _out0.val[1] = vmlaq_lane_f32(_out0.val[1], _v, vget_low_f32(_k0), 1);
                        vst2q_f32(outptr0, _out0);

                        _out0 = vld2q_f32(outptr0 + 2);
                        // 2,4,6,8
                        _out0.val[0] = vmlaq_lane_f32(_out0.val[0], _v, vget_high_f32(_k0), 0);
                        // 3,5,7,9
                        _out0.val[1] = vmlaq_lane_f32(_out0.val[1], _v, vget_high_f32(_k0), 1);
                        vst2q_f32(outptr0 + 2, _out0);

                        // row 1
                        float32x4x2_t _out1 = vld2q_f32(outptr1);
                        // 0,2,4,6
                        _out1.val[0] = vmlaq_lane_f32(_out1.val[0], _v, vget_low_f32(_k1), 0);
                        // 1,3,5,7
                        _out1.val[1] = vmlaq_lane_f32(_out1.val[1], _v, vget_low_f32(_k1), 1);
                        vst2q_f32(outptr1, _out1);

                        _out1 = vld2q_f32(outptr1 + 2);
                        // 2,4,6,8
                        _out1.val[0] = vmlaq_lane_f32(_out1.val[0], _v, vget_high_f32(_k1), 0);
                        // 3,5,7,9
                        _out1.val[1] = vmlaq_lane_f32(_out1.val[1], _v, vget_high_f32(_k1), 1);
                        vst2q_f32(outptr1 + 2, _out1);

                        // row 2
                        float32x4x2_t _out2 = vld2q_f32(outptr2);
                        _out2.val[0] = vmlaq_lane_f32(_out2.val[0], _v, vget_low_f32(_k2), 0);
                        _out2.val[1] = vmlaq_lane_f32(_out2.val[1], _v, vget_low_f32(_k2), 1);
                        vst2q_f32(outptr2, _out2);

                        _out2 = vld2q_f32(outptr2 + 2);
                        _out2.val[0] = vmlaq_lane_f32(_out2.val[0], _v, vget_high_f32(_k2), 0);
                        _out2.val[1] = vmlaq_lane_f32(_out2.val[1], _v, vget_high_f32(_k2), 1);
                        vst2q_f32(outptr2 + 2, _out2);

                        // row 3
                        float32x4x2_t _out3 = vld2q_f32(outptr3);
                        _out3.val[0] = vmlaq_lane_f32(_out3.val[0], _v, vget_low_f32(_k3), 0);
                        _out3.val[1] = vmlaq_lane_f32(_out3.val[1], _v, vget_low_f32(_k3), 1);
                        vst2q_f32(outptr3, _out3);

                        _out3 = vld2q_f32(outptr3 + 2);
                        _out3.val[0] = vmlaq_lane_f32(_out3.val[0], _v, vget_high_f32(_k3), 0);
                        _out3.val[1] = vmlaq_lane_f32(_out3.val[1], _v, vget_high_f32(_k3), 1);
                        vst2q_f32(outptr3 + 2, _out3);

                        r0 += 4;
                        outptr0 += 8;
                        outptr1 += 8;
                        outptr2 += 8;
                        outptr3 += 8;
                    }

    #endif // __ARM_NEON

                    for (; j < w; j++)
                    {
                        float val = r0[0];

                        outptr0[0] += val * k0[0];
                        outptr0[1] += val * k0[1];
                        outptr0[2] += val * k0[2];
                        outptr0[3] += val * k0[3];

                        outptr1[0] += val * k1[0];
                        outptr1[1] += val * k1[1];
                        outptr1[2] += val * k1[2];
                        outptr1[3] += val * k1[3];

                        outptr2[0] += val * k2[0];
                        outptr2[1] += val * k2[1];
                        outptr2[2] += val * k2[2];
                        outptr2[3] += val * k2[3];

                        outptr3[0] += val * k3[0];
                        outptr3[1] += val * k3[1];
                        outptr3[2] += val * k3[2];
                        outptr3[3] += val * k3[3];

                        r0++;
                        outptr0 += 2;
                        outptr1 += 2;
                        outptr2 += 2;
                        outptr3 += 2;
                    }
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

Tensor deconv2d_4x4s2_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding,
    IntArrayRef output_padding) {
    
    auto output = otter::empty({}, self.options());
    
    return deconv2d_4x4s2_neon_out(self, weight, bias, padding, output_padding, output);
}

}   // end namespace otter
