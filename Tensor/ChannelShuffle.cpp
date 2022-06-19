//
//  ChannelShuffle.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/31.
//

#include "Tensor.hpp"
#include "ChannelShuffle.hpp"
#include "TensorFactory.hpp"

namespace otter {

DEFINE_DISPATCH(channel_shuffle_stub);

Tensor& channel_shuffle_pack4_neon_out(Tensor& output, const Tensor& input, int64_t groups);

Tensor& channel_shuffle_out(Tensor& output, const Tensor& input, int64_t groups) {
    
    channel_shuffle_stub(Device::CPU, output, input, groups);
    
    return output;
}

Tensor channel_shuffle(const Tensor& input, int64_t groups) {
    Tensor output;
#if __ARM_NEON__
    if (input.elempack() == 4) {
        if (input.scalar_type() == ScalarType::Float4) {
            return channel_shuffle_pack4_neon_out(output, input, groups);
        }
        return channel_shuffle(input.packing(1), groups);
    }
#endif
    output = otter::empty_like(input, input.options());
    
    return channel_shuffle_out(output, input, groups);
}

Tensor& channel_shuffle_pack4_neon_out(Tensor& output, const Tensor& input, int64_t groups) {
    if (groups == 1) {
        output = input;
        
        return output;
    }
    
    int w = input.size(3);
    int h = input.size(2);
    int size = w * h;
    int channels = input.size(1);
    int channels_per_group = channels / groups;
    
    auto input_a = input.accessor<float, 4, 4>()[0];
    
    if (groups == 2 && channels % groups != 0) {
        output = otter::empty({1, channels, h, w}, otter::ScalarType::Float4);
        auto output_a = output.accessor<float, 4, 4>()[0];
        
        for (int q = 0; q < channels_per_group; q++) {
            const float* ptr0 = input_a[q].data();
            const float* ptr1 = input_a[channels_per_group + q].data();
            const float* ptr2 = input_a[channels_per_group + q + 1].data();
            float* outptr0 = output_a[q * 2].data();
            float* outptr1 = output_a[q * 2 + 1].data();

            for (int i = 0; i < size; i++) {
                float32x4_t _p0 = vld1q_f32(ptr0);
                float32x4_t _p1 = vld1q_f32(ptr1);
                float32x4_t _p2 = vld1q_f32(ptr2);

                float32x4_t _p12 = vextq_f32(_p1, _p2, 2);

                float32x4x2_t _p01 = vzipq_f32(_p0, _p12);

                vst1q_f32(outptr0, _p01.val[0]);
                vst1q_f32(outptr1, _p01.val[1]);

                ptr0 += 4;
                ptr1 += 4;
                ptr2 += 4;
                outptr0 += 4;
                outptr1 += 4;
            }
        }
        
        {
            const float* ptr0 = input_a[channels_per_group].data();
            const float* ptr1 = input_a[channels_per_group + channels_per_group].data();
            float* outptr0 = output_a[channels_per_group * 2].data();

            ptr1 += 2;

            for (int i = 0; i < size; i++) {
                float32x4_t _p0 = vld1q_f32(ptr0);
                float32x4_t _p1 = vld1q_f32(ptr1);

                float32x4x2_t _p01 = vzipq_f32(_p0, _p1);

                vst1q_f32(outptr0, _p01.val[0]);

                ptr0 += 4;
                ptr1 += 4;
                outptr0 += 4;
            }
        }
        
        return output;
    }
    
    if (groups > 4 || channels % groups != 0) {
        Tensor output_unpacked = channel_shuffle(input.packing(1), groups);
        output = output_unpacked.packing(4);
        
        return output;
    }
    
    output = otter::empty({1, channels, h, w}, otter::ScalarType::Float4);
    auto output_a = output.accessor<float, 4, 4>()[0];
    
    if (groups == 2) {
        for (int q = 0; q < channels_per_group; q++) {
            const float* ptr0 = input_a[q].data();
            const float* ptr1 = input_a[channels_per_group + q].data();
            float* outptr0 = output_a[q * 2].data();
            float* outptr1 = output_a[q * 2 + 1].data();

            for (int i = 0; i < size; i++) {
                float32x4_t _p0 = vld1q_f32(ptr0);
                float32x4_t _p1 = vld1q_f32(ptr1);

                float32x4x2_t _p01 = vzipq_f32(_p0, _p1);

                vst1q_f32(outptr0, _p01.val[0]);
                vst1q_f32(outptr1, _p01.val[1]);

                ptr0 += 4;
                ptr1 += 4;
                outptr0 += 4;
                outptr1 += 4;
            }
        }
    }

    if (groups == 3) {
        for (int q = 0; q < channels_per_group; q++) {
            const float* ptr0 = input_a[q].data();
            const float* ptr1 = input_a[channels_per_group + q].data();
            const float* ptr2 = input_a[channels_per_group * 2 + q].data();
            float* outptr0 = output_a[q * 3].data();
            float* outptr1 = output_a[q * 3 + 1].data();
            float* outptr2 = output_a[q * 3 + 2].data();

            for (int i = 0; i < size; i++) {
                float32x4_t _p0 = vld1q_f32(ptr0);
                float32x4_t _p1 = vld1q_f32(ptr1);
                float32x4_t _p2 = vld1q_f32(ptr2);

                float32x4x2_t _p01 = vzipq_f32(_p0, _p1);
                float32x4x2_t _p12 = vzipq_f32(_p1, _p2);

                float32x4_t _0415 = _p01.val[0];
                float32x4_t _2637 = _p01.val[1];
                float32x4_t _4859 = _p12.val[0];
                float32x4_t _6x7y = _p12.val[1];

                float32x2_t _15 = vget_high_f32(_0415);
                float32x2_t _37 = vget_high_f32(_2637);
                float32x2_t _48 = vget_low_f32(_4859);
                float32x2_t _6x = vget_low_f32(_6x7y);

                float32x2_t _81 = vext_f32(_48, _15, 1);
                float32x2_t _x3 = vext_f32(_6x, _37, 1);

                float32x4_t _0481 = vcombine_f32(vget_low_f32(_0415), _81);
                float32x4_t _5926 = vextq_f32(_4859, _2637, 2);
                float32x4_t _x37y = vcombine_f32(_x3, vget_high_f32(_6x7y));

                vst1q_f32(outptr0, _0481);
                vst1q_f32(outptr1, _5926);
                vst1q_f32(outptr2, _x37y);

                ptr0 += 4;
                ptr1 += 4;
                ptr2 += 4;
                outptr0 += 4;
                outptr1 += 4;
                outptr2 += 4;
            }
        }
    }

    if (groups == 4) {
        for (int q = 0; q < channels_per_group; q++) {
            const float* ptr0 = input_a[q].data();
            const float* ptr1 = input_a[channels_per_group + q].data();
            const float* ptr2 = input_a[channels_per_group * 2 + q].data();
            const float* ptr3 = input_a[channels_per_group * 3 + q].data();
            float* outptr0 = output_a[q * 4].data();
            float* outptr1 = output_a[q * 4 + 1].data();
            float* outptr2 = output_a[q * 4 + 2].data();
            float* outptr3 = output_a[q * 4 + 3].data();

            for (int i = 0; i < size; i++) {
                float32x4_t _p0 = vld1q_f32(ptr0);
                float32x4_t _p1 = vld1q_f32(ptr1);
                float32x4_t _p2 = vld1q_f32(ptr2);
                float32x4_t _p3 = vld1q_f32(ptr3);

                // transpose 4x4
                float32x4x2_t _p01 = vtrnq_f32(_p0, _p1);
                float32x4x2_t _p23 = vtrnq_f32(_p2, _p3);
                _p0 = vcombine_f32(vget_low_f32(_p01.val[0]), vget_low_f32(_p23.val[0]));
                _p1 = vcombine_f32(vget_low_f32(_p01.val[1]), vget_low_f32(_p23.val[1]));
                _p2 = vcombine_f32(vget_high_f32(_p01.val[0]), vget_high_f32(_p23.val[0]));
                _p3 = vcombine_f32(vget_high_f32(_p01.val[1]), vget_high_f32(_p23.val[1]));

                vst1q_f32(outptr0, _p0);
                vst1q_f32(outptr1, _p1);
                vst1q_f32(outptr2, _p2);
                vst1q_f32(outptr3, _p3);

                ptr0 += 4;
                ptr1 += 4;
                ptr2 += 4;
                ptr3 += 4;
                outptr0 += 4;
                outptr1 += 4;
                outptr2 += 4;
                outptr3 += 4;
            }
        }
    }
    
    return output;
}

}   // end namespace otter
