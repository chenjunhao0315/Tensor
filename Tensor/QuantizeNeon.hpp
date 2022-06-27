//
//  QuantizeNeon.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/27.
//

#ifndef QuantizeNeon_hpp
#define QuantizeNeon_hpp

#include "Tensor.hpp"

#if __ARM_NEON__
#include <arm_neon.h>
#include "neon_mathfun.hpp"

static inline signed char float2int8(float v)
{
    int int32 = round(v);
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

static inline int8x8_t float2int8(float32x4_t _vlow, float32x4_t _vhigh)
{
#if __aarch64__
    int32x4_t _vlow32 = vcvtaq_s32_f32(_vlow);
    int32x4_t _vhigh32 = vcvtaq_s32_f32(_vhigh);
#else
    // vcvtq_s32_f32 is round to zero
    // simulate round to nearest via +/-0.5
    float32x4_t _p5 = vdupq_n_f32(0.5f);
    int32x4_t _signmask = vdupq_n_s32(1 << 31);
    int32x4_t _signlow = vandq_s32(vreinterpretq_s32_f32(_vlow), _signmask);
    int32x4_t _signhigh = vandq_s32(vreinterpretq_s32_f32(_vhigh), _signmask);
    float32x4_t _p5low = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(_p5), _signlow));
    float32x4_t _p5high = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(_p5), _signhigh));
    float32x4_t _vlow5 = vaddq_f32(_vlow, _p5low);
    float32x4_t _vhigh5 = vaddq_f32(_vhigh, _p5high);
    int32x4_t _vlow32 = vcvtq_s32_f32(_vlow5);
    int32x4_t _vhigh32 = vcvtq_s32_f32(_vhigh5);
#endif
    int16x8_t _v16 = vcombine_s16(vqmovn_s32(_vlow32), vqmovn_s32(_vhigh32));
    int8x8_t _v8 = vqmovn_s16(_v16);
    return vmax_s8(_v8, vdup_n_s8(-127));
}

static inline int8x8_t float2int8relu(float32x4_t _vlow, float32x4_t _vhigh)
{
#if __aarch64__
    int32x4_t _vlow32 = vcvtaq_s32_f32(_vlow);
    int32x4_t _vhigh32 = vcvtaq_s32_f32(_vhigh);
#else
    // vcvtq_s32_f32 is round to zero
    // simulate round to nearest via +/-0.5
    float32x4_t _p5 = vdupq_n_f32(0.5f);
    int32x4_t _signmask = vdupq_n_s32(1 << 31);
    int32x4_t _signlow = vandq_s32(vreinterpretq_s32_f32(_vlow), _signmask);
    int32x4_t _signhigh = vandq_s32(vreinterpretq_s32_f32(_vhigh), _signmask);
    float32x4_t _p5low = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(_p5), _signlow));
    float32x4_t _p5high = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(_p5), _signhigh));
    float32x4_t _vlow5 = vaddq_f32(_vlow, _p5low);
    float32x4_t _vhigh5 = vaddq_f32(_vhigh, _p5high);
    int32x4_t _vlow32 = vcvtq_s32_f32(_vlow5);
    int32x4_t _vhigh32 = vcvtq_s32_f32(_vhigh5);
#endif
    int16x8_t _v16 = vcombine_s16(vqmovn_s32(_vlow32), vqmovn_s32(_vhigh32));
    int8x8_t _v8 = vqmovn_s16(_v16);
    return vmax_s8(_v8, vdup_n_s8(0));
}

static inline int8x8_t float2int8leakyrelu(float32x4_t _vlow, float32x4_t _vhigh, float32x4_t _slope)
{
    float32x4_t _vlow_leaky = vmulq_f32(_vlow, _slope);
    float32x4_t _vhigh_leaky = vmulq_f32(_vhigh, _slope);
#if __aarch64__
    int32x4_t _vlow32 = vcvtaq_s32_f32(_vlow);
    int32x4_t _vhigh32 = vcvtaq_s32_f32(_vhigh);
    int32x4_t _vlow32_leaky = vcvtaq_s32_f32(_vlow_leaky);
    int32x4_t _vhigh32_leaky = vcvtaq_s32_f32(_vhigh_leaky);
#else
    // vcvtq_s32_f32 is round to zero
    // simulate round to nearest via +/-0.5
    float32x4_t _p5 = vdupq_n_f32(0.5f);
    int32x4_t _signmask = vdupq_n_s32(1 << 31);
    int32x4_t _signlow = vandq_s32(vreinterpretq_s32_f32(_vlow), _signmask);
    int32x4_t _signhigh = vandq_s32(vreinterpretq_s32_f32(_vhigh), _signmask);
    float32x4_t _p5low = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(_p5), _signlow));
    float32x4_t _p5high = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(_p5), _signhigh));
    float32x4_t _vlow5 = vaddq_f32(_vlow, _p5low);
    float32x4_t _vhigh5 = vaddq_f32(_vhigh, _p5high);
    int32x4_t _vlow32 = vcvtq_s32_f32(_vlow5);
    int32x4_t _vhigh32 = vcvtq_s32_f32(_vhigh5);

    int32x4_t _signlow_leaky = vandq_s32(vreinterpretq_s32_f32(_vlow_leaky), _signmask);
    int32x4_t _signhigh_leaky = vandq_s32(vreinterpretq_s32_f32(_vhigh_leaky), _signmask);
    float32x4_t _p5low_leaky = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(_p5), _signlow_leaky));
    float32x4_t _p5high_leaky = vreinterpretq_f32_s32(vorrq_s32(vreinterpretq_s32_f32(_p5), _signhigh_leaky));
    float32x4_t _vlow5_leaky = vaddq_f32(_vlow_leaky, _p5low_leaky);
    float32x4_t _vhigh5_leaky = vaddq_f32(_vhigh_leaky, _p5high_leaky);
    int32x4_t _vlow32_leaky = vcvtq_s32_f32(_vlow5_leaky);
    int32x4_t _vhigh32_leaky = vcvtq_s32_f32(_vhigh5_leaky);
#endif
    int16x8_t _v16 = vcombine_s16(vqmovn_s32(_vlow32), vqmovn_s32(_vhigh32));
    int16x8_t _v16_leaky = vcombine_s16(vqmovn_s32(_vlow32_leaky), vqmovn_s32(_vhigh32_leaky));
    int8x8_t _v8 = vqmovn_s16(_v16);
    int8x8_t _v8_leaky = vqmovn_s16(_v16_leaky);
    return vmax_s8(_v8, _v8_leaky);
}

static inline float32x4_t activation_ps(float32x4_t _v, int activation_type, const otter::Tensor& activation_params) {
    if (activation_type == 1)
    {
        const float32x4_t _zero = vdupq_n_f32(0.f);
        _v = vmaxq_f32(_v, _zero);
    }
    else if (activation_type == 2)
    {
        const float32x4_t _zero = vdupq_n_f32(0.f);
        const float32x4_t _slope = vdupq_n_f32(activation_params.data_ptr<float>()[0]);
        const uint32x4_t _lemask = vcleq_f32(_v, _zero);
        float32x4_t _ps = vmulq_f32(_v, _slope);
        _v = vbslq_f32(_lemask, _ps, _v);
    }
    else if (activation_type == 3)
    {
        const float32x4_t _min = vdupq_n_f32(0);
        const float32x4_t _max = vdupq_n_f32(6);
        _v = vmaxq_f32(_v, _min);
        _v = vminq_f32(_v, _max);
    }
    else if (activation_type == 4)
    {
        _v = sigmoid_ps(_v);
    }
    else if (activation_type == 5)
    {
        _v = vmulq_f32(_v, tanh_ps(log_ps(vaddq_f32(exp_ps(_v), vdupq_n_f32(1.f)))));
    }
    else if (activation_type == 6)
    {
        const float alpha = activation_params.data_ptr<float>()[0];
        const float beta = activation_params.data_ptr<float>()[1];
        const float32x4_t _zero = vdupq_n_f32(0.f);
        const float32x4_t _one = vdupq_n_f32(1.f);
        float32x4_t _ans = vdupq_n_f32(beta);
        _ans = vmlaq_n_f32(_ans, _v, alpha);
        _ans = vmaxq_f32(_ans, _zero);
        _ans = vminq_f32(_ans, _one);
        _v = vmulq_f32(_ans, _v);
    }

    return _v;
}

namespace otter {

Tensor quantize_to_int8_neon(const Tensor& src, const Tensor& scale_data, bool pack);

Tensor dequantize_from_int32_neon(const Tensor& src, const Tensor& scale_data, const Tensor& bias_data, bool pack);

Tensor requantize_from_int32_to_int8_neon(const Tensor& src, const Tensor& scale_in_data, const Tensor& scale_out_data, const Tensor& bias_data, int activation_type, const Tensor& activation_params, bool pack);

}   // end namespace otter

#endif // __ARM_NEON__

#endif /* QuantizeNeon_hpp */
