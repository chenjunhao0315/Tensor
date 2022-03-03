//
//  Vec256_float_neon.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#ifndef Vec256_float_neon_h
#define Vec256_float_neon_h

#include "VecIntrinsic.hpp"
#include "VecBase.hpp"

#include "Utils.hpp"

namespace otter {
namespace vec {

#if defined(__aarch64__)

template<int index, bool mask_val>
struct BlendRegs {
    static float32x4_t impl(
                            const float32x4_t& a, const float32x4_t& b, float32x4_t& res);
};

template<int index>
struct BlendRegs<index, true>{
    static float32x4_t impl(
                            const float32x4_t& a, const float32x4_t& b, float32x4_t& res) {
        return vsetq_lane_f32(vgetq_lane_f32(b, index), res, index);
    }
};

template<int index>
struct BlendRegs<index, false>{
    static float32x4_t impl(
                            const float32x4_t& a, const float32x4_t& b, float32x4_t& res) {
        return vsetq_lane_f32(vgetq_lane_f32(a, index), res, index);
    }
};

template <>
class Vectorized<float> {
private:
    float32x4x2_t values;
public:
    using value_type = float;
    using size_type = int;
    static constexpr size_type size() {
        return 8;
    }
    Vectorized() {}
    Vectorized(float32x4x2_t v) : values(v) {}
    Vectorized(float val) : values{vdupq_n_f32(val), vdupq_n_f32(val) } {}
    Vectorized(float val0, float val1, float val2, float val3,
               float val4, float val5, float val6, float val7) :
    values{val0, val1, val2, val3, val4, val5, val6, val7} {}
    Vectorized(float32x4_t val0, float32x4_t val1) : values{val0, val1} {}
    operator float32x4x2_t() const {
        return values;
    }
    template <int64_t mask>
    static Vectorized<float> blend(const Vectorized<float>& a, const Vectorized<float>& b) {
        Vectorized<float> vec;
        // 0.
        vec.values.val[0] =
        BlendRegs<0, (mask & 0x01)!=0>::impl(
                                             a.values.val[0], b.values.val[0], vec.values.val[0]);
        vec.values.val[0] =
        BlendRegs<1, (mask & 0x02)!=0>::impl(
                                             a.values.val[0], b.values.val[0], vec.values.val[0]);
        vec.values.val[0] =
        BlendRegs<2, (mask & 0x04)!=0>::impl(
                                             a.values.val[0], b.values.val[0], vec.values.val[0]);
        vec.values.val[0] =
        BlendRegs<3, (mask & 0x08)!=0>::impl(
                                             a.values.val[0], b.values.val[0], vec.values.val[0]);
        // 1.
        vec.values.val[1] =
        BlendRegs<0, (mask & 0x10)!=0>::impl(
                                             a.values.val[1], b.values.val[1], vec.values.val[1]);
        vec.values.val[1] =
        BlendRegs<1, (mask & 0x20)!=0>::impl(
                                             a.values.val[1], b.values.val[1], vec.values.val[1]);
        vec.values.val[1] =
        BlendRegs<2, (mask & 0x40)!=0>::impl(
                                             a.values.val[1], b.values.val[1], vec.values.val[1]);
        vec.values.val[1] =
        BlendRegs<3, (mask & 0x80)!=0>::impl(
                                             a.values.val[1], b.values.val[1], vec.values.val[1]);
        return vec;
    }
    static Vectorized<float> blendv(const Vectorized<float>& a, const Vectorized<float>& b,
                                    const Vectorized<float>& mask) {
        // TODO
        // NB: This requires that each value, i.e., each uint value,
        // of the mask either all be zeros or all be 1s.
        // We perhaps need some kind of an assert?
        // But that will affect performance.
        Vectorized<float> vec(mask.values);
        vec.values.val[0] = vbslq_f32(
                                      vreinterpretq_u32_f32(vec.values.val[0]),
                                      b.values.val[0],
                                      a.values.val[0]);
        vec.values.val[1] = vbslq_f32(
                                      vreinterpretq_u32_f32(vec.values.val[1]),
                                      b.values.val[1],
                                      a.values.val[1]);
        return vec;
    }
    static Vectorized<float> set(const Vectorized<float>& a, const Vectorized<float>& b,
                                 int64_t count = size()) {
        switch (count) {
            case 0:
                return a;
            case 1:
            {
                Vectorized<float> vec;
                static uint32x4_t mask_low = {0xFFFFFFFF, 0x0, 0x0, 0x0};
                vec.values.val[0] = vreinterpretq_f32_u32(mask_low);
                vec.values.val[1] = a.values.val[1];
                vec.values.val[0] = vbslq_f32(
                                              vreinterpretq_u32_f32(vec.values.val[0]),
                                              b.values.val[0],
                                              a.values.val[0]);
                return vec;
            }
            case 2:
            {
                Vectorized<float> vec;
                static uint32x4_t mask_low = {0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0};
                vec.values.val[0] = vreinterpretq_f32_u32(mask_low);
                vec.values.val[1] = a.values.val[1];
                vec.values.val[0] = vbslq_f32(
                                              vreinterpretq_u32_f32(vec.values.val[0]),
                                              b.values.val[0],
                                              a.values.val[0]);
                return vec;
            }
            case 3:
            {
                Vectorized<float> vec;
                static uint32x4_t mask_low = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0};
                vec.values.val[0] = vreinterpretq_f32_u32(mask_low);
                vec.values.val[1] = a.values.val[1];
                vec.values.val[0] = vbslq_f32(
                                              vreinterpretq_u32_f32(vec.values.val[0]),
                                              b.values.val[0],
                                              a.values.val[0]);
                return vec;
            }
            case 4:
                return Vectorized<float>(b.values.val[0], a.values.val[1]);
            case 5:
            {
                Vectorized<float> vec;
                static uint32x4_t mask_high = {0xFFFFFFFF, 0x0, 0x0, 0x0};
                vec.values.val[0] = b.values.val[0];
                vec.values.val[1] = vreinterpretq_f32_u32(mask_high);
                vec.values.val[1] = vbslq_f32(
                                              vreinterpretq_u32_f32(vec.values.val[1]),
                                              b.values.val[1],
                                              a.values.val[1]);
                return vec;
            }
            case 6:
            {
                Vectorized<float> vec;
                static uint32x4_t mask_high = {0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0};
                vec.values.val[0] = b.values.val[0];
                vec.values.val[1] = vreinterpretq_f32_u32(mask_high);
                vec.values.val[1] = vbslq_f32(
                                              vreinterpretq_u32_f32(vec.values.val[1]),
                                              b.values.val[1],
                                              a.values.val[1]);
                return vec;
            }
            case 7:
            {
                Vectorized<float> vec;
                static uint32x4_t mask_high = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0};
                vec.values.val[0] = b.values.val[0];
                vec.values.val[1] = vreinterpretq_f32_u32(mask_high);
                vec.values.val[1] = vbslq_f32(
                                              vreinterpretq_u32_f32(vec.values.val[1]),
                                              b.values.val[1],
                                              a.values.val[1]);
                return vec;
            }
        }
        return b;
    }
    static Vectorized<float> loadu(const void* ptr, int64_t count = size()) {
        if (count == size()) {
            return vld1q_f32_x2(reinterpret_cast<const float*>(ptr));
        }
        else if (count == (size() >> 1)) {
            Vectorized<float> res;
            res.values.val[0] = vld1q_f32(reinterpret_cast<const float*>(ptr));
            res.values.val[1] = vdupq_n_f32(0.f);
            return res;
        }
        else {
            __otter_align__ float tmp_values[size()];
            for (const auto i : otter::irange(size())) {
                tmp_values[i] = 0.0;
            }
            std::memcpy(
                        tmp_values,
                        reinterpret_cast<const float*>(ptr),
                        count * sizeof(float));
            return vld1q_f32_x2(reinterpret_cast<const float*>(tmp_values));
        }
    }
    void store(void* ptr, int64_t count = size()) const {
        if (count == size()) {
            vst1q_f32_x2(reinterpret_cast<float*>(ptr), values);
        }
        else if (count == (size() >> 1)) {
            vst1q_f32(reinterpret_cast<float*>(ptr), values.val[0]);
        }
        else {
            float tmp_values[size()];
            vst1q_f32_x2(reinterpret_cast<float*>(tmp_values), values);
            std::memcpy(ptr, tmp_values, count * sizeof(float));
        }
    }
    inline const float32x4_t& get_low() const {
        return values.val[0];
    }
    inline float32x4_t& get_low() {
        return values.val[0];
    }
    inline const float32x4_t& get_high() const {
        return values.val[1];
    }
    inline float32x4_t& get_high() {
        return values.val[1];
    }
    float operator[](int idx) const {
        __otter_align__ float tmp[size()];
        store(tmp);
        return tmp[idx];
    }
    float operator[](int idx) {
        __otter_align__ float tmp[size()];
        store(tmp);
        return tmp[idx];
    }
    Vectorized<float> isnan() const {
        __otter_align__ float tmp[size()];
        __otter_align__ float res[size()];
        store(tmp);
        for (const auto i : otter::irange(size())) {
            if (std::isnan(tmp[i])) {
                std::memset(static_cast<void*>(&res[i]), 0xFF, sizeof(float));
            } else {
                std::memset(static_cast<void*>(&res[i]), 0, sizeof(float));
            }
        }
        return loadu(res);
    };
    
    Vectorized<float> map(float (*const f)(float)) const {
        __otter_align__ float tmp[size()];
        store(tmp);
        for (const auto i : otter::irange(size())) {
            tmp[i] = f(tmp[i]);
        }
        return loadu(tmp);
    }
    Vectorized<float> abs() const {
        return Vectorized<float>(vabsq_f32(values.val[0]), vabsq_f32(values.val[1]));
    }
    Vectorized<float> neg() const {
        return Vectorized<float>(vnegq_f32(values.val[0]), vnegq_f32(values.val[1]));
    }
    Vectorized<float> operator==(const Vectorized<float>& other) const {
        float32x4_t r0 =
        vreinterpretq_f32_u32(vceqq_f32(values.val[0], other.values.val[0]));
        float32x4_t r1 =
        vreinterpretq_f32_u32(vceqq_f32(values.val[1], other.values.val[1]));
        return Vectorized<float>(r0, r1);
    }
    
    Vectorized<float> operator!=(const Vectorized<float>& other) const {
        float32x4_t r0 = vreinterpretq_f32_u32(
                                               vmvnq_u32(vceqq_f32(values.val[0], other.values.val[0])));
        float32x4_t r1 = vreinterpretq_f32_u32(
                                               vmvnq_u32(vceqq_f32(values.val[1], other.values.val[1])));
        return Vectorized<float>(r0, r1);
    }
    
    Vectorized<float> operator<(const Vectorized<float>& other) const {
        float32x4_t r0 =
        vreinterpretq_f32_u32(vcltq_f32(values.val[0], other.values.val[0]));
        float32x4_t r1 =
        vreinterpretq_f32_u32(vcltq_f32(values.val[1], other.values.val[1]));
        return Vectorized<float>(r0, r1);
    }
    
    Vectorized<float> operator<=(const Vectorized<float>& other) const {
        float32x4_t r0 =
        vreinterpretq_f32_u32(vcleq_f32(values.val[0], other.values.val[0]));
        float32x4_t r1 =
        vreinterpretq_f32_u32(vcleq_f32(values.val[1], other.values.val[1]));
        return Vectorized<float>(r0, r1);
    }
    
    Vectorized<float> operator>(const Vectorized<float>& other) const {
        float32x4_t r0 =
        vreinterpretq_f32_u32(vcgtq_f32(values.val[0], other.values.val[0]));
        float32x4_t r1 =
        vreinterpretq_f32_u32(vcgtq_f32(values.val[1], other.values.val[1]));
        return Vectorized<float>(r0, r1);
    }
    
    Vectorized<float> operator>=(const Vectorized<float>& other) const {
        float32x4_t r0 =
        vreinterpretq_f32_u32(vcgeq_f32(values.val[0], other.values.val[0]));
        float32x4_t r1 =
        vreinterpretq_f32_u32(vcgeq_f32(values.val[1], other.values.val[1]));
        return Vectorized<float>(r0, r1);
    }
    
};

template <>
Vectorized<float> inline operator+(const Vectorized<float>& a, const Vectorized<float>& b) {
    float32x4_t r0 = vaddq_f32(a.get_low(), b.get_low());
    float32x4_t r1 = vaddq_f32(a.get_high(), b.get_high());
    return Vectorized<float>(r0, r1);
}

template <>
Vectorized<float> inline operator-(const Vectorized<float>& a, const Vectorized<float>& b) {
    float32x4_t r0 = vsubq_f32(a.get_low(), b.get_low());
    float32x4_t r1 = vsubq_f32(a.get_high(), b.get_high());
    return Vectorized<float>(r0, r1);
}

template <>
Vectorized<float> inline operator*(const Vectorized<float>& a, const Vectorized<float>& b) {
    float32x4_t r0 = vmulq_f32(a.get_low(), b.get_low());
    float32x4_t r1 = vmulq_f32(a.get_high(), b.get_high());
    return Vectorized<float>(r0, r1);
}

template <>
Vectorized<float> inline operator/(const Vectorized<float>& a, const Vectorized<float>& b) {
    float32x4_t r0 = vdivq_f32(a.get_low(), b.get_low());
    float32x4_t r1 = vdivq_f32(a.get_high(), b.get_high());
    return Vectorized<float>(r0, r1);
}

template <>
Vectorized<float> inline operator&(const Vectorized<float>& a, const Vectorized<float>& b) {
    float32x4_t r0 = vreinterpretq_f32_u32(vandq_u32(
                                                     vreinterpretq_u32_f32(a.get_low()),
                                                     vreinterpretq_u32_f32(b.get_low())));
    float32x4_t r1 = vreinterpretq_f32_u32(vandq_u32(
                                                     vreinterpretq_u32_f32(a.get_high()),
                                                     vreinterpretq_u32_f32(b.get_high())));
    return Vectorized<float>(r0, r1);
}

template <>
Vectorized<float> inline operator|(const Vectorized<float>& a, const Vectorized<float>& b) {
    float32x4_t r0 = vreinterpretq_f32_u32(vorrq_u32(
                                                     vreinterpretq_u32_f32(a.get_low()),
                                                     vreinterpretq_u32_f32(b.get_low())));
    float32x4_t r1 = vreinterpretq_f32_u32(vorrq_u32(
                                                     vreinterpretq_u32_f32(a.get_high()),
                                                     vreinterpretq_u32_f32(b.get_high())));
    return Vectorized<float>(r0, r1);
}

template <>
Vectorized<float> inline operator^(const Vectorized<float>& a, const Vectorized<float>& b) {
    float32x4_t r0 = vreinterpretq_f32_u32(veorq_u32(
                                                     vreinterpretq_u32_f32(a.get_low()),
                                                     vreinterpretq_u32_f32(b.get_low())));
    float32x4_t r1 = vreinterpretq_f32_u32(veorq_u32(
                                                     vreinterpretq_u32_f32(a.get_high()),
                                                     vreinterpretq_u32_f32(b.get_high())));
    return Vectorized<float>(r0, r1);
}

#endif


}   // end namespace vec
}   // end namespace otter


#endif /* Vec256_float_neon_h */
