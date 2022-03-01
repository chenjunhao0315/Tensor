//
//  Vec256_float.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/17.
//

#ifndef Vec256_float_h
#define Vec256_float_h

#include "VecIntrinsic.hpp"
#include "VecBase.hpp"

#include "Config.hpp"
#include "Utils.hpp"

namespace otter {
namespace vec {

#if CPU_CAPABILITY_AVX2

template <>
class Vectorized<float> {
private:
    __m256 values;
public:
    using value_type = float;
    using size_type = int;
    static constexpr size_type size() {
        return 8;
    }
    Vectorized() {}
    Vectorized(__m256 v) : values(v) {}
    Vectorized(float val) {
        values = _mm256_set1_ps(val);
    }
    Vectorized(float val1, float val2, float val3, float val4, float val5, float val6, float val7, float val8) {
        values = _mm256_setr_ps(val1, val2, val3, val4, val5, val6, val7, val8);
    }
    operator __m256() const {
        return values;
    }
    
    template <int64_t mask>
    static Vectorized<float> blend(const Vectorized<float>& a, const Vectorized<float>& b) {
        return _mm256_blend_ps(a.values, b.values, mask);
    }
    static Vectorized<float> blendv(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& mask) {
        return _mm256_blendv_ps(a.values, b.values, mask.values);
    }
    
    static Vectorized<float> loadu(const void* ptr, int64_t count = size()) {
        if (count == size())
            return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));
        __otter_align__ float tmp_values[size()];
        // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
        // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
        // instructions while a loop would be compiled to one instruction.
        for (const auto i : otter::irange(size())) {
            tmp_values[i] = 0.0;
        }
        std::memcpy(tmp_values, reinterpret_cast<const float*>(ptr), count * sizeof(float));
        return _mm256_loadu_ps(tmp_values);
    }
    void store(void* ptr, int64_t count = size()) const {
        if (count == size()) {
            _mm256_storeu_ps(reinterpret_cast<float*>(ptr), values);
        } else if (count > 0) {
            float tmp_values[size()];
            _mm256_storeu_ps(reinterpret_cast<float*>(tmp_values), values);
            std::memcpy(ptr, tmp_values, count * sizeof(float));
        }
    }
    const float& operator[](int idx) const  = delete;
    float& operator[](int idx) = delete;
    
    Vectorized<float> map(float (*const f)(float)) const {
        __otter_align__ float tmp[size()];
        store(tmp);
        for (const auto i : otter::irange(size())) {
            tmp[i] = f(tmp[i]);
        }
        return loadu(tmp);
    }
    Vectorized<float> abs() const {
        auto mask = _mm256_set1_ps(-0.f);
        return _mm256_andnot_ps(mask, values);
    }
    
    Vectorized<float> neg() const {
        return _mm256_xor_ps(_mm256_set1_ps(-0.f), values);
    }
};

template <>
Vectorized<float> inline operator+(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm256_add_ps(a, b);
}

template <>
Vectorized<float> inline operator-(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm256_sub_ps(a, b);
}

template <>
Vectorized<float> inline operator*(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm256_mul_ps(a, b);
}

template <>
Vectorized<float> inline operator/(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm256_div_ps(a, b);
}


#endif


}   // end namespace vec
}   // end namespace otter

#endif /* Vec256_float_h */
