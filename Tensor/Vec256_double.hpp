//
//  Vec256_double.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/5/26.
//

#ifndef Vec256_double_h
#define Vec256_double_h

#include "VecIntrinsic.hpp"
#include "VecBase.hpp"
#include "Avx_Math.hpp"

#include "Config.hpp"
#include "Utils.hpp"

namespace otter {
namespace vec {

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)

template <> class Vectorized<double> {
private:
  __m256d values;
public:
  using value_type = double;
  using size_type = int;
  static constexpr size_type size() {
    return 4;
  }
  Vectorized() {}
  Vectorized(__m256d v) : values(v) {}
  Vectorized(double val) {
    values = _mm256_set1_pd(val);
  }
  Vectorized(double val1, double val2, double val3, double val4) {
    values = _mm256_setr_pd(val1, val2, val3, val4);
  }
  operator __m256d() const {
    return values;
  }
  template <int64_t mask>
  static Vectorized<double> blend(const Vectorized<double>& a, const Vectorized<double>& b) {
    return _mm256_blend_pd(a.values, b.values, mask);
  }
  static Vectorized<double> blendv(const Vectorized<double>& a, const Vectorized<double>& b,
                               const Vectorized<double>& mask) {
    return _mm256_blendv_pd(a.values, b.values, mask.values);
  }
  template<typename step_t>
  static Vectorized<double> arange(double base = 0., step_t step = static_cast<step_t>(1)) {
    return Vectorized<double>(base, base + step, base + 2 * step, base + 3 * step);
  }
  static Vectorized<double> set(const Vectorized<double>& a, const Vectorized<double>& b,
                            int64_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
      case 2:
        return blend<3>(a, b);
      case 3:
        return blend<7>(a, b);
    }
    return b;
  }
  static Vectorized<double> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm256_loadu_pd(reinterpret_cast<const double*>(ptr));


    __otter_align__ double tmp_values[size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (const auto i : otter::irange(size())) {
      tmp_values[i] = 0.0;
    }
    std::memcpy(
        tmp_values,
        reinterpret_cast<const double*>(ptr),
        count * sizeof(double));
    return _mm256_load_pd(tmp_values);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm256_storeu_pd(reinterpret_cast<double*>(ptr), values);
    } else if (count > 0) {
      double tmp_values[size()];
      _mm256_storeu_pd(reinterpret_cast<double*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(double));
    }
  }
  const double& operator[](int idx) const  = delete;
  double& operator[](int idx) = delete;
  int zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
    __m256d cmp = _mm256_cmp_pd(values, _mm256_set1_pd(0.0), _CMP_EQ_OQ);
    return _mm256_movemask_pd(cmp);
  }
  Vectorized<double> isnan() const {
    return _mm256_cmp_pd(values, _mm256_set1_pd(0.0), _CMP_UNORD_Q);
  }
  Vectorized<double> map(double (*const f)(double)) const {
    __otter_align__ double tmp[size()];
    store(tmp);
    for (const auto i : otter::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vectorized<double> abs() const {
    auto mask = _mm256_set1_pd(-0.f);
    return _mm256_andnot_pd(mask, values);
  }
  Vectorized<double> real() const {
    return *this;
  }
  Vectorized<double> imag() const {
    return _mm256_set1_pd(0);
  }
  Vectorized<double> conj() const {
    return *this;
  }
  Vectorized<double> exp() const {
    return Vectorized<double>(exp256_ps(values));
  }
  Vectorized<double> log() const {
    return Vectorized<double>(log256_ps(values));
  }
  Vectorized<double> sin() const {
    return Vectorized<double>(sin256_ps(values));
  }
  Vectorized<double> cos() const {
    return Vectorized<double>(cos256_ps(values));
  }
  Vectorized<double> ceil() const {
    return _mm256_ceil_pd(values);
  }
  Vectorized<double> floor() const {
    return _mm256_floor_pd(values);
  }
  Vectorized<double> frac() const;
  Vectorized<double> neg() const {
    return _mm256_xor_pd(_mm256_set1_pd(-0.), values);
  }
  Vectorized<double> round() const {
    return _mm256_round_pd(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vectorized<double> trunc() const {
    return _mm256_round_pd(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vectorized<double> sqrt() const {
    return _mm256_sqrt_pd(values);
  }
  Vectorized<double> reciprocal() const {
    return _mm256_div_pd(_mm256_set1_pd(1), values);
  }
  Vectorized<double> rsqrt() const {
    return _mm256_div_pd(_mm256_set1_pd(1), _mm256_sqrt_pd(values));
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vectorized<double> operator==(const Vectorized<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_EQ_OQ);
  }

  Vectorized<double> operator!=(const Vectorized<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_NEQ_UQ);
  }

  Vectorized<double> operator<(const Vectorized<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_LT_OQ);
  }

  Vectorized<double> operator<=(const Vectorized<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_LE_OQ);
  }

  Vectorized<double> operator>(const Vectorized<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_GT_OQ);
  }

  Vectorized<double> operator>=(const Vectorized<double>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_GE_OQ);
  }

  Vectorized<double> eq(const Vectorized<double>& other) const;
  Vectorized<double> ne(const Vectorized<double>& other) const;
  Vectorized<double> lt(const Vectorized<double>& other) const;
  Vectorized<double> le(const Vectorized<double>& other) const;
  Vectorized<double> gt(const Vectorized<double>& other) const;
  Vectorized<double> ge(const Vectorized<double>& other) const;
};

template <>
Vectorized<double> inline operator+(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_add_pd(a, b);
}

template <>
Vectorized<double> inline operator-(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_sub_pd(a, b);
}

template <>
Vectorized<double> inline operator*(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_mul_pd(a, b);
}

template <>
Vectorized<double> inline operator/(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_div_pd(a, b);
}

// frac. Implement this here so we can use subtraction.
inline Vectorized<double> Vectorized<double>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<double> inline maximum(const Vectorized<double>& a, const Vectorized<double>& b) {
  Vectorized<double> max = _mm256_max_pd(a, b);
  Vectorized<double> isnan = _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_pd(max, isnan);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<double> inline minimum(const Vectorized<double>& a, const Vectorized<double>& b) {
  Vectorized<double> min = _mm256_min_pd(a, b);
  Vectorized<double> isnan = _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
  // Exploit the fact that all-ones is a NaN.
  return _mm256_or_pd(min, isnan);
}

template <>
Vectorized<double> inline clamp(const Vectorized<double>& a, const Vectorized<double>& min, const Vectorized<double>& max) {
  return _mm256_min_pd(max, _mm256_max_pd(min, a));
}

template <>
Vectorized<double> inline clamp_min(const Vectorized<double>& a, const Vectorized<double>& min) {
  return _mm256_max_pd(min, a);
}

template <>
Vectorized<double> inline clamp_max(const Vectorized<double>& a, const Vectorized<double>& max) {
  return _mm256_min_pd(max, a);
}

template <>
Vectorized<double> inline operator&(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_and_pd(a, b);
}

template <>
Vectorized<double> inline operator|(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_or_pd(a, b);
}

template <>
Vectorized<double> inline operator^(const Vectorized<double>& a, const Vectorized<double>& b) {
  return _mm256_xor_pd(a, b);
}

inline Vectorized<double> Vectorized<double>::eq(const Vectorized<double>& other) const {
  return (*this == other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::ne(const Vectorized<double>& other) const {
  return (*this != other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::gt(const Vectorized<double>& other) const {
  return (*this > other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::ge(const Vectorized<double>& other) const {
  return (*this >= other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::lt(const Vectorized<double>& other) const {
  return (*this < other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::le(const Vectorized<double>& other) const {
  return (*this <= other) & Vectorized<double>(1.0);
}

template <>
inline void convert(const double* src, double* dst, int64_t n) {
  int64_t i;
#pragma unroll
  for (i = 0; i <= (n - Vectorized<double>::size()); i += Vectorized<double>::size()) {
    _mm256_storeu_pd(dst + i, _mm256_loadu_pd(src + i));
  }
#pragma unroll
  for (; i < n; i++) {
    dst[i] = src[i];
  }
}

#ifdef CPU_CAPABILITY_AVX2
template <>
Vectorized<double> inline fmadd(const Vectorized<double>& a, const Vectorized<double>& b, const Vectorized<double>& c) {
  return _mm256_fmadd_pd(a, b, c);
}
#endif

#endif

}   // end namespace vec
}   // end namespace otter


#endif /* Vec256_double_h */
