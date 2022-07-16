//
//  HFloat-inl.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/16.
//

#ifndef HFloat_inl_h
#define HFloat_inl_h

#include <cstring>
#include <limits>


OTTER_CLANG_DIAGNOSTIC_PUSH()
#if OTTER_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
OTTER_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

namespace otter {

// Constructors
inline HFloat::HFloat(float value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  x = __half_as_short(__float2half(value));
#elif defined(__SYCL_DEVICE_ONLY__)
  x = sycl::bit_cast<uint16_t>(sycl::half(value));
#else
  x = fp16_ieee_from_fp32_value(value);
#endif
}

// Implicit conversions
inline HFloat::operator float() const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __half2float(*reinterpret_cast<const __half*>(&x));
#elif defined(__SYCL_DEVICE_ONLY__)
  return float(sycl::bit_cast<sycl::half>(x));
#else
  return fp16_ieee_to_fp32_value(x);
#endif
}

inline HFloat operator+(const HFloat& a, const HFloat& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}
inline HFloat operator-(const HFloat& a, const HFloat& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}
inline HFloat operator*(const HFloat& a, const HFloat& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}
inline HFloat operator/(const HFloat& a, const HFloat& b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / static_cast<float>(b);
}
inline HFloat operator-(const HFloat& a) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530) || \
    defined(__HIP_DEVICE_COMPILE__)
  return __hneg(a);
#elif defined(__SYCL_DEVICE_ONLY__)
  return -sycl::bit_cast<sycl::half>(a);
#else
  return -static_cast<float>(a);
#endif
}
inline HFloat& operator+=(HFloat& a, const HFloat& b) {
  a = a + b;
  return a;
}
inline HFloat& operator-=(HFloat& a, const HFloat& b) {
  a = a - b;
  return a;
}
inline HFloat& operator*=(HFloat& a, const HFloat& b) {
  a = a * b;
  return a;
}
inline HFloat& operator/=(HFloat& a, const HFloat& b) {
  a = a / b;
  return a;
}
/// Arithmetic with floats
inline float operator+(HFloat a, float b) {
  return static_cast<float>(a) + b;
}
inline float operator-(HFloat a, float b) {
  return static_cast<float>(a) - b;
}
inline float operator*(HFloat a, float b) {
  return static_cast<float>(a) * b;
}
inline float operator/(HFloat a, float b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / b;
}
inline float operator+(float a, HFloat b) {
  return a + static_cast<float>(b);
}
inline float operator-(float a, HFloat b) {
  return a - static_cast<float>(b);
}
inline float operator*(float a, HFloat b) {
  return a * static_cast<float>(b);
}
inline float operator/(float a, HFloat b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<float>(b);
}
inline float& operator+=(float& a, const HFloat& b) {
  return a += static_cast<float>(b);
}
inline float& operator-=(float& a, const HFloat& b) {
  return a -= static_cast<float>(b);
}
inline float& operator*=(float& a, const HFloat& b) {
  return a *= static_cast<float>(b);
}
inline float& operator/=(float& a, const HFloat& b) {
  return a /= static_cast<float>(b);
}
/// Arithmetic with doubles
inline double operator+(HFloat a, double b) {
  return static_cast<double>(a) + b;
}
inline double operator-(HFloat a, double b) {
  return static_cast<double>(a) - b;
}
inline double operator*(HFloat a, double b) {
  return static_cast<double>(a) * b;
}
inline double operator/(HFloat a, double b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<double>(a) / b;
}
inline double operator+(double a, HFloat b) {
  return a + static_cast<double>(b);
}
inline double operator-(double a, HFloat b) {
  return a - static_cast<double>(b);
}
inline double operator*(double a, HFloat b) {
  return a * static_cast<double>(b);
}
inline double operator/(double a, HFloat b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<double>(b);
}
/// Arithmetic with ints
inline HFloat operator+(HFloat a, int b) {
  return a + static_cast<HFloat>(b);
}
inline HFloat operator-(HFloat a, int b) {
  return a - static_cast<HFloat>(b);
}
inline HFloat operator*(HFloat a, int b) {
  return a * static_cast<HFloat>(b);
}
inline HFloat operator/(HFloat a, int b) {
  return a / static_cast<HFloat>(b);
}
inline HFloat operator+(int a, HFloat b) {
  return static_cast<HFloat>(a) + b;
}
inline HFloat operator-(int a, HFloat b) {
  return static_cast<HFloat>(a) - b;
}
inline HFloat operator*(int a, HFloat b) {
  return static_cast<HFloat>(a) * b;
}
inline HFloat operator/(int a, HFloat b) {
  return static_cast<HFloat>(a) / b;
}
//// Arithmetic with int64_t
inline HFloat operator+(HFloat a, int64_t b) {
  return a + static_cast<HFloat>(b);
}
inline HFloat operator-(HFloat a, int64_t b) {
  return a - static_cast<HFloat>(b);
}
inline HFloat operator*(HFloat a, int64_t b) {
  return a * static_cast<HFloat>(b);
}
inline HFloat operator/(HFloat a, int64_t b) {
  return a / static_cast<HFloat>(b);
}
inline HFloat operator+(int64_t a, HFloat b) {
  return static_cast<HFloat>(a) + b;
}
inline HFloat operator-(int64_t a, HFloat b) {
  return static_cast<HFloat>(a) - b;
}
inline HFloat operator*(int64_t a, HFloat b) {
  return static_cast<HFloat>(a) * b;
}
inline HFloat operator/(int64_t a, HFloat b) {
  return static_cast<HFloat>(a) / b;
}

}   // end namespace otter

namespace std {
template <>
class numeric_limits<otter::HFloat> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss =
      numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 11;
  static constexpr int digits10 = 3;
  static constexpr int max_digits10 = 5;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;
  static constexpr otter::HFloat min() {
    return otter::HFloat(0x0400, otter::HFloat::from_bits());
  }
  static constexpr otter::HFloat lowest() {
    return otter::HFloat(0xFBFF, otter::HFloat::from_bits());
  }
  static constexpr otter::HFloat max() {
    return otter::HFloat(0x7BFF, otter::HFloat::from_bits());
  }
  static constexpr otter::HFloat epsilon() {
    return otter::HFloat(0x1400, otter::HFloat::from_bits());
  }
  static constexpr otter::HFloat round_error() {
    return otter::HFloat(0x3800, otter::HFloat::from_bits());
  }
  static constexpr otter::HFloat infinity() {
    return otter::HFloat(0x7C00, otter::HFloat::from_bits());
  }
  static constexpr otter::HFloat quiet_NaN() {
    return otter::HFloat(0x7E00, otter::HFloat::from_bits());
  }
  static constexpr otter::HFloat signaling_NaN() {
    return otter::HFloat(0x7D00, otter::HFloat::from_bits());
  }
  static constexpr otter::HFloat denorm_min() {
    return otter::HFloat(0x0001, otter::HFloat::from_bits());
  }
};
} // namespace std

OTTER_CLANG_DIAGNOSTIC_POP()

#endif /* HFloat_inl_h */
