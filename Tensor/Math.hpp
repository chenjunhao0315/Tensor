//
//  Math.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/9.
//

#ifndef Math_hpp
#define Math_hpp

#include "cmath"
#include "Macro.hpp"
#include "HFloat.hpp"
#include <cstdint>

namespace otter {

template <typename T>
static inline T div_round_up(T x, T y) {
    int64_t q = x / y;
    int64_t r = x % y;
    if ((r!=0) && ((r<0) != (y<0))) --q;
    return static_cast<T>(q);
}

template <typename T>
static T abs_impl(T v) {
    return std::abs(v);
}

template <>
OTTER_UNUSED uint8_t abs_impl(uint8_t v) {
    return v;
}

// std::isnan isn't performant to use on integral types; it will
// (uselessly) convert to floating point and then do the test.
// This function is.
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline bool _isnan(T /*val*/) {
    return false;
}
template <typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
inline bool _isnan(T val) {
#if defined(__CUDACC__) || defined(__HIPCC__)
    return ::isnan(val);
#else
    return std::isnan(val);
#endif
}

template <typename T, typename std::enable_if<std::is_same<T, otter::HFloat>::value, int>::type = 0>
inline bool _isnan(T val) {
    return otter::_isnan(static_cast<float>(val));
}
//template <typename T, typename std::enable_if<std::is_same<T, at::BFloat16>::value, int>::type = 0>
//inline bool _isnan(otter::BFloat16 val) {
//  return otter::_isnan(static_cast<float>(val));
//}
//inline bool _isnan(otter::BFloat16 val) {
//  return otter::_isnan(static_cast<float>(val));
//}
// std::isinf isn't performant to use on integral types; it will
// (uselessly) convert to floating point and then do the test.
// This function is.
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
inline bool _isinf(T /*val*/) {
    return false;
}
template <typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
inline bool _isinf(T val) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return ::isinf(val);
#else
  return std::isinf(val);
#endif
}
inline bool _isinf(otter::HFloat val) {
    return otter::_isinf(static_cast<float>(val));
}
//inline bool _isinf(at::BFloat16 val) {
//  return otter::_isinf(static_cast<float>(val));
//}
template <typename T>
inline T exp(T x) {
    static_assert(!std::is_same<T, double>::value, "this template must be used with float or less precise type");
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
    // use __expf fast approximation for peak bandwidth
    return __expf(x);
#else
    return ::exp(x);
#endif
}
template <>
inline double exp<double>(double x) {
  return ::exp(x);
}
template <typename T>
inline T log(T x) {
    static_assert(!std::is_same<T, double>::value, "this template must be used with float or less precise type");
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
    // use __logf fast approximation for peak bandwidth
    return __logf(x);
#else
    return ::log(x);
#endif
}
template <>
inline double log<double>(double x) {
    return ::log(x);
}
template <typename T>
inline T tan(T x) {
    static_assert(!std::is_same<T, double>::value, "this template must be used with float or less precise type");
#if defined(__CUDA_ARCH__) || defined(__HIP_ARCH__)
    // use __tanf fast approximation for peak bandwidth
    return __tanf(x);
#else
    return ::tan(x);
#endif
}
template <>
inline double tan<double>(double x) {
  return ::tan(x);
}

}

#endif /* Math_hpp */
