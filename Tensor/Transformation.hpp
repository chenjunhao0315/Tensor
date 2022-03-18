//
//  Transformation.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/18.
//

#ifndef Transformation_hpp
#define Transformation_hpp

#include "Macro.hpp"

#include <limits>
#include <cstdint>
#include <cassert>
#include <cmath>

namespace otter {

template <typename T>
struct DistAccumType {};

template <> struct DistAccumType<float> { using type = float; };
template <> struct DistAccumType<double> { using type = double; };

template <typename T>
using dist_acctype = typename DistAccumType<T>::type;

namespace transformation {

template <typename T, typename V>
inline T uniform_int_from_to(V val, uint64_t range, int64_t base) {
    return static_cast<T>(static_cast<int64_t>((val % range) + base));
}

template <typename T, typename V>
inline T uniform_int_full_range(V val) {
    return static_cast<T>(static_cast<int64_t>(val));
}

template <typename T, typename V>
inline typename std::enable_if<!(std::is_floating_point<T>::value), T>::type uniform_int(V val) {
    if (std::is_same<T, bool>::value) {
        return static_cast<bool>(val & 1);
    } else if (std::is_same<T, int64_t>::value) {
        return static_cast<T>(val % (static_cast<uint64_t>(std::numeric_limits<T>::max()) + 1));
    } else if (std::is_integral<T>::value) {
        return static_cast<T>(val % (static_cast<uint64_t>(std::numeric_limits<T>::max()) + 1));
    } else {
        assert(false);
        return 0;
    }
    return 0;
}

template<typename T, typename V>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::type uniform_int(V val) {
    return static_cast<T>(val % static_cast<uint64_t>((1ULL << std::numeric_limits<T>::digits) + 1));
}

template <typename T, typename V>
inline dist_acctype<T> uniform_real(V val, T from, T to) {
    constexpr auto MASK = static_cast<V>((static_cast<uint64_t>(1) << std::numeric_limits<T>::digits) - 1);
    constexpr auto DIVISOR = static_cast<dist_acctype<T>>(1) / (static_cast<uint64_t>(1) << std::numeric_limits<T>::digits);
    dist_acctype<T> x = (val & MASK) * DIVISOR;
    return (x * (to - from) + from);
}

template <typename T>
inline T normal(T val, T mean, T std) {
    return val * std + mean;
}

template <typename T>
inline T cauchy(T val, T median, T sigma) {
    constexpr T eps = std::numeric_limits<T>::epsilon();
    constexpr T one_minus_eps = 1 - eps;
    constexpr T zero_plus_eps = 0 + eps;
    val = (val > one_minus_eps ? one_minus_eps : val);
    val = (val < zero_plus_eps ? zero_plus_eps : val);
    return median + sigma * tan((T)M_PI * (val - static_cast<T>(0.5)));
}

template <>
inline double cauchy(double val, double median, double sigma) {
    return median + sigma * std::tan((double)M_PI * (val - static_cast<double>(0.5)));
}


template <typename T>
__ubsan_ignore_float_divide_by_zero__ inline T exponential(T val, T lambda) {
    return static_cast<T>(-1.0) / lambda * std::log(static_cast<T>(1.0) - val);
}

template <typename T>
inline T geometric(T val, T p) {
    return static_cast<T>(std::ceil(std::log(val) / std::log(static_cast<T>(1.0) - p)));
}

template <typename T>
inline T log_normal(T val) {
    return std::exp(val);
}

template <typename T>
inline T bernoulli(T val, T p) {
    return val < p;
}


}   // end namespace transformation
}   // end namespace otter

#endif /* Transformation_hpp */
