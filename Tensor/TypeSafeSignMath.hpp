//
//  TypeSafeSignMath.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/16.
//

#ifndef TypeSafeSignMath_h
#define TypeSafeSignMath_h

#include "Macro.hpp"

#include <limits>
#include <type_traits>

OTTER_CLANG_DIAGNOSTIC_PUSH()
#if OTTER_CLANG_HAS_WARNING("-Wstring-conversion")
OTTER_CLANG_DIAGNOSTIC_IGNORE("-Wstring-conversion")
#endif
#if OTTER_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
OTTER_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

namespace otter {

// Returns false since we cannot have x < 0 if x is unsigned.
template <typename T>
static inline constexpr bool is_negative(const T& x, std::true_type is_unsigned) {
    (void)x;
    (void)is_unsigned;
    return false;
}

// Returns true if a signed variable x < 0
template <typename T>
static inline constexpr bool is_negative(const T& x, std::false_type is_unsigned) {
    (void)is_unsigned;
    return x < T(0);
}

// Returns true if x < 0
template <typename T>
inline constexpr bool is_negative(const T& x) {
    return is_negative(x, std::is_unsigned<T>());
}

// Returns the sign of an unsigned variable x as 0, 1
template <typename T>
static inline constexpr int signum(const T& x, std::true_type is_unsigned) {
    (void)is_unsigned;
    return T(0) < x;
}

// Returns the sign of a signed variable x as -1, 0, 1
template <typename T>
static inline constexpr int signum(const T& x, std::false_type is_unsigned) {
    (void)is_unsigned;
    return (T(0) < x) - (x < T(0));
}

// Returns the sign of x as -1, 0, 1
template <typename T>
inline constexpr int signum(const T& x) {
    return signum(x, std::is_unsigned<T>());
}

// Returns true if a and b are not both negative
template <typename T, typename U>
inline constexpr bool signs_differ(const T& a, const U& b) {
    return is_negative(a) != is_negative(b);
}

// Returns true if x is greater than the greatest value of the type Limit
template <typename Limit, typename T>
inline constexpr bool greater_than_max(const T& x) {
    constexpr bool can_overflow = std::numeric_limits<T>::digits> std::numeric_limits<Limit>::digits;
    return can_overflow && x > std::numeric_limits<Limit>::max();
}

// Returns true if x < lowest(Limit). Standard comparison
template <typename Limit, typename T>
static inline constexpr bool less_than_lowest(const T& x, std::false_type limit_is_unsigned, std::false_type x_is_unsigned) {
    (void)limit_is_unsigned;
    (void)x_is_unsigned;
    return x < std::numeric_limits<Limit>::lowest();
}

// Returns false since all the limit is signed and therefore includes
// negative values but x cannot be negative because it is unsigned
template <typename Limit, typename T>
static inline constexpr bool less_than_lowest(const T& x, std::false_type limit_is_unsigned, std::true_type x_is_unsigned) {
    (void)x;
    (void)limit_is_unsigned;
    (void)x_is_unsigned;
    return false;
}

// Returns true if x < 0, where 0 is constructed from T.
// Limit is not signed, so its lower value is zero
template <typename Limit, typename T>
static inline constexpr bool less_than_lowest(const T& x, std::true_type limit_is_unsigned, std::false_type x_is_unsigned) {
    (void)limit_is_unsigned;
    (void)x_is_unsigned;
    return x < T(0);
}

// Returns false sign both types are unsigned
template <typename Limit, typename T>
static inline constexpr bool less_than_lowest(const T& x, std::true_type limit_is_unsigned, std::true_type x_is_unsigned) {
    (void)x;
    (void)limit_is_unsigned;
    (void)x_is_unsigned;
    return false;
}

// Returns true if x is less than the lowest value of type T
template <typename Limit, typename T>
inline constexpr bool less_than_lowest(const T& x) {
    return less_than_lowest<Limit>(x, std::is_unsigned<Limit>(), std::is_unsigned<T>());
}

}   // end namespace otter


#endif /* TypeSafeSignMath_h */
