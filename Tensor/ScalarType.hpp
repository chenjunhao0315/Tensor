//
//  ScalarType.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef ScalarType_hpp
#define ScalarType_hpp

#include <string>
#include <limits>
#include <cmath>

#include "TypeCast.hpp"

namespace otter {

#define OTTER_ALL_SCALAR_TYPES_WO_BOOL(_) \
_(uint8_t, Byte)                                               \
_(int8_t, Char)                                                \
_(int16_t, Short)                                              \
_(int, Int)                                                    \
_(int64_t, Long)                                               \
_(float, Float)                                                \
_(double, Double)

#define OTTER_ALL_SCALAR_TYPES(_)       \
_(uint8_t, Byte)      /* 0 */       \
_(int8_t, Char)       /* 1 */       \
_(int16_t, Short)     /* 2 */       \
_(int, Int)           /* 3 */       \
_(int64_t, Long)      /* 4 */       \
_(float, Float)       /* 5 */       \
_(double, Double)     /* 6 */       \
_(bool, Bool)         /* 7 */

enum class ScalarType : int8_t {
#define DEFINE_ENUM(_1, n) n,
    OTTER_ALL_SCALAR_TYPES(DEFINE_ENUM)
#undef DEFINE_ENUM
    Undefined,
    NumOptions
};

std::string toString(ScalarType type);

static inline size_t elementSize(ScalarType t) {
#define CASE_ELEMENTSIZE_CASE(ctype, name) \
    case ScalarType::name:                 \
    return sizeof(ctype);

    switch (t) {
        OTTER_ALL_SCALAR_TYPES(CASE_ELEMENTSIZE_CASE)
    default:
        fprintf(stderr, "Undefined Scalar Type!\n");
    }
#undef CASE_ELEMENTSIZE_CASE
    return -1;
}

static inline bool isIntegralType(ScalarType t) {
  return (
      t == ScalarType::Byte || t == ScalarType::Char || t == ScalarType::Int ||
      t == ScalarType::Long || t == ScalarType::Short);
}

static inline bool isIntegralType(ScalarType t, bool includeBool) {
  bool isIntegral =
      (t == ScalarType::Byte || t == ScalarType::Char || t == ScalarType::Int ||
       t == ScalarType::Long || t == ScalarType::Short);

  return includeBool ? isIntegral || (t == ScalarType::Bool) : isIntegral;
}

static inline bool isFloatingType(ScalarType t) {
  return (
      t == ScalarType::Double || t == ScalarType::Float);
}

template <typename T>
struct CppTypeToScalarType;

#define SPECIALIZE_CppTypeToScalarType(cpp_type, scalar_type)                  \
  template <>                                                                  \
  struct CppTypeToScalarType<cpp_type>                                         \
      : std::                                                                  \
        integral_constant<otter::ScalarType, otter::ScalarType::scalar_type> { \
  };

OTTER_ALL_SCALAR_TYPES(SPECIALIZE_CppTypeToScalarType)

#undef SPECIALIZE_CppTypeToScalarType





template <typename To, typename From>
typename std::enable_if<std::is_same<From, bool>::value, bool>::type overflows(From f) {
    return false;
}

template <typename T>
struct scalar_value_type {
    using type = T;
};

// Returns false since we cannot have x < 0 if x is unsigned.
template <typename T>
static inline constexpr bool is_negative(const T& x, std::true_type is_unsigned) {
    return false;
}

// Returns true if a signed variable x < 0
template <typename T>
static inline constexpr bool is_negative(const T& x, std::false_type is_unsigned) {
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
    return T(0) < x;
}

// Returns the sign of a signed variable x as -1, 0, 1
template <typename T>
static inline constexpr int signum(const T& x, std::false_type is_unsigned) {
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
    return x < std::numeric_limits<Limit>::lowest();
}

// Returns false since all the limit is signed and therefore includes
// negative values but x cannot be negative because it is unsigned
template <typename Limit, typename T>
static inline constexpr bool less_than_lowest(const T& x, std::false_type limit_is_unsigned, std::true_type x_is_unsigned) {
    return false;
}

// Returns true if x < 0, where 0 is constructed from T.
// Limit is not signed, so its lower value is zero
template <typename Limit, typename T>
static inline constexpr bool less_than_lowest(const T& x, std::true_type limit_is_unsigned, std::false_type x_is_unsigned) {
    return x < T(0);
}

// Returns false sign both types are unsigned
template <typename Limit, typename T>
static inline constexpr bool less_than_lowest(const T& x, std::true_type limit_is_unsigned, std::true_type x_is_unsigned) {
    return false;
}

// Returns true if x is less than the lowest value of type T
template <typename Limit, typename T>
inline constexpr bool less_than_lowest(const T& x) {
    return less_than_lowest<Limit>(x, std::is_unsigned<Limit>(), std::is_unsigned<T>());
}

// skip isnan and isinf check for integral types
template <typename To, typename From>
typename std::enable_if<std::is_integral<From>::value && !std::is_same<From, bool>::value, bool>::type overflows(From f) {
    using limit = std::numeric_limits<typename scalar_value_type<To>::type>;
    if (!limit::is_signed && std::numeric_limits<From>::is_signed) {
        // allow for negative numbers to wrap using two's complement arithmetic.
        // For example, with uint8, this allows for `a - b` to be treated as
        // `a + 255 * b`.
        return greater_than_max<To>(f) || (is_negative(f) && -static_cast<uint64_t>(f) > limit::max());
    } else {
        return less_than_lowest<To>(f) || greater_than_max<To>(f);
    }
}

template <typename To, typename From>
typename std::enable_if<std::is_floating_point<From>::value, bool>::type overflows(From f) {
    using limit = std::numeric_limits<typename scalar_value_type<To>::type>;
    if (limit::has_infinity && std::isinf(static_cast<double>(f))) {
        return false;
    }
    if (!limit::has_quiet_NaN && (f != f)) {
        return true;
    }
    return f < limit::lowest() || f > limit::max();
}

#define ERROR_UNSUPPORTED_CAST fprintf(stderr, "[ScalarType] Unsupported cast!\n")

#define FETCH_AND_CAST_CASE(type, scalartype)   \
    case ScalarType::scalartype:                \
        return static_cast_with_inter_type<dest_t, type>::apply(*(const type*)ptr);
template <typename dest_t>
inline dest_t fetch_and_cast(const ScalarType src_type, const void* ptr) {
    switch (src_type) {
            OTTER_ALL_SCALAR_TYPES(FETCH_AND_CAST_CASE)
        default:
            ERROR_UNSUPPORTED_CAST;
    }
    return dest_t(0); // just to avoid compiler warning
}

// Cast a value with static type src_t into dynamic dest_type, and store it to ptr.
#define CAST_AND_STORE_CASE(type, scalartype)                                   \
    case ScalarType::scalartype:                                                \
        *(type*)ptr = static_cast_with_inter_type<type, src_t>::apply(value);   \
        return;
template <typename src_t>
inline void cast_and_store(const ScalarType dest_type, void* ptr, src_t value) {
    switch (dest_type) {
            OTTER_ALL_SCALAR_TYPES(CAST_AND_STORE_CASE)
        default:;
    }
    ERROR_UNSUPPORTED_CAST;
}

#undef FETCH_AND_CAST_CASE
#undef CAST_AND_STORE_CASE
#undef DEFINE_UNCASTABLE
#undef ERROR_UNSUPPORTED_CAST

template <typename To, typename From>
To convert(From f) {
    return static_cast_with_inter_type<To, From>::apply(f);
}

// Define separately to avoid being inlined and prevent code-size bloat
void report_overflow(const char* name);

template <typename To, typename From>
To checked_convert(From f, const char* name) {
    // Converting to bool can't overflow so we exclude this case from checking.
    if (!std::is_same<To, bool>::value && overflows<To, From>(f)) {
        report_overflow(name);
    }
    return convert<To, From>(f);
}

}   // end namespace otter

#endif /* ScalarType_hpp */
