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

#include "Exception.hpp"
#include "TypeCast.hpp"

#include "PackedData.hpp"

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

#define OTTER_ALL_SCALAR_TYPES_W_PACKED(_)       \
_(uint8_t, Byte)      /* 0 */       \
_(int8_t, Char)       /* 1 */       \
_(int16_t, Short)     /* 2 */       \
_(int, Int)           /* 3 */       \
_(int64_t, Long)      /* 4 */       \
_(float, Float)       /* 5 */       \
_(double, Double)     /* 6 */       \
_(bool, Bool)         /* 7 */       \
_(elempack4<signed char>, Byte4)    \
_(elempack4<int>, Int4)             \
_(elempack4<float>, Float4)         \
_(elempack8<signed char>, Byte8)    \
_(elempack8<int>, Int8)             \
_(elempack8<float>, Float8)         \

enum class ScalarType : int8_t {
#define DEFINE_ENUM(_1, n) n,
    OTTER_ALL_SCALAR_TYPES_W_PACKED(DEFINE_ENUM)
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
        OTTER_ALL_SCALAR_TYPES_W_PACKED(CASE_ELEMENTSIZE_CASE)
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

static inline bool isSignedType(ScalarType t) {
#define CASE_SIGNED(ctype, name) \
  case ScalarType::name:         \
    return std::numeric_limits<ctype>::is_signed;
    switch (t) {
        OTTER_ALL_SCALAR_TYPES(CASE_SIGNED)
        default:
            OTTER_CHECK(false, "Unknown ScalarType");
    }
#undef CASE_SIGNED
    return false;
}

template <ScalarType N>
struct ScalarTypeToCPPType;

#define SPECIALIZE_ScalarTypeToCPPType(cpp_type, scalar_type)                \
template <>                                                                \
struct ScalarTypeToCPPType<ScalarType::scalar_type> {                 \
    using type = cpp_type;                                                   \
    static type t;                                                           \
};

OTTER_ALL_SCALAR_TYPES(SPECIALIZE_ScalarTypeToCPPType)

#undef SPECIALIZE_ScalarTypeToCPPType

template <>
struct ScalarTypeToCPPType<ScalarType::Float4> {
    using type = float;
    static type t;
};

template <>
struct ScalarTypeToCPPType<ScalarType::Float8> {
    using type = float;
    static type t;
};

template <>
struct ScalarTypeToCPPType<ScalarType::Byte4> {
    using type = signed char;
    static type t;
};

template <>
struct ScalarTypeToCPPType<ScalarType::Byte8> {
    using type = signed char;
    static type t;
};

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
    (void)f;
    return false;
}

template <typename T>
struct scalar_value_type {
    using type = T;
};

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

// skip isnan and isinf check for integral types
template <typename To, typename From>
typename std::enable_if<std::is_integral<From>::value && !std::is_same<From, bool>::value, bool>::type
overflows(From f) {
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
typename std::enable_if<std::is_floating_point<From>::value, bool>::type
overflows(From f) {
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

// see tensor_attributes.rst for detailed explanation and examples
// of casting rules.
static inline bool canCast(const ScalarType from, const ScalarType to) {
  // We disallow float -> integral, e.g., int_tensor *= float is disallowed.
  if (isFloatingType(from) && isIntegralType(to, false)) {
    return false;
  }
  // Treat bool as a distinct "category," to be consistent with type promotion
  // rules (e.g. `bool_tensor + 5 -> int64_tensor`). If `5` was in the same
  // category as `bool_tensor`, we would not promote. Differing categories
  // implies `bool_tensor += 5` is disallowed.
  //
  // NB: numpy distinguishes "unsigned" as a category to get the desired
  // `bool_tensor + 5 -> int64_tensor` behavior. We don't, because:
  // * We don't want the performance hit of checking the runtime sign of
  // Scalars.
  // * `uint8_tensor + 5 -> int64_tensor` would be undesirable.
  if (from != ScalarType::Bool && to == ScalarType::Bool) {
    return false;
  }
  return true;
}

static inline ScalarType promoteTypes(ScalarType a, ScalarType b) {
  // This is generated according to NumPy's promote_types
  constexpr auto u1 = ScalarType::Byte;
  constexpr auto i1 = ScalarType::Char;
  constexpr auto i2 = ScalarType::Short;
  constexpr auto i4 = ScalarType::Int;
  constexpr auto i8 = ScalarType::Long;
  constexpr auto f4 = ScalarType::Float;
  constexpr auto f8 = ScalarType::Double;
  constexpr auto b1 = ScalarType::Bool;
  constexpr auto ud = ScalarType::Undefined;
  if (a == ud || b == ud) {
    return ScalarType::Undefined;
  }

  // this matrix has to be consistent with AT_FORALL_SCALAR_TYPES_WITH_COMPLEX
  // so that's why we have to add undefined as we are not sure what is the
  // corrent values for the type promotions in complex type cases.
  static constexpr ScalarType _promoteTypesLookup[static_cast<int>(
      ScalarType::NumOptions)][static_cast<int>(ScalarType::NumOptions)] = {
      /*        u1  i1  i2  i4  i8  f4  f8  b1*/
      /* u1 */ {u1, i2, i2, i4, i8, f4, f8, u1},
      /* i1 */ {i2, i1, i2, i4, i8, f4, f8, i1},
      /* i2 */ {i2, i2, i2, i4, i8, f4, f8, i2},
      /* i4 */ {i4, i4, i4, i4, i8, f4, f8, i4},
      /* i8 */ {i8, i8, i8, i8, i8, f4, f8, i8},
      /* f4 */ {f4, f4, f4, f4, f4, f4, f8, f4},
      /* f8 */ {f8, f8, f8, f8, f8, f8, f8, f8},
      /* b1 */ {u1, i1, i2, i4, i8, f4, f8, b1}
  };
  return _promoteTypesLookup[static_cast<int>(a)][static_cast<int>(b)];
}

inline std::ostream& operator<<(std::ostream& stream, ScalarType scalar_type) {
    return stream << toString(scalar_type);
}

}   // end namespace otter

#endif /* ScalarType_hpp */
