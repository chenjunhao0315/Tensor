//
//  Macro.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/16.
//

#ifndef Macro_h
#define Macro_h

#if defined(__clang__)
#define __ubsan_ignore_float_divide_by_zero__ \
__attribute__((no_sanitize("float-divide-by-zero")))
#define __ubsan_ignore_undefined__ __attribute__((no_sanitize("undefined")))
#define __ubsan_ignore_signed_int_overflow__ \
__attribute__((no_sanitize("signed-integer-overflow")))
#define __ubsan_ignore_function__ __attribute__((no_sanitize("function")))
#else
#define __ubsan_ignore_float_divide_by_zero__
#define __ubsan_ignore_undefined__
#define __ubsan_ignore_signed_int_overflow__
#define __ubsan_ignore_function__
#endif

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define OTTER_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define OTTER_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define OTTER_LIKELY(expr) (expr)
#define OTTER_UNLIKELY(expr) (expr)
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define OTTER_UNUSED __pragma(warning(suppress : 4100 4101))
#else
#define OTTER_UNUSED __attribute__((__unused__))
#endif //_MSC_VER

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define OTTER_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define OTTER_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define OTTER_LIKELY(expr) (expr)
#define OTTER_UNLIKELY(expr) (expr)
#endif

#define OTTER_RESTRICT __restrict

#define OTTER_STRINGIZE_IMPL(x) #x
#define OTTER_STRINGIZE(x) OTTER_STRINGIZE_IMPL(x)

#ifdef __clang__
#define _OTTER_PRAGMA__(string) _Pragma(#string)
#define _OTTER_PRAGMA_(string) _OTTER_PRAGMA__(string)
#define OTTER_CLANG_DIAGNOSTIC_PUSH() _Pragma("clang diagnostic push")
#define OTTER_CLANG_DIAGNOSTIC_POP() _Pragma("clang diagnostic pop")
#define OTTER_CLANG_DIAGNOSTIC_IGNORE(flag) \
  _OTTER_PRAGMA_(clang diagnostic ignored flag)
#define OTTER_CLANG_HAS_WARNING(flag) __has_warning(flag)
#else
#define OTTER_CLANG_DIAGNOSTIC_PUSH()
#define OTTER_CLANG_DIAGNOSTIC_POP()
#define OTTER_CLANG_DIAGNOSTIC_IGNORE(flag)
#define OTTER_CLANG_HAS_WARNING(flag) 0
#endif

#if defined(_MSC_VER)
#define OTTER_ALWAYS_INLINE __forceinline
#elif __has_attribute(always_inline) || defined(__GNUC__)
#define OTTER_ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define OTTER_ALWAYS_INLINE inline
#endif

#if defined(__GNUG__) && __GNUC__ < 5
#define OTTER_IS_TRIVIALLY_COPYABLE(T) __has_trivial_copy(T)
#else
#define OTTER_IS_TRIVIALLY_COPYABLE(T) std::is_trivially_copyable<T>::value
#endif

namespace otter {

template <int n>
struct ForcedUnroll {
    template <typename Func>
    OTTER_ALWAYS_INLINE void operator()(const Func& f) const {
        ForcedUnroll<n - 1>{}(f);
        f(n - 1);
    }
};
template <>
struct ForcedUnroll<1> {
    template <typename Func>
    OTTER_ALWAYS_INLINE void operator()(const Func& f) const {
        f(0);
    }
};

}

#endif /* Macro_h */
