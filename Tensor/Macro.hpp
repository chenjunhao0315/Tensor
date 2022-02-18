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


#endif /* Macro_h */
