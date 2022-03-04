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

namespace otter {

template <typename T>
static inline T div_round_up(T x, T y) {
    int q = x / y;
    int r = x % y;
    if ((r!=0) && ((r<0) != (y<0))) --q;
    return q;
}

template <typename T>
static T abs_impl(T v) {
    return std::abs(v);
}

template <>
OTTER_UNUSED uint8_t abs_impl(uint8_t v) {
    return v;
}



}

#endif /* Math_hpp */
