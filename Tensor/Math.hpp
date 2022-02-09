//
//  Math.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/9.
//

#ifndef Math_hpp
#define Math_hpp

#include "cmath"

namespace otter {

template <typename T>
static T abs_impl(T v) {
    return std::abs(v);
}

template <>
uint8_t abs_impl(uint8_t v) {
    return v;
}



}

#endif /* Math_hpp */
