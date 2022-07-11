//
//  zmath.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/11.
//

#ifndef zmath_h
#define zmath_h

namespace otter {

template <typename TYPE>
inline TYPE floor_impl (TYPE z) {
    return std::floor(z);
}

template <typename TYPE>
inline TYPE round_impl (TYPE z) {
    return std::nearbyint(z);
}

template <typename TYPE>
inline TYPE trunc_impl (TYPE z) {
    return std::trunc(z);
}

}   // end namespace otter

#endif /* zmath_h */
