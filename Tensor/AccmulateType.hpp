//
//  AccmulateType.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/18.
//

#ifndef AccmulateType_hpp
#define AccmulateType_hpp

#include "ScalarType.hpp"
#include "HFloat.hpp"

namespace otter {

template <typename T, bool is_cuda>
struct AccumulateType {};
#if defined(__CUDACC__) || defined(__HIPCC__)
template <>
struct AccumulateType<half, true> {
    using type = float;
};
#endif
template <>
struct AccumulateType<HFloat, true> {
    using type = float;
};
template <>
struct AccumulateType<float, true> {
    using type = float;
};
template <>
struct AccumulateType<double, true> {
    using type = double;
};
template <>
struct AccumulateType<int8_t, true> {
    using type = int64_t;
};
template <>
struct AccumulateType<uint8_t, true> {
    using type = int64_t;
};
template <>
struct AccumulateType<char, true> {
    using type = int64_t;
};
template <>
struct AccumulateType<int16_t, true> {
    using type = int64_t;
};
template <>
struct AccumulateType<int32_t, true> {
    using type = int64_t;
};
template <>
struct AccumulateType<int64_t, true> {
    using type = int64_t;
};
template <>
struct AccumulateType<bool, true> {
    using type = bool;
};
template <>
struct AccumulateType<HFloat, false> {
    using type = float;
};
template <>
struct AccumulateType<float, false> {
    using type = double;
};
template <>
struct AccumulateType<double, false> {
    using type = double;
};
template <>
struct AccumulateType<int8_t, false> {
    using type = int64_t;
};
template <>
struct AccumulateType<uint8_t, false> {
    using type = int64_t;
};
template <>
struct AccumulateType<char, false> {
    using type = int64_t;
};
template <>
struct AccumulateType<int16_t, false> {
    using type = int64_t;
};
template <>
struct AccumulateType<int32_t, false> {
    using type = int64_t;
};
template <>
struct AccumulateType<int64_t, false> {
    using type = int64_t;
};
template <>
struct AccumulateType<bool, false> {
    using type = bool;
};
template <typename T, bool is_cuda>
using acc_type = typename AccumulateType<T, is_cuda>::type;

ScalarType toAccumulateType(ScalarType type, bool is_cuda);

}   // end namespace otter

#endif /* AccmulateType_hpp */
