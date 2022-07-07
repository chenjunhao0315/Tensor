//
//  UnaryOps.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/6.
//

#ifndef UnaryOps_hpp
#define UnaryOps_hpp

#include "Generator.hpp"
#include "DispatchStub.hpp"

namespace otter {

class TensorBase;
class TensorIterator;
class Scalar;

using unary_fn = void(*)(TensorIterator&);
using unary_fn_with_scalar = void(*)(TensorIterator&, const Scalar& a);

DECLARE_DISPATCH(unary_fn, bitwise_not_stub);
DECLARE_DISPATCH(unary_fn, neg_stub);
DECLARE_DISPATCH(unary_fn, abs_stub);
DECLARE_DISPATCH(unary_fn, sin_stub);
DECLARE_DISPATCH(unary_fn, cos_stub);
DECLARE_DISPATCH(unary_fn, tan_stub);
DECLARE_DISPATCH(unary_fn, exp_stub);
DECLARE_DISPATCH(unary_fn, sqrt_stub);
DECLARE_DISPATCH(unary_fn, sigmoid_stub);

DECLARE_DISPATCH(void(*)(TensorIterator&, const double, const double, Generator), uniform_stub);
DECLARE_DISPATCH(void(*)(const TensorBase&, const double, const double, Generator), normal_stub);
DECLARE_DISPATCH(void(*)(TensorIterator&, const uint64_t, const int64_t, Generator), random_from_to_stub);
DECLARE_DISPATCH(void(*)(TensorIterator&, Generator), random_full_64_bits_range_stub);
DECLARE_DISPATCH(void(*)(TensorIterator&, Generator), random_stub);

}

#endif /* UnaryOps_hpp */
