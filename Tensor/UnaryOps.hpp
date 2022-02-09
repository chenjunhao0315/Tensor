//
//  UnaryOps.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/6.
//

#ifndef UnaryOps_hpp
#define UnaryOps_hpp

#include "DispatchStub.hpp"
#include "Scalar.hpp"
#include "TensorIterator.hpp"

namespace otter {

using unary_fn = void(*)(TensorIterator&);
using unary_fn_with_scalar = void(*)(TensorIterator&, const Scalar& a);

DECLARE_DISPATCH(unary_fn, bitwise_not_stub);
DECLARE_DISPATCH(unary_fn, neg_stub);
DECLARE_DISPATCH(unary_fn, abs_stub);
DECLARE_DISPATCH(unary_fn, sin_stub);
DECLARE_DISPATCH(unary_fn, cos_stub);
DECLARE_DISPATCH(unary_fn, tan_stub);
DECLARE_DISPATCH(unary_fn, exp_stub);

#define DECLEAR_UNARY_SELF(name) \
Tensor name(const Tensor& self); \
Tensor& name##_(Tensor& self)

DECLEAR_UNARY_SELF(neg);
DECLEAR_UNARY_SELF(bitwise_not);
DECLEAR_UNARY_SELF(abs);
DECLEAR_UNARY_SELF(sin);
DECLEAR_UNARY_SELF(cos);
DECLEAR_UNARY_SELF(tan);
DECLEAR_UNARY_SELF(exp);


}

#endif /* UnaryOps_hpp */
