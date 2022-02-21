//
//  TensorOperator.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/5.
//

#ifndef TensorOperator_hpp
#define TensorOperator_hpp

#include "UnaryOps.hpp"
#include "TensorFactory.hpp"

namespace otter {

#define AT_FORALL_BINARY_OPS(_) \
_(+,x.add(y), y.add(x)) \
_(*,x.mul(y), y.mul(x)) \
_(-,x.sub(y), otter::empty_like(y).fill_(x).sub_(y)) \
_(/,x.div(y), otter::empty_like(y).fill_(x).div_(y)) \
_(%,x.remainder(y), otter::empty_like(y).fill_(x).remainder_(y)) \
_(&,x.bitwise_and(y), y.bitwise_and(x)) \
_(|,x.bitwise_or(y), y.bitwise_or(x)) \
_(^,x.bitwise_xor(y), y.bitwise_xor(x))
//_(<,x.lt(y), y.gt(x)) \
//_(<=,x.le(y), y.ge(x)) \
//_(>,x.gt(y),y.lt(x)) \
//_(>=,x.ge(y), y.le(x)) \
//_(==,x.eq(y), y.eq(x)) \
//_(!=,x.ne(y), y.ne(x))

#define DEFINE_OPERATOR(op, body, reverse_scalar_body) \
static inline Tensor operator op(const Tensor& x, const Tensor& y) { \
  return body; \
} \
static inline Tensor operator op(const Tensor& x, const Scalar& y) { \
  return body; \
} \
static inline Tensor operator op(const Scalar& x, const Tensor & y) { \
  return reverse_scalar_body; \
}

AT_FORALL_BINARY_OPS(DEFINE_OPERATOR)
#undef DEFINE_OPERATOR
#undef AT_FORALL_BINARY_OPS

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
DECLEAR_UNARY_SELF(sqrt);

}   // end namespace otter

#endif /* TensorOperator_hpp */
