//
//  UnaryOps.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/6.
//

#include "UnaryOps.hpp"
#include "TensorFunction.hpp"

namespace otter {

#define DEFINE_UNARY_META_FUNCTION_SELF(name, overload) \
DEFINE_META_FUNCTION_OVERLOAD(name, overload) (const Tensor& self) { \
    build_borrowing_unary_float_op(maybe_get_output(), self); \
}

DEFINE_UNARY_META_FUNCTION_SELF(bitwise_not, Tensor);
DEFINE_UNARY_META_FUNCTION_SELF(neg, Tensor);
DEFINE_UNARY_META_FUNCTION_SELF(abs, Tensor);
DEFINE_UNARY_META_FUNCTION_SELF(sin, Tensor);
DEFINE_UNARY_META_FUNCTION_SELF(cos, Tensor);
DEFINE_UNARY_META_FUNCTION_SELF(tan, Tensor);
DEFINE_UNARY_META_FUNCTION_SELF(exp, Tensor);

DEFINE_DISPATCH(bitwise_not_stub);
DEFINE_DISPATCH(neg_stub);
DEFINE_DISPATCH(abs_stub);
DEFINE_DISPATCH(sin_stub);
DEFINE_DISPATCH(cos_stub);
DEFINE_DISPATCH(tan_stub);
DEFINE_DISPATCH(exp_stub);

#define DEFINE_UNARY_IMPL_FUNCTION(name, op) \
DEFINE_IMPL_FUNCTION(name) (const Tensor& self, const Tensor& out) { \
    op(Device::CPU, *this); \
}

DEFINE_UNARY_IMPL_FUNCTION(neg_out, neg_stub)
DEFINE_UNARY_IMPL_FUNCTION(bitwise_not_out, bitwise_not_stub)
DEFINE_UNARY_IMPL_FUNCTION(abs_out, abs_stub)
DEFINE_UNARY_IMPL_FUNCTION(sin_out, sin_stub)
DEFINE_UNARY_IMPL_FUNCTION(cos_out, cos_stub)
DEFINE_UNARY_IMPL_FUNCTION(tan_out, tan_stub)
DEFINE_UNARY_IMPL_FUNCTION(exp_out, exp_stub)

#define DEFINE_UNARY_FUNCTION_SELF(name) \
Tensor name(const Tensor& self) { return self.name(); } \
Tensor& name##_(Tensor& self) { return self.name##_(); }

DEFINE_UNARY_FUNCTION_SELF(bitwise_not);
DEFINE_UNARY_FUNCTION_SELF(neg);
DEFINE_UNARY_FUNCTION_SELF(abs);
DEFINE_UNARY_FUNCTION_SELF(sin);
DEFINE_UNARY_FUNCTION_SELF(cos);
DEFINE_UNARY_FUNCTION_SELF(tan);
DEFINE_UNARY_FUNCTION_SELF(exp);


}
