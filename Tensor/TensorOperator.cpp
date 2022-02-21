//
//  TensorOperator.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/5.
//

#include "TensorOperator.hpp"

namespace otter {

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
DEFINE_UNARY_FUNCTION_SELF(sqrt);

}   // end namespace otter
