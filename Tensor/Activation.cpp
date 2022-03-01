//
//  Activation.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/1.
//

#include "Activation.hpp"
#include "TensorFunction.hpp"

namespace otter {

DEFINE_DISPATCH(leaky_relu_stub);

DEFINE_META_FUNCTION(leaky_relu) (const Tensor& self, const Scalar& negative_slope) {
    build_unary_op(maybe_get_output(), self);
}

DEFINE_IMPL_FUNCTION(leaky_relu_out) (const Tensor& self, const Scalar& negval, const Tensor& result) {
    leaky_relu_stub(Device::CPU, *this, negval);
}

}
