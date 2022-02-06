//
//  BinaryOps.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/30.
//

#include "BinaryOps.hpp"
#include "TensorFunction.hpp"

namespace otter {

DEFINE_META_FUNCTION_OTHER(add, Tensor) (const Tensor& self, const Tensor& other, const Scalar& value) {
    build_borrowing_binary_op(maybe_get_output(), self, other);
}

DEFINE_META_FUNCTION_OTHER(sub, Tensor) (const Tensor& self, const Tensor& other, const Scalar& value) {
    build_borrowing_binary_op(maybe_get_output(), self, other);
}

DEFINE_META_FUNCTION_OTHER(mul, Tensor) (const Tensor& self, const Tensor& other) {
    build_borrowing_binary_op(maybe_get_output(), self, other);
}

DEFINE_META_FUNCTION_OTHER(div, Tensor) (const Tensor& self, const Tensor& other) {
    build_borrowing_binary_op(maybe_get_output(), self, other);
}


DEFINE_DISPATCH(add_stub);
DEFINE_DISPATCH(sub_stub);
DEFINE_DISPATCH(mul_stub);
DEFINE_DISPATCH(div_true_stub);

}
