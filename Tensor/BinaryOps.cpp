//
//  BinaryOps.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/30.
//

#include "BinaryOps.hpp"
#include "TensorFunction.hpp"

namespace otter {

DEFINE_META_FUNCTION_OVERLOAD(add, Tensor) (const Tensor& self, const Tensor& other, const Scalar& value) {
    build_borrowing_binary_op(maybe_get_output(), self, other);
}

DEFINE_META_FUNCTION_OVERLOAD(sub, Tensor) (const Tensor& self, const Tensor& other, const Scalar& value) {
    build_borrowing_binary_op(maybe_get_output(), self, other);
}

DEFINE_META_FUNCTION_OVERLOAD(mul, Tensor) (const Tensor& self, const Tensor& other) {
    build_borrowing_binary_op(maybe_get_output(), self, other);
}

DEFINE_META_FUNCTION_OVERLOAD(div, Tensor) (const Tensor& self, const Tensor& other) {
    build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

DEFINE_META_FUNCTION_OVERLOAD(remainder, Tensor) (const Tensor& self, const Tensor& other) {
    build_borrowing_binary_op(maybe_get_output(), self, other);
}

DEFINE_META_FUNCTION_OVERLOAD(bitwise_and, Tensor) (const Tensor& self, const Tensor& other) {
    build_borrowing_binary_op(maybe_get_output(), self, other);
}

DEFINE_META_FUNCTION_OVERLOAD(bitwise_or, Tensor) (const Tensor& self, const Tensor& other) {
    build_borrowing_binary_op(maybe_get_output(), self, other);
}

DEFINE_META_FUNCTION_OVERLOAD(bitwise_xor, Tensor) (const Tensor& self, const Tensor& other) {
    build_borrowing_binary_op(maybe_get_output(), self, other);
}


DEFINE_DISPATCH(add_stub);
DEFINE_DISPATCH(sub_stub);
DEFINE_DISPATCH(mul_stub);
DEFINE_DISPATCH(div_true_stub);
DEFINE_DISPATCH(remainder_stub);
DEFINE_DISPATCH(bitwise_and_stub);
DEFINE_DISPATCH(bitwise_or_stub);
DEFINE_DISPATCH(bitwise_xor_stub);

DEFINE_IMPL_FUNCTION(add_out) (const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& result) {
    add_stub(Device::CPU, *this, alpha);
    assert(result.scalar_type() == output().scalar_type());
}

DEFINE_IMPL_FUNCTION(sub_out) (const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& result) {
    sub_stub(Device::CPU, *this, alpha);
    assert(result.scalar_type() == output().scalar_type());
}

DEFINE_IMPL_FUNCTION(mul_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
    mul_stub(Device::CPU, *this);
    assert(result.scalar_type() == output().scalar_type());
}

DEFINE_IMPL_FUNCTION(div_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
    div_true_stub(Device::CPU, *this);
    assert(result.scalar_type() == output().scalar_type());
}

DEFINE_IMPL_FUNCTION(remainder_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
    remainder_stub(Device::CPU, *this);
    assert(result.scalar_type() == output().scalar_type());
}

DEFINE_IMPL_FUNCTION(bitwise_and_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
    bitwise_and_stub(Device::CPU, *this);
    assert(result.scalar_type() == output().scalar_type());
}

DEFINE_IMPL_FUNCTION(bitwise_or_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
    bitwise_or_stub(Device::CPU, *this);
    assert(result.scalar_type() == output().scalar_type());
}

DEFINE_IMPL_FUNCTION(bitwise_xor_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
    bitwise_xor_stub(Device::CPU, *this);
    assert(result.scalar_type() == output().scalar_type());
}

}
