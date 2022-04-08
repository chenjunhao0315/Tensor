//
//  BinaryOps.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/30.
//

#include "BinaryOps.hpp"
#include "TensorFunction.hpp"
#include "Dispatch.hpp"
#include "ScalarOps.hpp"

namespace otter {

namespace native {
static void check_convert(const Scalar& scalar, ScalarType scalarType) {
    // Validate that is possible to convert scalar to tensor dtype without
    // overflow
    OTTER_DISPATCH_ALL_TYPES_AND(
      otter::ScalarType::Bool,
      scalarType,
      "check_convert",
      [&] { scalar.to<scalar_t>(); });
}

static Tensor wrapped_scalar_tensor_and_check_convert(const Scalar& scalar, Tensor tensor) {
    check_convert(scalar, tensor.scalar_type());
    return otter::native::wrapped_scalar_tensor(scalar);
}

}   // end namespace native

DEFINE_META_FUNCTION_OVERLOAD(add, Tensor) (const Tensor& self, const Tensor& other, const Scalar& /*value*/) {
    build_borrowing_binary_op(maybe_get_output(), self, other);
}

DEFINE_META_FUNCTION_OVERLOAD(sub, Tensor) (const Tensor& self, const Tensor& other, const Scalar& /*value*/) {
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

void comparison_op_check(const Tensor& self, const Tensor& other, const Tensor& /*result*/) {
    // Validate that is possible to convert zero-dim tensor's dtype to other dtype
    // without overflow
    if (self.scalar_type() != other.scalar_type()) {
        if (self.dim() != 0 && other.dim() == 0) {
            native::check_convert(other.item(), self.scalar_type());
        } else if (self.dim() == 0 && other.dim() != 0) {
            native::check_convert(self.item(), other.scalar_type());
        }
    }
}

#define CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(func)                     \
  DEFINE_META_FUNCTION_OVERLOAD(func, Tensor)(const Tensor& self, const Tensor& other) { \
    const Tensor& result = maybe_get_output();                              \
    comparison_op_check(self, other, result);                               \
    build_borrowing_comparison_op(result, self, other);                     \
  }                                                                         \
                                                                            \
  DEFINE_META_FUNCTION_OVERLOAD(func, Scalar)(const Tensor& self, const Scalar& other) { \
    auto other_tensor =                                                     \
        native::wrapped_scalar_tensor_and_check_convert(other, self);       \
    build_borrowing_except_last_argument_comparison_op(maybe_get_output(), self, other_tensor);  \
  }

CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(eq);
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(ne);
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(lt);
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(le);
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(gt);
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(ge);


DEFINE_DISPATCH(add_stub);
DEFINE_DISPATCH(sub_stub);
DEFINE_DISPATCH(add_clamp_stub);
DEFINE_DISPATCH(mul_stub);
DEFINE_DISPATCH(div_true_stub);
DEFINE_DISPATCH(remainder_stub);
DEFINE_DISPATCH(bitwise_and_stub);
DEFINE_DISPATCH(bitwise_or_stub);
DEFINE_DISPATCH(bitwise_xor_stub);

DEFINE_DISPATCH(lt_stub);
DEFINE_DISPATCH(le_stub);
DEFINE_DISPATCH(gt_stub);
DEFINE_DISPATCH(ge_stub);
DEFINE_DISPATCH(eq_stub);
DEFINE_DISPATCH(ne_stub);

DEFINE_IMPL_FUNCTION(add_out) (const Tensor& /*self*/, const Tensor& /*other*/, const Scalar& alpha, const Tensor& /*result*/) {
    add_stub(Device::CPU, *this, alpha);
    assert(result.scalar_type() == output().scalar_type());
}

DEFINE_IMPL_FUNCTION(sub_out) (const Tensor& /*self*/, const Tensor& /*other*/, const Scalar& alpha, const Tensor& /*result*/) {
    sub_stub(Device::CPU, *this, alpha);
    assert(result.scalar_type() == output().scalar_type());
}

DEFINE_IMPL_FUNCTION(mul_out) (const Tensor& /*self*/, const Tensor& /*other*/, const Tensor& /*result*/) {
    mul_stub(Device::CPU, *this);
    assert(result.scalar_type() == output().scalar_type());
}

DEFINE_IMPL_FUNCTION(div_out) (const Tensor& /*self*/, const Tensor& /*other*/, const Tensor& /*result*/) {
    div_true_stub(Device::CPU, *this);
    assert(result.scalar_type() == output().scalar_type());
}

DEFINE_IMPL_FUNCTION(remainder_out) (const Tensor& /*self*/, const Tensor& /*other*/, const Tensor& /*result*/) {
    remainder_stub(Device::CPU, *this);
    assert(result.scalar_type() == output().scalar_type());
}

DEFINE_IMPL_FUNCTION(bitwise_and_out) (const Tensor& /*self*/, const Tensor& /*other*/, const Tensor& /*result*/) {
    bitwise_and_stub(Device::CPU, *this);
    assert(result.scalar_type() == output().scalar_type());
}

DEFINE_IMPL_FUNCTION(bitwise_or_out) (const Tensor& /*self*/, const Tensor& /*other*/, const Tensor& /*result*/) {
    bitwise_or_stub(Device::CPU, *this);
    assert(result.scalar_type() == output().scalar_type());
}

DEFINE_IMPL_FUNCTION(bitwise_xor_out) (const Tensor& /*self*/, const Tensor& /*other*/, const Tensor& /*result*/) {
    bitwise_xor_stub(Device::CPU, *this);
    assert(result.scalar_type() == output().scalar_type());
}

#define CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(func)             \
  DEFINE_IMPL_FUNCTION(func##_Tensor_out)                                \
  (const Tensor& /*self*/, const Tensor& /*other*/, const Tensor& /*result*/) { \
    func##_stub(Device::CPU, *this);                              \
  }                                                                 \
                                                                        \
  DEFINE_IMPL_FUNCTION(func##_Scalar_out)                                \
  (const Tensor& /*self*/, const Scalar& /*other*/, const Tensor& /*result*/) { \
    func##_stub(Device::CPU, *this);                              \
  }

CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(eq);
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(ne);
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(gt);
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(ge);
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(lt);
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(le);

}   // end namespace otter
