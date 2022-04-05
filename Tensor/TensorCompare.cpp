//
//  TensorCompare.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/6.
//

#include "TensorCompare.hpp"
#include "TensorFunction.hpp"
#include "TensorFactory.hpp"
#include "TensorIterator.hpp"
#include "ScalarOps.hpp"

namespace otter {

DEFINE_DISPATCH(clamp_stub);
DEFINE_DISPATCH(clamp_min_stub);
DEFINE_DISPATCH(clamp_max_stub);
DEFINE_DISPATCH(clamp_scalar_stub);
DEFINE_DISPATCH(clamp_min_scalar_stub);
DEFINE_DISPATCH(clamp_max_scalar_stub);

// Maybe be abandoned
DEFINE_META_FUNCTION(clamp) (const Tensor& self, const Scalar min, Scalar max) {
    build_borrowing_unary_op(maybe_get_output(), self);
}

DEFINE_IMPL_FUNCTION(clamp_out)(const Tensor& /*self*/, const Scalar min, Scalar max, const Tensor& /*result*/) {
    clamp_scalar_stub(Device::CPU, *this, min, max);
}

Tensor& clamp_out(const Tensor& self, const Tensor& min, const Tensor& max, Tensor& result) {
    auto iter = TensorIteratorConfig()
        .set_check_mem_overlap(true)
        .add_output(result)
        .add_input(self)
        .add_input(min)
        .add_input(max)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .enforce_safe_casting_to_output(true)
        .build();
    clamp_stub(Device::CPU, iter);
    return result;
}

Tensor clamp(const Tensor& self, const Scalar& min, const Scalar& max) {
    Tensor result = otter::empty_like(self);
    return clamp_out(self, native::wrapped_scalar_tensor(min), native::wrapped_scalar_tensor(max), result);
}

Tensor clamp(const Tensor& self, const Tensor& min, const Tensor& max) {
    Tensor result = otter::empty_like(self);
    return clamp_out(self, min, max, result);
}

Tensor& clamp_(Tensor& self, const Scalar& min, const Scalar& max) {
    return clamp_out(self, native::wrapped_scalar_tensor(min), native::wrapped_scalar_tensor(max), self);
}

Tensor& clamp_(Tensor& self, const Tensor& min, const Tensor& max) {
    return clamp_out(self, min, max, self);
}

Tensor& clamp_max_out(const Tensor& self, const Scalar& max, Tensor& result) {
    auto iter = TensorIterator::unary_op(result, self);
    clamp_max_scalar_stub(Device::CPU, iter, max);
    return result;
}

Tensor& clamp_max_out(const Tensor& self, const Tensor& max, Tensor& result) {
    auto iter = TensorIterator::borrowing_binary_op(result, self, max);
    clamp_max_stub(Device::CPU, iter);
    return result;
}

Tensor clamp_max(const Tensor& self, const Scalar& max) {
    Tensor result = otter::empty_like(self);
    return clamp_max_out(self, max, result);
}

Tensor clamp_max(const Tensor& self, const Tensor& max) {
    Tensor result = otter::empty_like(self);
    return clamp_max_out(self, max, result);
}

Tensor& clamp_max_(Tensor& self, const Scalar& max) {
    return clamp_max_out(self, max, self);
}

Tensor& clamp_max_(Tensor& self, const Tensor& max) {
    return clamp_max_out(self, max, self);
}

Tensor& clamp_min_out(const Tensor& self, const Scalar& min, Tensor& result) {
    auto iter = TensorIterator::unary_op(result, self);
    clamp_min_scalar_stub(Device::CPU, iter, min);
    return result;
}

Tensor& clamp_min_out(const Tensor& self, const Tensor& min, Tensor& result) {
    auto iter = TensorIterator::borrowing_binary_op(result, self, min);
    clamp_min_stub(Device::CPU, iter);
    return result;
}

Tensor clamp_min(const Tensor& self, const Scalar& min) {
    Tensor result = otter::empty_like(self);
    return clamp_min_out(self, min, result);
}

Tensor clamp_min(const Tensor& self, const Tensor& min) {
    Tensor result = otter::empty_like(self);
    return clamp_min_out(self, min, result);
}

Tensor& clamp_min_(Tensor& self, const Scalar& min) {
    return clamp_min_out(self, min, self);
}

Tensor& clamp_min_(Tensor& self, const Tensor& min) {
    return clamp_min_out(self, min, self);
}


}
