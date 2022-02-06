//
//  Tensor.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "Tensor.hpp"
#include "Fill.hpp"
#include "BinaryOps.hpp"
#include "ScalarOps.hpp"
#include "TensorFunction.hpp"
#include "TensorConversion.hpp"

namespace otter {

Tensor& Tensor::zero_() {
    return native::zero_(*this);
}

Tensor& Tensor::fill_(const Scalar &value) {
    return native::fill_out(*this, value);
}

Tensor Tensor::to(ScalarType dtype) const {
    return native::to(*this, dtype);
}

Tensor& Tensor::add_(const Tensor &other, const Scalar &alpha) {
    structured_add_Tensor_out op(*this);
    op.meta(*this, other, alpha);
    add_stub(Device::CPU, op, alpha);
    
    return *this;
}

Tensor Tensor::add(const Tensor& other, const Scalar& alpha) const {
    structured_add_Tensor_functional op;
    op.meta(*this, other, alpha);
    add_stub(Device::CPU, op, alpha);
    
    return std::move(op.outputs_[0]).take();
}

Tensor& Tensor::add_(const Scalar &other, const Scalar &alpha) {
    return add_(scalar_to_tensor(other), alpha);
}

Tensor Tensor::add(const Scalar& other, const Scalar& alpha) const {
    return add(scalar_to_tensor(other), alpha);
}

Tensor& Tensor::sub_(const Tensor &other, const Scalar &alpha) {
    structured_sub_Tensor_out op(*this);
    op.meta(*this, other, alpha);
    sub_stub(Device::CPU, op, alpha);
    
    return *this;
}

Tensor Tensor::sub(const Tensor& other, const Scalar& alpha) const {
    structured_sub_Tensor_functional op;
    op.meta(*this, other, alpha);
    sub_stub(Device::CPU, op, alpha);
    
    return std::move(op.outputs_[0]).take();
}

Tensor& Tensor::sub_(const Scalar &other, const Scalar &alpha) {
    return sub_(scalar_to_tensor(other), alpha);
}

Tensor Tensor::sub(const Scalar& other, const Scalar& alpha) const {
    return sub(scalar_to_tensor(other), alpha);
}

Tensor& Tensor::mul_(const Tensor &other) {
    structured_mul_Tensor_out op(*this);
    op.meta(*this, other);
    mul_stub(Device::CPU, op);
    
    return *this;
}

Tensor Tensor::mul(const Tensor& other) const {
    structured_mul_Tensor_functional op;
    op.meta(*this, other);
    mul_stub(Device::CPU, op);
    
    return std::move(op.outputs_[0]).take();
}

Tensor& Tensor::mul_(const Scalar &other) {
    return mul_(scalar_to_tensor(other));
}

Tensor Tensor::mul(const Scalar& other) const {
    return mul(scalar_to_tensor(other));
}

Tensor& Tensor::div_(const Tensor &other) {
    structured_div_Tensor_out op(*this);
    op.meta(*this, other);
    div_true_stub(Device::CPU, op);
    
    return *this;
}

Tensor Tensor::div(const Tensor& other) const {
    structured_div_Tensor_functional op;
    op.meta(*this, other);
    div_true_stub(Device::CPU, op);
    
    return std::move(op.outputs_[0]).take();
}

Tensor& Tensor::div_(const Scalar &other) {
    return div_(scalar_to_tensor(other));
}

Tensor Tensor::div(const Scalar& other) const {
    return div(scalar_to_tensor(other));
}


}   // namespace otter
