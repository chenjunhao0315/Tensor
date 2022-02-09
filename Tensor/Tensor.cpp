//
//  Tensor.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "Tensor.hpp"
#include "Fill.hpp"
#include "UnaryOps.hpp"
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

Tensor& Tensor::remainder_(const Tensor &other) {
    structured_remainder_Tensor_out op(*this);
    op.meta(*this, other);
    remainder_stub(Device::CPU, op);
    
    return *this;
}

Tensor Tensor::remainder(const Tensor& other) const {
    structured_remainder_Tensor_functional op;
    op.meta(*this, other);
    remainder_stub(Device::CPU, op);
    
    return std::move(op.outputs_[0]).take();
}

Tensor& Tensor::remainder_(const Scalar &other) {
    return remainder_(scalar_to_tensor(other));
}

Tensor Tensor::remainder(const Scalar& other) const {
    return remainder(scalar_to_tensor(other));
}

Tensor& Tensor::bitwise_and_(const Tensor &other) {
    structured_bitwise_and_Tensor_out op(*this);
    op.meta(*this, other);
    bitwise_and_stub(Device::CPU, op);
    
    return *this;
}

Tensor Tensor::bitwise_and(const Tensor& other) const {
    structured_bitwise_and_Tensor_functional op;
    op.meta(*this, other);
    bitwise_and_stub(Device::CPU, op);
    
    return std::move(op.outputs_[0]).take();
}

Tensor& Tensor::bitwise_and_(const Scalar &other) {
    return bitwise_and_(scalar_to_tensor(other));
}

Tensor Tensor::bitwise_and(const Scalar& other) const {
    return bitwise_and(scalar_to_tensor(other));
}

Tensor& Tensor::bitwise_or_(const Tensor &other) {
    structured_bitwise_or_Tensor_out op(*this);
    op.meta(*this, other);
    bitwise_or_stub(Device::CPU, op);
    
    return *this;
}

Tensor Tensor::bitwise_or(const Tensor& other) const {
    structured_bitwise_or_Tensor_functional op;
    op.meta(*this, other);
    bitwise_or_stub(Device::CPU, op);
    
    return std::move(op.outputs_[0]).take();
}

Tensor& Tensor::bitwise_or_(const Scalar &other) {
    return bitwise_or_(scalar_to_tensor(other));
}

Tensor Tensor::bitwise_or(const Scalar& other) const {
    return bitwise_or(scalar_to_tensor(other));
}

Tensor& Tensor::bitwise_xor_(const Tensor &other) {
    structured_bitwise_xor_Tensor_out op(*this);
    op.meta(*this, other);
    bitwise_xor_stub(Device::CPU, op);
    
    return *this;
}

Tensor Tensor::bitwise_xor(const Tensor& other) const {
    structured_bitwise_xor_Tensor_functional op;
    op.meta(*this, other);
    bitwise_xor_stub(Device::CPU, op);
    
    return std::move(op.outputs_[0]).take();
}

Tensor& Tensor::bitwise_xor_(const Scalar &other) {
    return bitwise_xor_(scalar_to_tensor(other));
}

Tensor Tensor::bitwise_xor(const Scalar& other) const {
    return bitwise_xor(scalar_to_tensor(other));
}

Tensor& Tensor::bitwise_not_() {
    structured_bitwise_not_Tensor_out op(*this);
    op.meta(*this);
    bitwise_not_stub(Device::CPU, op);
    
    return *this;
}

Tensor Tensor::bitwise_not() const {
    structured_bitwise_not_Tensor_functional op;
    op.meta(*this);
    bitwise_not_stub(Device::CPU, op);
    
    return std::move(op.outputs_[0]).take();
}

Tensor& Tensor::neg_() {
    structured_neg_Tensor_out op(*this);
    op.meta(*this);
    neg_stub(Device::CPU, op);
    
    return *this;
}

Tensor Tensor::neg() const {
    structured_neg_Tensor_functional op;
    op.meta(*this);
    neg_stub(Device::CPU, op);
    
    return std::move(op.outputs_[0]).take();
}

Tensor& Tensor::abs_() {
    structured_abs_Tensor_out op(*this);
    op.meta(*this);
    abs_stub(Device::CPU, op);
    
    return *this;
}

Tensor Tensor::abs() const {
    structured_abs_Tensor_functional op;
    op.meta(*this);
    abs_stub(Device::CPU, op);
    
    return std::move(op.outputs_[0]).take();
}

Tensor& Tensor::sin_() {
    structured_sin_Tensor_out op(*this);
    op.meta(*this);
    sin_stub(Device::CPU, op);
    
    return *this;
}

Tensor Tensor::sin() const {
    structured_sin_Tensor_functional op;
    op.meta(*this);
    sin_stub(Device::CPU, op);
    
    return std::move(op.outputs_[0]).take();
}

Tensor& Tensor::cos_() {
    structured_cos_Tensor_out op(*this);
    op.meta(*this);
    cos_stub(Device::CPU, op);
    
    return *this;
}

Tensor Tensor::cos() const {
    structured_cos_Tensor_functional op;
    op.meta(*this);
    cos_stub(Device::CPU, op);
    
    return std::move(op.outputs_[0]).take();
}

Tensor& Tensor::tan_() {
    structured_tan_Tensor_out op(*this);
    op.meta(*this);
    tan_stub(Device::CPU, op);
    
    return *this;
}

Tensor Tensor::tan() const {
    structured_tan_Tensor_functional op;
    op.meta(*this);
    tan_stub(Device::CPU, op);
    
    return std::move(op.outputs_[0]).take();
}

Tensor& Tensor::exp_() {
    structured_exp_Tensor_out op(*this);
    op.meta(*this);
    exp_stub(Device::CPU, op);
    
    return *this;
}

Tensor Tensor::exp() const {
    structured_exp_Tensor_functional op;
    op.meta(*this);
    exp_stub(Device::CPU, op);
    
    return std::move(op.outputs_[0]).take();
}


}   // namespace otter
