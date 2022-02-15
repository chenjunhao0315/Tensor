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
#include "TensorResize.hpp"
#include "TensorCopy.hpp"
#include "TensorFactory.hpp"
#include "TensorShape.hpp"
#include "TensorBlas.hpp"
#include "TensorProperties.hpp"

namespace otter {

Tensor Tensor::select(int64_t dim, int64_t index) const {
    return otter::native::select(*this, dim, index);
}

Tensor Tensor::operator[](int64_t index) const {
    return this->select(0, index);
}

Tensor& Tensor::copy_(const Tensor &src, bool non_blocking) const {
    return otter::copy_(const_cast<Tensor&>(*this), src, non_blocking);
}

Tensor Tensor::clone() const {
    return otter::clone(*this);
}

Tensor Tensor::clone(MemoryFormat memory_format) const {
    return otter::clone(*this, memory_format);
}

Tensor Tensor::contiguous(MemoryFormat memory_format) const {
    if (is_contiguous()) {
        return *this;
    } else {
        return otter::contiguous(*this, memory_format);
    }
}

Tensor Tensor::permute(IntArrayRef dims) const {
    return otter::native::permute(*this, dims);
};

Tensor& Tensor::transpose_(int64_t dim0, int64_t dim1) const {
    return otter::native::transpose_(const_cast<Tensor&>(*this), dim0, dim1);
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    return otter::native::transpose(*this, dim0, dim1);
}

Tensor Tensor::expand(IntArrayRef sizes) const {
    return otter::native::expand(*this, sizes);
}

Tensor Tensor::expand_as(const Tensor &other) const {
    return otter::native::expand_as(*this, other);
}

const Tensor& Tensor::resize_(IntArrayRef shape) const {
    return otter::native::resize_(*this, shape);
}

const Tensor& Tensor::resize_as_(const Tensor& the_template) const {
    return otter::native::resize_as_(*this, the_template);
}

Tensor Tensor::as_strided(IntArrayRef sizes, IntArrayRef strides) const {
    return otter::native::as_strided_tensorimpl(*this, sizes, strides);
}

Tensor Tensor::as_strided(IntArrayRef sizes, IntArrayRef strides, int64_t memory_offset) const {
    return otter::native::as_strided_tensorimpl(*this, sizes, strides, memory_offset);
}

const Tensor& Tensor::as_strided_(IntArrayRef sizes, IntArrayRef strides) const {
    return otter::native::as_strided_(*this, sizes, strides);
}

const Tensor& Tensor::as_strided_(IntArrayRef sizes, IntArrayRef strides, int64_t memory_offset) const {
    return otter::native::as_strided_(*this, sizes, strides, memory_offset);
}

Tensor Tensor::view(IntArrayRef sizes) const {
    return otter::native::view(*this, sizes);
}

Tensor& Tensor::zero_() {
    return native::zero_(*this);
}

Tensor& Tensor::fill_(const Scalar &value) {
    return native::fill_out(*this, value);
}

Tensor Tensor::to(ScalarType dtype) const {
    return native::to(*this, dtype);
}

Tensor& Tensor::add_(const Tensor &other, const Scalar &alpha) const {
    return otter::cpu::add_(const_cast<Tensor&>(*this), other, alpha);
}

Tensor Tensor::add(const Tensor& other, const Scalar& alpha) const {
    return otter::cpu::add(*this, other, alpha);
}

Tensor& Tensor::add_(const Scalar &other, const Scalar &alpha) const {
    return add_(scalar_to_tensor(other), alpha);
}

Tensor Tensor::add(const Scalar& other, const Scalar& alpha) const {
    return add(scalar_to_tensor(other), alpha);
}

Tensor& Tensor::sub_(const Tensor &other, const Scalar &alpha) const {
    return otter::cpu::sub_(const_cast<Tensor&>(*this), other, alpha);
}

Tensor Tensor::sub(const Tensor& other, const Scalar& alpha) const {
    return otter::cpu::sub(*this, other, alpha);
}

Tensor& Tensor::sub_(const Scalar &other, const Scalar &alpha) const {
    return sub_(scalar_to_tensor(other), alpha);
}

Tensor Tensor::sub(const Scalar& other, const Scalar& alpha) const {
    return sub(scalar_to_tensor(other), alpha);
}

Tensor& Tensor::mul_(const Tensor &other) const {
    return otter::cpu::mul_(const_cast<Tensor&>(*this), other);
}

Tensor Tensor::mul(const Tensor& other) const {
    return otter::cpu::mul(*this, other);
}

Tensor& Tensor::mul_(const Scalar &other) const {
    return mul_(scalar_to_tensor(other));
}

Tensor Tensor::mul(const Scalar& other) const {
    return mul(scalar_to_tensor(other));
}

Tensor& Tensor::div_(const Tensor &other) const {
    return otter::cpu::div_(const_cast<Tensor&>(*this), other);
}

Tensor Tensor::div(const Tensor& other) const {
    return otter::cpu::div(*this, other);
}

Tensor& Tensor::div_(const Scalar &other) const {
    return div_(scalar_to_tensor(other));
}

Tensor Tensor::div(const Scalar& other) const {
    return div(scalar_to_tensor(other));
}

Tensor& Tensor::remainder_(const Tensor &other) const {
    return otter::cpu::remainder_(const_cast<Tensor&>(*this), other);
}

Tensor Tensor::remainder(const Tensor& other) const {
    return otter::cpu::remainder(*this, other);
}

Tensor& Tensor::remainder_(const Scalar &other) const {
    return remainder_(scalar_to_tensor(other));
}

Tensor Tensor::remainder(const Scalar& other) const {
    return remainder(scalar_to_tensor(other));
}

Tensor& Tensor::bitwise_and_(const Tensor &other) const {
    return otter::cpu::bitwise_and_(const_cast<Tensor&>(*this), other);
}

Tensor Tensor::bitwise_and(const Tensor& other) const {
    return otter::cpu::bitwise_and(*this, other);
}

Tensor& Tensor::bitwise_and_(const Scalar &other) const {
    return bitwise_and_(scalar_to_tensor(other));
}

Tensor Tensor::bitwise_and(const Scalar& other) const {
    return bitwise_and(scalar_to_tensor(other));
}

Tensor& Tensor::bitwise_or_(const Tensor &other) const {
    return otter::cpu::bitwise_or_(const_cast<Tensor&>(*this), other);
}

Tensor Tensor::bitwise_or(const Tensor& other) const {
    return otter::cpu::bitwise_or(*this, other);
}

Tensor& Tensor::bitwise_or_(const Scalar &other) const {
    return bitwise_or_(scalar_to_tensor(other));
}

Tensor Tensor::bitwise_or(const Scalar& other) const {
    return bitwise_or(scalar_to_tensor(other));
}

Tensor& Tensor::bitwise_xor_(const Tensor &other) const {
    return otter::cpu::bitwise_xor_(const_cast<Tensor&>(*this), other);
}

Tensor Tensor::bitwise_xor(const Tensor& other) const {
    return otter::cpu::bitwise_xor(*this, other);
}

Tensor& Tensor::bitwise_xor_(const Scalar &other) const {
    return bitwise_xor_(scalar_to_tensor(other));
}

Tensor Tensor::bitwise_xor(const Scalar& other) const {
    return bitwise_xor(scalar_to_tensor(other));
}

Tensor& Tensor::bitwise_not_() const {
    return otter::cpu::bitwise_not_(const_cast<Tensor&>(*this));
}

Tensor Tensor::bitwise_not() const {
    return otter::cpu::bitwise_not(*this);
}

Tensor& Tensor::neg_() const {
    return otter::cpu::neg_(const_cast<Tensor&>(*this));
}

Tensor Tensor::neg() const {
    return otter::cpu::neg(*this);
}

Tensor& Tensor::abs_() const {
    return otter::cpu::abs_(const_cast<Tensor&>(*this));
}

Tensor Tensor::abs() const {
    return otter::cpu::abs(*this);
}

Tensor& Tensor::sin_() const {
    return otter::cpu::sin_(const_cast<Tensor&>(*this));
}

Tensor Tensor::sin() const {
    return otter::cpu::sin(*this);
}

Tensor& Tensor::cos_() const {
    return otter::cpu::cos_(const_cast<Tensor&>(*this));
}

Tensor Tensor::cos() const {
    return otter::cpu::cos(*this);
}

Tensor& Tensor::tan_() const {
    return otter::cpu::tan_(const_cast<Tensor&>(*this));
}

Tensor Tensor::tan() const {
    return otter::cpu::tan(*this);
}

Tensor& Tensor::exp_() const {
    return otter::cpu::exp_(const_cast<Tensor&>(*this));
}

Tensor Tensor::exp() const {
    return otter::cpu::exp(*this);
}

Tensor Tensor::dot(const Tensor& other) const {
    return otter::dot(*this, other);
}

Tensor Tensor::addmm(const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) const {
    return otter::cpu::addmm(*this, mat1, mat2, beta, alpha);
}

Tensor& Tensor::addmm_(const Tensor &mat1, const Tensor &mat2, const Scalar& beta, const Scalar& alpha) const {
    return otter::cpu::addmm_(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
}

Tensor Tensor::mm(const Tensor &other) const {
    return otter::cpu::mm(*this, other);
}


}   // namespace otter
