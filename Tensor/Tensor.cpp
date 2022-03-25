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
#include "TensorScalar.hpp"
#include "TensorDistribution.hpp"
#include "DType.hpp"

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

const Tensor& Tensor::resize_(IntArrayRef shape, MemoryFormat memory_format) const {
    return otter::native::resize_(*this, shape, memory_format);
}

const Tensor& Tensor::resize_as_(const Tensor& the_template, MemoryFormat memory_format) const {
    return otter::native::resize_as_(*this, the_template, memory_format);
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

Tensor Tensor::reshape(IntArrayRef sizes) const {
    return otter::native::reshape(*this, sizes);
}

Tensor Tensor::reshape_as(const Tensor &other) const {
    return otter::native::reshape_as(*this, other);
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
    return otter::native::slice(*this, dim, start, end, step);
}

Tensor Tensor::unsqueeze(int64_t dim) const {
    return otter::native::unsqueeze(*this, dim);
}

Tensor& Tensor::unsqueeze_(int64_t dim) const {
    return otter::native::unsqueeze_(const_cast<Tensor&>(*this), dim);
}

Tensor Tensor::squeeze(int64_t dim) const {
    return otter::native::squeeze(*this, dim);
}

Tensor& Tensor::squeeze_(int64_t dim) const {
    return otter::native::squeeze_(const_cast<Tensor&>(*this), dim);
}

Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
    return otter::native::narrow(*this, dim, start, length);
}

Tensor Tensor::unfold(int64_t dim, int64_t size, int64_t step) const {
    return otter::native::unfold(*this, dim, size, step);
}

Tensor Tensor::detach() const {
    return otter::native::detach(*this);
}

Tensor& Tensor::uniform_(double from, double to) const {
    return otter::native::uniform_(const_cast<Tensor&>(*this), from, to);
}

Tensor& Tensor::uniform_(double from, double to, Generator generator) const {
    return otter::native::uniform_(const_cast<Tensor&>(*this), from, to, generator);
}

Tensor& Tensor::normal_(double mean, double std) const {
    return otter::native::normal_(const_cast<Tensor&>(*this), mean, std);
}

Tensor& Tensor::normal_(double mean, double std, Generator generator) const {
    return otter::native::normal_(const_cast<Tensor&>(*this), mean, std, generator);
}

Tensor& Tensor::random_(int64_t from, int64_t to) const {
    return otter::native::random_(const_cast<Tensor&>(*this), from, to);
}

Tensor& Tensor::random_(int64_t from, int64_t to, Generator generator) const {
    return otter::native::random_(const_cast<Tensor&>(*this), from, to, generator);
}

Tensor& Tensor::random_(int64_t to) const {
    return otter::native::random_(const_cast<Tensor&>(*this), to);
}

Tensor& Tensor::random_(int64_t to, Generator generator) const {
    return otter::native::random_(const_cast<Tensor&>(*this), to, generator);
}

Tensor& Tensor::random_() const {
    return otter::native::random_(const_cast<Tensor&>(*this));
}

Tensor& Tensor::random_(Generator generator) const {
    return otter::native::random_(const_cast<Tensor&>(*this), generator);
}

Tensor& Tensor::zero_() {
    return native::zero_(*this);
}

Tensor& Tensor::fill_(const Scalar &value) {
    return native::fill_out(*this, value);
}

Tensor& Tensor::fill_(const Tensor &value) {
    return native::fill_(*this, value);
}

Tensor Tensor::to(ScalarType dtype, bool non_blocking, bool copy, MemoryFormat memory_format) const {
    return native::to(*this, dtype, non_blocking, copy, memory_format);
}

Tensor Tensor::to(TensorOptions options, bool non_blocking, bool copy, MemoryFormat memory_format) const {
    return native::to(*this, typeMetaToScalarType(options.dtype()), non_blocking, copy, memory_format);
}

Tensor& Tensor::add_(const Tensor &other, const Scalar &alpha) const {
    return otter::native::add_(const_cast<Tensor&>(*this), other, alpha);
}

Tensor Tensor::add(const Tensor& other, const Scalar& alpha) const {
    return otter::native::add(*this, other, alpha);
}

Tensor& Tensor::add_(const Scalar &other, const Scalar &alpha) const {
    return add_(native::wrapped_scalar_tensor(other), alpha);
}

Tensor Tensor::add(const Scalar& other, const Scalar& alpha) const {
    return add(native::wrapped_scalar_tensor(other), alpha);
}

Tensor& Tensor::sub_(const Tensor &other, const Scalar &alpha) const {
    return otter::native::sub_(const_cast<Tensor&>(*this), other, alpha);
}

Tensor Tensor::sub(const Tensor& other, const Scalar& alpha) const {
    return otter::native::sub(*this, other, alpha);
}

Tensor& Tensor::sub_(const Scalar &other, const Scalar &alpha) const {
    return sub_(native::wrapped_scalar_tensor(other), alpha);
}

Tensor Tensor::sub(const Scalar& other, const Scalar& alpha) const {
    return sub(native::wrapped_scalar_tensor(other), alpha);
}

Tensor& Tensor::mul_(const Tensor &other) const {
    return otter::native::mul_(const_cast<Tensor&>(*this), other);
}

Tensor Tensor::mul(const Tensor& other) const {
    return otter::native::mul(*this, other);
}

Tensor& Tensor::mul_(const Scalar &other) const {
    return mul_(native::wrapped_scalar_tensor(other));
}

Tensor Tensor::mul(const Scalar& other) const {
    return mul(native::wrapped_scalar_tensor(other));
}

Tensor& Tensor::div_(const Tensor &other) const {
    return otter::native::div_(const_cast<Tensor&>(*this), other);
}

Tensor Tensor::div(const Tensor& other) const {
    return otter::native::div(*this, other);
}

Tensor& Tensor::div_(const Scalar &other) const {
    return div_(native::wrapped_scalar_tensor(other));
}

Tensor Tensor::div(const Scalar& other) const {
    return div(native::wrapped_scalar_tensor(other));
}

Tensor& Tensor::remainder_(const Tensor &other) const {
    return otter::native::remainder_(const_cast<Tensor&>(*this), other);
}

Tensor Tensor::remainder(const Tensor& other) const {
    return otter::native::remainder(*this, other);
}

Tensor& Tensor::remainder_(const Scalar &other) const {
    return remainder_(native::wrapped_scalar_tensor(other));
}

Tensor Tensor::remainder(const Scalar& other) const {
    return remainder(native::wrapped_scalar_tensor(other));
}

Tensor& Tensor::bitwise_and_(const Tensor &other) const {
    return otter::native::bitwise_and_(const_cast<Tensor&>(*this), other);
}

Tensor Tensor::bitwise_and(const Tensor& other) const {
    return otter::native::bitwise_and(*this, other);
}

Tensor& Tensor::bitwise_and_(const Scalar &other) const {
    return bitwise_and_(native::wrapped_scalar_tensor(other));
}

Tensor Tensor::bitwise_and(const Scalar& other) const {
    return bitwise_and(native::wrapped_scalar_tensor(other));
}

Tensor& Tensor::bitwise_or_(const Tensor &other) const {
    return otter::native::bitwise_or_(const_cast<Tensor&>(*this), other);
}

Tensor Tensor::bitwise_or(const Tensor& other) const {
    return otter::native::bitwise_or(*this, other);
}

Tensor& Tensor::bitwise_or_(const Scalar &other) const {
    return bitwise_or_(native::wrapped_scalar_tensor(other));
}

Tensor Tensor::bitwise_or(const Scalar& other) const {
    return bitwise_or(native::wrapped_scalar_tensor(other));
}

Tensor& Tensor::bitwise_xor_(const Tensor &other) const {
    return otter::native::bitwise_xor_(const_cast<Tensor&>(*this), other);
}

Tensor Tensor::bitwise_xor(const Tensor& other) const {
    return otter::native::bitwise_xor(*this, other);
}

Tensor& Tensor::bitwise_xor_(const Scalar &other) const {
    return bitwise_xor_(native::wrapped_scalar_tensor(other));
}

Tensor Tensor::bitwise_xor(const Scalar& other) const {
    return bitwise_xor(native::wrapped_scalar_tensor(other));
}

Tensor& Tensor::bitwise_not_() const {
    return otter::native::bitwise_not_(const_cast<Tensor&>(*this));
}

Tensor Tensor::bitwise_not() const {
    return otter::native::bitwise_not(*this);
}

Tensor Tensor::eq(const Scalar & other) const {
    return otter::native::eq(const_cast<Tensor&>(*this), other);
}
Tensor Tensor::eq(const Tensor & other) const {
    return otter::native::eq(const_cast<Tensor&>(*this), other);
}
Tensor & Tensor::eq_(const Scalar & other) const {
    return otter::native::eq_(const_cast<Tensor&>(*this), other);
}
Tensor & Tensor::eq_(const Tensor & other) const {
    return otter::native::eq_(const_cast<Tensor&>(*this), other);
}

Tensor Tensor::ne(const Scalar & other) const {
    return otter::native::ne(const_cast<Tensor&>(*this), other);
}
Tensor Tensor::ne(const Tensor & other) const {
    return otter::native::ne(const_cast<Tensor&>(*this), other);
}
Tensor & Tensor::ne_(const Scalar & other) const {
    return otter::native::ne_(const_cast<Tensor&>(*this), other);
}
Tensor & Tensor::ne_(const Tensor & other) const {
    return otter::native::ne_(const_cast<Tensor&>(*this), other);
}

Tensor Tensor::ge(const Scalar & other) const {
    return otter::native::ge(const_cast<Tensor&>(*this), other);
}
Tensor Tensor::ge(const Tensor & other) const {
    return otter::native::ge(const_cast<Tensor&>(*this), other);
}
Tensor & Tensor::ge_(const Scalar & other) const {
    return otter::native::ge_(const_cast<Tensor&>(*this), other);
}
Tensor & Tensor::ge_(const Tensor & other) const {
    return otter::native::ge_(const_cast<Tensor&>(*this), other);
}

Tensor Tensor::le(const Scalar & other) const {
    return otter::native::le(const_cast<Tensor&>(*this), other);
}
Tensor Tensor::le(const Tensor & other) const {
    return otter::native::le(const_cast<Tensor&>(*this), other);
}
Tensor & Tensor::le_(const Scalar & other) const {
    return otter::native::le_(const_cast<Tensor&>(*this), other);
}
Tensor & Tensor::le_(const Tensor & other) const {
    return otter::native::le_(const_cast<Tensor&>(*this), other);
}

Tensor Tensor::gt(const Scalar & other) const {
    return otter::native::gt(const_cast<Tensor&>(*this), other);
}
Tensor Tensor::gt(const Tensor & other) const {
    return otter::native::gt(const_cast<Tensor&>(*this), other);
}
Tensor & Tensor::gt_(const Scalar & other) const {
    return otter::native::gt_(const_cast<Tensor&>(*this), other);
}
Tensor & Tensor::gt_(const Tensor & other) const {
    return otter::native::gt_(const_cast<Tensor&>(*this), other);
}

Tensor Tensor::lt(const Scalar & other) const {
    return otter::native::lt(const_cast<Tensor&>(*this), other);
}
Tensor Tensor::lt(const Tensor & other) const {
    return otter::native::lt(const_cast<Tensor&>(*this), other);
}
Tensor & Tensor::lt_(const Scalar & other) const {
    return otter::native::lt_(const_cast<Tensor&>(*this), other);
}
Tensor & Tensor::lt_(const Tensor & other) const {
    return otter::native::lt_(const_cast<Tensor&>(*this), other);
}

Tensor& Tensor::neg_() const {
    return otter::native::neg_(const_cast<Tensor&>(*this));
}

Tensor Tensor::neg() const {
    return otter::native::neg(*this);
}

Tensor& Tensor::abs_() const {
    return otter::native::abs_(const_cast<Tensor&>(*this));
}

Tensor Tensor::abs() const {
    return otter::native::abs(*this);
}

Tensor& Tensor::sin_() const {
    return otter::native::sin_(const_cast<Tensor&>(*this));
}

Tensor Tensor::sin() const {
    return otter::native::sin(*this);
}

Tensor& Tensor::cos_() const {
    return otter::native::cos_(const_cast<Tensor&>(*this));
}

Tensor Tensor::cos() const {
    return otter::native::cos(*this);
}

Tensor& Tensor::tan_() const {
    return otter::native::tan_(const_cast<Tensor&>(*this));
}

Tensor Tensor::tan() const {
    return otter::native::tan(*this);
}

Tensor& Tensor::exp_() const {
    return otter::native::exp_(const_cast<Tensor&>(*this));
}

Tensor Tensor::exp() const {
    return otter::native::exp(*this);
}

Tensor& Tensor::sqrt_() const {
    return otter::native::sqrt_(const_cast<Tensor&>(*this));
}

Tensor Tensor::sqrt() const {
    return otter::native::sqrt(*this);
}

Tensor Tensor::dot(const Tensor& other) const {
    return otter::dot(*this, other);
}

Tensor Tensor::addmm(const Tensor& mat1, const Tensor& mat2, const Scalar& beta, const Scalar& alpha) const {
    return otter::native::addmm(*this, mat1, mat2, beta, alpha);
}

Tensor& Tensor::addmm_(const Tensor &mat1, const Tensor &mat2, const Scalar& beta, const Scalar& alpha) const {
    return otter::native::addmm_(const_cast<Tensor&>(*this), mat1, mat2, beta, alpha);
}

Tensor Tensor::mm(const Tensor &other) const {
    return otter::native::mm(*this, other);
}

#define DEFINE_ITEM(T, name)      \
    template <>                         \
    T Tensor::item() const {            \
        return item().to##name();       \
    }

OTTER_ALL_SCALAR_TYPES(DEFINE_ITEM)
#undef DEFINE_ITEM

Scalar Tensor::item() const {
    return otter::item(*this);
}


}   // namespace otter
