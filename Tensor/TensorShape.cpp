//
//  TensorShape.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/10.
//

#include "TensorUtils.hpp"
#include "ExpandUtils.hpp"
#include "TensorShape.hpp"

namespace otter {

namespace native {

Tensor select(const Tensor& self, int64_t dim, int64_t index) {
    int64_t ndim = self.dim();
    
    dim = maybe_wrap_dim(dim, ndim);
    auto size = self.size(dim);
    if (index < -size || index >= size) {
        // "select(): index ", index, " out of range for tensor of size ", self.sizes(), " at dimension ", dim
        assert(false);
    }
    if (index < 0) {
        index += size;
    }
    DimVector sizes(self.sizes().begin(), self.sizes().end());
    DimVector strides(self.strides().begin(), self.strides().end());
    auto memory_offset = self.memory_offset() + index * strides[dim];
    sizes.erase(sizes.begin() + dim);
    strides.erase(strides.begin() + dim);
    auto result = (memory_offset) ? self.as_strided(sizes, strides, memory_offset) : self.as_strided(sizes, strides);
    
    return result;
}

Tensor permute(const Tensor& self, IntArrayRef dims) {
    auto nDims = self.dim();
    // number of dims don't match in permute
    assert(dims.size() == (size_t)nDims);
    auto oldSizes = self.sizes();
    auto oldStrides = self.strides();
    DimVector newSizes(nDims);
    DimVector newStrides(nDims);
    std::vector<bool> seen(nDims);
    for (const auto i : otter::irange(nDims)) {
        auto dim = maybe_wrap_dim(dims[i], nDims);
        // "repeated dim in permute"
        assert(!seen[dim]);
        seen[dim] = true;
        newSizes[i] = oldSizes[dim];
        newStrides[i] = oldStrides[dim];
    }
    
    return self.as_strided(newSizes, newStrides);
}

Tensor& transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
    auto ndims = self.dim();
    dim0 = maybe_wrap_dim(dim0, ndims);
    dim1 = maybe_wrap_dim(dim1, ndims);
    if (dim0 == dim1) {
        return self;
    }
    DimVector sizes(self.sizes().begin(), self.sizes().end());
    DimVector strides(self.strides().begin(), self.strides().end());
    std::swap(strides[dim0], strides[dim1]);
    std::swap(sizes[dim0], sizes[dim1]);
    self.as_strided_(sizes, strides);
    return self;
}

Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
    auto ndims = self.dim();
    dim0 = maybe_wrap_dim(dim0, ndims);
    dim1 = maybe_wrap_dim(dim1, ndims);
    
    if (dim0 == dim1) {
        return self;
    }
    
    DimVector sizes(self.sizes().begin(), self.sizes().end());
    DimVector strides(self.strides().begin(), self.strides().end());
    std::swap(strides[dim0], strides[dim1]);
    std::swap(sizes[dim0], sizes[dim1]);
    auto result = self.as_strided(sizes, strides);
    return result;
}

Tensor expand(const Tensor& self, IntArrayRef sizes) {
    // Ensure the size is more bigger
    assert(sizes.size() >= (size_t)self.dim());
    auto expandedPerspectiveView = inferExpandGeometry_dimvector(self.sizes(), self.strides(), sizes);
    
    auto result = self.as_strided(expandedPerspectiveView.sizes, expandedPerspectiveView.strides);
    return result;
}

Tensor expand_as(const Tensor& self, const Tensor& other) {
    return self.expand(other.sizes());
}

template <typename Vec>
Tensor alias_with_sizes_and_strides(const Tensor& self, const Vec& sizes, const Vec& strides) {
    Tensor self_;
    self_ = otter::make_tensor<TensorNucleus>(Memory(self.memory()), self.dtype());
    setStrided(self_, sizes, strides, self.memory_offset());
    
    return self_;
}

Tensor view(const Tensor& self, IntArrayRef sizes) {
    DimVector infer_size = otter::infer_size_dimvector(sizes, self.numel());
    auto stride = otter::computeStride(self.sizes(), self.strides(), infer_size);
    // view size is not compactible with input tensor's size and stride -> trun to use .reshape(...)
    assert(!stride.empty());
    
    return alias_with_sizes_and_strides(self, infer_size, stride);
}

Tensor as_strided_tensorimpl(const Tensor& self, IntArrayRef sizes, IntArrayRef strides) {
    return as_strided_tensorimpl(self, sizes, strides, self.memory_offset());
}

Tensor as_strided_tensorimpl(const Tensor& self, IntArrayRef sizes, IntArrayRef strides, int64_t memory_offset_) {
    auto result = otter::make_tensor<TensorNucleus>(Memory(self.memory()), self.dtype());
    
    setStrided(result, sizes, strides, memory_offset_);
    return result;
}

const Tensor &as_strided_(const Tensor& self, IntArrayRef size, IntArrayRef stride) {
    return as_strided_(self, size, stride, self.memory_offset());
}

const Tensor &as_strided_(const Tensor& self, IntArrayRef size, IntArrayRef stride, int64_t memory_offset_) {
    setStrided(self, size, stride, memory_offset_);
    return self;
}

}   // end namespace native
    
}   // end namespace otter
