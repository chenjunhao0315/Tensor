//
//  TensorShape.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/10.
//

#include "TensorShape.hpp"

namespace otter {

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

namespace native {

Tensor as_strided_tensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride) {
    return as_strided_tensorimpl(self, size, stride, self.memory_offset());
}

Tensor as_strided_tensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride, int64_t memory_offset_) {
    auto result = otter::make_tensor<TensorNucleus>(Memory(self.memory()), self.dtype());
    
    setStrided(result, size, stride, memory_offset_);
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
