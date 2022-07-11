//
//  Sorting.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/11.
//

#include "Sorting.hpp"
#include "TensorFunction.hpp"
#include "ExpandUtils.hpp"
#include "Tensor.hpp"
#include "TensorFactory.hpp"
#include "TensorIterator.hpp"

namespace otter {

DEFINE_DISPATCH(sort_stub);
DEFINE_DISPATCH(topk_stub);

DEFINE_META_FUNCTION(topk)(const Tensor& self, int64_t k, int64_t dim_, bool largest, bool sorted) {
    int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
    OTTER_CHECK(
                k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
                "selected index k out of range");
    int64_t sliceSize = self.dim() == 0 ? 1 : self.size(dim);
    OTTER_CHECK(k >= 0 && k <= sliceSize, "k not in range for dimension");
    
    // Build the output size, which is the dim being selected set to
    // size k
    DimVector topKSize(self.sizes().vec());
    if (topKSize.size() > 0) {
        topKSize[dim] = k;
    }
    set_output(0, topKSize, {}, self.options());
    set_output(1, topKSize, {}, self.options().dtype(otter::ScalarType::Long));
}

DEFINE_META_FUNCTION_OVERLOAD(sort, stable)(const Tensor& self, bool stable, int64_t dim, bool descending) {
    maybe_wrap_dim(dim, self.dim());
    
    // See issue: https://github.com/pytorch/pytorch/issues/65863
    // Strides should be dense, so as not to allocate too much memory.
    // We either use 'self' strides, or infer dense strides from them.
    std::vector<int64_t> strides = (self.is_non_overlapping_and_dense())
    ? self.strides().vec()
    : otter::infer_dense_strides(self.sizes(), self.strides());
    
    set_output(0, self.sizes(), strides, self.options());
    set_output(1, self.sizes(), strides, self.options().dtype(otter::ScalarType::Long));
}

void _fill_indices(const TensorBase &indices, int64_t dim) {
    auto ndim = indices.dim();
    assert(0 <= dim && dim < ndim);
    auto dim_size = indices.size(dim);
    auto idx_dim = otter::arange(0, dim_size, 1, otter::ScalarType::Long);
    auto idx_dim_sizes = std::vector<int64_t>(ndim, 1);
    auto idx_dim_strides = std::vector<int64_t>(ndim, 0);
    idx_dim_sizes[dim] = dim_size;
    idx_dim_strides[dim] = 1;
    auto idx_dim_restrided = idx_dim.as_strided(idx_dim_sizes, idx_dim_strides);
    TensorRef(indices)->copy_(idx_dim_restrided);
}

DEFINE_IMPL_FUNCTION(sort_stable_out)
(const Tensor& self,
 bool stable,
 int64_t dim,
 bool descending,
 Tensor& values,
 Tensor& indices) {
    values.copy_(self);
    // check if self is scalar
    if (self.dim() == 0 && self.numel() == 1) {
        indices.zero_();
    } else {
        dim = maybe_wrap_dim(dim, self.dim());
        sort_stub(Device::CPU, self, values, indices, dim, descending, stable);
    }
}

DEFINE_IMPL_FUNCTION(topk_out_cpu)
(const Tensor& self,
 int64_t k,
 int64_t dim_,
 bool largest,
 bool sorted,
 Tensor& values,
 Tensor& indices) {
    int64_t dim = maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
    OTTER_CHECK(
                k >= 0 && k <= (self.dim() > 0 ? self.size(dim) : 1),
                "selected index k out of range");
    
    if (self.dim() == 0 && self.numel() == 1) {
        values.copy_(self);
        indices.zero_();
    } else {
        topk_stub(Device::CPU, values, indices, self, k, dim, largest, sorted);
    }
}


}   // end namespace otter
