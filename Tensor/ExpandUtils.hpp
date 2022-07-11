//
//  ExpandUtils.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/12.
//

#ifndef ExpandUtils_hpp
#define ExpandUtils_hpp

#include "Utils.hpp"
#include "MaybeOwned.hpp"
#include "ArrayRef.hpp"
#include "Tensor.hpp"

namespace otter {

// Use for TensorIterator -> compute_shape()
DimVector infer_size_dimvector(IntArrayRef a, IntArrayRef b);

// Use for Tensor -> view
DimVector infer_size_dimvector(IntArrayRef shape, int64_t numel);

std::vector<int64_t> infer_dense_strides(IntArrayRef tensor_sizes, IntArrayRef tensor_strides);

template <typename Container>
struct InferExpandGeometryResult {
    Container sizes;
    Container strides;
    explicit InferExpandGeometryResult(size_t ndim)
        : sizes(ndim), strides(ndim) {}
    explicit InferExpandGeometryResult(IntArrayRef sizes_, size_t ndim)
        : sizes(sizes_.begin(), sizes_.end()), strides(ndim) {}
};

std::tuple<std::vector<int64_t>, std::vector<int64_t>>
inferExpandGeometry(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides,
    IntArrayRef sizes);

InferExpandGeometryResult<DimVector>
inferExpandGeometry_dimvector(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides,
    IntArrayRef sizes);

inline MaybeOwned<Tensor> expand_size(const Tensor &to_expand, IntArrayRef sizes) {
    if (to_expand.sizes().equals(sizes)) {
        return MaybeOwned<Tensor>::borrowed(to_expand);
    }

    return MaybeOwned<Tensor>::owned(to_expand.expand(sizes));
}

inline MaybeOwned<Tensor> expand_size(Tensor &&to_expand, IntArrayRef sizes) = delete;




}

#endif /* ExpandUtils_hpp */
