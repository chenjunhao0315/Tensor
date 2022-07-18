//
//  WarpDimUtils.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/10.
//

#ifndef WarpDimUtils_hpp
#define WarpDimUtils_hpp

#include "ArrayRef.hpp"
#include "Utils.hpp"
#include "WarpDimMinimal.hpp"
#include "Tensor.hpp"

namespace otter {

// wrap each dim in the dims array, taking dim_post_expr as the true number of
// dimensions
static inline void maybe_wrap_dims_n(
    int64_t* dims,
    int64_t ndims,
    int64_t dim_post_expr) {
    if (dim_post_expr <= 0) {
        dim_post_expr = 1; // this will make range [-1, 0]
    }
    int64_t min = -dim_post_expr;
    int64_t max = dim_post_expr - 1;
    for (const auto i : otter::irange(ndims)) {
        auto& dim = dims[i];
        if (dim < min || dim > max) {
            OTTER_CHECK(
                false,
                "Dimension out of range (expected to be in range of [",
                min,
                ", ",
                max,
                "], but got ",
                dim,
                ")");
        }
        if (dim < 0)
            dim += dim_post_expr;
    }
}
// Wrap each dim in a contiguous container, taking dim_post_expr as the true
// number of dimensions E.g. could also be std::array or c10::SmallVector
template <typename Container>
inline void maybe_wrap_dims(Container& dims, int64_t dim_post_expr) {
    return maybe_wrap_dims_n(dims.data(), dims.size(), dim_post_expr);
}

static inline int64_t maybe_wrap_dim(int64_t dim, TensorList tensors) {
    if (tensors.size() == 0) {
        // can't wrap empty TensorList; rely on underlying implementation to throw
        // error if necessary.
        return dim;
    }
    return maybe_wrap_dim(dim, tensors[0].dim());
}

static inline int64_t legacy_cat_wrap_dim(int64_t dim, const std::vector<std::vector<int64_t>>& tensor_sizes) {
    for (auto& sizes : tensor_sizes) {
        if (sizes == std::vector<int64_t>({0})) {
            continue;
        }
        return maybe_wrap_dim(dim, sizes.size());
    }
    return dim;
}

static inline int64_t legacy_cat_wrap_dim(int64_t dim, TensorList tensors) {
    for (auto& tensor : tensors) {
        if (tensor.dim() == 1 && tensor.sizes()[0] == 0) {
            continue;
        }
        return maybe_wrap_dim(dim, tensor.dim());
    }
    return dim;
}

// wrap negative dims in a vector
static inline void wrap_all_dims(std::vector<int64_t>& dims_to_wrap, int64_t tensor_total_dims) {
    for (const auto i : otter::irange(dims_to_wrap.size())) {
        dims_to_wrap[i] = maybe_wrap_dim(dims_to_wrap[i], tensor_total_dims);
    }
}

}   // end namespace otter

#endif /* WarpDimUtils_hpp */
