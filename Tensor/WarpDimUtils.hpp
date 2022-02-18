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


}

#endif /* WarpDimUtils_hpp */
