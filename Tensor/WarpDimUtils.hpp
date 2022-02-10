//
//  WarpDimUtils.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/10.
//

#ifndef WarpDimUtils_hpp
#define WarpDimUtils_hpp

#include <cassert>

namespace otter {

static inline int64_t maybe_wrap_dim(int64_t dim, int64_t dim_post_expr, bool wrap_scalar = true) {
    if (dim_post_expr <= 0) {
        if (!wrap_scalar) {
            // "dimension specified as ", dim, " but tensor has no dimensions"
            assert(false);
        }
        dim_post_expr = 1; // this will make range [-1, 0]
    }

    int64_t min = -dim_post_expr;
    int64_t max = dim_post_expr - 1;
    if (dim < min || dim > max) {
        // "Dimension out of range (expected to be in range of [", min, ", ", max, "], but got ", dim, ")"
        assert(false);
    }
    if (dim < 0)
        dim += dim_post_expr;
    return dim;
}


}

#endif /* WarpDimUtils_hpp */
