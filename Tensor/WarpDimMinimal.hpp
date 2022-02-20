//
//  WarpDimMinimal.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/18.
//

#ifndef WarpDimMinimal_hpp
#define WarpDimMinimal_hpp

#include "Exception.hpp"

namespace otter {

static inline int64_t maybe_wrap_dim(int64_t dim, int64_t dim_post_expr, bool wrap_scalar = true) {
    if (dim_post_expr <= 0) {
        if (!wrap_scalar) {
            OTTER_CHECK(false, "dimension specified as ", dim, " but tensor has no dimensions");
        }
        dim_post_expr = 1; // this will make range [-1, 0]
    }

    int64_t min = -dim_post_expr;
    int64_t max = dim_post_expr - 1;
    if (dim < min || dim > max) {
        OTTER_CHECK(false, "Dimension out of range (expected to be in range of [", min, ", ", max, "], but got ", dim, ")");
    }
    if (dim < 0)
        dim += dim_post_expr;
    return dim;
}

}

#endif /* WarpDimMinimal_hpp */
