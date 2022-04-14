//
//  TensorInterpolation.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/12.
//

#include "TensorInterpolation.hpp"
#include "Tensor.hpp"

namespace otter {

Tensor Interpolate(const Tensor& input, IntArrayRef size, ArrayRef<double> scale_factor, InterpolateMode mode, bool align_corners) {
    
    if (scale_factor.empty()) {
        scale_factor = {0, 0};
    } else if (size.empty()) {
        size = {static_cast<long long>(input.size(2) * scale_factor[0]), static_cast<long long>(input.size(3) * scale_factor[1])};
    } else if (scale_factor.empty() && size.empty()) {
        OTTER_CHECK(false, "Invalid interpolation");
    }
    
    switch (mode) {
        case InterpolateMode::NEAREST:
            return otter::native::upsample_nearest2d(input, size, scale_factor[0], scale_factor[1]);
            break;
        case InterpolateMode::BILINEAR:
            return otter::native::upsample_bilinear2d(input, size, align_corners, scale_factor[0], scale_factor[1]);
            break;
    }
    OTTER_CHECK(false, "Unsupport interpolation mode");
    return Tensor();
}

}
