//
//  TensorInterpolation.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/12.
//

#include "TensorInterpolation.hpp"

namespace otter {

Tensor Interpolate(const Tensor& input, IntArrayRef size, IntArrayRef scale_factor, InterpolateMode mode, bool align_corners) {
    
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
