//
//  TensorInterpolation.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/12.
//

#ifndef TensorInterpolation_hpp
#define TensorInterpolation_hpp

#include "TensorFunction.hpp"

namespace otter {

enum InterpolateMode {
    NEAREST,
    BILINEAR
};

Tensor Interpolate(const Tensor& input, IntArrayRef size, IntArrayRef scale_factor, InterpolateMode mode, bool align_corners = false);

};

#endif /* TensorInterpolation_hpp */
