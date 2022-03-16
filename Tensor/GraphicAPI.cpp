//
//  GraphicAPI.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/14.
//

#include "GraphicAPI.hpp"

#include <cassert>

namespace otter {
namespace cv {

void colorToRawData(const Color& color, void *buf_, otter::ScalarType dtype, int channels, int unroll_to) {
    OTTER_DISPATCH_ALL_TYPES(dtype, "colorToRawData", [&] {
        scalar_t *buf = (scalar_t*)buf_;
        int i;
        for (i = 0; i < channels; ++i) {
            buf[i] = static_cast<scalar_t>(color.val[i]);
        }
    });
}

}   // end namespace cv
}   // end namespace otter
