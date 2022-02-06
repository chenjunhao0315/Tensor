//
//  TensorFunction.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/31.
//

#include "TensorResize.hpp"
#include "EmptyTensor.hpp"
#include "TensorFunction.hpp"

namespace otter {

Tensor create_out(IntArrayRef sizes, IntArrayRef strides, TensorOptions options) {
    if (strides.empty()) {
        return otter::empty_cpu(sizes, options);
    } else {
        return otter::empty_strided_cpu(sizes, strides, options);
    }
}

void resize_out(const Tensor &out, IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {
    assert(out.dtype() == options.dtype());
    assert(out.device() == options.device());
    
    const bool resized = resize_output(out, sizes);
    if (resized) {
        if (!strides.empty()) {
            
        }
    }
}



}
