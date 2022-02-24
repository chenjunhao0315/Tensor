//
//  Unfold2D.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/15.
//

#ifndef Unfold2D_hpp
#define Unfold2D_hpp

#include "DispatchStub.hpp"
#include "Scalar.hpp"

namespace otter {

using unfold2d_fn = void (*)(
    ScalarType dtype,
    void* finput, void* input,
    int64_t kernel_height, int64_t kernel_width,
    int64_t stride_height, int64_t stride_width,
    int64_t pad_height, int64_t pad_width,
    int64_t n_input_plane, int64_t input_height, int64_t input_width,
    int64_t output_height, int64_t output_width);

DECLARE_DISPATCH(unfold2d_fn, unfold2d_copy_stub);

}

#endif /* Unfold2D_hpp */
