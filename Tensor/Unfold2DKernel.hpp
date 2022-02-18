//
//  Unfold2DKernel.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/15.
//

#ifndef Unfold2DKernel_hpp
#define Unfold2DKernel_hpp

namespace otter {

void unfold2d_copy_kernel(
    ScalarType dtype,
    void* finput_data, void* input_datta,
    int64_t kernel_height, int64_t kernel_width,
    int64_t stride_height, int64_t stride_width,
    int64_t pad_height, int64_t pad_width,
    int64_t n_input_plane, int64_t input_height, int64_t input_width,
    int64_t output_height, int64_t output_width);

}

#endif /* Unfold2DKernel_hpp */
