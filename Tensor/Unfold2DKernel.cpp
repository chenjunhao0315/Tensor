//
//  Unfold2DKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/15.
//

#include "Dispatch.hpp"
#include "Unfold2D.hpp"
#include "Unfold2DKernel.hpp"
#include "Parallel.hpp"

namespace otter {

template <typename scalar_t>
static void unfold2d_copy(
    scalar_t* input_data,
    scalar_t* finput_data,
    int64_t kernel_height, int64_t kernel_width,
    int64_t stride_height, int64_t stride_width,
    int64_t pad_height, int64_t pad_width,
    int64_t n_input_plane, int64_t input_height, int64_t input_width,
    int64_t output_height, int64_t output_width) {
    
    otter::parallel_for(0, (int64_t)n_input_plane * kernel_height * kernel_width, 0, [&](int64_t start, int64_t end) {
        for (const auto k : otter::irange(start, end)) {
            int64_t nip  = k / (kernel_height * kernel_width);
            int64_t rest = k % (kernel_height * kernel_width);
            int64_t kh   = rest / kernel_width;
            int64_t kw   = rest % kernel_width;
            int64_t x,  y;
            int64_t ix, iy;
            
            scalar_t* dst = finput_data
                + nip * ((size_t)kernel_height * kernel_width * output_height * output_width)
                + kh * ((size_t)kernel_width * output_height * output_width)
                + kw * ((size_t)output_height * output_width);
            scalar_t* src = input_data
                + nip * ((size_t)input_height * input_width);
            
            if (pad_width > 0 || pad_height > 0) {
                int64_t lpad, rpad;
                for (y = 0; y < output_height; ++y) {
                    iy = (int64_t)y * stride_height - pad_height + kh;
                    if (iy < 0 || iy >= input_height) {
                        memset(
                               dst + (size_t)y * output_width,
                               0,
                               sizeof(scalar_t) * output_width);
                    } else {
                        if (stride_width == 1) {
                            ix = 0 - pad_width + kw;
                            lpad = std::max<int64_t>(0, pad_width - kw);
                            rpad = std::max<int64_t>(0, pad_width - (kernel_width - kw - 1));
                            if (output_width - rpad - lpad <= 0) {
                                memset(
                                       dst + (size_t)y * output_width,
                                       0,
                                       sizeof(scalar_t) * output_width);
                            } else {
                                if (lpad > 0)
                                    memset(
                                           dst + (size_t)y * output_width,
                                           0,
                                           sizeof(scalar_t) * lpad);
                                memcpy(
                                       dst + (size_t)y * output_width + lpad,
                                       src + (size_t)iy * input_width + ix + lpad,
                                       sizeof(scalar_t) * (output_width - rpad - lpad));
                                if (rpad > 0)
                                    memset(
                                           dst + (size_t)y * output_width + output_width - rpad,
                                           0,
                                           sizeof(scalar_t) * rpad);
                            }
                        } else {    // stride_width != 1
                            for (x = 0; x < output_width; x++) {
                                ix = (int64_t)x * stride_width - pad_width + kw;
                                if (ix < 0 || ix >= input_width) {
                                    memset(
                                           dst + (size_t)y * output_width + x,
                                           0,
                                           sizeof(scalar_t) * 1);
                                } else {
                                    memcpy(
                                           dst + (size_t)y * output_width + x,
                                           src + (size_t)iy * input_width + ix,
                                           sizeof(scalar_t) * (1));
                                }
                            }
                        }
                    }
                }
            } else {    // pad_width <= 0 && pad_height <= 0
                for (y = 0; y < output_height; y++) {
                    iy = (int64_t)y * stride_height + kh;
                    ix = 0 + kw;
                    if (stride_width == 1) {
                        memcpy(
                               dst + (size_t)y * output_width,
                               src + (size_t)iy * input_width + ix,
                               sizeof(scalar_t) * output_width);
                    } else {
                        for (x = 0; x < output_width; x++)
                            memcpy(
                                   dst + (size_t)y * output_width + x,
                                   src + (size_t)iy * input_width + ix + (int64_t)x * stride_width,
                                   sizeof(scalar_t) * (1));
                    }
                }
            }
        }
    });
}

void unfold2d_copy_kernel(
    ScalarType dtype,
    void* finput_data, void* input_data,
    int64_t kernel_height, int64_t kernel_width,
    int64_t stride_height, int64_t stride_width,
    int64_t pad_height, int64_t pad_width,
    int64_t n_input_plane, int64_t input_height, int64_t input_width,
    int64_t output_height, int64_t output_width) {
    
    OTTER_DISPATCH_ALL_TYPES(dtype, "unfold2d_copy", [&] {
        unfold2d_copy(
            static_cast<scalar_t*>(input_data), static_cast<scalar_t*>(finput_data),
            kernel_height, kernel_width,
            stride_height, stride_width,
            pad_height, pad_width,
            n_input_plane, input_height, input_width,
            output_height, output_width);
    });
}

REGISTER_DISPATCH(unfold2d_copy_stub, &unfold2d_copy_kernel);

}   // end namespace otter
