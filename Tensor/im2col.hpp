//
//  im2col.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/24.
//

#ifndef im2col_hpp
#define im2col_hpp

#include "Exception.hpp"
#include "Tensor.hpp"
#include "Utils.hpp"
#include "Math.hpp"

namespace otter {

template <typename T>
static void im2col(
    const T* data_im,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,    // (input_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1) / stride_height + 1
    const int64_t output_width,     // (input_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) / stride_width + 1
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    T* data_col) {
    
    const int64_t height_col = output_height;
    const int64_t width_col = output_width;
    const int64_t channels_col = channels * kernel_h * kernel_w;

    for (const auto c_col : otter::irange(channels_col)) {
        int64_t w_offset = c_col % kernel_w;
        int64_t h_offset = (c_col / kernel_w) % kernel_h;
        int64_t c_im = c_col / kernel_h / kernel_w;

        for (const auto h_col : otter::irange(height_col)) {
            int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;

            for (const auto w_col : otter::irange(width_col)) {
                int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;
                data_col[(c_col * height_col + h_col) * width_col + w_col] =
                    (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                    ? data_im[(c_im * height + h_im) * width + w_im]
                    : static_cast<T>(0);
            }
        }
    }
}

template <typename T>
static void col2im(
    const T* data_col,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int64_t output_height,
    const int64_t output_width,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t dilation_h,
    const int64_t dilation_w,
    T* data_im) {
    
    std::fill_n(data_im, height * width * channels, T(0));

    const int64_t height_col = output_height;
    const int64_t width_col = output_width;
    const int64_t channels_col = channels * kernel_h * kernel_w;

    for (const auto c_col : otter::irange(channels_col)) {
        int64_t w_offset = c_col % kernel_w;
        int64_t h_offset = (c_col / kernel_w) % kernel_h;
        int64_t c_im = c_col / kernel_h / kernel_w;

        for (const auto h_col : otter::irange(height_col)) {
            int64_t h_im = h_col * stride_h - pad_h + h_offset * dilation_h;

            for (const auto w_col : otter::irange(width_col)) {
                int64_t w_im = w_col * stride_w - pad_w + w_offset * dilation_w;

                if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
                    data_im[(c_im * height + h_im) * width + w_im] +=
                    data_col[(c_col * height_col + h_col) * width_col + w_col];
            }
        }
    }
}

Tensor im2col_cpu(const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation);

Tensor& im2col_out_cpu(const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor& output);

static inline void im2col_shape_check(
    const Tensor& input,
    const Tensor& /*grad_output*/,
    int64_t kernel_height,
    int64_t kernel_width,
    int64_t dilation_height,
    int64_t dilation_width,
    int64_t pad_height,
    int64_t pad_width,
    int64_t stride_height,
    int64_t stride_width) {
    
    OTTER_CHECK(
                kernel_width > 0 && kernel_height > 0,
                "kernel size should be greater than zero, but got kernel_height: ",
                kernel_height,
                " kernel_width: ",
                kernel_width);

    OTTER_CHECK(
                dilation_width > 0 && dilation_height > 0,
                "dilation should be greater than zero, but got dilation_height: ",
                dilation_height,
                " dilation_width: ",
                dilation_width);

    OTTER_CHECK(
                pad_width >= 0 && pad_height >= 0,
                "padding should be non-negative, but got pad_height: ",
                pad_height,
                " pad_width: ",
                pad_width);

    OTTER_CHECK(
                stride_width > 0 && stride_height > 0,
                "stride should be greater than zero, but got stride_height: ",
                stride_height,
                " stride_width: ",
                stride_width);

    int64_t ndim = input.dim();

    // allow dim=0 only the batch dimension.
    bool valid_dims = input.size(1) != 0 && input.size(2) != 0;
    OTTER_CHECK(
                (ndim == 3 && input.size(0) && valid_dims) ||
                (ndim == 4 && valid_dims && input.size(3) != 0),
                "Expected 3D or 4D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
                input.sizes());

    int64_t dim_batch = 0;

    if (ndim == 3) {
        dim_batch = -1;
    }

    int64_t input_height = input.size(dim_batch + 2);
    int64_t input_width = input.size(dim_batch + 3);
    int64_t output_height = div_round_up<int64_t>(
                                             input_height + 2 * pad_height -
                                             (dilation_height * (kernel_height - 1) + 1),
                                             stride_height) + 1;
    
    int64_t output_width = div_round_up<int64_t>(
                                            input_width + 2 * pad_width -
                                            (dilation_width * (kernel_width - 1) + 1),
                                            stride_width) + 1;

    if (output_height < 1 || output_width < 1) {
        OTTER_CHECK(false,
                 "Given input with spatial size (",
                 input_height,
                 ", ",
                 input_height,
                 "), kernel_size=(",
                 kernel_height,
                 ", ",
                 kernel_width,
                 "), dilation=(",
                 dilation_height,
                 ", ",
                 dilation_width,
                 "), padding=(",
                 pad_height,
                 ", ",
                 pad_width,
                 "), calculated shape of the array of sliding blocks as (",
                 output_height,
                 ", ",
                 output_width,
                 "), which is too small (non-positive).");
    }
}

}   // end namespace otter

#endif /* im2col_hpp */
