//
//  DilatedConvolutionUtils.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/23.
//

#ifndef DilatedConvolutionUtils_hpp
#define DilatedConvolutionUtils_hpp

#include "Math.hpp"
#include "Tensor.hpp"

#define OTTER_CHECK_DIM_SIZE(T, DIM, DIM_SIZE, SIZE) \
    OTTER_CHECK(                                       \
        T.dim() == DIM && T.size(DIM_SIZE) == SIZE,    \
        "Need " #T " of dimension ",                   \
        DIM,                                           \
        " and " #T ".size[",                           \
        DIM_SIZE,                                      \
        "] == ",                                       \
        SIZE,                                          \
        " but got input to be of shape ",              \
        T.sizes())

namespace otter {

namespace {
inline bool all_positive(IntArrayRef& arr) {
    return std::all_of(arr.begin(), arr.end(), [](int64_t item) { return item > 0; });
}

inline bool all_nonnegative(std::vector<int64_t>& arr) {
    return std::all_of(arr.begin(), arr.end(), [](int64_t item) { return item >= 0; });
}
}   //

// calculate the rear part of output tensor sizes
template <int64_t dim>
std::vector<int64_t> get_output_size(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
    std::vector<int64_t> sizes;
    for (const auto index : otter::irange(dim)) {
        sizes.push_back(
            div_round_up<int64_t>(
                input.size(index + input.dim() - dim) + 2 * pad_size[index] -
                (dilation_size[index] * (kernel_size[index] - 1) + 1),
                stride_size[index]) +
        1);
  }
  return sizes;
}

// calculate the sizes of output tensor
template <int64_t dim>
std::vector<int64_t> get_output_size(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
    
    auto output_size = get_output_size<dim>(input, kernel_size, stride_size, pad_size, dilation_size);
    output_size.insert(output_size.begin(), weight.size(0));
    if (input.dim() == dim + 2) {
        output_size.insert(output_size.begin(), input.size(0));
    }
    return output_size;
}

template <int64_t dim>
void slow_conv_dilated_shape_check(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& grad_output,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
    
    OTTER_CHECK(kernel_size.size() == dim,
        "kernel sizes length should be ", dim,
        ", but got ", kernel_size.size());
    OTTER_CHECK(stride_size.size() == dim,
        "strides length should be ", dim,
        ", but got ", stride_size.size());
    OTTER_CHECK(dilation_size.size() == dim,
        "dilations length should be ", dim,
        ", but got ", dilation_size.size());
    OTTER_CHECK(pad_size.size() == dim,
        "pads length should be ", dim,
        ", but got ", pad_size.size());

    OTTER_CHECK(all_positive(kernel_size),
        "kernel size should be greater than zero, but got ", kernel_size);
    OTTER_CHECK(all_positive(stride_size),
        "stride should be greater than zero, but got ", stride_size);
    OTTER_CHECK(all_positive(dilation_size),
        "dilation should be greater than zero, but got ", dilation_size);
    
    // check input
    OTTER_CHECK(input.defined(), "input must be defined");
    bool is_batch = input.dim() == dim + 2;
    int64_t n = (is_batch ? 2 : 1);
    int64_t ndim = n + dim;
    if (!is_batch) {
        // input dim has to be dim + 1 if not batched
        OTTER_CHECK(input.dim() == dim + 1,
            "input must be 4D or 5D tensor but got ", input.dim(), "D tensor");
    }
    
    // check output sizes
    auto output_size = get_output_size<dim>(input, kernel_size, stride_size, pad_size, dilation_size);
    
    OTTER_CHECK(all_nonnegative(output_size),
        "calculated output size ", IntArrayRef(output_size),
        " is too small (all sizes must be non-negative)");
    
    // check weight
    OTTER_CHECK(weight.defined(), "weight must be defined");
    OTTER_CHECK(weight.dim() == dim + 2,
        "weight must be ", dim + 2,
        "D tensor but got ", weight.dim(),
        "D tensor dim=", dim);
    OTTER_CHECK(weight.sizes().slice(2) == kernel_size,
        "weight[2:] shape ", weight.sizes().slice(2),
        " must be equal to kernel_size ", kernel_size);
    
    OTTER_CHECK_DIM_SIZE(input, input.dim(), (is_batch ? 1 : 0), weight.size(1));
    
    // check bias when present
    if (bias.defined()) {
        OTTER_CHECK(bias.dim() == 1, "bias must be 1D tensor but got ", bias.dim(), "D tensor");
        OTTER_CHECK_DIM_SIZE(bias, 1, 0, weight.size(0));
    }
    
    if (grad_output.defined()) {
        OTTER_CHECK(
            grad_output.dim() == ndim,
            "grad_output must be ",
            ndim,
            "D tensor but got ",
            grad_output.dim(),
            "D tensor");
        if (is_batch) {
            OTTER_CHECK(
                grad_output.size(0) == input.size(0),
                "grad_output.size(0)=",
                grad_output.size(0),
                " must be input.size(0)=",
                input.size(0));
        }
        OTTER_CHECK(
            grad_output.size(n - 1) == weight.size(0),
            "grad_output.size(",
            n - 1,
            ")=",
            grad_output.size(n - 1),
            " must be weight.size(0)=",
            weight.size(0));
        OTTER_CHECK(
            grad_output.sizes().slice(n) == output_size,
            "grad_output[",
            n,
            ":] shape",
            grad_output.sizes().slice(n),
            " must be equal to output size ",
            IntArrayRef(output_size));
      }
}

}   // end namespace otter

#endif /* DilatedConvolutionUtils_hpp */
