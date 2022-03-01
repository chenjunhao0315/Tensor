//
//  ConvolutionUtils.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/15.
//

#include "Tensor.hpp"
#include "ConvolutionUtils.hpp"

namespace otter {

void ConvParams::view_1d_as_2d() {
    if (stride.size() == 1) {
        stride.insert(stride.begin(), 1);
        padding.insert(padding.begin(), 0);
        dilation.insert(dilation.begin(), 1);
        output_padding.insert(output_padding.begin(), 0);
    }
}

bool ConvParams::is_strided() const {
    bool is_strided = false;
    for (auto d : stride) {
        is_strided |= (d != 1);
    }
    return is_strided;
}

bool ConvParams::is_dilated() const {
    bool is_dilated = false;
    for (auto d : dilation) {
        is_dilated |= (d != 1);
    }
    return is_dilated;
}

bool ConvParams::is_padded() const {
    bool is_padded = false;
    for (auto p : padding) {
        is_padded |= (p != 0);
    }
    return is_padded;
}

bool ConvParams::is_padding_neg() const {
    bool is_non_neg = false;
    for (auto p : padding) {
        is_non_neg |= (p < 0);
    }
    return is_non_neg;
}

bool ConvParams::is_output_padding_neg() const {
    bool is_non_neg = false;
    for (auto p : output_padding) {
        is_non_neg |= (p < 0);
    }
    return is_non_neg;
}

bool ConvParams::is_stride_nonpos() const {
    bool is_nonpos = false;
    for (auto s : stride) {
        is_nonpos |= (s <= 0);
    }
    return is_nonpos;
}

bool ConvParams::use_cpu_depthwise3x3_winograd(const Tensor& input, const Tensor& weight) const {
#if defined(__ARM_NEON__)
    // Currently only 3x3 depthwise convolutions on tensors of float are supported.
    return (input.dim() == 4) &&
        (input.size(1) == groups) &&
        (weight.dim() == 4) &&
        (weight.size(0) % input.size(1) == 0) &&
        (weight.size(1) == 1) &&
        (weight.size(2) == 3) &&
        (weight.size(3) == 3) &&
        (input.device() == Device::CPU) &&
        (input.scalar_type() == ScalarType::Float) &&
        input.is_contiguous() &&
        (weight.device() == Device::CPU) &&
        (weight.scalar_type() == ScalarType::Float) &&
        weight.is_contiguous() &&
        !is_strided() &&
        !is_dilated() &&
        !transposed;
#else
    return false;
#endif
}

bool ConvParams::use_cpu_neon() const {
#if defined(__ARM_NEON__)
    return (input.scalar_type() == ScalarType::Float) &&
    (input.scalar_type() == ScalarType::Float) &&
    (input.device() == Device::CPU) &&
    (weight.device() == Device::CPU);
#else
    return false;
#endif
}

}   // end namespace otter
