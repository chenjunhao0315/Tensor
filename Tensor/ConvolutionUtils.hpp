//
//  ConvolutionUtils.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/15.
//

#ifndef ConvolutionUtils_hpp
#define ConvolutionUtils_hpp

#include <vector>
#include "Exception.hpp"
#include "ArrayRef.hpp"

namespace otter {

class Tensor;

struct ConvParams {
    std::vector<int64_t> stride;
    std::vector<int64_t> padding;
    std::vector<int64_t> dilation;
    std::vector<int64_t> output_padding;
    bool transposed;
    bool benchmark;
    int64_t groups;
    
    void view_1d_as_2d();
    bool is_strided() const;
    bool is_dilated() const;
    bool is_padded() const;
    bool is_padding_neg() const;
    bool is_output_padding_neg() const;
    bool is_stride_nonpos() const;
    bool is_int8(const Tensor& input, const Tensor& weight) const;
    bool is_depthwise(const Tensor& input, const Tensor& weight) const;
    bool is_transpose_depthwise(const Tensor& input, const Tensor& weight) const;
    bool use_cpu_depthwise3x3_winograd(const Tensor& input, const Tensor& weight) const;
    bool use_cpu_neon(const Tensor& input, const Tensor& weight) const;
    bool use_cpu_x86(const Tensor& input, const Tensor& weight) const;
};

enum class ConvBackend {
    Winograd3x3Depthwise,
    SlowDilated2d,
    SlowDilated3d,
    Slow2d,
    SlowTranspose2d,
    SlideWinTranspose2d,
    Transpose2dNeon_4x4s2,
    Sgemm2dNeon,
    Sgemm2dNeon_1x1s1,
    Sgemm2dNeon_1x1s2,
    Sgemm2dX86,
    SlideWin2dNeon_1x1s1,
    SlideWin2dNeon_3x3s1,
    WinogradNeon_3x3s1,
    Packed2DNeon_3x3s2,
    WinogradX86_3x3s1,
    SlideWin2d,
    SlideWin2dInt8,
    DepthwiseNeon_3x3s1,
    DepthwiseNeon_3x3s2,
    DepthwiseNeon_5x5s1,
    DepthwiseNeon_5x5s2,
    DepthwiseX86_3x3s1,
    DepthwiseX86_3x3s2,
    DepthwiseTransposeNeon,
    Slow3d,
    Overrideable
};

inline std::vector<int64_t> expand_param_if_needed(IntArrayRef list_param, const char* /*param_name*/, int64_t expected_dim) {
    if (list_param.size() == 1) {
        return std::vector<int64_t>(expected_dim, list_param[0]);
    } else if ((int64_t)list_param.size() != expected_dim) {
        OTTER_CHECK(false, "Expected param to be a single integer value or a list of expected_dim values to match the convolution dimension, but got param_name = list_param");
    }
    return list_param.vec();
}

inline std::vector<int64_t> calculate_conv_output_size(
    const IntArrayRef input_size,
    const IntArrayRef weight_size,
    const IntArrayRef stride,
    const IntArrayRef padding) {
    
    const auto calc_output_dimension = [](
        const int64_t input, const int64_t kernel, const int64_t stride, const int64_t padding) {
            return 1 + (input - kernel + 2 * padding) / stride;
        };

    return std::vector<int64_t> {
        input_size[0],
        weight_size[0],
        calc_output_dimension(input_size[2], weight_size[2], stride[0], padding[0]),
        calc_output_dimension(input_size[3], weight_size[3], stride[1], padding[1]),
    };
}

inline std::vector<int64_t> calculate_deconv_output_size_without_padding(
    const IntArrayRef input_size,
    const IntArrayRef weight_size,
    const IntArrayRef stride,
    const IntArrayRef dilation,
    const IntArrayRef padding,
    const IntArrayRef output_padding) {
    
    const auto calc_output_dimension = [](
        const int64_t input, const int64_t kernel, const int64_t stride, const int64_t dilation, const int64_t /*padding*/, const int64_t output_padding) {
            return (input - 1) * stride + (dilation * (kernel - 1) + 1) + output_padding;
        };

    return std::vector<int64_t> {
        input_size[0],
        weight_size[1],
        calc_output_dimension(input_size[2], weight_size[2], stride[0], dilation[0], padding[0], output_padding[0]),
        calc_output_dimension(input_size[3], weight_size[3], stride[1], dilation[1], padding[1], output_padding[1]),
    };
}

}

#endif /* ConvolutionUtils_hpp */
