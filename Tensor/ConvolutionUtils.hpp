//
//  ConvolutionUtils.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/15.
//

#ifndef ConvolutionUtils_hpp
#define ConvolutionUtils_hpp

#include <vector>
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
    bool is_depthwise(const Tensor& input, const Tensor& weight) const;
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
    Sgemm2dNeon,
    Sgemm2dNeon_1x1s1,
    Sgemm2dNeon_1x1s2,
    Sgemm2dX86,
    SlideWin2dNeon_1x1s1,
    SlideWin2d,
    DepthwiseNeon_3x3s2,
    DepthwiseNeon_5x5s1,
    DepthwiseNeon_5x5s2,
    DepthwiseX86_3x3s1,
    DepthwiseX86_3x3s2,
    Slow3d,
    Overrideable
};

inline std::vector<int64_t> expand_param_if_needed(IntArrayRef list_param, const char* param_name, int64_t expected_dim) {
    if (list_param.size() == 1) {
        return std::vector<int64_t>(expected_dim, list_param[0]);
    } else if ((int64_t)list_param.size() != expected_dim) {
        assert(false);  // Expected param to be a single integer value or a list of expected_dim values to match the convolution dimension, but got param_name = list_param
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

}

#endif /* ConvolutionUtils_hpp */
