//
//  ConvolutionUtils.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/15.
//

#ifndef ConvolutionUtils_hpp
#define ConvolutionUtils_hpp

#include <vector>
#include "Tensor.hpp"

namespace otter {

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
    bool use_cpu_depthwise3x3_winograd(const Tensor& input, const Tensor& weight) const;
};

enum class ConvBackend {
    Winograd3x3Depthwise,
    SlowDilated2d,
    SlowDilated3d,
    Slow2d,
    Slow3d,
    Overrideable
};

inline std::vector<int64_t> expand_param_if_needed(IntArrayRef list_param, const char* param_name, int64_t expected_dim) {
    if (list_param.size() == 1) {
        return std::vector<int64_t>(expected_dim, list_param[0]);
    } else if ((int64_t)list_param.size() != expected_dim) {
        assert(false);  // Expected param to be a single integer value or a list of expected_dim values to match the convolution dimension, but got param_name = list_param
    } else {
        return list_param.vec();
    }
}

}

#endif /* ConvolutionUtils_hpp */
