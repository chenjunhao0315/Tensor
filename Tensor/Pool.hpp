//
//  Pool.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/1.
//

#ifndef Pool_hpp
#define Pool_hpp

#include "Math.hpp"
#include "DispatchStub.hpp"
#include "Tensor.hpp"

namespace otter {

using max_pool2d_fn = void(*)(const Tensor& output, const Tensor& indices, const Tensor& input, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH);
DECLARE_DISPATCH(max_pool2d_fn, max_pool2d_stub);

using avg_pool2d_fn = void(*)(const Tensor& output, const Tensor& input, int64_t kW, int64_t kH, int64_t dW, int64_t dH, int64_t padW, int64_t padH, bool count_include_pad, int64_t divisor_override);
DECLARE_DISPATCH(avg_pool2d_fn, avg_pool2d_kernel);

Tensor max_pool2d(const Tensor& self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode);

template <typename dest_t, typename src_t>
static inline dest_t
safe_downcast(src_t v) {
    OTTER_CHECK(std::numeric_limits<dest_t>::min() <= v && v <= std::numeric_limits<dest_t>::max(), "integer out of range");

    return static_cast<dest_t>(v);
}

template<typename T>
static inline T pooling_output_shape_pad_lr(T inputSize, T kernelSize, T pad_l, T pad_r, T stride, T dilation, bool ceil_mode) {
    T outputSize = div_round_up<T>(
                              inputSize + pad_l + pad_r - dilation * (kernelSize - 1) - 1 +
                              (ceil_mode ? stride - 1 : 0), stride) + 1;
    if (ceil_mode) {
        if ((outputSize - 1) * stride >= inputSize + pad_l) {
            --outputSize;
        }
    }
    return outputSize;
}

template<typename T>
static inline T pooling_output_shape(T inputSize, T kernelSize, T pad, T stride, T dilation, bool ceil_mode) {
    OTTER_CHECK(stride != 0, "stride should not be zero");
    return pooling_output_shape_pad_lr(inputSize, kernelSize, pad, pad, stride, dilation, ceil_mode);
}

inline std::pair<int64_t, int64_t> pooling_same_mode_padding_lr(int64_t inputSize, int64_t kernelSize, int64_t stride, int64_t dilation) {
    auto total_padding = dilation * (kernelSize - 1);

    // Prefer symmetric padding if possible
    if (stride > 2 && (total_padding % 2 == 1)) {
        // The floor in the output size calculation gives us a little wiggle room
        auto wiggle_room = inputSize % stride - 1;
        if (wiggle_room > 0) {
            --total_padding;
        }
    }

    auto left = total_padding / 2;
    return {left, total_padding - left};
}

static inline void
pool2d_shape_check(
    const Tensor& input,
    int kH, int kW, int dH, int dW, int padH, int padW, int dilationH, int dilationW,
    int64_t nInputPlane,
    int64_t inputHeight, int64_t inputWidth,
    int64_t outputHeight, int64_t outputWidth, MemoryFormat memory_format) {
    
    const int64_t ndim = input.dim();
    const int64_t nOutputPlane = nInputPlane;

    OTTER_CHECK(kW > 0 && kH > 0,
                "kernel size should be greater than zero, but got ",
                "kH: ", kH, " kW: ", kW);
    OTTER_CHECK(dW > 0 && dH > 0,
                "stride should be greater than zero, but got "
                "dH: ", dH, " dW: ", dW);
    OTTER_CHECK(dilationH > 0 && dilationW > 0,
                "dilation should be greater than zero, but got ",
                "dilationH: ", dilationH, " dilationW: ", dilationW);

    bool valid_dims = input.size(1) != 0 && input.size(2) != 0;
    if (memory_format == MemoryFormat::ChannelsLast){
        // Expect tensor in NHWC format and allow 0-dim only for N.
        OTTER_CHECK((ndim == 4 && valid_dims && input.size(3) != 0),
                    "Expected 4D (batch mode) tensor expected for input with channels_last layout"
                    " with optional 0 dim batch size for input, but got: ", input.sizes());
    } else {
        OTTER_CHECK((ndim == 3 && input.size(0) != 0 && valid_dims) ||
                    (ndim == 4 && valid_dims && input.size(3) != 0),
                    "Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input, but got:",
                    input.sizes());
    }

    OTTER_CHECK(kW/2 >= padW && kH/2 >= padH,
                "pad should be smaller than or equal to half of kernel size, but got ",
                "padW = ", padW, ", padH = ", padH, ", kW = ", kW, ", kH = ", kH);

    OTTER_CHECK(outputWidth >= 1 && outputHeight >= 1,
                "Given input size: (",
                nInputPlane, "x", inputHeight, "x", inputWidth, "). ",
                "Calculated output size: (",
                nOutputPlane, "x", outputHeight, "x", outputWidth, "). ",
                "Output size is too small");
}

}   // end namespace otter

#endif /* Pool_hpp */
