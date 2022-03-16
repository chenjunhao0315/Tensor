//
//  Pool.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/1.
//

#include "TensorFunction.hpp"
#include "Pool.hpp"

namespace otter {

DEFINE_DISPATCH(max_pool2d_stub);

DEFINE_META_FUNCTION(max_pool2d_with_indices) (const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
    OTTER_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2, "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
    const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
    const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);
    
    OTTER_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2, "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
    const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
    const int dW = stride.empty() ? kW : stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);
    
    OTTER_CHECK(padding.size() == 1 || padding.size() == 2, "max_pool2d: padding must be either be a single int, or a tuple of two ints");
    const int padH = safe_downcast<int, int64_t>(padding[0]);
    const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);
    
    OTTER_CHECK(dilation.size() == 1 || dilation.size() == 2, "max_pool2d: dilation must be either a single int, or a tuple of two ints");
    const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
    const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);
    
    const auto memory_format = input.suggest_memory_format();
    
    if (memory_format == MemoryFormat::ChannelsLast) {
        OTTER_CHECK(input.dim() == 4, "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
    } else if (memory_format == MemoryFormat::Contiguous) {
        OTTER_CHECK((input.dim() == 3 || input.dim() == 4), "non-empty 3D or 4D (batch mode) tensor expected for input");
    } else {
        OTTER_CHECK(false, "Unsupport memory format. Supports only ChannelsLast, Contiguous");
    }
    
    const int64_t nbatch = input.dim() == 4 ? input.size(-4) : 1;
    const int64_t nInputPlane = input.size(-3);
    const int64_t inputHeight = input.size(-2);
    const int64_t inputWidth = input.size(-1);

    const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
    const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);
    
    pool2d_shape_check(
                       input,
                       kH, kW, dH, dW, padH, padW, dilationH, dilationW,
                       nInputPlane,
                       inputHeight, inputWidth,
                       outputHeight, outputWidth, memory_format);
    
    if (input.dim() == 3) {
        set_output(0, {nInputPlane, outputHeight, outputWidth}, {}, input.options().memory_format(memory_format));
        /* indices will contain the locations for each output point */
        set_output(1, {nInputPlane, outputHeight, outputWidth}, {}, input.options().memory_format(memory_format).dtype(ScalarType::Long));
    } else {
        set_output(0, {nbatch, nInputPlane, outputHeight, outputWidth}, {}, input.options().memory_format(memory_format));
        /* indices will contain the locations for each output point */
        set_output(1, {nbatch, nInputPlane, outputHeight, outputWidth}, {}, input.options().memory_format(memory_format).dtype(ScalarType::Long));
    }
}

DEFINE_IMPL_FUNCTION(max_pool2d_with_indices_out_cpu) (const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor& output, const Tensor& indices) {
    
    const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
    const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

    const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
    const int dW = stride.empty() ? kW : stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

    const int padH = safe_downcast<int, int64_t>(padding[0]);
    const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

    const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
    const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);
    
    max_pool2d_stub(Device::CPU, output, indices, input, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
}

Tensor max_pool2d(const Tensor& self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
    auto output_and_indices = otter::native::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
    
    return std::get<0>(output_and_indices);
}

}   // end namespace otter
