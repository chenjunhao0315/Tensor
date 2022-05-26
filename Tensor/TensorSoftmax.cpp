//
//  TensorSoftmax.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/5/26.
//

#include "Tensor.hpp"
#include "TensorSoftmax.hpp"
#include "TensorFactory.hpp"
#include "TensorFunction.hpp"
#include "WarpDimMinimal.hpp"

namespace otter {

DEFINE_DISPATCH(softmax_kernel);
DEFINE_DISPATCH(softmax_lastdim_kernel);

DEFINE_META_FUNCTION(softmax)(const Tensor& input, const int64_t dim, const bool half_to_float) {
    int64_t dim_ = maybe_wrap_dim(dim, input.dim());
    
    auto output_options = input.options().memory_format(otter::MemoryFormat::Contiguous);

    if (half_to_float) {
        output_options = output_options.dtype(ScalarType::Float);
    }
    
    int64_t input_dim = input.dim() > 0 ? input.dim() : 1;
    
    OTTER_CHECK(dim_ >= 0 && dim_ < input_dim, "dim must be non-negative and less than input dimensions");
    
    set_output(0, input.sizes(), {}, output_options);
}

DEFINE_IMPL_FUNCTION(softmax_cpu_out)(const Tensor& input, const int64_t dim, const bool half_to_float, const Tensor& output) {
    OTTER_CHECK(!half_to_float, "softmax with half to float conversion is not supported on CPU");

    if (input.numel() == 0) {
        return;
    }

    auto input_ = input.contiguous();
    int64_t dim_ = maybe_wrap_dim(dim, input_.dim());

    if (input_.dim() == 0) {
        input_ = input_.view({1});
    }

    OTTER_CHECK(dim_ >= 0 && dim_ < input_.dim(), "dim must be non-negative and less than input dimensions");

    if (input_.dim() > 0 && dim_ == input_.dim() - 1) {
        softmax_lastdim_kernel(Device::CPU, output, input_);
    } else {
        softmax_kernel(Device::CPU, output, input_, dim_);
    }
}


}   // end namespace otter
