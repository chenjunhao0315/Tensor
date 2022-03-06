//
//  UpSample.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/3.
//

#include "UpSample.hpp"
#include "TensorFunction.hpp"
#include "Accumulator.hpp"

namespace otter {

DEFINE_DISPATCH(upsampling_nearest2d_stub);

DEFINE_META_FUNCTION(upsample_nearest2d) (const Tensor& input, IntArrayRef output_size, double scales_h, double scales_w) {
    auto full_output_size = otter::upsample_2d_common_check(input.sizes(), output_size);

    // Allow for empty batch size but not other dimensions
    OTTER_CHECK(
                input.numel() != 0 || otter::multiply_integers(input.sizes().begin() + 1, input.sizes().end()),
                "Non-empty 4D data tensor expected but got a tensor with sizes ",
                input.sizes());

    set_output(0, full_output_size, {}, input.options().memory_format(input.suggest_memory_format()));
}

DEFINE_IMPL_FUNCTION(upsample_nearest2d_out_cpu) (const Tensor& input, IntArrayRef output_size, double scales_h, double scales_w, const Tensor& output) {
    upsampling_nearest2d_stub(Device::CPU, output, input, scales_h, scales_w);
}

}   // end namespace otter
