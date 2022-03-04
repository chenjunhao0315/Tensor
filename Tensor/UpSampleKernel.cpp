//
//  UpSampleKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/3.
//

#include "UpSample.hpp"
#include "UpSampleKernel.hpp"
#include "Dispatch.hpp"
#include "Tensor.hpp"

namespace otter {

template <int out_ndims, typename scale_type, class F>
void upsample_generic_Nd_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    bool align_corners,
    const scale_type& scales) {
    
    // input can be NCHW, NCL or NCKHW
    auto shape = input.sizes().vec();
    auto strides = input.strides().vec();
    auto oshape = output.sizes();

    OTTER_INTERNAL_ASSERT(shape.size() == oshape.size() && shape.size() == 2 + out_ndims);
    OTTER_INTERNAL_ASSERT(strides.size() == 2 + out_ndims);

    for (const auto i : otter::irange(out_ndims)) {
        shape[i + 2] = oshape[i + 2];
        strides[i + 2] = 0;
    }
    auto restrided_input = input.as_strided(shape, strides);
}

void upsample_nearest2d_kernel_impl(
    const Tensor& output,
    const Tensor& input,
    double scales_h,
    double scales_w) {
    
    if (input.is_contiguous(MemoryFormat::ChannelsLast)) {
        OTTER_DISPATCH_FLOATING_TYPES(input.scalar_type(), "upsample_nearest2d_channels_last", [&] {
//            cpu_upsample_nearest_channels_last<scalar_t, scale_t, nearest_idx>(output, input, {scales_h, scales_w});
        });
    } else {
//        upsample_generic_Nd_kernel_impl<2, scale_t, HelperInterpNearest>(output, input, false, {scales_h, scales_w});
    }
}

}   // end namespace otter
