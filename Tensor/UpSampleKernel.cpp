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
