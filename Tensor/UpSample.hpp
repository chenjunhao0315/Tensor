//
//  UpSample.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/3.
//

#ifndef UpSample_hpp
#define UpSample_hpp

#include "Macro.hpp"
#include "Exception.hpp"
#include "ArrayRef.hpp"
#include "DispatchStub.hpp"
#include <cmath>
#include <algorithm>
#include <array>

namespace otter {

class Tensor;

using upsampling_nearest2d_fn = void(*)(const Tensor& output, const Tensor& input, double scales_h, double scales_w);

DECLARE_DISPATCH(upsampling_nearest2d_fn, upsampling_nearest2d_stub);

static inline int64_t nearest_neighbor_compute_source_index(
    const float scale,
    int64_t dst_index,
    int64_t input_size) {
    // Index computation matching OpenCV INTER_NEAREST
    // which is buggy and kept for BC
    const int64_t src_index = std::min(static_cast<int64_t>(std::floor(dst_index * scale)), input_size - 1);
    return src_index;
}

template <typename scalar_t>
static inline scalar_t compute_scales_value(
    const double scale,
    int64_t input_size,
    int64_t output_size) {
    
    // see Note [compute_scales_value]
    // FIXME: remove magic > 0 after we ensure no models were serialized with -1 defaults.
    return (scale > 0.)
          ? static_cast<scalar_t>(1.0 / scale)
          : (static_cast<scalar_t>(input_size) / output_size);
}

template <typename scalar_t>
static inline scalar_t area_pixel_compute_scale(
    int64_t input_size,
    int64_t output_size,
    bool align_corners,
    const double scale) {
    // see Note [area_pixel_compute_scale]
    if(align_corners){
        if(output_size > 1) {
            return static_cast<scalar_t>(input_size - 1) / (output_size - 1);
        } else {
            return static_cast<scalar_t>(0);
        }
    } else{
        return compute_scales_value<scalar_t>(scale, input_size, output_size);
    }
}

template <typename scalar_t>
static inline scalar_t area_pixel_compute_source_index(
    scalar_t scale,
    int64_t dst_index,
    bool align_corners,
    bool cubic) {
    if (align_corners) {
        return scale * dst_index;
    } else {
        scalar_t src_idx = scale * (dst_index + 0.5) - 0.5;
        // [Note] Follow Opencv resize logic:
        // We allow negative src_idx here and later will use
        //   dx = src_idx - floorf(src_idx)
        // to compute the "distance"(which affects weights).
        // For linear modes, weight distribution doesn't matter
        // for negative indices as they use 2 pixels to interpolate.
        // For example, [-1, 0], they both use pixel 0 value so it
        // doesn't affect if we bound the src_idx to 0 or not.
        // TODO: Our current linear mode impls use unbound indices
        // where we should and then remove this cubic flag.
        // This matters in cubic mode, as we might need [-1, 0, 1, 2]
        // to interpolate and the weights can be affected.
        return (!cubic && src_idx < 0) ? scalar_t(0) : src_idx;
    }
}

static OTTER_UNUSED std::array<int64_t, 4> upsample_2d_common_check(IntArrayRef input_size, IntArrayRef output_size) {
    OTTER_CHECK(
                output_size.size() == 2,
                "It is expected output_size equals to 2, but got size ",
                output_size.size());

    OTTER_CHECK(
                input_size.size() == 4,
                "It is expected input_size equals to 4, but got size ",
                input_size.size());

    int64_t output_height = output_size[0];
    int64_t output_width = output_size[1];

    int64_t nbatch = input_size[0];
    int64_t channels = input_size[1];
    int64_t input_height = input_size[2];
    int64_t input_width = input_size[3];

    OTTER_CHECK(
                input_height > 0 && input_width > 0 && output_height > 0 && output_width > 0,
                "Input and output sizes should be greater than 0,"
                " but got input (H: ",
                input_height,
                ", W: ",
                input_width,
                ") output (H: ",
                output_height,
                ", W: ",
                output_width,
                ")");

    return {nbatch, channels, output_height, output_width};
}

}   // end namespace otter

#endif /* UpSample_hpp */
