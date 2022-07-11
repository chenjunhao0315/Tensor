//
//  GridSampler.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/11.
//

#include "GridSampler.hpp"
#include "Tensor.hpp"
#include "Parallel.hpp"
#include "Dispatch.hpp"
#include "DispatchStub.hpp"
#include "TensorFactory.hpp"
#include "Vec.hpp"
#include "UpSample.hpp"

#include "GridSamplerKernel.hpp"

namespace otter {

template<typename scalar_t>
Tensor grid_sampler_3d_cpu_impl(const Tensor& input, const Tensor& grid,
                                GridSamplerInterpolation interpolation_mode,
                                GridSamplerPadding padding_mode,
                                bool align_corners) {
    // See NOTE [ grid_sampler Native Functions ].
    // Add checks here in case this is called instead of grid_sampler.
    check_grid_sampler_common(input, grid);
    check_grid_sampler_3d(
                          input, grid, static_cast<int64_t>(interpolation_mode));
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t inp_D = input.size(2);
    int64_t inp_H = input.size(3);
    int64_t inp_W = input.size(4);
    int64_t out_D = grid.size(1);
    int64_t out_H = grid.size(2);
    int64_t out_W = grid.size(3);
    auto output = otter::empty({N, C, out_D, out_H, out_W}, input.options());
    int64_t inp_sN = input.stride(0);
    int64_t inp_sC = input.stride(1);
    int64_t inp_sD = input.stride(2);
    int64_t inp_sH = input.stride(3);
    int64_t inp_sW = input.stride(4);
    int64_t grid_sN = grid.stride(0);
    int64_t grid_sD = grid.stride(1);
    int64_t grid_sH = grid.stride(2);
    int64_t grid_sW = grid.stride(3);
    int64_t grid_sCoor = grid.stride(4);
    int64_t out_sN = output.stride(0);
    int64_t out_sC = output.stride(1);
    int64_t out_sD = output.stride(2);
    int64_t out_sH = output.stride(3);
    int64_t out_sW = output.stride(4);
    scalar_t *inp_ptr = input.data_ptr<scalar_t>();
    scalar_t *out_ptr = output.data_ptr<scalar_t>();
    scalar_t *grid_ptr = grid.data_ptr<scalar_t>();
    // loop over each output pixel
    otter::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
        for (const auto n : otter::irange(start, end)) {
            scalar_t *grid_ptr_N = grid_ptr + n * grid_sN;
            scalar_t *inp_ptr_N = inp_ptr + n * inp_sN;
            for (const auto d : otter::irange(out_D)) {
                for (const auto h : otter::irange(out_H)) {
                    for (const auto w : otter::irange(out_W)) {
                        // get the corresponding input x, y, z co-ordinates from grid
                        scalar_t *grid_ptr_NDHW = grid_ptr_N + d * grid_sD + h * grid_sH + w * grid_sW;
                        scalar_t ix = *grid_ptr_NDHW;
                        scalar_t iy = grid_ptr_NDHW[grid_sCoor];
                        scalar_t iz = grid_ptr_NDHW[2 * grid_sCoor];
                        ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode, align_corners);
                        iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode, align_corners);
                        iz = grid_sampler_compute_source_index(iz, inp_D, padding_mode, align_corners);
                        if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
                            // get corner pixel values from (x, y, z)
                            // for 4d, we used north-east-south-west
                            // for 5d, we add top-bottom
                            int64_t ix_tnw = static_cast<int64_t>(std::floor(ix));
                            int64_t iy_tnw = static_cast<int64_t>(std::floor(iy));
                            int64_t iz_tnw = static_cast<int64_t>(std::floor(iz));
                            int64_t ix_tne = ix_tnw + 1;
                            int64_t iy_tne = iy_tnw;
                            int64_t iz_tne = iz_tnw;
                            int64_t ix_tsw = ix_tnw;
                            int64_t iy_tsw = iy_tnw + 1;
                            int64_t iz_tsw = iz_tnw;
                            int64_t ix_tse = ix_tnw + 1;
                            int64_t iy_tse = iy_tnw + 1;
                            int64_t iz_tse = iz_tnw;
                            int64_t ix_bnw = ix_tnw;
                            int64_t iy_bnw = iy_tnw;
                            int64_t iz_bnw = iz_tnw + 1;
                            int64_t ix_bne = ix_tnw + 1;
                            int64_t iy_bne = iy_tnw;
                            int64_t iz_bne = iz_tnw + 1;
                            int64_t ix_bsw = ix_tnw;
                            int64_t iy_bsw = iy_tnw + 1;
                            int64_t iz_bsw = iz_tnw + 1;
                            int64_t ix_bse = ix_tnw + 1;
                            int64_t iy_bse = iy_tnw + 1;
                            int64_t iz_bse = iz_tnw + 1;
                            // get surfaces to each neighbor:
                            scalar_t tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
                            scalar_t tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
                            scalar_t tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
                            scalar_t tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
                            scalar_t bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
                            scalar_t bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
                            scalar_t bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
                            scalar_t bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);
                            // calculate bilinear weighted pixel value and set output pixel
                            scalar_t *out_ptr_NCDHW = out_ptr + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
                            scalar_t *inp_ptr_NC = inp_ptr_N;
                            for (int64_t c = 0; c < C; ++c, out_ptr_NCDHW += out_sC, inp_ptr_NC += inp_sC) {
                                //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
                                // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
                                // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) * bne
                                // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) * bse
                                *out_ptr_NCDHW = static_cast<scalar_t>(0);
                                if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
                                    *out_ptr_NCDHW += inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] * tnw;
                                }
                                if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
                                    *out_ptr_NCDHW += inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] * tne;
                                }
                                if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
                                    *out_ptr_NCDHW += inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] * tsw;
                                }
                                if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
                                    *out_ptr_NCDHW += inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] * tse;
                                }
                                if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
                                    *out_ptr_NCDHW += inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] * bnw;
                                }
                                if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
                                    *out_ptr_NCDHW += inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] * bne;
                                }
                                if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
                                    *out_ptr_NCDHW += inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] * bsw;
                                }
                                if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
                                    *out_ptr_NCDHW += inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] * bse;
                                }
                            }
                        } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
                            int64_t ix_nearest = static_cast<int64_t>(std::round(ix));
                            int64_t iy_nearest = static_cast<int64_t>(std::round(iy));
                            int64_t iz_nearest = static_cast<int64_t>(std::round(iz));
                            // assign nearest neighor pixel value to output pixel
                            scalar_t *out_ptr_NCDHW = out_ptr + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
                            scalar_t *inp_ptr_NC = inp_ptr_N;
                            for (int64_t c = 0; c < C; ++c, out_ptr_NCDHW += out_sC, inp_ptr_NC += inp_sC) {
                                if (within_bounds_3d(iz_nearest, iy_nearest, ix_nearest, inp_D, inp_H, inp_W)) {
                                    *out_ptr_NCDHW = inp_ptr_NC[iz_nearest * inp_sD + iy_nearest * inp_sH + ix_nearest * inp_sW];
                                } else {
                                    *out_ptr_NCDHW = static_cast<scalar_t>(0);
                                }
                            }
                        }
                    }
                }
            }
        }
    });
    return output;
}

Tensor _grid_sampler_2d_cpu_fallback(const Tensor& input, const Tensor& grid,
                                     int64_t interpolation_mode_,
                                     int64_t padding_mode_,
                                     bool align_corners) {
    // See NOTE [ grid_sampler Native Functions ].
    // Add checks here in case this is called instead of grid_sampler.
    check_grid_sampler_common(input, grid);
    check_grid_sampler_2d(input, grid);
    auto interpolation_mode = static_cast<GridSamplerInterpolation>(interpolation_mode_);
    auto padding_mode = static_cast<GridSamplerPadding>(padding_mode_);
    using scalar_t = float;
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t inp_H = input.size(2);
    int64_t inp_W = input.size(3);
    int64_t out_H = grid.size(1);
    int64_t out_W = grid.size(2);
    auto output = otter::empty({N, C, out_H, out_W}, input.options());
    int64_t inp_sN = input.stride(0);
    int64_t inp_sC = input.stride(1);
    int64_t inp_sH = input.stride(2);
    int64_t inp_sW = input.stride(3);
    int64_t grid_sN = grid.stride(0);
    int64_t grid_sH = grid.stride(1);
    int64_t grid_sW = grid.stride(2);
    int64_t grid_sCoor = grid.stride(3);
    int64_t out_sN = output.stride(0);
    int64_t out_sC = output.stride(1);
    int64_t out_sH = output.stride(2);
    int64_t out_sW = output.stride(3);
    scalar_t *inp_ptr = input.data_ptr<scalar_t>();
    scalar_t *out_ptr = output.data_ptr<scalar_t>();
    scalar_t *grid_ptr = grid.data_ptr<scalar_t>();
    // loop over each output pixel
    otter::parallel_for(0, N, 0, [&](int64_t start, int64_t end) {
        for (const auto n : otter::irange(start, end)) {
            scalar_t *grid_ptr_N = grid_ptr + n * grid_sN;
            scalar_t *inp_ptr_N = inp_ptr + n * inp_sN;
            for (const auto h : otter::irange(out_H)) {
                for (const auto w : otter::irange(out_W)) {
                    // get the corresponding input x, y, z co-ordinates from grid
                    scalar_t *grid_ptr_NHW = grid_ptr_N + h * grid_sH + w * grid_sW;
                    scalar_t x = *grid_ptr_NHW;
                    scalar_t y = grid_ptr_NHW[grid_sCoor];
                    scalar_t ix = grid_sampler_compute_source_index(x, inp_W, padding_mode, align_corners);
                    scalar_t iy = grid_sampler_compute_source_index(y, inp_H, padding_mode, align_corners);
                    if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
                        // get corner pixel values from (x, y)
                        // for 4d, we use north-east-south-west
                        int64_t ix_nw = static_cast<int64_t>(std::floor(ix));
                        int64_t iy_nw = static_cast<int64_t>(std::floor(iy));
                        int64_t ix_ne = ix_nw + 1;
                        int64_t iy_ne = iy_nw;
                        int64_t ix_sw = ix_nw;
                        int64_t iy_sw = iy_nw + 1;
                        int64_t ix_se = ix_nw + 1;
                        int64_t iy_se = iy_nw + 1;
                        // get surfaces to each neighbor:
                        scalar_t nw = (ix_se - ix)    * (iy_se - iy);
                        scalar_t ne = (ix    - ix_sw) * (iy_sw - iy);
                        scalar_t sw = (ix_ne - ix)    * (iy    - iy_ne);
                        scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);
                        // calculate bilinear weighted pixel value and set output pixel
                        scalar_t *inp_ptr_NC = inp_ptr_N;
                        scalar_t *out_ptr_NCHW = out_ptr + n * out_sN + h * out_sH + w * out_sW;
                        for (int64_t c = 0; c < C; ++c, out_ptr_NCHW += out_sC, inp_ptr_NC += inp_sC) {
                            auto res = static_cast<scalar_t>(0);
                            if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
                                res += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
                            }
                            if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
                                res += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
                            }
                            if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
                                res += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
                            }
                            if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
                                res += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
                            }
                            *out_ptr_NCHW = res;
                        }
                    } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
                        int64_t ix_nearest = static_cast<int64_t>(std::nearbyint(ix));
                        int64_t iy_nearest = static_cast<int64_t>(std::nearbyint(iy));
                        // assign nearest neighor pixel value to output pixel
                        scalar_t *out_ptr_NCHW = out_ptr + n * out_sN + h * out_sH + w * out_sW;
                        scalar_t *inp_ptr_NC = inp_ptr_N;
                        for (int64_t c = 0; c < C; ++c, out_ptr_NCHW += out_sC, inp_ptr_NC += inp_sC) {
                            if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
                                *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
                            } else {
                                *out_ptr_NCHW = static_cast<scalar_t>(0);
                            }
                        }
                    } else if (interpolation_mode == GridSamplerInterpolation::Bicubic) {
                        // grid_sampler_compute_source_index will "clip the value" of idx depends on the padding,
                        // which would cause calculation to be wrong,
                        // for example x = -0.1 -> ix = 0 for zero padding, but in bicubic ix = floor(x) = -1
                        // There would be more problem in reflection padding, since the -1 and +1 direction is not fixed in boundary condition
                        ix = grid_sampler_unnormalize(x, inp_W, align_corners);
                        iy = grid_sampler_unnormalize(y, inp_H, align_corners);
                        scalar_t ix_nw = std::floor(ix);
                        scalar_t iy_nw = std::floor(iy);
                        const scalar_t tx = ix - ix_nw;
                        const scalar_t ty = iy - iy_nw;
                        scalar_t *inp_ptr_NC = inp_ptr_N;
                        scalar_t *out_ptr_NCHW = out_ptr + n * out_sN + h * out_sH + w * out_sW;
                        for (int64_t c = 0; c < C; ++c, out_ptr_NCHW += out_sC, inp_ptr_NC += inp_sC) {
                            // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
                            scalar_t coefficients[4];
                            // Interpolate 4 values in the x directon
                            for (const auto i : otter::irange(4)) {
                                coefficients[i] = cubic_interp1d<scalar_t>(
                                                                           get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw - 1, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
                                                                           get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 0, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
                                                                           get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 1, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
                                                                           get_value_bounded<scalar_t>(inp_ptr_NC, ix_nw + 2, iy_nw - 1 + i, inp_W, inp_H, inp_sW, inp_sH, padding_mode, align_corners),
                                                                           tx);
                            }
                            // Interpolate in the y direction
                            *out_ptr_NCHW = cubic_interp1d<scalar_t>(
                                                                     coefficients[0],
                                                                     coefficients[1],
                                                                     coefficients[2],
                                                                     coefficients[3],
                                                                     ty);
                        }
                    }
                }
            }
        }
    });
    return output;
}

Tensor grid_sampler_2d_cpu(const Tensor& input, const Tensor& grid,
                           int64_t interpolation_mode, int64_t padding_mode,
                           bool align_corners) {
    // See NOTE [ grid_sampler Native Functions ].
    // Add checks here in case this is called instead of grid_sampler.
    check_grid_sampler_common(input, grid);
    check_grid_sampler_2d(input, grid);

    // AVX gather instructions use signed 32-bit offsets to gather float values.
    // Check for possible overflow and fallback to scalar implementation
    if (input.scalar_type() != otter::ScalarType::Double) {
        OTTER_CHECK(input.scalar_type() == otter::ScalarType::Float,
                    "grid_sampler_2d_cpu not implemented for ", input.scalar_type());
        auto sizes = input.sizes();
        auto strides = input.strides();
        const auto grid_sW = grid.strides()[2];
        // NOTE: Gather offsets are only used for the input H, W dimensions
        //       or only for strided access to the grid tensor
        auto max_gather_offset = std::max(
                                          (sizes[2] - 1) * strides[2] + (sizes[3] - 1) * strides[3],
                                          grid_sW * (vec::Vectorized<float>::size() - 1));
        if (max_gather_offset > std::numeric_limits<int32_t>::max()) {
            return otter::_grid_sampler_2d_cpu_fallback(
                                                         input, grid, interpolation_mode, padding_mode, align_corners);
        }
    }
    auto in_size = input.sizes();
    auto grid_size = grid.sizes();
    auto output = otter::empty(
                               {in_size[0], in_size[1], grid_size[1], grid_size[2]}, input.options());
    grid_sampler_2d_cpu_kernel(
                               Device::CPU, output, input, grid, interpolation_mode, padding_mode, align_corners);
    return output;
}
DEFINE_DISPATCH(grid_sampler_2d_cpu_kernel);
Tensor grid_sampler_3d_cpu(const Tensor& input, const Tensor& grid,
                           int64_t interpolation_mode, int64_t padding_mode,
                           bool align_corners) {
    // See NOTE [ grid_sampler Native Functions ].
    // Add checks here in case this is called instead of grid_sampler.
    check_grid_sampler_common(input, grid);
    check_grid_sampler_3d(input, grid, interpolation_mode);
    return OTTER_DISPATCH_FLOATING_TYPES(input.scalar_type(), "grid_sampler3d_cpu", [&] {
        return grid_sampler_3d_cpu_impl<scalar_t>(
                                                  input, grid, static_cast<GridSamplerInterpolation>(interpolation_mode),
                                                  static_cast<GridSamplerPadding>(padding_mode), align_corners);
    });
}

// See NOTE [ grid_sampler Native Functions ].
Tensor grid_sampler(const Tensor& input,
                    const Tensor& grid,
                    int64_t interpolation_mode,
                    int64_t padding_mode,
                    bool align_corners
                    ) {
    if (input.dim() == 4) {
        return otter::grid_sampler_2d_cpu(
                                      input, grid, interpolation_mode, padding_mode, align_corners);
    } else {
        return otter::grid_sampler_3d_cpu(
                                      input, grid, interpolation_mode, padding_mode, align_corners);
    }
}

}   // end namespace otter
