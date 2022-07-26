//
//  AffineGridGenerator.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/12.
//

#include "AffineGridGenerator.hpp"

#include "Tensor.hpp"
#include "TensorMaker.hpp"
#include "TensorFactory.hpp"
#include "TensorOperator.hpp"

namespace otter {

Tensor linspace_from_neg_one(const Tensor& grid, int64_t num_steps, bool align_corners) {
    if (num_steps <= 1) {
        return otter::tensor(0, grid.options());
    }
    auto range = otter::linspace(-1, 1, num_steps, grid.options());
    if (!align_corners) {
        range = range * (num_steps - 1) / num_steps;
    }
    return range;
}
Tensor make_base_grid_4D(
    const Tensor& theta,
    int64_t N,
    int64_t /*C*/,
    int64_t H,
    int64_t W,
    bool align_corners) {
    auto base_grid = otter::empty({N, H, W, 3}, theta.options());
    base_grid.select(-1, 0).copy_(linspace_from_neg_one(theta, W, align_corners));
    base_grid.select(-1, 1).copy_(linspace_from_neg_one(theta, H, align_corners).unsqueeze_(-1));
    base_grid.select(-1, 2).fill_(1);
    return base_grid;
}
Tensor make_base_grid_5D(
    const Tensor& theta,
    int64_t N,
    int64_t /*C*/,
    int64_t D,
    int64_t H,
    int64_t W,
    bool align_corners) {
    auto base_grid = otter::empty({N, D, H, W, 4}, theta.options());
    base_grid.select(-1, 0).copy_(linspace_from_neg_one(theta, W, align_corners));
    base_grid.select(-1, 1).copy_(linspace_from_neg_one(theta, H, align_corners).unsqueeze_(-1));
    base_grid.select(-1, 2).copy_(linspace_from_neg_one(theta, D, align_corners).unsqueeze_(-1).unsqueeze_(-1));
    base_grid.select(-1, 3).fill_(1);
    return base_grid;
}
Tensor affine_grid_generator_4D(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    bool align_corners) {
    Tensor base_grid = make_base_grid_4D(theta, N, C, H, W, align_corners);
    auto grid = base_grid.view({N, H * W, 3}).bmm(theta.transpose(1, 2));
    return grid.view({N, H, W, 2});
}
Tensor affine_grid_generator_5D(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    bool align_corners) {
    Tensor base_grid = make_base_grid_5D(theta, N, C, D, H, W, align_corners);
    auto grid = base_grid.view({N, D * H * W, 4}).bmm(theta.transpose(1, 2));
    return grid.view({N, D, H, W, 3});
}
Tensor affine_grid_generator(const Tensor& theta, IntArrayRef size, bool align_corners) {
    OTTER_CHECK(size.size() == 4 || size.size() == 5,
                "AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs.");
    if (size.size() == 4) {
        return affine_grid_generator_4D(theta, size[0], size[1], size[2], size[3], align_corners);
    } else {
        return affine_grid_generator_5D(theta, size[0], size[1], size[2], size[3], size[4], align_corners);
    }
}

}   // end namespace otter
