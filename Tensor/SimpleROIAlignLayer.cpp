//
//  SimpleROIAlignLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/12.
//

#include "SimpleROIAlignLayer.hpp"
#include "Tensor.hpp"
#include "TensorMaker.hpp"
#include "TensorFactory.hpp"
#include "AffineGridGenerator.hpp"
#include "TensorOperator.hpp"
#include "GridSampler.hpp"
#include "TensorShape.hpp"
#include "Formatting.hpp"

namespace otter {

SimpleROIAlignLayer::SimpleROIAlignLayer() {}

int SimpleROIAlignLayer::parse_param(LayerOption& option, ParamDict& pd) {
    int aligned = opt_find_int(option, "aligned", 0);
    int pooled_width = opt_find_int(option, "pooled_width", 0);
    int pooled_height = opt_find_int(option, "pooled_height", 0);
    float spatial_scale = opt_find_float(option, "spatial_scale", 1.f);
    
    pd.set((int)SimpleROIAlignParam::Aligned, aligned);
    pd.set((int)SimpleROIAlignParam::PooledWidth, pooled_width);
    pd.set((int)SimpleROIAlignParam::PooledHeight, pooled_height);
    pd.set((int)SimpleROIAlignParam::SpatialScale, spatial_scale);
    
    return 0;
}

int SimpleROIAlignLayer::compute_output_shape(ParamDict& pd) {
    auto shape_a = bottom_shapes[0].accessor<int, 2>()[0];
    int input_channels = shape_a[1];
    auto shape_b = bottom_shapes[1].accessor<int, 2>()[0];
    int output_batch = shape_b[2];
    
    int pooled_width = pd.get((int)SimpleROIAlignParam::PooledWidth, 0);
    int pooled_height = pd.get((int)SimpleROIAlignParam::PooledHeight, 0);
    
    pd.set(OUTPUT_SHAPE_HINT, otter::tensor({output_batch, input_channels, pooled_height, pooled_width}, ScalarType::Int).view({1, -1}));
    
    return 0;
}

int SimpleROIAlignLayer::load_param(const ParamDict &pd) {
    aligned = pd.get((int)SimpleROIAlignParam::Aligned, 0);
    pooled_width = pd.get((int)SimpleROIAlignParam::PooledWidth, 0);
    pooled_height = pd.get((int)SimpleROIAlignParam::PooledHeight, 0);
    spatial_scale = pd.get((int)SimpleROIAlignParam::SpatialScale, 1.f);
    
    return 0;
}

otter::Tensor normalize(otter::Tensor& grid) {
    return (grid + 1.0) / 2.0;
}

otter::Tensor denormalize(otter::Tensor& grid) {
    return grid * 2.0 - 1.0;
}

otter::Tensor generate_grid(int num_grid, int height, int width) {
    auto affine_trans = otter::tensor({1., 0., 0., 0., 1., 0.}, otter::ScalarType::Float).view({1, 2, 3});
    auto grid = otter::affine_grid_generator(affine_trans, {1, 1, height, width}, false);
    grid = normalize(grid);
    
    return grid.view({1, -1, 2}).expand({num_grid, -1, -1});
}

otter::Tensor point_sample(otter::Tensor& input, otter::Tensor& points, bool align_corners) {
    bool add_dim = false;
    
    if (points.dim() == 3) {
        add_dim = true;
        points = points.unsqueeze(2);
    }
    
    auto output = otter::grid_sampler(input, denormalize(points), 0, 0, align_corners);
    
    if (add_dim)
        output = output.squeeze(3);
    
    return output;
}

otter::Tensor abs_img_point_to_rel_img_point(otter::Tensor& abs_img_points, otter::Tensor& img, float spatial_scale) {
    auto scale = otter::tensor({img.size(3), img.size(2)}, otter::ScalarType::Float);
    
    return abs_img_points / scale * spatial_scale;
}

otter::Tensor rel_roi_point_to_abs_img_point(otter::Tensor& rois, otter::Tensor& rel_roi_points) {
    if (rois.size(1) == 5) {
        rois = rois.slice(1, 1, 5, 1);
    }
    
    auto abs_img_points = otter::empty_like(rel_roi_points);
    abs_img_points.copy_(rel_roi_points);
    
    auto xs = abs_img_points.slice(2, 0, 1, 1).squeeze(2) * (rois.slice(1, 2, 3, 1) - rois.slice(1, 0, 1, 1));
    auto ys = abs_img_points.slice(2, 1, 2, 1).squeeze(2) * (rois.slice(1, 3, 4, 1) - rois.slice(1, 1 ,2, 1));
    xs += rois.slice(1, 0, 1, 1);
    ys += rois.slice(1, 1, 2, 1);
    
    abs_img_points = otter::native::stack({xs, ys}, 2);
    
    return abs_img_points;
}

otter::Tensor rel_roi_point_to_rel_img_point(otter::Tensor& rois, otter::Tensor& rel_roi_points, otter::Tensor& img, float spatial_scale) {
    
    auto abs_img_point = rel_roi_point_to_abs_img_point(rois, rel_roi_points);
    auto rel_img_point = abs_img_point_to_rel_img_point(abs_img_point, img, spatial_scale);
    
    return rel_img_point;
}

int SimpleROIAlignLayer::forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& opt) const {
    auto features = bottom_blobs[0];
    auto rois     = bottom_blobs[1];

    int num_rois = rois.size(0);
    
    auto rel_roi_points = generate_grid(num_rois, pooled_height, pooled_width);
    
    auto rel_img_points = rel_roi_point_to_rel_img_point(rois, rel_roi_points, features, spatial_scale).unsqueeze(0);
    
    auto point_feats = point_sample(features, rel_img_points, !aligned);
    point_feats = point_feats.transpose(1, 2);
    
    int channels = features.size(1);
    auto roi_feats = point_feats.reshape({num_rois, channels, pooled_height, pooled_width});
    
    top_blobs[0] = roi_feats;
    
    return 0;
}

}   // end namespace otter
