//
//  UpsampleLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/6.
//

#include "UpsampleLayer.hpp"
#include "TensorFunction.hpp"

namespace otter {

UpsampleLayer::UpsampleLayer() {
    one_blob_only = true;
    support_inplace = false;
}

int UpsampleLayer::parse_param(LayerOption& option, ParamDict& pd) {
    pd.clear();
    int mode = -1;
    std::string upsample_mode = opt_find_string(option, "upsample_mode", "nearest");
    if (upsample_mode == "nearest") {
        mode = 1;
    } else if (upsample_mode == "bilinear") {
        mode = 2;
    } else if (upsample_mode == "bicubic") {
        mode = 3;
    }
    int output_height = opt_find_int(option, "output_height", 0);
    int output_width = opt_find_int(option, "output_width", 0);
    float scale_height = opt_find_float(option, "scale_height", 1.f);
    float scale_width = opt_find_float(option, "scale_width", 1.f);
    int align_corner = 0;
    if (opt_find(option, "align_corner")) {
        if (option["align_corner"] == "false")
            align_corner = 0;
        else
            align_corner = 1;
    }
    
    float stride = opt_find_float(option, "stride", 1.f);
    
    OTTER_CHECK(mode <= 2, "Unsupport upsample type");
    
    pd.set((int)UpsampleParam::Mode, mode);
    pd.set((int)UpsampleParam::Output_height, output_height);
    pd.set((int)UpsampleParam::Output_width, output_width);
    pd.set((int)UpsampleParam::Height_scale, scale_height);
    pd.set((int)UpsampleParam::Width_scale, scale_width);
    pd.set((int)UpsampleParam::Stride, stride);
    pd.set((int)UpsampleParam::Align_corner, align_corner);
    
    return 0;
}

int UpsampleLayer::compute_output_shape(ParamDict& pd) {
    auto shape = bottom_shapes[0][0].clone();
    auto shape_a = shape.accessor<int, 1>();
    auto bottom_shape_a = bottom_shapes[0].accessor<int, 2>()[0];
    int output_height = pd.get((int)UpsampleParam::Output_height, 0);
    int output_width = pd.get((int)UpsampleParam::Output_width, 0);
    float scale_height = pd.get((int)UpsampleParam::Height_scale, 0.f);
    float scale_width = pd.get((int)UpsampleParam::Width_scale, 0.f);
    float stride = pd.get((int)UpsampleParam::Stride, 1.f);
    
    if (output_height && output_width) {
        scale_height = output_height / bottom_shape_a[2];
        scale_width  = output_width  / bottom_shape_a[3];
    }
    
    if (stride != 0) {
        scale_height = stride;
        scale_width  = stride;
    }
    
    if (scale_height && scale_width) {
        output_height = bottom_shape_a[2] * scale_height;
        output_width = bottom_shape_a[3] * scale_width;
    }
    
    pd.set((int)UpsampleParam::Height_scale, scale_height);
    pd.set((int)UpsampleParam::Width_scale, scale_width);
    pd.set((int)UpsampleParam::Output_height, output_height);
    pd.set((int)UpsampleParam::Output_width, output_width);
    
    shape_a[2] = output_height;
    shape_a[3] = output_width;
    
    pd.set(OUTPUT_SHAPE_HINT, shape.view({1, -1}));
    
    return 0;
}

int UpsampleLayer::load_param(const ParamDict &pd) {
    mode = pd.get((int)UpsampleParam::Mode, 0);
    output_height = pd.get((int)UpsampleParam::Output_height, 0);
    output_width = pd.get((int)UpsampleParam::Output_width, 0);
    scale_height = pd.get((int)UpsampleParam::Height_scale, 0.f);
    scale_width = pd.get((int)UpsampleParam::Width_scale, 0.f);
    stride = pd.get((int)UpsampleParam::Stride, -1);
    align_corner = pd.get((int)UpsampleParam::Align_corner, 0);
    
    return 0;
}

int UpsampleLayer::forward(const Tensor& bottom_blob, Tensor& top_blob, const NetOption& /*opt*/) const {
    int input_height = (int)bottom_blob.size(2);
    int input_width = (int)bottom_blob.size(3);
    
    int output_height = input_height * scale_height;
    int output_width = input_width * scale_width;
    
    if (mode == 1) {
        top_blob = otter::native::upsample_nearest2d(bottom_blob, {output_height, output_width}, scale_height, scale_width);
    } else if (mode == 2) {
        top_blob = otter::native::upsample_bilinear2d(bottom_blob, {output_height, output_width}, align_corner, scale_height, scale_width);
    }
    
    return 0;
}

}   // end namespace otter
