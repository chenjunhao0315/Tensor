//
//  UpsampleLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/6.
//

#include "UpsampleLayer.hpp"
#include "TensorFunction.hpp"
#include "LayerRegistry.hpp"

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
        mode = 0;
    }
    int output_height = opt_find_int(option, "output_height", 0);
    int output_width = opt_find_int(option, "output_width", 0);
    float scale_height = opt_find_float(option, "scale_height", 0.f);
    float scale_width = opt_find_float(option, "scale_width", 0.f);
    
    int stride = -1;
    if (opt_check_string(option, "darknet_mode")) {
        stride = opt_find_int(option, "stride", 1);
    }
    
    OTTER_CHECK(mode >= 0, "Unsupport upsample type");
    
    pd.set((int)UpsampleParam::Mode, mode);
    pd.set((int)UpsampleParam::Output_height, output_height);
    pd.set((int)UpsampleParam::Output_width, output_width);
    pd.set((int)UpsampleParam::Height_scale, scale_height);
    pd.set((int)UpsampleParam::Width_scale, scale_width);
    pd.set((int)UpsampleParam::Stride, stride);
    
    
    return 0;
}

int UpsampleLayer::load_param(const ParamDict &pd) {
    mode = pd.get((int)UpsampleParam::Mode, 0);
    output_height = pd.get((int)UpsampleParam::Output_height, 0);
    output_width = pd.get((int)UpsampleParam::Output_width, 0);
    scale_height = pd.get((int)UpsampleParam::Height_scale, 0.f);
    scale_width = pd.get((int)UpsampleParam::Width_scale, 0.f);
    stride = pd.get((int)UpsampleParam::Stride, -1);
    
    return 0;
}

int UpsampleLayer::compute_output_shape(ParamDict& pd) {
    auto shape = bottom_shapes[0].clone();
    auto shape_a = shape.accessor<int, 1>();
    auto bottom_shape_a = bottom_shapes[0].accessor<int, 1>();
    int output_height = pd.get((int)UpsampleParam::Output_height, 0);
    int output_width = pd.get((int)UpsampleParam::Output_width, 0);
    int stride = pd.get((int)UpsampleParam::Stride, -1);
    if (stride > 0) {
        output_height = bottom_shape_a[2] * stride;
        output_width = bottom_shape_a[3] * stride;
    }
    pd.set((int)UpsampleParam::Output_height, output_height);
    pd.set((int)UpsampleParam::Output_width, output_width);
    
    shape_a[2] = output_height;
    shape_a[3] = output_width;
    
    pd.set(OUTPUT_SHAPE_HINT, shape);
    
    return 0;
}

int UpsampleLayer::forward(const Tensor& bottom_blob, Tensor& top_blob, const NetOption& opt) const {
    if (mode == 0) {
        top_blob = otter::native::upsample_nearest2d(bottom_blob, {output_height, output_width}, scale_height, scale_width);
    }
    
    return 0;
}

REGISTER_LAYER_CLASS(Upsample);


}   // end namespace otter
