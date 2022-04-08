//
//  SliceLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/1.
//

#include "SliceLayer.hpp"
#include "TensorShape.hpp"
#include "TensorMaker.hpp"
#include "Formatting.hpp"

namespace otter {

SliceLayer::SliceLayer() {
    one_blob_only = true;
    support_inplace = false;
}

int SliceLayer::parse_param(LayerOption& option, ParamDict& pd) {
    pd.clear();
    int axis = opt_find_int(option, "axis", 1);
    int start = opt_find_int(option, "start", -1);
    int end = opt_find_int(option, "end", -1);
    
    pd.set((int)SliceParam::Axis, axis);
    pd.set((int)SliceParam::Start, start);
    pd.set((int)SliceParam::End, end);
    
    return 0;
}

int SliceLayer::compute_output_shape(ParamDict& pd) {
    auto shape_a = bottom_shapes[0].accessor<int, 1>();
    int input_batch    = shape_a[0];
    int input_channels = shape_a[1];
    int input_height   = shape_a[2];
    int input_width    = shape_a[3];
    
    int axis = pd.get((int)SliceParam::Axis, 1);
    int start = pd.get((int)SliceParam::Start, -1);
    int end = pd.get((int)SliceParam::End, -1);
    
    otter::Tensor shape;
    
    if (axis == 1)
        shape = otter::tensor({input_batch, end - start, input_height, input_width});
    else if (axis == 2)
        shape = otter::tensor({input_batch, input_channels, end - start, input_width});
    else if (axis == 3)
        shape = otter::tensor({input_batch, input_channels, input_height, end - start});
    else
        OTTER_CHECK(false, "[Slice] Shape fatal error, expect [1, 3] but get", axis);
    
    pd.set(OUTPUT_SHAPE_HINT, shape);
    
    return 0;
}

int SliceLayer::load_param(const ParamDict &pd) {
    axis = pd.get((int)SliceParam::Axis, 1);
    start = pd.get((int)SliceParam::Start, -1);
    end = pd.get((int)SliceParam::End, -1);
    
    return 0;
}

int SliceLayer::forward(const Tensor& bottom_blob, Tensor& top_blob, const NetOption& opt) const {
    
    int ends = (end == -1) ? (int)bottom_blob.size(1) : end;
    
    top_blob = otter::native::slice(bottom_blob, axis, start, ends);
    
    return 0;
}

REGISTER_LAYER_CLASS(Slice);

}   // end namespace otter
