//
//  ConcatLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/3.
//

#include "ConcatLayer.hpp"
#include "TensorShape.hpp"

namespace otter {

ConcatLayer::ConcatLayer() {
    one_blob_only = false;
    support_inplace = false;
}

int ConcatLayer::parse_param(LayerOption& option, ParamDict& pd) {
    pd.clear();
    int axis = opt_find_int(option, "axis", 1);
    
    pd.set((int)ConcatParam::Axis, axis);
    
    return 0;
}

int ConcatLayer::compute_output_shape(ParamDict& pd) {
    int axis = pd.get((int)ConcatParam::Axis, 1);
    
    auto shape = bottom_shapes[0].clone();
    auto shape_a = shape.accessor<int, 1>();
    for (size_t i = 1; i < bottom_shapes.size(); ++i) {
        auto bottom_shape_a = bottom_shapes[i].accessor<int, 1>();
        shape_a[axis] += bottom_shape_a[axis];
    }
    
    pd.set(OUTPUT_SHAPE_HINT, shape);
    
    return 0;
}

int ConcatLayer::load_param(const ParamDict &pd) {
    axis = pd.get((int)ConcatParam::Axis, axis);
    
    return 0;
}

int ConcatLayer::forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& opt) const {
    top_blobs[0] = otter::native::cat(bottom_blobs, axis);
        
    return 0;
}

}
