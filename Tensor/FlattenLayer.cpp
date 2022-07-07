//
//  FlattenLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/7.
//

#include "FlattenLayer.hpp"
#include "TensorMaker.hpp"

namespace otter {

FlattenLayer::FlattenLayer() {
    one_blob_only = true;
    support_inplace = true;
}

int FlattenLayer::compute_output_shape(ParamDict& pd) {
    auto shape_a = bottom_shapes[0].accessor<int, 2>()[0];
    
    int dims = bottom_shapes[0][0].numel();
    
    if (dims == 4) {
        int size = shape_a[3] * shape_a[2] * shape_a[1] * shape_a[0];
        pd.set(OUTPUT_SHAPE_HINT, otter::tensor({size}, ScalarType::Int).view({1, -1}));
    } else if (dims == 3) {
        int size = shape_a[2] * shape_a[1] * shape_a[0];
        pd.set(OUTPUT_SHAPE_HINT, otter::tensor({size}, ScalarType::Int).view({1, -1}));
    } else if (dims == 2) {
        int size = shape_a[1] * shape_a[0];
        pd.set(OUTPUT_SHAPE_HINT, otter::tensor({size}, ScalarType::Int).view({1, -1}));
    } else if (dims == 1) {
        int size = shape_a[0];
        pd.set(OUTPUT_SHAPE_HINT, otter::tensor({size}, ScalarType::Int).view({1, -1}));
    } else {
        OTTER_CHECK(false, "Flatten shape error!");
    }
    
    return 0;
}

int FlattenLayer::forward_inplace(Tensor& bottom_blob, const NetOption& opt) const {
    bottom_blob = bottom_blob.view({-1});
    
    return 0;
}

}   // end namespace otter
