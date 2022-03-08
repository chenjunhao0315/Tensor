//
//  SplitLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/2.
//

#include "SplitLayer.hpp"
#include "LayerRegistry.hpp"

namespace otter {

SplitLayer::SplitLayer() {
    one_blob_only = false;
    support_inplace = false;
}

int SplitLayer::compute_output_shape(ParamDict &pd) {
    pd.set(OUTPUT_SHAPE_HINT, bottom_shapes[0]);
    
    return 0;
}

int SplitLayer::forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& opt) const {
    const Tensor& bottom_blob = bottom_blobs[0];
    for (size_t i = 0; i < top_blobs.size(); i++) {
        top_blobs[i] = bottom_blob;
    }
    
    return 0;
}

REGISTER_LAYER_CLASS(Split);

}   // end namespace otter
