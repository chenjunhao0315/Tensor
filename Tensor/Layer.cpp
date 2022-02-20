//
//  Layer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#include "Layer.hpp"

namespace otter {

Layer::Layer() {
    one_blob_only = false;
    support_inplace = false;
}

Layer::~Layer() {
    
}

int Layer::prase_param(LayerOption& option, ParamDict& pd) {
    return -1;
}

int Layer::load_param(const ParamDict &pd) {
    return 0;
}

int Layer::forward(const Tensor &bottom_blob, Tensor &top_blob, const NetOption &opt) const {
    if (!support_inplace)
        return -1;
    
    top_blob = bottom_blob.clone();
    if (!top_blob.defined())
        return -100;
    
    return forward_inplace(top_blob, opt);
}

int Layer::forward_inplace(const Tensor &bottom_blob, const NetOption &opt) const {
    return -1;
}

int Layer::forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& opt) const {
    if (!support_inplace)
        return -1;
    
    top_blobs = bottom_blobs;
    for (const auto i : otter::irange(top_blobs.size())) {
        top_blobs[i] = bottom_blobs[i].clone();
        if (!top_blobs[i].defined())
            return -100;
    }
    
    return forward_inplace(top_blobs, opt);
}

int Layer::forward_inplace(const std::vector<Tensor>& bottom_blobs, const NetOption& opt) const {
    return -1;
}



}   // end namespace otter