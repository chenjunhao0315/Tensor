//
//  ReluLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/15.
//

#include "ReluLayer.hpp"
#include "Activation.hpp"

namespace otter {

ReluLayer::ReluLayer() {
    one_blob_only = true;
    support_inplace = true;
}

int ReluLayer::parse_param(LayerOption& option, ParamDict& pd) {
    pd.clear();
    
    int relu6 = opt_find(option, "relu6");
    
    if (relu6) {
        pd.set((int)ReluParam::Relu6, 1);
    }
    
    return 0;
}

int ReluLayer::load_param(const ParamDict& pd) {
    relu6 = pd.get((int)ReluParam::Relu6, 0);
    
    return 0;
}

int ReluLayer::forward_inplace(Tensor& bottom_blob, const NetOption& opt) const {
    
    if (relu6)
        otter::relu6_(bottom_blob);
    
    otter::relu_(bottom_blob);
    
    return 0;
}

}   // end namespace otter
