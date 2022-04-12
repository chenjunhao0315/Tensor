//
//  LReluLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/1.
//

#include "LReluLayer.hpp"
#include "TensorFunction.hpp"

namespace otter {

LReluLayer::LReluLayer() {
    one_blob_only = true;
    support_inplace = true;
}

int LReluLayer::parse_param(LayerOption& option, ParamDict &pd) {
    pd.clear();
    
    float neg_slope = opt_find_float(option, "alpha", 0.1);
    
    pd.set((int)LReluParam::Neg_slope, neg_slope);
    
    return 0;
}

int LReluLayer::load_param(const ParamDict &pd) {
    neg_slope = pd.get((int)LReluParam::Neg_slope, 0.1f);
    
    return 0;
}

int LReluLayer::forward_inplace(Tensor &bottom_blob, const NetOption &opt) const {
    if (opt.use_non_lib_optimize) {
        // TODO: leaky relu enhancement
    } else {
        bottom_blob = otter::native::leaky_relu(bottom_blob, neg_slope);
    }
    
    return 0;
}

}   // end namespace otter
