//
//  DropoutLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/3.
//

#include "DropoutLayer.hpp"
#include "Dropout.hpp"
#include "NetOption.hpp"
#include "LayerRegistry.hpp"

namespace otter {

DropoutLayer::DropoutLayer() {
    one_blob_only = true;
    support_inplace = true;
}

int DropoutLayer::parse_param(LayerOption& option, ParamDict &pd) {
    pd.clear();
    float prob = opt_find_float(option, "probability", 1.f);
    
    pd.set((int)DropoutParam::Probability, prob);
    
    return 0;
}

int DropoutLayer::load_param(const ParamDict &pd) {
    probability = pd.get((int)DropoutParam::Probability, 1.f);
    
    return 0;
}

int DropoutLayer::forward_inplace(Tensor &bottom_blob, const NetOption &opt) const {
    if (opt.train) {
        if (opt.use_non_lib_optimize) {
            // TODO: dropout enhancement
        } else {
            bottom_blob = std::get<0>(otter::dropout(bottom_blob, probability, true));
        }
    }
    return 0;
}

REGISTER_LAYER_CLASS(Dropout);

}   // end namespace otter
