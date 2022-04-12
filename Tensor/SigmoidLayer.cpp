//
//  SigmoidLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/2.
//

#include "SigmoidLayer.hpp"

namespace otter {

SigmoidLayer::SigmoidLayer() {
    one_blob_only = true;
    support_inplace = true;
}

int SigmoidLayer::load_param(const ParamDict& /*pd*/) {
    return 0;
}

int SigmoidLayer::forward_inplace(Tensor& bottom_blob, const NetOption& /*opt*/) const {
    (void)bottom_blob;
    
    return 0;
}

}
