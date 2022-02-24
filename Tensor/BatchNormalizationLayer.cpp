//
//  BatchNormalizationLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/22.
//

#include "LayerRegistry.hpp"
#include "Normalization.hpp"
#include "BatchNormalizationLayer.hpp"
#include "TensorFactory.hpp"

namespace otter {

BatchNormalizationLayer::BatchNormalizationLayer() {
    one_blob_only = true;
    support_inplace = true;
}

int BatchNormalizationLayer::prase_param(LayerOption& option, ParamDict &pd) {
    pd.clear();
    float eps = opt_find_float(option, "eps", 0.00001f);
    
    pd.set((int)BnParam::Eps, eps);
    
    return 0;
}

int BatchNormalizationLayer::load_param(const ParamDict &pd) {
    eps = pd.get((int)BnParam::Eps, 0.00001f);
    
    return 0;
}

int BatchNormalizationLayer::init_model() {
    auto shape_a = bottom_shapes[0].accessor<int, 1>();
    alpha = otter::rand({shape_a[1]}, ScalarType::Float);
    beta = otter::rand({shape_a[1]}, ScalarType::Float);
    
    return 0;
}

int BatchNormalizationLayer::forward_inplace(Tensor& bottom_blob, const NetOption& opt) const {
    
    bottom_blob = otter::batchnorm_alpha_beta(bottom_blob, alpha, beta);
    
    return 0;
}

REGISTER_LAYER_CLASS(BatchNormalization);

}   // end namespace otter
