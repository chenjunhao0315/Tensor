//
//  BatchNormalizationLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/22.
//

#include "Normalization.hpp"
#include "BatchNormalizationLayer.hpp"
#include "TensorFactory.hpp"
#include "Formatting.hpp"

namespace otter {

BatchNormalizationLayer::BatchNormalizationLayer() {
    one_blob_only = true;
    support_inplace = true;
}

int BatchNormalizationLayer::parse_param(LayerOption& option, ParamDict &pd) {
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
    bias_data = otter::rand({shape_a[1]}, ScalarType::Float);
    scale_data = otter::rand({shape_a[1]}, ScalarType::Float);
    mean_data = otter::rand({shape_a[1]}, ScalarType::Float);
    var_data = otter::rand({shape_a[1]}, ScalarType::Float);
    
    return 0;
}

int BatchNormalizationLayer::load_model(const Initializer& initializer) {
    auto shape_a = bottom_shapes[0].accessor<int, 1>();
    
    if (initializer.type != InitializerType::Ncnn) {
        bias_data = initializer.load({shape_a[1]}, 1);
        scale_data = initializer.load({shape_a[1]}, 1);
        mean_data = initializer.load({shape_a[1]}, 1);
        var_data = initializer.load({shape_a[1]}, 1);
    } else {
        scale_data = initializer.load({shape_a[1]}, 1);
        mean_data = initializer.load({shape_a[1]}, 1);
        var_data = initializer.load({shape_a[1]}, 1);
        bias_data = initializer.load({shape_a[1]}, 1);
    }
    
    return 0;
}

int BatchNormalizationLayer::forward_inplace(Tensor& bottom_blob, const NetOption& opt) const {
    
//    bottom_blob = otter::batchnorm_alpha_beta(bottom_blob, alpha, beta);
    bottom_blob = otter::batchnorm(bottom_blob, scale_data, bias_data, mean_data, var_data, false, 0, 0.000001);
    
    return 0;
}

REGISTER_LAYER_CLASS(BatchNormalization);

}   // end namespace otter
