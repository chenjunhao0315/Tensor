//
//  ActivationLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/5.
//

#ifndef ActivationLayer_h
#define ActivationLayer_h

#include "Layer.hpp"
#include "Tensor.hpp"
#include "LayerRegistry.hpp"

namespace otter {

static OTTER_ALWAYS_INLINE float activation_ss(float v, int activation_type, const Tensor& activation_params) {
    switch (activation_type) {
        case 1: {
            v = fmax(v, 0.f);
            break;
        }
        case 2: {
            float slope = activation_params.item().toFloat();
            v = v > 0.f ? v : v * slope;
            break;
        }
        case 3: {
            float min = 0;
            float max = 6;
            if (v < min)
                v = min;
            if (v > max)
                v = max;
            break;
        }
        case 4: {
            v = std::min(v, 88.3762626647949f);
            v = std::max(v, -88.3762626647949f);
            v = 1.f / (1.f + exp(-v));
            break;
        }
    }
            
    return v;
}

static Layer* create_activation_layer(int activation_type, const Tensor& activation_params) {
    Layer* activation = nullptr;
    
    if (activation_type == 1) {
        activation = LayerRegistry::CreateLayer("Relu");
    } else if (activation_type == 2) {
        activation = LayerRegistry::CreateLayer("LRelu");
        
        float default_value[] = {0.1f};
        auto activation_params_data = activation_params.defined() ? activation_params.data_ptr<float>() : default_value;
        ParamDict pd;
        pd.set(0, activation_params_data[0]);
        
        activation->load_param(pd);
    } else if (activation_type == 3) {
        activation = LayerRegistry::CreateLayer("Relu6");
    } else if (activation_type == 4) {
        activation = LayerRegistry::CreateLayer("Sigmoid");
    }
    
    return activation;
}

}   // end namespace otter

#endif /* ActivationLayer_h */
