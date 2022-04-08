//
//  EltwiseLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/5.
//

#include "EltwiseLayer.hpp"
#include "Tensor.hpp"
#include "TensorFactory.hpp"
#include "TensorOperator.hpp"

namespace otter {

EltwiseLayer::EltwiseLayer() {
    one_blob_only = false;
    support_inplace = false;
}

int EltwiseLayer::parse_param(LayerOption& option, ParamDict& pd) {
    std::string type_str = opt_find_string(option, "type", "Sum");
    
    int type = 2;
    
    if (type_str == "Prod") {
        type = 1;
    } else if (type_str == "Sum") {
        type = 2;
    } else if (type_str == "Max") {
        type = 3;
    }
    
    pd.set((int)EltwiseParam::Type, type);
    
    return 0;
}

int EltwiseLayer::load_param(const ParamDict& pd) {
    operation_type = pd.get((int)EltwiseParam::Type, 2);
    
    return 0;
}

int EltwiseLayer::compute_output_shape(ParamDict &pd) {
    pd.set(OUTPUT_SHAPE_HINT, bottom_shapes[0]);
    
    return 0;
}

int EltwiseLayer::forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& /*opt*/) const {
    
    const Tensor& bottom_blob = bottom_blobs[0];
    
    Tensor& top_blob = top_blobs[0];
    top_blob = otter::empty_like(bottom_blob);
    
    if (operation_type == 1) {
        const Tensor& bottom_blob1 = bottom_blobs[1];
        
        top_blob = bottom_blob * bottom_blob1;
        
        for (size_t i = 2; i < bottom_blobs.size(); ++i) {
            const Tensor& bottom_blob1 = bottom_blobs[i];
            
            top_blob *= bottom_blob1;
        }
    } else if (operation_type == 2) {
        const Tensor& bottom_blob1 = bottom_blobs[1];
        
        top_blob = bottom_blob + bottom_blob1;
        
        for (size_t i = 2; i < bottom_blobs.size(); ++i) {
            const Tensor& bottom_blob1 = bottom_blobs[i];
            
            top_blob += bottom_blob1;
        }
    }
    
    return 0;
}

}   // end namespace otter
