//
//  BatchNormalizationLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/22.
//

#ifndef BatchNormalizationLayer_hpp
#define BatchNormalizationLayer_hpp

#include "Layer.hpp"

namespace otter {

class BatchNormalizationLayer : public Layer {
public:
    BatchNormalizationLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int load_param(const ParamDict& pd);
    
    virtual int init_model();
    
    virtual int forward_inplace(Tensor& bottom_blob, const NetOption& opt) const;
    
    virtual std::string type() const { return "BatchNorm"; }
public:
    float eps; 
    
    Tensor alpha;
    Tensor beta;
};

enum class BnParam {
    Eps
};

}   // end namespace otter

#endif /* BatchNormalizationLayer_hpp */
