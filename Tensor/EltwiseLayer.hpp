//
//  EltwiseLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/5.
//

#ifndef EltwiseLayer_hpp
#define EltwiseLayer_hpp

#include "Layer.hpp"

namespace otter {

class EltwiseLayer : public Layer {
public:
    EltwiseLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int load_param(const ParamDict& pd);
    
    virtual int compute_output_shape(ParamDict &pd);
    
    virtual int forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& opt) const;
    
    virtual std::string type() const { return "Eltwise"; }
public:
    int operation_type;
};

enum class EltwiseParam {
    Type
};

}

#endif /* EltwiseLayer_hpp */
