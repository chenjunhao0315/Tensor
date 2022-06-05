//
//  LReluLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/1.
//

#ifndef LReluLayer_hpp
#define LReluLayer_hpp

#include "Layer.hpp"

namespace otter {

class LReluLayer : public Layer {
public:
    LReluLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);  
    
    virtual int load_param(const ParamDict& pd);
    
    virtual int forward_inplace(Tensor& bottom_blob, const NetOption& opt) const;
    
    virtual std::string type() const { return "LRelu"; }
private:
    float neg_slope;
};

enum class LReluParam {
    Neg_slope = 0
};

}   // end namespace otter

#endif /* LReluLayer_hpp */
