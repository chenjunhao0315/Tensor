//
//  DropoutLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/3.
//

#ifndef DropoutLayer_hpp
#define DropoutLayer_hpp

#include "Layer.hpp"

namespace otter {

class DropoutLayer : public Layer {
public:
    DropoutLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int load_param(const ParamDict &pd);
    
    virtual int forward_inplace(Tensor& bottom_blob, const NetOption& opt) const;
private:
    float probability;
};

enum class DropoutParam {
    Probability
};

}

#endif /* DropoutLayer_hpp */
