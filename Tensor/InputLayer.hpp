//
//  InputLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#ifndef InputLayer_hpp
#define InputLayer_hpp

#include "Layer.hpp"

namespace otter {

class InputLayer : public Layer {
public:
    InputLayer();
    
    virtual int prase_param(LayerOption& option, ParamDict& pd);
    
    virtual int load_param(const ParamDict& pd);
    
    virtual std::string type() const { return "Input"; }
    
    virtual int forward_inplace(Tensor& bottom_top_blob, const NetOption& opt) const;
public:
    DimVector shape;
};

enum class InputParam : int {
    Batch   = 0,
    Channel = 1,
    Height  = 2,
    Width   = 3
};

}

#endif /* InputLayer_hpp */
