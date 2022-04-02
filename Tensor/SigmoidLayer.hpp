//
//  SigmoidLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/2.
//

#ifndef SigmoidLayer_hpp
#define SigmoidLayer_hpp

#include "Layer.hpp"

namespace otter {

class SigmoidLayer : public Layer {
public:
    SigmoidLayer();
    
    virtual int load_param(const ParamDict &pd);
    
    virtual int forward_inplace(Tensor& bottom_blob, const NetOption& opt) const;
    
    virtual std::string type() const { return "Sigmoid"; }
};

}

#endif /* SigmoidLayer_hpp */
