//
//  Relu6Layer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/15.
//

#ifndef ReluLayer_hpp
#define ReluLayer_hpp

#include "Layer.hpp"

namespace otter {

class ReluLayer : public Layer {
public:
    ReluLayer();
    
    virtual int forward_inplace(Tensor& bottom_blob, const NetOption& opt) const;
    
    virtual std::string type() const { return "Relu"; }
};

}   // end namespace otter

#endif /* ReluLayer_hpp */
