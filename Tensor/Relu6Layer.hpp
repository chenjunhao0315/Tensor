//
//  Relu6Layer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/1.
//

#ifndef Relu6Layer_hpp
#define Relu6Layer_hpp

#include "Layer.hpp"

namespace otter {

class Relu6Layer : public Layer {
public:
    Relu6Layer();
    
    virtual int forward_inplace(Tensor& bottom_blob, const NetOption& opt) const;
    
    virtual std::string type() const { return "Relu6"; }
};

}   // end namespace otter

#endif /* Relu6Layer_hpp */
