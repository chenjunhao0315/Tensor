//
//  FlattenLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/7.
//

#ifndef FlattenLayer_hpp
#define FlattenLayer_hpp

#include "Layer.hpp"

namespace otter {

class FlattenLayer : public Layer {
public:
    FlattenLayer();
    
    virtual int compute_output_shape(ParamDict& pd);
    
    virtual int forward_inplace(Tensor& bottom_blob, const NetOption& opt) const;
    
    virtual std::string type() const { return "Flatten"; }
};

}   // end namespace otter

#endif /* FlattenLayer_hpp */
