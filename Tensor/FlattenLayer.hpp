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
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int compute_output_shape(ParamDict& pd);
    
    virtual int load_param(const ParamDict& pd);
    
    virtual int forward_inplace(Tensor& bottom_blob, const NetOption& opt) const;
    
    virtual std::string type() const { return "Flatten"; }
private:
    int start_dim;
    int end_dim;
};

enum class FlattenParam {
    StartDim,
    EndDim
};

}   // end namespace otter

#endif /* FlattenLayer_hpp */
