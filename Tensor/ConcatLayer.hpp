//
//  ConcatLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/3.
//

#ifndef ConcatLayer_hpp
#define ConcatLayer_hpp

#include "Layer.hpp"

namespace otter {

class ConcatLayer : public Layer {
public:
    ConcatLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int compute_output_shape(ParamDict& pd);
    
    virtual int load_param(const ParamDict &pd);
    
    virtual int forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& opt) const;
    
private:
    int axis;
};

enum class ConcatParam {
    Axis
};

}

#endif /* ConcatLayer_hpp */
