//
//  SplitLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/2.
//

#ifndef SplitLayer_hpp
#define SplitLayer_hpp

#include "Layer.hpp"

namespace otter {

class SplitLayer : public Layer {
public:
    SplitLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int compute_output_shape(ParamDict &pd);
    
    virtual int load_param(const ParamDict &pd);
    
    virtual int forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& opt) const;
    
    virtual std::string type() const { return "Split"; }
private:
};

}

#endif /* SplitLayer_hpp */
