//
//  ShortCutLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/1.
//

#ifndef ShortCutLayer_hpp
#define ShortCutLayer_hpp

#include "Layer.hpp"

namespace otter {

class ShortCutLayer : public Layer {
public:
    ShortCutLayer();
    
    virtual int prase_param(LayerOption& option, ParamDict& pd);
    
    virtual int load_param(const ParamDict& pd);
    
    virtual int init_model();
    
    virtual int forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& opt) const;
    
    virtual std::string type() const { return "ShortCut"; }
};

enum class ShortCutBackend {
    Eltwise_add,
    Darknet_shortcut
};

}   // end namespace otter

#endif /* ShortCutLayer_hpp */
