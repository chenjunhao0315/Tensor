//
//  SliceLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/5/31.
//

#ifndef SliceLayer_hpp
#define SliceLayer_hpp

#include "Layer.hpp"

namespace otter {

class SliceLayer : public Layer {
public:
    SliceLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int compute_output_shape(ParamDict& pd);
    
    virtual int load_param(const ParamDict &pd);
    
    virtual int forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& opt) const;
    
    virtual std::string type() const { return "Slice"; }
private:
    int axis;
    Tensor slice;
};

enum class SliceParam {
    Axis,
    Slice
};

}   // end namespace otter

#endif /* SliceLayer_hpp */
