//
//  SliceLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/1.
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
    
    virtual int forward(const Tensor& bottom_blob, Tensor& top_blob, const NetOption& opt) const;
    
    virtual std::string type() const { return "Slice"; }
private:
    int axis;
    int start;
    int end;
};

enum class SliceParam : int {
    Axis,
    Start,
    End
};

}   // end namespace otter

#endif /* SliceLayer_hpp */
