//
//  UpsampleLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/6.
//

#ifndef UpsampleLayer_hpp
#define UpsampleLayer_hpp

#include "Layer.hpp"

namespace otter {

class UpsampleLayer : public Layer {
public:
    UpsampleLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int compute_output_shape(ParamDict& pd);
    
    virtual int load_param(const ParamDict &pd);
    
    virtual int forward(const Tensor& bottom_blob, Tensor& top_blob, const NetOption& opt) const;
private:
    int mode;
    int output_height;
    int output_width;
    float scale_height;
    float scale_width;
    int stride;
};

enum class UpsampleParam {
    Mode,
    Output_height,
    Output_width,
    Height_scale,
    Width_scale,
    Stride
};

}   // end namespace otter

#endif /* UpsampleLayer_hpp */
