//
//  CropLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/1.
//

#ifndef CropLayer_hpp
#define CropLayer_hpp

#include "Layer.hpp"

namespace otter {

class CropLayer : public Layer {
public:
    CropLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int compute_output_shape(ParamDict& pd);
    
    virtual int load_param(const ParamDict &pd);
    
    virtual int forward(const Tensor& bottom_blob, Tensor& top_blob, const NetOption& opt) const;
    
    virtual std::string type() const { return "Crop"; }
private:
    int axis;
    int start;
    int end;
};

enum class CropParam : int {
    Axis,
    Start,
    End
};

}   // end namespace otter

#endif /* CropLayer_hpp */
