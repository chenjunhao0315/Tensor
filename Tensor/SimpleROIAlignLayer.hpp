//
//  SimpleROIAlignLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/12.
//

#ifndef SimpleROIAlignLayer_hpp
#define SimpleROIAlignLayer_hpp

#include "Layer.hpp"

namespace otter {

class SimpleROIAlignLayer : public Layer {
public:
    SimpleROIAlignLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int compute_output_shape(ParamDict& pd);
    
    virtual int load_param(const ParamDict &pd);
    
    virtual int forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& opt) const;
    
    virtual std::string type() const { return "SimpleROIAlign"; }
    
private:
    int aligned;
    int pooled_width;
    int pooled_height;
    float spatial_scale;
};

enum class SimpleROIAlignParam {
    Aligned,
    PooledWidth,
    PooledHeight,
    SpatialScale
};

}   // end namespace otter

#endif /* SimpleROIAlignLayer_hpp */
