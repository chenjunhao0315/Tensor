//
//  ROIAlignLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/7.
//

#ifndef ROIAlignLayer_hpp
#define ROIAlignLayer_hpp

#include "Layer.hpp"

namespace otter {

// Detectron2 version
class ROIAlignLayer : public Layer {
public:
    ROIAlignLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int compute_output_shape(ParamDict& pd);
    
    virtual int load_param(const ParamDict &pd);
    
    virtual int forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& opt) const;
    
    virtual std::string type() const { return "ROIAlign"; }
    
public:
    int aligned;
    int pooled_width;
    int pooled_height;
    float spatial_scale;
    float sampling_ratio;
};

enum class ROIAlignParam {
    Aligned,
    PooledWidth,
    PooledHeight,
    SpatialScale,
    SamplingRatio
};

}

#endif /* ROIAlignLayer_hpp */
