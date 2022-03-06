//
//  Yolov3DetectionOutputLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/6.
//

#ifndef Yolov3DetectionOutputLayer_hpp
#define Yolov3DetectionOutputLayer_hpp

#include "Layer.hpp"

namespace otter {

class Yolov3DetectionOutputLayer : public Layer {
public:
    Yolov3DetectionOutputLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int compute_output_shape(ParamDict& pd);
    
    virtual int load_param(const ParamDict &pd);
    
    virtual int forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& opt) const;
    
    virtual std::string type() const { return "Yolov3"; }
};

}

#endif /* Yolov3DetectionOutputLayer_hpp */
