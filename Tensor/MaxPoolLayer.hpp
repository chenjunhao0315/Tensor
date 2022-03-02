//
//  MaxPoolLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/2.
//

#ifndef MaxPoolLayer_hpp
#define MaxPoolLayer_hpp

#include "Layer.hpp"

namespace otter {

class MaxPoolLayer : public Layer {
public:
    MaxPoolLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int compute_output_shape(ParamDict &pd);
    
    virtual int load_param(const ParamDict &pd);
    
    virtual int forward(const Tensor& bottom_blob, Tensor& top_blob, const NetOption& opt) const;
    
    virtual std::string type() const { return "MaxPool"; }
private:
    int kernel_height;
    int kernel_width;
    int stride_height;
    int stride_width;
    int padding_height;
    int padding_width; 
    int dilation_height;
    int dilation_width;
    int ceil_mode;
    int darknet_mode;
};

enum class MaxPoolParam {
    Kernel_height,
    Kernel_width,
    Stride_height,
    Stride_width,
    Padding_height,
    Padding_width,
    Dilation_height,
    Dilation_width,
    Ceil_mode,
    Darknet_mode
};

}   // end namespace otter

#endif /* MaxPoolLayer_hpp */
