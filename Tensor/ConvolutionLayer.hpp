//
//  ConvolutionLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/21.
//

#ifndef ConvolutionLayer_hpp
#define ConvolutionLayer_hpp

#include "Layer.hpp"

namespace otter {

class ConvolutionLayer : public Layer {
public:
    ConvolutionLayer();
    
    virtual int prase_param(LayerOption& option, ParamDict& pd);
    
    virtual int load_param(const ParamDict& pd);
    
    virtual int forward(const Tensor& bottom_blob, Tensor& top_blob, const NetOption& opt) const;
    
    virtual std::string type() const { return "Convoltuion"; }
public:
    int stride_height;
    int stride_width;
    int padding_height;
    int padding_width;
    int dilation_height;
    int dilation_width;
    int output_padding_height;
    int output_padding_width;
    
    int groups;
    
    Tensor weight_data;
    Tensor bias_data;
};

enum class ConvParam : int {
    Stride_height,
    Stride_width,
    Padding_height,
    Padding_width,
    Dilation_height,
    Dilation_width,
    Output_padding_height,
    Output_padding_width,
    Group
};

}

#endif /* ConvolutionLayer_hpp */
