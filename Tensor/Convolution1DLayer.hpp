//
//  Convolution1DLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/11.
//

#ifndef Convolution1D1DLayer_hpp
#define Convolution1D1DLayer_hpp

#include "Layer.hpp"

namespace otter {

class Convolution1DLayer : public Layer {
public:
    ~Convolution1DLayer();
    Convolution1DLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int compute_output_shape(ParamDict& pd);
    
    virtual int load_param(const ParamDict& pd);
    
    virtual int init_model();
    
    virtual int load_model(const Initializer& initializer);
    
    virtual int create_pipeline(const NetOption& opt);
    
    virtual int forward(const Tensor& bottom_blob, Tensor& top_blob, const NetOption& opt) const;
    
#if __F16C__
    virtual int create_pipeline_fp16s(const NetOption& opt);
    
    virtual int forward_fp16s(const Tensor& bottom_blob, Tensor& top_blob, const NetOption& opt) const;
#endif
    
    virtual std::string type() const { return "Convolution1D"; }
public:
    int in_channels;
    int out_channels;
    int kernel_w;
    int stride_w;
    int padding_w;
    int dilation_w;
    
    int groups;
    
    int bias_term;
    
    int activation_type;
    Layer* activation;
    Tensor activation_params;
    
    Tensor weight_data;
    Tensor weight_data_packed;
    Tensor bias_data;
    
    Tensor weight_data_fp16s;
    Tensor weight_data_packed_fp16s;
    
};

enum class Conv1DParam : int {
    In_channels,
    Out_channels,
    Kernel_width,
    Stride_width,
    Padding_width,
    Dilation_width,
    Groups,
    Bias_term,
    Activation_type,
    Activation_params
};

}

#endif /* Convolution1DLayer_hpp */
