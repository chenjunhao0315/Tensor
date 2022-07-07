//
//  InnerProductLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/7.
//

#ifndef InnerProductLayer_hpp
#define InnerProductLayer_hpp

#include "Layer.hpp"

namespace otter {

class InnerProductLayer : public Layer {
public:
    InnerProductLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int compute_output_shape(ParamDict& pd);
    
    virtual int load_param(const ParamDict& pd);
    
    virtual int init_model();
    
    virtual int load_model(const Initializer& initializer);
    
    virtual int create_pipeline(const NetOption& opt);
    
    virtual int forward(const Tensor& bottom_blob, Tensor& top_blob, const NetOption& opt) const;
    
    virtual std::string type() const { return "InnerProduct"; }
private:
    int out_features;
    int in_features;
    int bias_term;
    
    int activation_type;
    Layer* activation;
    Tensor activation_params;
    
    Tensor weight_data;
    Tensor bias_data;
};

enum class InnerProductParam : int {
    OutFeatures,
    InFeatures,
    Bias_term,
    Activation_type,
    Activation_params
};

}   // end namespace otter

#endif /* InnerProductLayer_hpp */
