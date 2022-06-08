//
//  ReshapeLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/1.
//

#ifndef ReshapeLayer_hpp
#define ReshapeLayer_hpp

#include "Layer.hpp"

namespace otter {

class ReshapeLayer : public Layer {
public:
    ReshapeLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int compute_output_shape(ParamDict& pd);
    
    virtual int load_param(const ParamDict &pd);
    
    virtual int forward_inplace(Tensor& bottom_blob, const NetOption& opt) const;
    
    virtual std::string type() const { return "Reshape"; }
private:
    std::vector<long int> shape;
};

enum class ReshapeParam {
    Shape
};

}   // end namespace otter

#endif /* ReshapeLayer_hpp */
