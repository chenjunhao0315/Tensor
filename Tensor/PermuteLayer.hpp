//
//  PermuteLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/1.
//

#ifndef PermuteLayer_hpp
#define PermuteLayer_hpp

#include "Layer.hpp"

namespace otter {

class PermuteLayer : public Layer {
public:
    PermuteLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int compute_output_shape(ParamDict& pd);
    
    virtual int load_param(const ParamDict &pd);
    
    virtual int forward_inplace(Tensor& bottom_blob, const NetOption& opt) const;
    
    virtual std::string type() const { return "Permute"; }
private:
    std::vector<long int> permute;
};

enum class PermuteParam {
    Permute
};

}   // end namespace otter

#endif /* PermuteLayer_hpp */
