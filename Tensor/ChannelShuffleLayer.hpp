//
//  ChannelShuffleLayer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/1.
//

#ifndef ChannelShuffleLayer_hpp
#define ChannelShuffleLayer_hpp

#include "Layer.hpp"

namespace otter {

class ChannelShuffleLayer : public Layer {
public:
    ChannelShuffleLayer();
    
    virtual int parse_param(LayerOption& option, ParamDict& pd);
    
    virtual int load_param(const ParamDict &pd);
    
    virtual int forward(const Tensor& bottom_blob, Tensor& top_blob, const NetOption& opt) const;
    
    virtual std::string type() const { return "ChannelShuffle"; }
private:
    int groups;
};

enum class ChannelShuffleParam : int {
    Groups
};

}   // end namespace otter

#endif /* ChannelShuffleLayer_hpp */
