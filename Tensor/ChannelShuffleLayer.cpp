//
//  ChannelShuffleLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/1.
//

#include "ChannelShuffleLayer.hpp"
#include "ChannelShuffle.hpp"
#include "LayerRegistry.hpp"

namespace otter {

ChannelShuffleLayer::ChannelShuffleLayer() {
    one_blob_only = true;
    support_inplace = false;
}

int ChannelShuffleLayer::parse_param(LayerOption& option, ParamDict& pd) {
    pd.clear();
    int groups = opt_find_int(option, "groups", 1);
    
    pd.set((int)ChannelShuffleParam::Groups, groups);
    
    return 0;
}

int ChannelShuffleLayer::load_param(const ParamDict &pd) {
    groups = pd.get((int)ChannelShuffleParam::Groups, 1);
    
    return 0;
}

int ChannelShuffleLayer::forward(const Tensor& bottom_blob, Tensor& top_blob, const NetOption& opt) const {
    top_blob = otter::channel_shuffle(bottom_blob, groups);
    
    return 0;
}

REGISTER_LAYER_CLASS(ChannelShuffle);

}   // end namespace otter
