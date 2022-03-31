//
//  ChannelShuffle.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/31.
//

#include "Tensor.hpp"
#include "ChannelShuffle.hpp"
#include "TensorFactory.hpp"

namespace otter {

DEFINE_DISPATCH(channel_shuffle_stub);

Tensor& channel_shuffle_out(Tensor& output, const Tensor& input, int64_t groups) {
    
    channel_shuffle_stub(Device::CPU, output, input, groups);
    
    return output;
}

Tensor channel_shuffle(const Tensor& input, int64_t groups) {
    auto output = otter::empty_like(input);
    
    return channel_shuffle_out(output, input, groups);
}

}   // end namespace otter
