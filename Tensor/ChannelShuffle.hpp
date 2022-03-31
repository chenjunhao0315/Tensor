//
//  ChannelShuffle.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/31.
//

#ifndef ChannelShuffle_hpp
#define ChannelShuffle_hpp

#include "DispatchStub.hpp"

namespace otter {

class Tensor;

using channel_shuffle_fn = void(*)(Tensor&, const Tensor&, int64_t);
DECLARE_DISPATCH(channel_shuffle_fn, channel_shuffle_stub);

Tensor& channel_shuffle_out(Tensor& output, const Tensor& input, int64_t groups);

Tensor channel_shuffle(const Tensor& input, int64_t groups);

}   // end namespace otter

#endif /* ChannelShuffle_hpp */
