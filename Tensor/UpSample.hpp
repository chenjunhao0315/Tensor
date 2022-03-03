//
//  UpSample.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/3.
//

#ifndef UpSample_hpp
#define UpSample_hpp

#include "DispatchStub.hpp"

namespace otter {

class Tensor;

using upsampling_nearest2d_fn = void(*)(const Tensor& output, const Tensor& input, double scales_h, double scales_w);

DECLARE_DISPATCH(upsampling_nearest2d_fn, upsmapling_nearest2d_stub);

}

#endif /* UpSample_hpp */
