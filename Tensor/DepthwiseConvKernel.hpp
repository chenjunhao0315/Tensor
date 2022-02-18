//
//  DepthwiseConvKernel.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#ifndef DepthwiseConvKernel_hpp
#define DepthwiseConvKernel_hpp

#include "Tensor.hpp"
#include "DispatchStub.hpp"

namespace otter {

using convolution_depthwise3x3_winograd_fn = Tensor (*)(const Tensor &, const Tensor &, const Tensor &, IntArrayRef, IntArrayRef, int64_t);
DECLARE_DISPATCH(convolution_depthwise3x3_winograd_fn, convolution_depthwise3x3_winograd_stub);

}

#endif /* DepthwiseConvKernel_hpp */
