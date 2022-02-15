//
//  TensorUtils.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/11.
//

#ifndef TensorUtils_hpp
#define TensorUtils_hpp

#include "Utils.hpp"

namespace otter {

DimVector computeStride(IntArrayRef oldshape, IntArrayRef oldstride, const DimVector& newshape);


}   // end namespace otter

#endif /* TensorUtils_hpp */
