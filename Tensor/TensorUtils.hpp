//
//  TensorUtils.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/11.
//

#ifndef TensorUtils_hpp
#define TensorUtils_hpp

#include "Tensor.hpp"
#include "Utils.hpp"

namespace otter {

void check_dim_size(const Tensor& tensor, int64_t dim, int64_t dim_size, int64_t size);

DimVector computeStride(IntArrayRef oldshape, IntArrayRef oldstride, const DimVector& newshape);


}   // end namespace otter

#endif /* TensorUtils_hpp */
