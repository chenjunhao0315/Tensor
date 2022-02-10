//
//  RangeFactoryKernel.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/9.
//

#ifndef RangeFactoryKernel_hpp
#define RangeFactoryKernel_hpp

#include "TensorIterator.hpp"
#include "Scalar.hpp"

namespace otter {

void linspace_kernel(TensorIterator& iter, const Scalar& scalar_start, const Scalar& scalar_end, int64_t steps);



}

#endif /* RangeFactoryKernel_hpp */
