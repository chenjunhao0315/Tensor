//
//  FillKernel.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/30.
//

#ifndef FillKernel_hpp
#define FillKernel_hpp

namespace otter {

class Scalar;
class TensorIterator;

void fill_kernel(TensorIterator& iter, const Scalar& value_scalar);

}

#endif /* FillKernel_hpp */
