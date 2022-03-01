//
//  ActivationKernel.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/1.
//

#ifndef ActivationKernel_hpp
#define ActivationKernel_hpp

namespace otter {

void leaky_relu_kernel(TensorIterator& iter, const Scalar& value);

}

#endif /* ActivationKernel_hpp */
