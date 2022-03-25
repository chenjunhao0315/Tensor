//
//  Activation.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/1.
//

#ifndef Activation_hpp
#define Activation_hpp

#include "DispatchStub.hpp"

namespace otter {

class TensorIterator;
class Scalar;

using leaky_relu_fn = void(*)(TensorIterator&, const Scalar& a);
using threshold_fn = void (*)(TensorIterator&, const Scalar&, const Scalar&);

DECLARE_DISPATCH(leaky_relu_fn, leaky_relu_stub);
DECLARE_DISPATCH(threshold_fn, threshold_stub);

}

#endif /* Activation_hpp */
