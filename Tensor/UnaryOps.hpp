//
//  UnaryOps.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/6.
//

#ifndef UnaryOps_hpp
#define UnaryOps_hpp

#include "DispatchStub.hpp"
#include "Scalar.hpp"
#include "TensorIterator.hpp"

namespace otter {

using unary_fn = void(*)(TensorIterator&);
using unary_fn_with_scalar = void(*)(TensorIterator&, const Scalar& a);

DECLARE_DISPATCH(unary_fn, sin_stub);


}

#endif /* UnaryOps_hpp */
