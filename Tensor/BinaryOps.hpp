//
//  BinaryOps.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/30.
//

#ifndef BinaryOps_hpp
#define BinaryOps_hpp

#include "DispatchStub.hpp"

namespace otter {

class Scalar;
class TensorIterator;

using structured_binary_fn = void(*)(TensorIterator&);
using structured_binary_fn_alpha = void(*)(TensorIterator&, const Scalar& alpha);

DECLARE_DISPATCH(structured_binary_fn_alpha, add_stub);
DECLARE_DISPATCH(structured_binary_fn_alpha, sub_stub);
DECLARE_DISPATCH(structured_binary_fn, mul_stub);
DECLARE_DISPATCH(structured_binary_fn, div_true_stub);
DECLARE_DISPATCH(structured_binary_fn, remainder_stub);
DECLARE_DISPATCH(structured_binary_fn, bitwise_and_stub);
DECLARE_DISPATCH(structured_binary_fn, bitwise_or_stub);
DECLARE_DISPATCH(structured_binary_fn, bitwise_xor_stub);

 
}



#endif /* BinaryOps_hpp */
