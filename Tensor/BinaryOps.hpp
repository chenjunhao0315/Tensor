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

class Tensor;
class Scalar;
class TensorIterator;

using structured_binary_fn = void(*)(TensorIterator&);
using structured_binary_fn_alpha = void(*)(TensorIterator&, const Scalar& alpha);
using binary_clamp_fn_alpha = void(*)(TensorIterator&, const Scalar& alpha, const Scalar& min_val, const Scalar& max_val);

DECLARE_DISPATCH(structured_binary_fn_alpha, add_stub);
DECLARE_DISPATCH(structured_binary_fn_alpha, sub_stub);
DECLARE_DISPATCH(binary_clamp_fn_alpha, add_clamp_stub);
DECLARE_DISPATCH(structured_binary_fn, mul_stub);
DECLARE_DISPATCH(structured_binary_fn, div_true_stub);
DECLARE_DISPATCH(structured_binary_fn, remainder_stub);
DECLARE_DISPATCH(structured_binary_fn, bitwise_and_stub);
DECLARE_DISPATCH(structured_binary_fn, bitwise_or_stub);
DECLARE_DISPATCH(structured_binary_fn, bitwise_xor_stub);

DECLARE_DISPATCH(structured_binary_fn, lt_stub);
DECLARE_DISPATCH(structured_binary_fn, le_stub);
DECLARE_DISPATCH(structured_binary_fn, gt_stub);
DECLARE_DISPATCH(structured_binary_fn, ge_stub);
DECLARE_DISPATCH(structured_binary_fn, eq_stub);
DECLARE_DISPATCH(structured_binary_fn, ne_stub);

Tensor& add_relu_out(const Tensor& self, const Tensor& other, const Scalar& alpha, Tensor& result);

Tensor add_relu(const Tensor& self, const Tensor& other, const Scalar& alpha);

Tensor add_relu(const Tensor& self, const Scalar& other, const Scalar& alpha);

Tensor& add_relu_(Tensor& self, const Tensor& other, const Scalar& alpha);

Tensor& add_relu_(Tensor& self, const Scalar& other, const Scalar& alpha);

}



#endif /* BinaryOps_hpp */
