//
//  BinaryOpsKernel.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/30.
//

#ifndef BinaryOpsKernel_hpp
#define BinaryOpsKernel_hpp

#include "TensorIterator.hpp"
#include "Scalar.hpp"

namespace otter {

void add_kernel(TensorIterator& iter, const Scalar& alpha_scalar);

void sub_kernel(TensorIterator& iter, const Scalar& alpha_sclaar);

void mul_kernel(TensorIterator& iter);

void div_true_kernel(TensorIterator& iter);

void remainder_kernel(TensorIterator& iter);

void bitwise_and_kernel(TensorIterator& iter);

void bitwise_or_kernel(TensorIterator& iter);

void bitwise_xor_kernel(TensorIterator& iter);


}

#endif /* BinaryOpsKernel_hpp */
