//
//  UnaryOpsKernel.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/6.
//

#ifndef UnaryOpsKernel_hpp
#define UnaryOpsKernel_hpp

namespace otter {

void bitwise_not_kernel(TensorIterator& iter);

void neg_kernel(TensorIterator& iter);

void abs_kernel(TensorIterator& iter);

void sin_kernel(TensorIterator& iter);

void cos_kernel(TensorIterator& iter);

void tan_kernel(TensorIterator& iter);

void exp_kernel(TensorIterator& iter);

void sqrt_kernel(TensorIterator& iter);

void sigmoid_kernel(TensorIterator& iter);


}

#endif /* UnaryOpsKernel_hpp */
