//
//  TensorLinearAlgebra.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/11.
//

#ifndef TensorLinearAlgebra_hpp
#define TensorLinearAlgebra_hpp

namespace otter {

void addmm_impl_cpu_(Tensor &result, const Tensor &self, Tensor m1, Tensor m2, const Scalar& beta, const Scalar& alpha);

Tensor matmul(Tensor& output, const Tensor& tensor1, const Tensor& tensor2);

}

#endif /* TensorLinearAlgebra_hpp */
