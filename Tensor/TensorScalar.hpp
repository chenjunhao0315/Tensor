//
//  TensorScalar.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/24.
//

#ifndef TensorScalar_hpp
#define TensorScalar_hpp

namespace otter {

class Scalar;
class Tensor;

Scalar item(const Tensor& self);
Scalar _local_scalar_dense_cpu(const Tensor& self);

}

#endif /* TensorScalar_hpp */
