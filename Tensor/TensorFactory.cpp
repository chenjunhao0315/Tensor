//
//  TensorFactory.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "TensorFactory.hpp"
#include "EmptyTensor.hpp"

namespace otter {

Tensor empty(IntArrayRef size, ScalarType dtype) {
    return otter::empty_cpu(size, dtype);
}

Tensor empty_strided(IntArrayRef size, IntArrayRef stride, ScalarType dtype) {
    return otter::empty_strided_cpu(size, stride, dtype);
}

Tensor empty_like(const Tensor& self) {
    return empty_like(self, self.scalar_type());
}

Tensor empty_like(const Tensor& self, ScalarType dtype) {
    auto result = empty(self.sizes(), dtype);
    
    return result;
}

Tensor full(IntArrayRef size, const Scalar& fill_value, ScalarType dtype) {
    auto result = empty(size, dtype);
    
    return result.fill_(fill_value);
}

Tensor ones(IntArrayRef size, ScalarType dtype) {
    auto result = empty(size, dtype);
    
    return result.fill_(1);
}




}
