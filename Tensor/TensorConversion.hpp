//
//  TensorConversion.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/6.
//

#ifndef TensorConversion_hpp
#define TensorConversion_hpp

#include "Tensor.hpp"

namespace otter {

namespace native{
Tensor to(const Tensor& self, ScalarType dtype);
}

}

#endif /* TensorConversion_hpp */
