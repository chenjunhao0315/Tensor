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

Tensor to(const Tensor& self, ScalarType dtype, bool non_blocking, bool copy, MemoryFormat optional_memory_format);


}   // end namespace native
}   // end namespace otter

#endif /* TensorConversion_hpp */
