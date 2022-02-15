//
//  EmptyTensor.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/1.
//

#ifndef EmptyTensor_hpp
#define EmptyTensor_hpp

#include "Tensor.hpp"
#include "TensorOptions.hpp"
#include "Accumulator.hpp"

namespace otter {

size_t computeStorageNbytes(IntArrayRef sizes, IntArrayRef strides, size_t itemsize_bytes);

Tensor empty_cpu(IntArrayRef size, TensorOptions option);
Tensor empty_cpu(IntArrayRef size, TensorOptions option, MemoryFormat memory_format);
Tensor empty_cpu(IntArrayRef size, ScalarType dtype);

Tensor empty_generic(IntArrayRef size, Allocator* allocator, ScalarType dtype);
Tensor empty_generic(IntArrayRef size, Allocator* allocator, ScalarType dtype, MemoryFormat memory_format);

Tensor empty_strided_cpu(IntArrayRef size, IntArrayRef stride, TensorOptions option);
Tensor empty_strided_cpu(IntArrayRef size, IntArrayRef stride, ScalarType dtype);

Tensor empty_strided_generic(IntArrayRef size, IntArrayRef stride, Allocator* allocator, ScalarType scalar_type);


}

#endif /* EmptyTensor_hpp */
