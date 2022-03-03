//
//  TensorFactory.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "TensorFactory.hpp"
#include "EmptyTensor.hpp"
#include "RangeFactory.hpp"

namespace otter {

Tensor empty(IntArrayRef size, ScalarType dtype) {
    return otter::empty_cpu(size, dtype);
}

Tensor empty(IntArrayRef size, TensorOptions options) {
    return otter::empty_cpu(size, options);
}

Tensor empty_strided(IntArrayRef size, IntArrayRef stride, ScalarType dtype) {
    return otter::empty_strided_cpu(size, stride, dtype);
}

Tensor empty_strided(IntArrayRef size, IntArrayRef stride, TensorOptions options) {
    return otter::empty_strided_cpu(size, stride, options);
}

Tensor empty_like(const Tensor& self) {
    return empty_like(self, self.scalar_type());
}

Tensor empty_like(const Tensor& self, const TensorOptions& options) {
    return empty_like(self, typeMetaToScalarType(options.dtype()));
}

Tensor empty_like(const Tensor& self, const TensorOptions& options, MemoryFormat memory_format) {
    return otter::empty_cpu(self.sizes(), options, memory_format);
}

Tensor empty_like(const Tensor& self, ScalarType dtype) {
    auto result = empty(self.sizes(), dtype);
    
    return result;
}

Tensor clone(const Tensor& src, MemoryFormat memory_format) {
    Tensor self;
    if (memory_format == MemoryFormat::Preserve) {
        if (self.is_non_overlapping_and_dense()) {
            self = empty_strided(src.sizes(), src.strides(), src.options());
        } else {
            self = empty_like(src);
        }
    } else {
        self = empty_like(src, src.options(), memory_format);
    }
    
    self.copy_(src);
    
    return self;
}

Tensor full(IntArrayRef size, const Scalar& fill_value, ScalarType dtype) {
    auto result = empty(size, dtype);
    
    return result.fill_(fill_value);
}

Tensor ones(IntArrayRef size, ScalarType dtype) {
    auto result = empty(size, dtype);
    
    return result.fill_(1);
}

Tensor ones(IntArrayRef size, TensorOptions options) {
    auto result = empty(size, options);
    
    return result.fill_(1);
}

Tensor ones_like(const Tensor& self, TensorOptions options) {
    auto result = empty_like(self, options);
    
    return result.fill_(1);
}

Tensor zeros(IntArrayRef size, ScalarType dtype) {
    auto result = empty_cpu(size, dtype);
    return result.zero_();
}

Tensor zeros(IntArrayRef size, TensorOptions options) {
    return empty_cpu(size, options);
}

Tensor linspace(const Scalar& start, const Scalar& end, int64_t steps, ScalarType dtype) {
    assert(steps >= 0);
    
    Tensor result = empty({steps}, dtype);
    return otter::linspace_out(start, end, steps, result);
}

Tensor range(const Scalar& start, const Scalar& end, const Scalar& step, ScalarType dtype) {
    Tensor result = empty({}, dtype);
    
    return otter::range_out(start, end, step, result);
}

Tensor rand(IntArrayRef size, ScalarType dtype) {
    return ones(size, dtype);
}

Tensor rand(IntArrayRef size, TensorOptions options) {
    return ones(size, options);
}


}
