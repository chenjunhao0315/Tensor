//
//  TensorFactory.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef TensorFactory_hpp
#define TensorFactory_hpp

#include "MemoryFormat.hpp"
#include "ScalarType.hpp"
#include "ArrayRef.hpp"

namespace otter {

class Scalar;
class Tensor;
struct TensorOptions;

Tensor empty(IntArrayRef size, ScalarType dtype);
Tensor empty(IntArrayRef size, TensorOptions options);
Tensor empty(IntArrayRef size, TensorOptions options, MemoryFormat memory_format);

Tensor empty_strided(IntArrayRef size, IntArrayRef stride, ScalarType dtype);
Tensor empty_strided(IntArrayRef size, IntArrayRef stride, TensorOptions options);

Tensor empty_like(const Tensor& self);
Tensor empty_like(const Tensor& self, const TensorOptions& options);
Tensor empty_like(const Tensor& self, const TensorOptions& options, MemoryFormat memory_format);
Tensor empty_like(const Tensor& self, MemoryFormat memory_format);
Tensor empty_like(const Tensor& self, ScalarType dtype);

Tensor clone(const Tensor& src, MemoryFormat memory_format = MemoryFormat::Preserve);

Tensor full(IntArrayRef size, const Scalar& fill_value, ScalarType dtype);
Tensor full(IntArrayRef size, const Scalar& fill_value, TensorOptions options);

Tensor zeros(IntArrayRef size, ScalarType dtype);
Tensor zeros(IntArrayRef size, TensorOptions options);
Tensor zeros_like(const Tensor& self, ScalarType dtype);
Tensor zeros_like(const Tensor& self, TensorOptions options);
Tensor ones(IntArrayRef size, ScalarType dtype);
Tensor ones(IntArrayRef size, TensorOptions options);
Tensor ones_like(const Tensor& self, ScalarType dtype);
Tensor ones_like(const Tensor& self, TensorOptions options);

Tensor linspace(const Scalar& start, const Scalar& end, int64_t steps, ScalarType dtype);

Tensor range(const Scalar& start, const Scalar& end, const Scalar& step, ScalarType dtype);

Tensor rand(IntArrayRef size, ScalarType dtype);
Tensor rand(IntArrayRef size, TensorOptions options);
Tensor rand_like(IntArrayRef size, ScalarType dtype);
Tensor rand_like(IntArrayRef size, TensorOptions options);

Tensor randn(IntArrayRef size, ScalarType dtype);
Tensor randn(IntArrayRef size, TensorOptions options);
Tensor randn_like(IntArrayRef size, ScalarType dtype);
Tensor randn_like(IntArrayRef size, TensorOptions options);


}   // namespace otter

#endif /* TensorFactory_hpp */
