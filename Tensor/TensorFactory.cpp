//
//  TensorFactory.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "TensorFactory.hpp"
#include "EmptyTensor.hpp"
#include "RangeFactory.hpp"
#include "Parallel.hpp"
#include "Dispatch.hpp"

namespace otter {

Tensor empty(IntArrayRef size, ScalarType dtype) {
    return otter::empty_cpu(size, dtype);
}

Tensor empty(IntArrayRef size, TensorOptions options) {
    return otter::empty_cpu(size, options);
}

Tensor empty(IntArrayRef size, TensorOptions options, MemoryFormat memory_format) {
    return otter::empty_cpu(size, options, memory_format);
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
    return otter::empty(self.sizes(), options, memory_format);
}

Tensor empty_like(const Tensor& self, MemoryFormat memory_format) {
    return otter::empty(self.sizes(), self.options(), memory_format);
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

Tensor full(IntArrayRef size, const Scalar& fill_value, TensorOptions options) {
    auto result = empty(size, options);
    
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

Tensor ones_like(const Tensor& self) {
    auto result = empty_like(self, self.options());
    
    return result.fill_(1);
}

Tensor ones_like(const Tensor& self, ScalarType dtype) {
    auto result = empty_like(self, dtype);
    
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
    auto result = empty_cpu(size, options);
    
    return result.zero_();
}

Tensor zeros_like(const Tensor& self) {
    auto result = empty_like(self, self.options());
    
    return result.zero_();
}

Tensor zeros_like(const Tensor& self, ScalarType dtype) {
    auto result = empty_like(self, dtype);
    
    return result.zero_();
}

Tensor zeros_like(const Tensor& self, TensorOptions options) {
    auto result = empty_like(self, options);
    
    return result.zero_();
}

Tensor linspace(const Scalar& start, const Scalar& end, int64_t steps, ScalarType dtype) {
    assert(steps >= 0);
    
    Tensor result = empty({steps}, dtype);
    return otter::linspace_out(start, end, steps, result);
}

Tensor linspace(const Scalar& start, const Scalar& end, int64_t steps, TensorOptions options) {
    assert(steps >= 0);
    
    Tensor result = empty({steps}, options);
    return otter::linspace_out(start, end, steps, result);
}

Tensor range(const Scalar& start, const Scalar& end, const Scalar& step, ScalarType dtype) {
    Tensor result = empty({}, dtype);
    
    return otter::range_out(start, end, step, result);
}

Tensor range(const Scalar& start, const Scalar& end, const Scalar& step, TensorOptions options) {
    Tensor result = empty({}, options);
    
    return otter::range_out(start, end, step, result);
}

Tensor arange(const Scalar& start, const Scalar& end, const Scalar& step, ScalarType dtype) {
    Tensor result = empty({}, dtype);
    
    return otter::arange_out(start, end, step, result);
}

Tensor arange(const Scalar& start, const Scalar& end, const Scalar& step, TensorOptions options) {
    Tensor result = empty({}, options);
    
    return otter::arange_out(start, end, step, result);
}

Tensor rand(IntArrayRef size, ScalarType dtype) {
    auto result = empty(size, dtype);
    result.uniform_(0, 1);
    
    return result;
}

Tensor rand(IntArrayRef size, TensorOptions options) {
    auto result = empty(size, options);
    result.uniform_(0, 1);
    
    return result;
}

Tensor rand_like(const Tensor& self, ScalarType dtype) {
    auto result = empty_like(self, dtype);
    result.uniform_(0, 1);
    
    return result;
}

Tensor rand_like(const Tensor& self, TensorOptions options) {
    auto result = empty_like(self, options);
    result.uniform_(0, 1);
    
    return result;
}

Tensor randn(IntArrayRef size, ScalarType dtype) {
    auto result = empty(size, dtype);
    result.normal_(0, 1);
    
    return result;
}

Tensor randn(IntArrayRef size, TensorOptions options) {
    auto result = empty(size, options);
    result.normal_(0, 1);
    
    return result;
}

Tensor randn_like(const Tensor& self, ScalarType dtype) {
    auto result = empty_like(self, dtype);
    result.normal_(0, 1);
    
    return result;
}

Tensor randn_like(const Tensor& self, TensorOptions options) {
    auto result = empty_like(self, options);
    result.normal_(0, 1);
    
    return result;
}

Tensor eye(int64_t n, ScalarType dtype) {
    return eye(n, n, dtype);
}

Tensor eye(int64_t n, int64_t m, ScalarType dtype) {
    auto output = otter::empty({}, dtype);
    
    return eye_out(n, m, output);
}

Tensor& eye_out(int64_t n, Tensor& result) {
    return eye_out(n, n, result);
}

Tensor& eye_out(int64_t n, int64_t m, Tensor& result) {
    OTTER_CHECK(n >= 0, "n must be greater or equal to 0, got ", n);
    OTTER_CHECK(m >= 0, "m must be greater or equal to 0, got ", m);

    result.resize_({n, m});
    result.zero_();
    
    int64_t sz = std::min<int64_t>(n, m);
    OTTER_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, result.scalar_type(), "eye", [&]() -> void {
        scalar_t* result_data = result.data_ptr<scalar_t>();
        otter::parallel_for(0, sz, 32768, [&](int64_t p_begin, int64_t p_end) {
            for (const auto i : otter::irange(p_begin, p_end))result_data[i*(result.strides()[0] + result.strides()[1])] = 1;
        });
    });
    
    return result;
}


}
