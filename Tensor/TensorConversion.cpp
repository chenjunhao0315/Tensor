//
//  TensorConversion.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/6.
//

#include "TensorFactory.hpp"
#include "TensorConversion.hpp"
#include "Parallel.hpp"
#include "Loop.hpp"
#include "Dispatch.hpp"
#include "TypeCast.hpp"

namespace otter {
namespace native {

Tensor _to_copy(
    const Tensor& self,
    ScalarType dtype,
    Device device,
    bool non_blocking,
    MemoryFormat memory_format) {
    
    auto options = TensorOptions()
        .dtype(dtype)
        .device(device);

    if (memory_format == MemoryFormat::Preserve) {
        if (self.is_non_overlapping_and_dense()) {
            Tensor r;

            r = otter::empty_strided(self.sizes(), self.strides(), options);
            r.copy_(self, non_blocking);
            
            return r;
        } else {
            memory_format = self.suggest_memory_format();
        }
    }
    // See Note [Explicit nullopt MemoryFormat argument]
    auto r = otter::empty(self.sizes(), options.memory_format(memory_format));
    r.copy_(self, non_blocking);
    return r;
}

bool to_will_alias(
    const Tensor& self,
    ScalarType dtype,
    Device device,
    bool copy,
    MemoryFormat memory_format) {
    
    return (self.scalar_type() == dtype) &&
    (self.device() == device) &&
    !copy &&
    (memory_format == MemoryFormat::Preserve || self.suggest_memory_format() == memory_format);
}

Tensor to_impl(const Tensor& self, ScalarType dtype, Device device, bool non_blocking, bool copy, MemoryFormat optional_memory_format) {
    if (to_will_alias(self, dtype, device, copy, optional_memory_format)) {
        return self;
    }
    return _to_copy(self, dtype, device, non_blocking, optional_memory_format);
}

Tensor to(const Tensor& self, ScalarType dtype, bool non_blocking, bool copy, MemoryFormat optional_memory_format) {
    return to_impl(self, dtype, self.device(), non_blocking, copy, optional_memory_format);
}

}   // end namespace native
}   // end namespace otter
