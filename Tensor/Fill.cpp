//
//  Fill.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/30.
//

#include "Accumulator.hpp"
#include "Fill.hpp"

namespace otter {

namespace native {
Tensor& fill_out(Tensor& self, const Scalar& value) {
    auto iter = TensorIteratorConfig()
        .add_output(self)
        .resize_outputs(false)
        .build();
    
    fill_stub(Device::CPU, iter, value);
    return self;
}

Tensor& fill_(Tensor& self, const Scalar& value) {
    return fill_out(self, value);
}

Tensor& zero_cpu_(Tensor& self, int64_t numel) {
    void* ptr = self.data_ptr();
    if (ptr == nullptr) {
        return self.fill_(0);
    }
    int64_t size_bytes = numel * self.dtype().itemsize();
    if (size_bytes > 0) {
        std::memset(ptr, 0, size_bytes);
    }
    
    return self;
}

Tensor& zero_(Tensor& self) {
    int64_t nelements = otter::multiply_integers(self.sizes());
    if (self.device() == Device::CPU && nelements < otter::GRAIN_SIZE) {
        return zero_cpu_(self, nelements);
    }
    return self.fill_(0);
}
}   // end namespace native

DEFINE_DISPATCH(fill_stub);

}   // end namespace otter

