//
//  TensorCopy.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/10.
//

#include "TensorIterator.hpp"
#include "TensorCopy.hpp"

namespace otter {

static Tensor& copy_impl(Tensor & self, const Tensor & src, bool non_blocking) {
    assert(self.defined());
    assert(src.defined());
    
    if (self.is_same(src)) {
        return self;
    }
    
    auto iter = TensorIteratorConfig()
        .add_output(self)
        .add_input(src)
        .resize_outputs(false)
        .check_all_same_dtype(false)
        .check_all_same_device(false)
        .build();
    
    if (iter.numel() == 0) {
        return self;
    }
    copy_stub(Device::CPU, iter, non_blocking);
    
    return self;
}

static Tensor& copy_packed_impl(Tensor& self, const Tensor& src) {
    assert(self.defined());
    assert(src.defined());
    
    memcpy(self.raw_data(), src.raw_data(), src.numel() * src.itemsize());
    
    return self;
}

Tensor& copy_(Tensor& self, const Tensor& src, bool non_blocking) {
    if (src.elempack() != 1)
        return copy_packed_impl(self, src);
    
    return copy_impl(self, src, non_blocking);
}

DEFINE_DISPATCH(copy_stub);

}
