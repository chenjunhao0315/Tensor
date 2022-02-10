//
//  TensorCopy.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/10.
//

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

Tensor& copy_(Tensor& self, const Tensor& src, bool non_blocking) {
    copy_impl(self, src, non_blocking);
    
    return self;
}

DEFINE_DISPATCH(copy_stub);

}
