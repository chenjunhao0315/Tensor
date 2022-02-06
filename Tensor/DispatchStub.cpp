//
//  DispatchStub.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/30.
//

#include "DispatchStub.hpp"

namespace otter {

static CPUCapability compute_cpu_capability() {
    return CPUCapability::DEFAULT;
}

CPUCapability get_cpu_capability() {
    static CPUCapability capability = compute_cpu_capability();
    return capability;
}

void* DispatchStubImpl::get_call_ptr(Device device_type, void *DEFAULT) {
    switch (device_type) {
        case Device::CPU: {
            auto fptr = cpu_dispatch_ptr.load(std::memory_order_relaxed);
            if (!fptr) {
                fptr = choose_cpu_impl(DEFAULT);
                cpu_dispatch_ptr.store(fptr, std::memory_order_relaxed);
            }
            return fptr;
        }
        default:
            fprintf(stderr, "DispatchStub: unsupported device type");
    }
}

void* DispatchStubImpl::choose_cpu_impl(void *DEFAULT) {
    auto capability = static_cast<int>(get_cpu_capability());
    (void)capability;
    return DEFAULT;
}




}
