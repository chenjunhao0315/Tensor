//
//  DispatchStub.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/30.
//

#ifndef DispatchStub_hpp
#define DispatchStub_hpp

#include <type_traits>
#include <atomic>
#include <utility>

#include "Device.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-var-template"
#endif

namespace otter {

enum class CPUCapability {
    DEFAULT = 0,
    NUM_OPTIONS
};

CPUCapability get_cpu_capability();

template <typename FnPtr, typename T>
struct DispatchStub;

struct DispatchStubImpl {
    void* get_call_ptr(Device device , void *DEFAULT);

    void* choose_cpu_impl(void *DEFAULT);

#if defined(_MSC_VER) && defined(_DEBUG)
    std::atomic<void*> cpu_dispatch_ptr;
    void* cuda_dispatch_ptr;
    void* hip_dispatch_ptr;
#else
    std::atomic<void*> cpu_dispatch_ptr{nullptr};
    void* cuda_dispatch_ptr = nullptr;
    void* hip_dispatch_ptr = nullptr;
#endif
};

template <typename rT, typename T, typename... Args>
struct DispatchStub<rT (*)(Args...), T> {
    using FnPtr = rT (*) (Args...);
    
    DispatchStub() = default;
    DispatchStub(const DispatchStub&) = delete;
    DispatchStub& operator=(const DispatchStub&) = delete;
  
private:
    FnPtr get_call_ptr(Device device_type) {
        return reinterpret_cast<FnPtr>(impl.get_call_ptr(device_type, reinterpret_cast<void*>(DEFAULT)));
    }

public:
    template <typename... ArgTypes>
    rT operator()(Device device_type, ArgTypes&&... args) {
        FnPtr call_ptr = get_call_ptr(device_type);
        return (*call_ptr)(std::forward<ArgTypes>(args)...);
    }

    static FnPtr DEFAULT;
    
private:
    DispatchStubImpl impl;
};
    
#define DECLARE_DISPATCH(fn, name)         \
  struct name : DispatchStub<fn, name> {   \
    name() = default;                      \
    name(const name&) = delete;            \
    name& operator=(const name&) = delete; \
  };                                       \
  extern struct name name

#define DEFINE_DISPATCH(name) struct name name

#define REGISTER_ARCH_DISPATCH(name, arch, fn) \
    template <> name::FnPtr DispatchStub<name::FnPtr, struct name>::arch = fn;

#define REGISTER_DISPATCH(name, fn) REGISTER_ARCH_DISPATCH(name, DEFAULT, fn)
    
}

#endif /* DispatchStub_hpp */
