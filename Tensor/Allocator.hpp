//
//  Allocator.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef Allocator_hpp
#define Allocator_hpp

#include <stdio.h>
#include <memory>
#include <cassert>
#include <functional>

#include "Config.hpp"
#include "Device.hpp"

#ifndef OTTER_LOC
#define OTTER_LOC __FILE__, __func__, __LINE__
#endif

void *otter_malloc_log(const size_t size, const char * const filename, const char * const funcname, const int line);
void *otter_calloc_log(const size_t nmemb, const size_t size, const char * const filename, const char * const funcname, const int line);
void *otter_realloc_log(void *ptr, const size_t size, const char * const filename, const char * const funcname, const int line);
void otter_free(void *ptr);

#define otter_malloc(s)      otter_malloc_log(s, OTTER_LOC)
#define otter_calloc(m, s)   otter_calloc_log(m, s, OTTER_LOC)
#define otter_realloc(p, s)  otter_realloc_log(p, s, OTTER_LOC)

#if OTTER_MOBILE
// Use 16-byte alignment on mobile
// - ARM NEON AArch32 and AArch64
// - x86[-64] < AVX
constexpr size_t gAlignment = 16;
#else
// Use 64-byte alignment should be enough for computation up to AVX512.
constexpr size_t gAlignment = 64;
#endif

using DeleterFnPtr = void (*)(void*);
void deleteNothing(void*);

class VoidPtr {
public:
    VoidPtr() : data_(nullptr), ctx_(nullptr, &deleteNothing) {}
    explicit VoidPtr(void* data) : data_(data), ctx_(nullptr, &deleteNothing) {}
    VoidPtr(void* data, void* ctx, DeleterFnPtr ctx_deleter) : data_(data), ctx_(ctx, ctx_deleter ? ctx_deleter : &deleteNothing) {}
    
    void* operator->() const { return data_; }
    
    void* get() const { return data_; }
    void* get_context() const { return ctx_.get(); }
    void* release_context() { return ctx_.release(); }
    void clear() { data_ = nullptr; ctx_ = nullptr; }
    DeleterFnPtr get_deleter() const { return ctx_.get_deleter(); }
    
    operator bool() const { return data_ || ctx_; }
private:
    void* data_;
    std::unique_ptr<void, DeleterFnPtr> ctx_;
};

class DataPtr {
public:
    DataPtr() : ptr_(), device_(Device::CPU) {}
    DataPtr(void* data, Device device) : ptr_(data), device_(device) {}
    DataPtr(void* data, void* ctx, DeleterFnPtr ctx_deleter, Device device) : ptr_(data, ctx, ctx_deleter), device_(device) {}
    
    void* operator->() const {
        return ptr_.get();
    }
    void clear() {
        ptr_.clear();
    }
    
    void* get() const {
        return ptr_.get();
    }
    
    void* get_context() const {
        return ptr_.get_context();
    }
    
    void* release_context() {
        return ptr_.release_context();
    }

    operator bool() const {
        return static_cast<bool>(ptr_);
    }

    DeleterFnPtr get_deleter() const {
        return ptr_.get_deleter();
    }
    
    Device device() const {
        return device_;
    }
private:
    VoidPtr ptr_;
    Device device_;
};

inline bool operator==(const DataPtr& dp, std::nullptr_t) noexcept {
    return !dp;
}
inline bool operator==(std::nullptr_t, const DataPtr& dp) noexcept {
    return !dp;
}
inline bool operator!=(const DataPtr& dp, std::nullptr_t) noexcept {
    return dp;
}
inline bool operator!=(std::nullptr_t, const DataPtr& dp) noexcept {
    return dp;
}

struct Allocator {
    virtual ~Allocator() = default;
    
    virtual DataPtr allocate(size_t n) const = 0;
    
    virtual DeleterFnPtr raw_deleter() const {
        return nullptr;
    }
    
    void* raw_allocate(size_t n) {
        auto dptr = allocate(n);
        assert(dptr.get() == dptr.get_context());
        return dptr.release_context();
    }
    void raw_deallocate(void* ptr) {
        auto d = raw_deleter();
        assert(d);
        d(ptr);
    }
};

void* alloc_cpu(size_t nbytes);
void free_cpu(void* data);

Allocator* get_default_allocator();
Allocator* GetAllocator(Device device);

struct InefficientStdFunctionContext {
  std::unique_ptr<void, std::function<void(void*)>> ptr_;
  InefficientStdFunctionContext(
      std::unique_ptr<void, std::function<void(void*)>>&& ptr)
      : ptr_(std::move(ptr)) {}
  static DataPtr makeDataPtr(
      void* ptr,
      const std::function<void(void*)>& deleter,
      Device device);
};

#endif /* Allocator_hpp */
