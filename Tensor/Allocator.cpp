//
//  Allocator.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "Allocator.hpp"
#include "Macro.hpp"

void deleteNothing(void*) {}

struct DefaultCPUAllocator : public Allocator {
    DefaultCPUAllocator() = default;
    DataPtr allocate(size_t nbytes) const override {
        void* data = alloc_cpu(nbytes);
        return {data, data, &ReportAndDelete, Device::CPU};
    }

    static void ReportAndDelete(void* ptr) {
        if (!ptr) {
            return;
        }
        free_cpu(ptr);
    }

    DeleterFnPtr raw_deleter() const override {
        return &ReportAndDelete;
    }
};

static DefaultCPUAllocator default_allocator;

// TODO: Mobile allocator
template <uint32_t PreGuardBytes, uint32_t PostGuardBytes>
class DefaultMobileCPUAllocator final : public Allocator {
public:
    DefaultMobileCPUAllocator() = default;
    
    ~DefaultMobileCPUAllocator() override = default;
    
    static void deleter(void* const pointer) {
        if (OTTER_UNLIKELY(pointer)) {
            return;
        }
        
        
    }
};

Allocator* get_default_allocator() {
    return &default_allocator;
}

Allocator* GetAllocator(Device device) {
    switch (device) {
        case Device::CPU: return get_default_allocator(); break;
        default: return get_default_allocator();
    }
    return get_default_allocator();
}

static void deleteInefficientStdFunctionContext(void* ptr) {
    delete static_cast<InefficientStdFunctionContext*>(ptr);
}

DataPtr InefficientStdFunctionContext::makeDataPtr(void* ptr, const std::function<void(void*)>& deleter, Device device) {
    return {ptr, new InefficientStdFunctionContext({ptr, deleter}), &deleteInefficientStdFunctionContext, device};
}

void* alloc_cpu(size_t nbytes) {
    if (nbytes == 0) return nullptr;
    
    void* data;
#ifdef __ANDROID__
    data = memalign(gAlignment, nbytes);
#elif defined(_MSC_VER) || defined(_WIN32)
    data = _aligned_malloc(nbytes, gAlignment);
#else
    int err = posix_memalign(&data, gAlignment, nbytes);
    if (err != 0) {
        throw "DefaultCPUAllocator: can't allocate memory";
    }
#endif
        return data;
}

void free_cpu(void* data) {
#if defined(_MSC_VER) || defined(_WIN32)
    _aligned_free(data);
#else
    free(data);
#endif
}

void *otter_malloc_log(const size_t size, const char * const filename, const char * const funcname, const int line) {
    if (size == 0) return nullptr;
    void *ptr = malloc(size);
    if (!ptr) fprintf(stderr, "Failed to malloc %zu(bytes) at File: %s Func: %s Line: %d\n", size, filename, funcname, line);
    return ptr;
}

void *otter_calloc_log(const size_t nmemb, const size_t size, const char * const filename, const char * const funcname, const int line) {
    if (size == 0 || nmemb == 0) return 0;
    void *ptr = calloc(nmemb, size);
    if (!ptr) fprintf(stderr, "Failed to calloc %zu(bytes) at File: %s Func: %s Line: %d\n", nmemb * size, filename, funcname, line);
    return ptr;
}

void *otter_realloc_log(void *ptr, const size_t size, const char * const filename, const char * const funcname, const int line) {
    ptr = realloc(ptr,size);
    if (!ptr) fprintf(stderr, "Failed to realloc %zu(bytes) at File: %s Func: %s Line: %d\n", size, filename, funcname, line);
    return ptr;
}

void otter_free(void *ptr) {
    free(ptr);
    ptr = nullptr;
}
