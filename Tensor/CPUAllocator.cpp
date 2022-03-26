//
//  CPUAllocator.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/27.
//

#include "CPUCachingAllocator.hpp"
#include "CPUProfilingAllocator.hpp"
#include "CPUAllocator.hpp"
#include "Macro.hpp"
#include "Config.hpp"

namespace otter {

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
        
        auto allocator_ptr = GetThreadLocalCachingAllocator();
        auto profiling_allocator_ptr = GetThreadLocalProfilingAllocator();
        if (allocator_ptr != nullptr) {
            allocator_ptr->free(pointer);
        } else if (profiling_allocator_ptr != nullptr) {
            profiling_allocator_ptr->free(pointer);
        } else {
            otter::free_cpu(pointer);
            // This adds extra cost to freeing memory to the default case when
            // caching allocator is not enabled.
            // NOLINTNEXTLINE(clang-analyzer-unix.Malloc)
            CPUCachingAllocator::record_free(pointer);
            auto allocation_planner = GetThreadLocalAllocationPlanner();
            if (allocation_planner != nullptr) {
                allocation_planner->record_free(pointer);
            }
        }
        
    }
    
    DataPtr allocate(size_t nbytes) const override {
        if (OTTER_UNLIKELY(0u == nbytes)) {
            return {
                nullptr,
                nullptr,
                &deleter,
                Device::CPU
            };
        }
        auto alloc_size = PreGuardBytes + nbytes + PostGuardBytes;
        void* data;
        
        auto allocator_ptr = GetThreadLocalCachingAllocator();
        auto profiling_allocator_ptr = GetThreadLocalProfilingAllocator();
        if (allocator_ptr != nullptr) {
            data = allocator_ptr->allocate(alloc_size);
        } else if (profiling_allocator_ptr != nullptr) {
            data = profiling_allocator_ptr->allocate(alloc_size);
        } else {
            data = alloc_cpu(alloc_size);
            auto allocation_planner = GetThreadLocalAllocationPlanner();
            if (allocation_planner != nullptr) {
                allocation_planner->record_allocation(alloc_size, data);
            }
        }
        
        return {
            reinterpret_cast<uint8_t*>(data) + PreGuardBytes,
            data,
            &deleter,
            Device::CPU
        };
    }
    
};

static DefaultMobileCPUAllocator<gAlignment, 16u> g_mobile_cpu_allocator;

Allocator* GetDefaultMobileCPUAllocator() {
    return &g_mobile_cpu_allocator;
}

#if OTTER_MOBILE
Allocator* GetDefaultCPUAllocator() {
    return GetDefaultMobileCPUAllocator();
}
#else
static DefaultCPUAllocator g_cpu_alloc;

Allocator* GetDefaultCPUAllocator() {
    return &g_cpu_alloc;
}
#endif  // OTTER_MOBILE

Allocator* GetCPUAllocator() {
    return GetAllocator(Device::CPU);
}

Allocator* GetAllocator(Device device) {
    switch (device) {
        case Device::CPU: return GetDefaultCPUAllocator(); break;
        default: return GetDefaultCPUAllocator();
    }
    return GetDefaultCPUAllocator();
}


}
