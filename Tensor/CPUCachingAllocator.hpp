//
//  CPUCachingAllocator.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/27.
//

#ifndef CPUCachingAllocator_hpp
#define CPUCachingAllocator_hpp

#include <cstdint>
#include <mutex>

#include "flat_hash_map.hpp"
#include "SmallVector.hpp"

namespace otter {

class CPUCachingAllocator {
    /*
     * What it does:
     * Caches all the allocations carried out by this allocator.
     * Cache key is the size of the allocation.
     * If requested size is found in the cache returns the cached pointer.
     * What it does not do:
     * No speculative allocation for any future allocations.
     */
private:
    inline void* allocate_and_cache(const size_t bytes);
    void free_cached();
    
protected:
    // Invariants.
    // 1. If memory is ever allocated via this allocator then
    //    the pointer will exist in allocation_map_, unless the allocator
    //    returned the memory to OS via free_cached.
    //  1.1. Therefore even when the said memory is "freed" via this
    //       allocator (and thus cached), it will continue to stay
    //       in allocation_map_. Furthermore it will also exist in
    //       available_map_. Thus an allocated memory pointer can be in both
    //       allocation_map_ and available_map_ simultaneously.
    // 2. Memory pointer maybe removed from allocation_map_, when it
    //    is freed outside of the scope of this allocator, but was allocated
    //    by this allocator.
    // 3. Available map only contains that memory which was allocated
    //    by this allocator and subsequently freed by this allocator.
    // As a result of above invariants, allocated memory ptr cannot be in
    // available_map_ unless it is in allocation_map_ as well.
    ska::flat_hash_map<size_t, otter::SmallVector<void*, 16>> available_map_;
    static ska::flat_hash_map<void*, size_t> allocation_map_;
    // Since allocation_map, which is a global instance, is mutated/read via
    // all public APIs we need a global mutex.
    static std::mutex mutex_;
    
public:
    static void record_free(void* ptr);
    virtual ~CPUCachingAllocator();
    // Checks the cache to see if allocation of size bytes can be found.
    // If so return cached memory, else
    // allocates memory, records it for caching and returns.
    virtual void* allocate(const size_t bytes);
    // Checks if the memory being freed is was marked for allocation by
    // an earlier call to allocate. If so cache the allocation.
    // Otherwise free.
    virtual void free(void* ptr);
};

CPUCachingAllocator* GetDefaultCPUCachingAllocator();

bool ThreadLocalCachingAllocatorEnabled();
CPUCachingAllocator* GetThreadLocalCachingAllocator();

class WithCPUCachingAllocatorGuard {
public:
    WithCPUCachingAllocatorGuard(CPUCachingAllocator* allocator);
    ~WithCPUCachingAllocatorGuard();
    
private:
    CPUCachingAllocator* prev_caching_allocator_ptr_{nullptr};
};

}

#endif /* CPUCachingAllocator_hpp */
