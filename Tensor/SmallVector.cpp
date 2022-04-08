//
//  SmallVector.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "SmallVector.hpp"

using namespace otter;

[[noreturn]] static void report_size_overflow(size_t MinSize, size_t MaxSize);
static void report_size_overflow(size_t MinSize, size_t MaxSize) {
    std::string Reason = "SmallVector unable to grow. Requested capacity (" +
    std::to_string(MinSize) +
    ") is larger than maximum value for size type (" +
    std::to_string(MaxSize) + ")";
    throw std::length_error(Reason);
}

[[noreturn]] static void report_at_maximum_capacity(size_t MaxSize);
static void report_at_maximum_capacity(size_t MaxSize) {
    std::string Reason =
    "SmallVector capacity unable to grow. Already at maximum size " +
    std::to_string(MaxSize);
    throw std::length_error(Reason);
}

template <class Size_T>
static size_t getNewCapacity(size_t MinSize, size_t /*TSize*/, size_t OldCapacity) {
    constexpr size_t MaxSize = std::numeric_limits<Size_T>::max();
    
    if (MinSize > MaxSize)
        report_size_overflow(MinSize, MaxSize);
    
    if (OldCapacity == MaxSize)
        report_at_maximum_capacity(MaxSize);
    
    size_t NewCapacity = 2 * OldCapacity + 1;
    return std::min(std::max(NewCapacity, MinSize), MaxSize);
}

template <class Size_T>
void* SmallVectorBase<Size_T>::mallocForGrow(size_t MinSize, size_t TSize, size_t& NewCapacity) {
    NewCapacity = getNewCapacity<Size_T>(MinSize, TSize, this->capacity());
    auto Result = std::malloc(NewCapacity * TSize);
    if (Result == nullptr) {
        throw std::bad_alloc();
    }
    return Result;
}

template <class Size_T>
void SmallVectorBase<Size_T>::grow_pod(void* FirstEl, size_t MinSize, size_t TSize) {
    size_t NewCapacity = getNewCapacity<Size_T>(MinSize, TSize, this->capacity());
    void* NewElts;
    if (BeginX == FirstEl) {
        NewElts = std::malloc(NewCapacity * TSize);
        if (NewElts == nullptr) {
            throw std::bad_alloc();
        }
        
        // Copy the elements over.  No need to run dtors on PODs.
        memcpy(NewElts, this->BeginX, size() * TSize);
    } else {
        // If this wasn't grown from the inline copy, grow the allocated space.
        NewElts = std::realloc(this->BeginX, NewCapacity * TSize);
        if (NewElts == nullptr) {
            throw std::bad_alloc();
        }
    }
    
    this->BeginX = NewElts;
    this->Capacity = NewCapacity;
}

template class otter::SmallVectorBase<uint32_t>;

#if SIZE_MAX > UINT32_MAX
template class otter::SmallVectorBase<uint64_t>;
#endif
