//
//  Memory.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef Memory_hpp
#define Memory_hpp

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <iostream>

#include "RefPtr.hpp"
#include "ExclusivelyOwned.hpp"
#include "Allocator.hpp"

namespace otter {

class Ref;
class Data;
class Memory_Tiny;

class Memory_Tiny {
public:
    ~Memory_Tiny();
    Memory_Tiny();
    Memory_Tiny(size_t size);
    Memory_Tiny(const Memory_Tiny& other);
    Memory_Tiny& operator=(const Memory_Tiny& other);
    Memory_Tiny clone();
    
    void copyFrom(Memory_Tiny &other);
    void copyTo(Memory_Tiny &other);
    
    const void *cpu_data();
    void *mutable_cpu_data();
    
    enum MemoryState { UNINITIALIZED, OWN_CPU, REFERENCE_CPU };
    MemoryState state() { return state_; }
    size_t size() const { return size_; }
    void status();
private:
    void to_cpu();
    
    Data *cpu_data_;
    MemoryState state_;
    size_t size_;
};

class Ref {
public:
    Ref() : refCount_(1) {}
    int reference() { return ++refCount_; }
    int unreference() { return --refCount_; }
    int refCount() { return refCount_; }
private:
    int refCount_;
};

class Data : public Ref {
public:
    ~Data() { otter_free(cpu_data_); }
    Data() : Ref(), cpu_data_(nullptr) {}
    Data(size_t size) : Data() { cpu_data_ = otter_calloc(1, size); }
    void *cpu_data() { return cpu_data_; }
private:
    void *cpu_data_;
};

class MemoryNucleus : public Ptr_quantum {
public:
    MemoryNucleus(size_t size_bytes, DataPtr data_ptr, Allocator* allocator) : data_ptr_(std::move(data_ptr)), size_bytes_(size_bytes), allocator_(allocator) {}
    
    MemoryNucleus(size_t size_bytes, Allocator* allocator) : MemoryNucleus(size_bytes, allocator->allocate(size_bytes), allocator) {}
    
    MemoryNucleus() = delete;
    MemoryNucleus(const MemoryNucleus&) = delete;
    MemoryNucleus(MemoryNucleus&& other) = default;
    ~MemoryNucleus() override = default;
    MemoryNucleus& operator=(MemoryNucleus&& other) = default;
    MemoryNucleus& operator=(const MemoryNucleus&) = delete;
    
    void reset() {
        data_ptr_.clear();
        size_bytes_ = 0;
    }
    
    template <typename T>
    inline T* data() const {
        return unsafe_data<T>();
    }

    template <typename T>
    inline T* unsafe_data() const {
        return static_cast<T*>(this->data_ptr_.get());
    }

    void release_resources() override {
        data_ptr_.clear();
    }

    size_t nbytes() const {
        return size_bytes_;
    }
    
    void set_nbytes(size_t size_bytes) {
        size_bytes_ = size_bytes;
    }
    
    DataPtr& data_ptr() {
        return data_ptr_;
    }
    
    const DataPtr& data_ptr() const {
        return data_ptr_;
    }
    
    DataPtr set_data_ptr(DataPtr&& data_ptr) {
        std::swap(data_ptr_, data_ptr);
        return std::move(data_ptr);
    };

    void set_data_ptr_noswap(DataPtr&& data_ptr) {
        data_ptr_ = std::move(data_ptr);
    }
    
    void* data() {
        return data_ptr_.get();
    }
    
    void* data() const {
        return data_ptr_.get();
    }
    
    Allocator* allocator() {
        return allocator_;
    }
    
    const Allocator* allocator() const {
        return allocator_;
    }
    
    Device device() const {
        return data_ptr_.device();
    }
    
private:
    DataPtr data_ptr_;
    size_t size_bytes_;
    Allocator* allocator_;
};

struct Memory {
public:
    Memory() {}
    Memory(Ptr<MemoryNucleus> ptr) : memory_nucleus_(std::move(ptr)) {}
    Memory(size_t size, Allocator* allocator = nullptr) : memory_nucleus_(make_otterptr<MemoryNucleus>(size, allocator)) {}
    
    Memory(size_t size, DataPtr data_ptr, Allocator* allocator = nullptr) : memory_nucleus_(make_otterptr<MemoryNucleus>(size, std::move(data_ptr), allocator)) {}
    
    operator bool() const {
        return memory_nucleus_;
    }
    
    template <typename T>
    inline T* data() const {
        return memory_nucleus_->data<T>();
    }

    template <typename T>
    inline T* unsafe_data() const {
        return memory_nucleus_->unsafe_data<T>();
    }

    size_t nbytes() const {
        return memory_nucleus_->nbytes();
    }
    
    void set_nbytes(size_t size_bytes) const {
        memory_nucleus_.get()->set_nbytes(size_bytes);
    }
    
    void* data() const {
        return memory_nucleus_.get()->data();
    }
    
    DataPtr& data_ptr() {
        return memory_nucleus_->data_ptr();
    }
    
    const DataPtr& data_ptr() const {
        return memory_nucleus_->data_ptr();
    }
    
    DataPtr set_data_ptr(DataPtr&& data_ptr) const {
        return memory_nucleus_.get()->set_data_ptr(std::move(data_ptr));
    }
    
    void set_data_ptr_noswap(DataPtr&& data_ptr) const {
        return memory_nucleus_.get()->set_data_ptr_noswap(std::move(data_ptr));
    }
    
    Allocator* allocator() const {
        return memory_nucleus_.get()->allocator();
    }
    
    Device device() const {
        return memory_nucleus_->device();
    }
    
    static Memory create_empty(Device device) {
        Allocator* allocator = GetAllocator(device);
        return Memory(make_otterptr<MemoryNucleus>(0, allocator->allocate(0), allocator));
    }
    
    bool is_alias_of(const Memory& other) const {
        return memory_nucleus_ == other.memory_nucleus_;
    }
    
    MemoryNucleus* unsafeGetMemoryNucleus() const noexcept { return memory_nucleus_.get(); }
    
protected:
    Ptr<MemoryNucleus> memory_nucleus_;
};

}   // end namespace otter

#endif /* Memory_hpp */
