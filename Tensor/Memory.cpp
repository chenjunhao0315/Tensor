//
//  Memory.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "Memory.hpp"

namespace otter {

Memory_Tiny::~Memory_Tiny() {
    if (cpu_data_ != nullptr && cpu_data_->unreference() == 0) {
        delete cpu_data_;
    }
}

Memory_Tiny::Memory_Tiny() : cpu_data_(nullptr), size_(0), state_(UNINITIALIZED) {}

Memory_Tiny::Memory_Tiny(size_t size) : cpu_data_(nullptr), size_(size), state_(UNINITIALIZED) {}

Memory_Tiny::Memory_Tiny(const Memory_Tiny &other) {
    size_ = other.size_;
    cpu_data_ = other.cpu_data_;
    state_ =  (cpu_data_ && cpu_data_->reference()) ? REFERENCE_CPU : UNINITIALIZED;
}

Memory_Tiny& Memory_Tiny::operator=(const Memory_Tiny& other) {
    if (this == &other) {
        return *this;
    }
    if (cpu_data_ != nullptr && cpu_data_->unreference() == 0) {
        delete cpu_data_;
    }
    size_ = other.size_;
    cpu_data_ = other.cpu_data_;
    state_ =  (cpu_data_ && cpu_data_->reference()) ? REFERENCE_CPU : UNINITIALIZED;
    
    return *this;
}

Memory_Tiny Memory_Tiny::clone() {
    Memory_Tiny clone(this->size());
    if (state_ == UNINITIALIZED) return clone;
    
    this->copyTo(clone);
    
    return clone;
}

void Memory_Tiny::copyFrom(Memory_Tiny &other) {
    const void *src_data = other.cpu_data();
    void *dst_data = this->mutable_cpu_data();
    memcpy(dst_data, src_data, std::min(size(), other.size()));
}

void Memory_Tiny::copyTo(Memory_Tiny &other) {
    const void *src_data = this->cpu_data();
    void *dst_data = other.mutable_cpu_data();
    memcpy(dst_data, src_data, std::min(size(), other.size()));
}

void Memory_Tiny::to_cpu() {
    switch (state_) {
        case UNINITIALIZED:
            cpu_data_ = new Data(size_);
            state_ = OWN_CPU;
            break;
        case OWN_CPU:
        case REFERENCE_CPU:
            break;
    }
}

const void* Memory_Tiny::cpu_data() {
    to_cpu();
    return (const void*)cpu_data_->cpu_data();
}

void* Memory_Tiny::mutable_cpu_data() {
    to_cpu();
    return (void*)cpu_data_->cpu_data();
}

void Memory_Tiny::status() {
    std::cout << "<Memory_Tiny at " << this << ">" << std::endl;
    std::cout << "-> Size: " << size() << "(bytes) Status: ";
    switch (state_) {
        case UNINITIALIZED: std::cout << "UNINITIALIZED"; break;
        case OWN_CPU: std::cout << "OWN_CPU"; break;
        case REFERENCE_CPU: std::cout << "REFERENCE_CPU"; break;
    }
    std::cout << " Physical: " << cpu_data_ << std::endl;
}

}   // end namespace otter
