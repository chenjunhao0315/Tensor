//
//  SmallBuffer.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef SmallBuffer_hpp
#define SmallBuffer_hpp

#include <type_traits>
#include <cstdint>
#include <cstddef>

namespace otter {

template <typename T, size_t N>
class SmallBuffer {
    static_assert(std::is_pod<T>::value, "SmallBuffer is intended for POD types");
    
    T storage_[N];
    size_t size_;
    T* data_;
    
public:
    SmallBuffer(size_t size) : size_(size) {
        if (size > N) {
            data_ = new T[size];
        } else {
            data_ = &storage_[0];
        }
    }
    
    ~SmallBuffer() {
        if (size_ > N) {
            delete[] data_;
        }
    }
    
    T& operator[](int64_t idx) {
        return data()[idx];
    }
    const T& operator[](int64_t idx) const {
        return data()[idx];
    }
    T* data() {
        return data_;
    }
    const T* data() const {
        return data_;
    }
    size_t size() const {
        return size_;
    }
    T* begin() {
        return data_;
    }
    const T* begin() const {
        return data_;
    }
    T* end() {
        return data_ + size_;
    }
    const T* end() const {
        return data_ + size_;
    }
};

}

#endif /* SmallBuffer_hpp */
