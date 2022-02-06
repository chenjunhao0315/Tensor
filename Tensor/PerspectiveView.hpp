//
//  PerspectiveView.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef PerspectiveView_hpp
#define PerspectiveView_hpp

#include "Memory.hpp"
#include "ArrayRef.hpp"

#define MAX_RANK_OF_TENSOR 5

namespace otter {

class PerspectiveView {
public:
    using sizes_iterator = int64_t*;
    using const_sizes_iterator = const int64_t*;
    using strides_iterator = int64_t*;
    using const_strides_iterator = const int64_t*;
    
    PerspectiveView() : size_(1) {
        size_at(0) = 0;
        stride_at(0) = 1;
    }
    
    ~PerspectiveView() {
        if (!isInline()) {
            otter_free(outOflineStorage_);
        }
    }
    
    PerspectiveView(const PerspectiveView& other) : size_(other.size_) {
        if (other.isInline()) {
            this->copyDataInline(other);
        } else {
            this->allocOutOfLineStorage(size_);
            copyOutOfLine(other);
        }
    }
    
    PerspectiveView& operator=(const PerspectiveView& other) {
        if (this == &other) {
            return *this;
        }
        if (other.isInline()) {
            if (!isInline()) {
                otter_free(outOflineStorage_);
            }
            this->copyDataInline(other);
        } else {
            if (isInline()) {
                allocOutOfLineStorage(other.size_);
            } else {
                resizeOutOfLineStorage(other.size_);
            }
            copyOutOfLine(other);
        }
        size_ = other.size_;
        
        return *this;
    }
    
    PerspectiveView(PerspectiveView&& other) noexcept : size_(other.size_) {
        if (isInline()) {
            memcpy(inlineStorage_, other.inlineStorage_, sizeof(inlineStorage_));
        } else {
            outOflineStorage_ = other.outOflineStorage_;
            other.outOflineStorage_ = nullptr;
        }
        other.size_ = 0;
    }
    
    PerspectiveView& operator=(PerspectiveView&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (other.isInline()) {
            if (!isInline()) {
                otter_free(outOflineStorage_);
            }
            copyDataInline(other);
        } else {
            if (!isInline()) {
                otter_free(outOflineStorage_);
            }
            outOflineStorage_ = other.outOflineStorage_;
            other.outOflineStorage_ = nullptr;
        }
        size_ = other.size_;
        other.size_ = 0;
        
        return *this;
    }
    
    size_t size() const noexcept {
        return size_;
    }
    
    int64_t* sizes_data() noexcept {
        if (isInline()) {
            return &inlineStorage_[0];
        } else {
            return &outOflineStorage_[0];
        }
    }
    
    const int64_t* sizes_data() const noexcept {
        if (isInline()) {
            return &inlineStorage_[0];
        } else {
            return &outOflineStorage_[0];
        }
    }
    
    sizes_iterator sizes_begin() noexcept {
        return sizes_data();
    }
    
    const_sizes_iterator sizes_begin() const noexcept {
        return sizes_data();
    }
    
    sizes_iterator sizes_end() noexcept {
        return sizes_data() + size();
    }
    
    const_sizes_iterator sizes_end() const noexcept {
        return sizes_data() + size();
    }
    
    IntArrayRef sizes_arrayref() const noexcept {
        return IntArrayRef(sizes_data(), size());
    }
    
    void set_sizes(IntArrayRef newSizes) {
        this->resize(newSizes.size());
        std::copy(newSizes.begin(), newSizes.end(), sizes_begin());
    }
    
    int64_t* strides_data() noexcept {
        if (isInline()) {
            return &inlineStorage_[MAX_RANK_OF_TENSOR];
        } else {
            return &outOflineStorage_[size()];
        }
    }
    
    const int64_t* strides_data() const noexcept {
        if (isInline()) {
            return &inlineStorage_[MAX_RANK_OF_TENSOR];
        } else {
            return &outOflineStorage_[size()];
        }
    }
    
    strides_iterator strides_begin() noexcept {
        return strides_data();
    }
    
    const_strides_iterator strides_begin() const noexcept {
        return strides_data();
    }
    
    strides_iterator strides_end() noexcept {
        return strides_data() + size();
    }
    
    IntArrayRef strides_arrayref() const noexcept {
        return IntArrayRef(strides_data(), size());
    }
    
    void set_strides(IntArrayRef newStrides) {
        this->resize(newStrides.size());
        std::copy(newStrides.begin(), newStrides.end(), strides_begin());
    }
    
    const_strides_iterator strides_end() const noexcept {
        return strides_data() + size();
    }
    
    int64_t size_at(size_t idx) const noexcept {
        return sizes_data()[idx];
    }
    
    int64_t& size_at(size_t idx) noexcept {
        return sizes_data()[idx];
    }
    
    int64_t stride_at(size_t idx) const noexcept {
        return strides_data()[idx];
    }
    
    int64_t& stride_at(size_t idx) noexcept {
        return strides_data()[idx];
    }
    
    void resize(size_t newSize) {
        const auto oldSize = size();
        if (newSize == oldSize) {
            return;
        }
        if (newSize <= MAX_RANK_OF_TENSOR && isInline()) {
            if (oldSize < newSize) {
                const auto bytesToZero = (newSize - oldSize) * sizeof(inlineStorage_[0]);
                memset(&inlineStorage_[oldSize], 0, bytesToZero);
                memset(&inlineStorage_[MAX_RANK_OF_TENSOR + oldSize], 0, bytesToZero);
            }
            size_ = newSize;
        } else {
            resizeSlowPath(newSize, oldSize);
        }
    }
    
    void resizeSlowPath(size_t newSize, size_t oldSize);
private:
    bool isInline() const noexcept {
        return size_ <= MAX_RANK_OF_TENSOR;
    }
    
    static size_t storageBytes(size_t size) {
        return 2 * size * sizeof(int64_t);
    }
    
    void allocOutOfLineStorage(size_t size) {
        outOflineStorage_ = static_cast<int64_t*>(otter_malloc(storageBytes(size)));
    }
    
    void resizeOutOfLineStorage(size_t newSize) {
        outOflineStorage_ = static_cast<int64_t*>(otter_realloc(outOflineStorage_, storageBytes(newSize)));
    }
    
    void copyDataInline(const PerspectiveView& other) {
        memcpy(inlineStorage_, other.inlineStorage_, sizeof(inlineStorage_));
    }
    
    void copyOutOfLine(const PerspectiveView& other) {
        memcpy(outOflineStorage_, other.outOflineStorage_, storageBytes(other.size_));
    }
    
    int64_t size_;
    union {
        int64_t inlineStorage_[MAX_RANK_OF_TENSOR * 2]{};
        int64_t* outOflineStorage_;
    };
};

}   // end namespace otter

#endif /* PerspectiveView_hpp */
