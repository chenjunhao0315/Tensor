//
//  TensorMaker.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/10.
//

#ifndef TensorMaker_hpp
#define TensorMaker_hpp

#include "Tensor.hpp"

namespace otter {

namespace detail {

inline void noopDelete(void*) {}

}

class TensorMaker {
    friend TensorMaker for_blob(void* data, IntArrayRef sizes) noexcept;
public:
    using ContextDeleter = DeleterFnPtr;
    
    TensorMaker& dtype(ScalarType dtype) noexcept {
        dtype_ = dtype;
        
        return *this;
    }
    
    TensorMaker& strides(IntArrayRef strides) noexcept {
        strides_ = strides;
        
        return *this;
    }
    
    Tensor make_tensor();
private:
    explicit TensorMaker(void* data, IntArrayRef sizes) noexcept : data_(data), sizes_(sizes) {}
    
    size_t computeStorageSize();
    
    DataPtr makeDataPtrFromDeleter();
    DataPtr makeDataPtrFromContext();
    
    void* data_;
    IntArrayRef sizes_;
    IntArrayRef strides_;
    
    size_t memory_offset_ = 0;
    
    std::function<void(void*)> deleter_{};
    std::unique_ptr<void, ContextDeleter> ctx_{nullptr, detail::noopDelete};
    
    TensorOptions opts_{};
    ScalarType dtype_ = ScalarType::Undefined;
    Device device_ = Device::CPU;
};

inline TensorMaker for_blob(void* data, IntArrayRef sizes) noexcept {
    return TensorMaker{data, sizes};
}

inline Tensor from_blob(void* data, IntArrayRef sizes, ScalarType dtype) {
    return for_blob(data, sizes).dtype(dtype).make_tensor();
}




}

#endif /* TensorMaker_hpp */
