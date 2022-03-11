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

class Tensor;

namespace detail {

inline void noopDelete(void*) {}

}

class TensorMaker {
    friend TensorMaker for_blob(void* data, IntArrayRef sizes) noexcept;
public:
    using ContextDeleter = DeleterFnPtr;
    
    TensorMaker& options(TensorOptions value) noexcept {
        opts_ = value;

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

inline Tensor from_blob(void* data, IntArrayRef sizes, TensorOptions options) {
    return for_blob(data, sizes).options(options).make_tensor();
}

template <typename T>
Tensor tensor_cpu(ArrayRef<T> values, const TensorOptions& options);

#define TENSOR(T, S)                                                                    \
Tensor tensor(ArrayRef<T> values, const TensorOptions& options);                        \
inline Tensor tensor(std::initializer_list<T> values, const TensorOptions& options) {   \
    return otter::tensor(ArrayRef<T>(values), options);                                 \
}                                                                                       \
inline Tensor tensor(T value, const TensorOptions& options) {                           \
    return otter::tensor(ArrayRef<T>(value), options);                                  \
}                                                                                       \
inline Tensor tensor(ArrayRef<T> values) {                                              \
    return otter::tensor(std::move(values), TensorOptions(ScalarType::S));              \
}                                                                                       \
inline Tensor tensor(std::initializer_list<T> values) {                                 \
    return otter::tensor(ArrayRef<T>(values));                                          \
}                                                                                       \
inline Tensor tensor(T value) {                                                         \
    return otter::tensor(ArrayRef<T>(value));                                           \
}
OTTER_ALL_SCALAR_TYPES(TENSOR)
#undef TENSOR


}

#endif /* TensorMaker_hpp */
