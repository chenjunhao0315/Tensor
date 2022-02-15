//
//  TensorMaker.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/10.
//

#include "Dispatch.hpp"
#include "EmptyTensor.hpp"
#include "TensorMaker.hpp"

namespace otter {

Tensor TensorMaker::make_tensor() {
    // check size nonnegative
    // check_size_nonnegative(sizes_);
    
    size_t size_bytes = this->computeStorageSize();
    DataPtr data_ptr{};
    if (deleter_) {
        data_ptr = makeDataPtrFromDeleter();
    } else {
        data_ptr = makeDataPtrFromContext();
    }
    Memory memory{size_bytes, std::move(data_ptr)};
    Tensor tensor = otter::make_tensor<otter::TensorNucleus>(std::move(memory), opts_.dtype());
    
    if (sizes_.size() != 1 || sizes_[0] != 0) {
        TensorNucleus* tensor_nucleus = tensor.unsafeGetTensorNucleus();
        
        if (!strides_.empty()) {
            tensor_nucleus->set_sizes_and_strides(sizes_, strides_);
        } else {
            tensor_nucleus->set_sizes_contiguous(sizes_);
        }
        if (memory_offset_) {
            tensor_nucleus->set_memory_offset(memory_offset_);
        }
    }
    
    return tensor;
}

size_t TensorMaker::computeStorageSize() {
    size_t itemsize = opts_.dtype().itemsize();
    
    if (!strides_.empty()) {
        auto memory_size = otter::computeStorageNbytes(sizes_, strides_, itemsize);
        if (memory_offset_) {
            memory_size += memory_offset_;
        }
        return memory_size;
    }
    
    size_t size = 1;
    for (size_t s : sizes_) {
        size *= static_cast<int64_t>(s);
    }
    auto memory_size = size * itemsize;
    if (memory_offset_) {
        memory_size += memory_offset_;
    }
    return memory_size;
}

DataPtr TensorMaker::makeDataPtrFromDeleter() {
    return InefficientStdFunctionContext::makeDataPtr(data_, deleter_, device_);
}

DataPtr TensorMaker::makeDataPtrFromContext() {
    return DataPtr{data_, ctx_.release(), ctx_.get_deleter(), device_};
}

template <typename T>
Tensor tensor_cpu(ArrayRef<T> values, const TensorOptions& options) {
    auto result = otter::empty_cpu({static_cast<long long>(values.size())}, options);
    OTTER_DISPATCH_ALL_TYPES(result.scalar_type(), "tensor_cpu", [&] {
        std::copy(values.begin(), values.end(), result.template data_ptr<scalar_t>());
    });
    return result;
}

#define TENSOR(T, _) \
Tensor tensor(ArrayRef<T> values, const TensorOptions& options) { \
    return tensor_cpu(values, options); \
}
OTTER_ALL_SCALAR_TYPES(TENSOR)
#undef TENSOR


}
