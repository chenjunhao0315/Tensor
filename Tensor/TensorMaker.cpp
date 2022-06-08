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

float float16_to_float32(unsigned short value)
{
    // 1 : 5 : 10
    unsigned short sign = (value & 0x8000) >> 15;
    unsigned short exponent = (value & 0x7c00) >> 10;
    unsigned short significand = value & 0x03FF;

    //     NCNN_LOGE("%d %d %d", sign, exponent, significand);

    // 1 : 8 : 23
    union
    {
        unsigned int u;
        float f;
    } tmp;
    if (exponent == 0)
    {
        if (significand == 0)
        {
            // zero
            tmp.u = (sign << 31);
        }
        else
        {
            // denormal
            exponent = 0;
            // find non-zero bit
            while ((significand & 0x200) == 0)
            {
                significand <<= 1;
                exponent++;
            }
            significand <<= 1;
            significand &= 0x3FF;
            tmp.u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
        }
    }
    else if (exponent == 0x1F)
    {
        // infinity or NaN
        tmp.u = (sign << 31) | (0xFF << 23) | (significand << 13);
    }
    else
    {
        // normalized
        tmp.u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
    }

    return tmp.f;
}

Tensor from_float16(const unsigned short* data, IntArrayRef size) {
    auto result = otter::empty_cpu(size, otter::ScalarType::Float);
    
    float* ptr = result.data_ptr<float>();
    
    int64_t remain = otter::multiply_integers(size);
    
    for (; remain > 0; --remain) {
        *ptr = float16_to_float32(*data);
        
        ptr++;
        data++;
    }
    
    return result;
}


}
