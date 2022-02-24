//
//  EmptyTensor.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/1.
//

#include "Utils.hpp"
#include "EmptyTensor.hpp"

namespace otter {

size_t computeStorageNbytes(
    IntArrayRef sizes,
    IntArrayRef strides,
    size_t itemsize_bytes) {
    size_t size = 1;
    for (const auto i : otter::irange(sizes.size())) {
        if(sizes[i] == 0) {
            return 0;
        }
        size += strides[i]*(sizes[i]-1);
    }
    return size * itemsize_bytes;
}

Tensor empty_cpu(IntArrayRef size, TensorOptions option) {
    ScalarType dtype = typeMetaToScalarType(option.dtype());
    
    return empty_cpu(size, dtype);
}

Tensor empty_cpu(IntArrayRef size, TensorOptions option, MemoryFormat memory_format) {
    ScalarType dtype = typeMetaToScalarType(option.dtype());
    
    return empty_generic(size, GetAllocator(Device::CPU), dtype, memory_format);
}

Tensor empty_cpu(IntArrayRef size, ScalarType dtype) {
    return empty_generic(size, GetAllocator(Device::CPU), dtype);
}

Tensor empty_generic(
    IntArrayRef size,
    Allocator* allocator,
    ScalarType scalar_type) {
    int64_t nelements = multiply_integers(size);
    TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
    int64_t size_bytes = nelements * dtype.itemsize();
    
    Memory memory = make_otterptr<MemoryNucleus>(size_bytes, allocator);
    Tensor tensor = otter::make_tensor<otter::TensorNucleus>(std::move(memory), dtype);
    
    if (size.size() != 1 || size[0] != 0) {
        tensor.unsafeGetTensorNucleus()->set_sizes_contiguous(size);
    }
    
    return tensor;
}

Tensor empty_generic(
    IntArrayRef size,
    Allocator* allocator,
    ScalarType scalar_type,
    MemoryFormat memory_format) {
    int64_t nelements = multiply_integers(size);
    TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
    int64_t size_bytes = nelements * dtype.itemsize();
    
    Memory memory = make_otterptr<MemoryNucleus>(size_bytes, allocator);
    Tensor tensor = otter::make_tensor<otter::TensorNucleus>(std::move(memory), dtype);
    
    if (size.size() != 1 || size[0] != 0) {
        tensor.unsafeGetTensorNucleus()->set_sizes_contiguous(size);
    }
    
    if (memory_format != MemoryFormat::Contiguous) {
        tensor.unsafeGetTensorNucleus()->empty_tensor_restride(memory_format);
    }
    
    return tensor;
}

Tensor empty_strided_cpu(IntArrayRef size, IntArrayRef stride, TensorOptions option) {
    ScalarType dtype = typeMetaToScalarType(option.dtype());
    
    return empty_strided_cpu(size, stride, dtype);
}

Tensor empty_strided_cpu(IntArrayRef size, IntArrayRef stride, ScalarType dtype) {
    return empty_strided_generic(size, stride, GetAllocator(Device::CPU), dtype);
}

Tensor empty_strided_generic(
    IntArrayRef size,
    IntArrayRef stride,
    Allocator* allocator,
    ScalarType scalar_type) {
    
    TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
    int64_t size_bytes = computeStorageNbytes(size, stride, dtype.itemsize());
    
    Memory memory = make_otterptr<MemoryNucleus>(size_bytes, allocator);
    Tensor tensor = otter::make_tensor<otter::TensorNucleus>(std::move(memory), dtype);

    tensor.unsafeGetTensorNucleus()->set_sizes_and_strides(size, stride);
    
    return tensor;
}

}   // namespace otter
