//
//  TensorResize.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef TensorResize_hpp
#define TensorResize_hpp

#include "Utils.hpp"
#include "Tensor.hpp"

namespace otter {

bool resize_output(const Tensor& output, IntArrayRef shape);
bool resize_output_check(const Tensor& output, IntArrayRef shape);
void resize_bytes_cpu(MemoryNucleus* memory, size_t size_bytes);

inline int64_t storage_size_for(IntArrayRef size, IntArrayRef stride) {
    assert(size.size() == stride.size());
    int64_t storage_size = 1;
    for (const auto dim : otter::irange(size.size())) {
        if (size[dim] == 0) {
            storage_size = 0;
            break;
        }
        storage_size += (size[dim] - 1) * stride[dim];
    }
    return storage_size;
}

static inline void maybe_resize_storage_cpu(TensorNucleus* self, uint64_t new_size) {
    if (new_size == 0) {
        return;
    }
    
    const auto new_size_bytes_i = (new_size + self->memory_offset()) * self->dtype().itemsize();
    assert(!overflows<size_t>(new_size_bytes_i));
    const auto new_size_bytes = static_cast<size_t>(new_size_bytes_i);
    
    const Memory& memory = self->memory();
    if (!memory) {
        auto new_memory = make_otterptr<MemoryNucleus>(new_size_bytes, GetAllocator(Device::CPU));
        self->set_storage_keep_dtype(std::move(new_memory));
    } else if (new_size_bytes > memory.nbytes()) {
        resize_bytes_cpu(memory.unsafeGetMemoryNucleus(), new_size_bytes);
    }
}

inline TensorNucleus* resize_impl_cpu_(TensorNucleus* self, IntArrayRef size, IntArrayRef stride, bool resize_storage = true) {
    if (self->sizes() == size && (stride.empty() || self->strides() == stride)) {
        return self;
    }
    
    int64_t storage_size = 1;
    if (!stride.empty()) {
        self->set_sizes_and_strides(size, stride);
        storage_size = storage_size_for(size, stride);
    } else {
        self->set_sizes_contiguous(size);
        storage_size = self->numel();
    }
    if (resize_storage) {
        maybe_resize_storage_cpu(self, storage_size);
    }
    
    return self;
}

namespace native {

const Tensor& resize_as_(const Tensor& self, const Tensor& the_template);
const Tensor& resize_(const Tensor& self, IntArrayRef size);
const Tensor& resize_as_(const Tensor& self, const Tensor& the_template, MemoryFormat memory_format);
const Tensor& resize_(const Tensor& self, IntArrayRef size, MemoryFormat memory_format);

}


inline void setStrided(const Tensor& self, IntArrayRef size, IntArrayRef stride, int64_t memory_offset) {
    assert(size.size() == stride.size());
    auto* self_ = self.unsafeGetTensorNucleus();
    
    assert(memory_offset >= 0);
    self_->set_memory_offset(memory_offset);
    if (self_->sizes() == size && self_->strides() == stride) {
        return;
    }
    for (auto val : stride) {
        assert(val >= 0);
    }
    self_->set_sizes_and_strides(size, stride);
}



}   // end namespace otter



#endif /* TensorResize_hpp */
