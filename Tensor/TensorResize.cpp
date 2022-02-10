//
//  TensorResize.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "TensorResize.hpp"

namespace otter {

bool resize_output_check(const Tensor& output, IntArrayRef shape) {
    // Tests for resizing of tensors with one or more elements
    if (output.sizes().equals(shape)) {
        return false;
    }
    if (output.numel() != 0) {
        fprintf(stderr, "Weird!\n");
    }
    return true;
}

bool resize_output(const Tensor& output, IntArrayRef shape) {
    if (resize_output_check(output, shape)) {
        native::resize_(output, shape);
        return true;
    } else {
        return false;
    }
}

void resize_bytes_cpu(MemoryNucleus* memory, size_t size_bytes) {
    //  TORCH_CHECK(memory->resizable(), "Trying to resize memory that is not resizable");
    
    DataPtr new_data;
    if (size_bytes != 0) {
        new_data = memory->allocator()->allocate(size_bytes);
    }
    DataPtr old_data = memory->set_data_ptr(std::move(new_data));
    const auto old_capacity = memory->nbytes();
    memory->set_nbytes(size_bytes);
    const auto copy_capacity = std::min(size_bytes, old_capacity);
    if (old_data != nullptr && copy_capacity > 0) {
        memcpy(memory->data(), old_data.get(), copy_capacity);
    }
}

namespace native {

const Tensor& resize_as_(const Tensor& self, const Tensor& the_template) {
    const Tensor& result = self.resize_(the_template.sizes());
    self.unsafeGetTensorNucleus()->empty_tensor_restride();
    return result;
}

const Tensor& resize_(const Tensor& self, IntArrayRef size) {
    auto* self_ = self.unsafeGetTensorNucleus();
    resize_impl_cpu_(self_, size, {});
    self_->empty_tensor_restride();
    return self;
}

}   // end namespace native


}   // end namespace otter
