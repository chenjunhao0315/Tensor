//
//  TensorBase.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/1.
//

#include "TensorBase.hpp"

namespace otter {

// TensorNucleus main constructor
TensorNucleus::TensorNucleus(Memory&& memory, const TypeMeta data_type, TensorOptions) : memory_(std::move(memory)), data_type_(data_type), memory_offset_(0), numel_(0) {}

TensorNucleus::TensorNucleus(Memory&& memory, const TypeMeta data_type) : TensorNucleus(std::forward<Memory>(memory), data_type, TensorOptions{}) {}

int64_t TensorNucleus::dim() const {
    return perspective_view_.size();
}

int64_t TensorNucleus::size(size_t idx) const {
    idx = maybe_wrap_dim(idx, dim(), false);
    return perspective_view_.size_at(idx);
}

int64_t TensorNucleus::stride(size_t idx) const {
    idx = maybe_wrap_dim(idx, dim(), false);
    return perspective_view_.stride_at(idx);
}

void TensorBase::print() const {
    if (this->defined()) {
        std::cerr << "[" << toString() << " " << sizes() << "]" << std::endl;
    } else {
        std::cerr << "[UndefinedTensor]" << std::endl;
    }
}

UndefinedTensorNucleus::UndefinedTensorNucleus()
    : TensorNucleus({}, otter::TypeMeta()) {
//  set_storage_access_should_throw();
}

UndefinedTensorNucleus UndefinedTensorNucleus::_singleton;

std::string TensorBase::toString() const {
    return ::toString(this->unsafeGetTensorNucleus()->scalar_type()) + "Type";
}


}
