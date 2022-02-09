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
