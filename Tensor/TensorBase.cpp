//
//  TensorBase.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/1.
//

#include "TensorBase.hpp"

namespace otter {

// TensorNucleus main constructor
TensorNucleus::TensorNucleus(Memory&& memory, const TypeMeta data_type, Device device) : memory_(std::move(memory)), data_type_(data_type), memory_offset_(0), numel_(0), device_(device) {
    init_bitfields();
}

TensorNucleus::TensorNucleus(Memory&& memory, const TypeMeta data_type) : TensorNucleus(std::forward<Memory>(memory), data_type, memory.device()) {
    init_bitfields();
}

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

bool TensorNucleus::compute_contiguous() const {
    bool is_contiguous = true;
    if (is_empty())
        return is_contiguous;
    int64_t z = 1;
    for (int64_t d = dim() - 1; d >= 0; d--) {
        const auto size_d = perspective_view_.size_at(d);
        if (size_d != 1) {
            if (perspective_view_.stride_at(d) == z) {
                z *= size_d;
            } else {
                is_contiguous = false;
                break;
            }
        }
    }
  return is_contiguous;
}

bool TensorNucleus::compute_non_overlapping_and_dense() const {
    if (dim() == 1) {
        return perspective_view_.size_at(0) < 2 ||
            perspective_view_.stride_at(0) == 1;
    }
    SmallVector<int64_t, 5> perm;
    perm.resize(dim());
    for (const auto i : otter::irange(dim())) {
        perm[i] = i;
    }
    // Sort by strides, leaving 0 and 1 sized dims at the end of the array
    std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
        if (perspective_view_.size_at(a) < 2) {
            return false;
        } else if (perspective_view_.size_at(b) < 2) {
            return true;
        }
        return perspective_view_.stride_at(a) < perspective_view_.stride_at(b);
    });
    auto require_stride = 1;
    for (const auto i : otter::irange(dim())) {
        const auto size_perm_i = perspective_view_.size_at(perm[i]);
        if (size_perm_i < 2) {
            return true;
        }
        if (perspective_view_.stride_at(perm[i]) != require_stride) {
            return false;
        }
        require_stride *= size_perm_i;
    }
    return true;
}

bool TensorNucleus::compute_channels_last_contiguous_2d() const {
    switch (perspective_view_.size()) {
        case 4: {
            int64_t expected = 1;
            for (auto& d : {1, 3, 2, 0}) {
                const auto size_d = perspective_view_.size_at(d);
                if (size_d != 1) {
                    if (perspective_view_.stride_at(d) != expected) {
                        return false;
                    }
                    expected *= size_d;
                }
            }
            return true;
        }
        case 3:
            return false;
        default:
            return false;
    }
}

bool TensorNucleus::compute_strides_like_channels_last_2d() const {
    return is_channels_last_strides_2d(TensorNucleus::sizes(), TensorNucleus::strides());
}

void TensorBase::print() const {
    if (this->defined()) {
        std::cerr << "[" << toString() << " " << sizes() << "]" << std::endl;
    } else {
        std::cerr << "[UndefinedTensor]" << std::endl;
    }
}

UndefinedTensorNucleus::UndefinedTensorNucleus()
    : TensorNucleus({}, otter::TypeMeta(), Device::Undefined) {
//  set_storage_access_should_throw();
}

UndefinedTensorNucleus UndefinedTensorNucleus::_singleton;

std::string TensorBase::toString() const {
    return otter::toString(this->unsafeGetTensorNucleus()->scalar_type()) + "Type";
}

#define DEFINE_ALL_TYPES_DATA_PTR(T, name) \
template <> \
T* TensorBase::data_ptr() const {   \
    assert(scalar_type() == ScalarType::name);  \
    return this->unsafeGetTensorNucleus()->data_ptr_nucleus<T>();  \
}
OTTER_ALL_SCALAR_TYPES(DEFINE_ALL_TYPES_DATA_PTR)
#undef DEFINE_ALL_TYPES_DATA_PTR

}
