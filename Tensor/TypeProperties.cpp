//
//  TypeProperties.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/24.
//

#include "TypeProperties.hpp"
#include "DefaultDtype.hpp"
#include "Tensor.hpp"

namespace otter {
namespace native {

//bool is_cuda(const Tensor& self) {
//  return self.is_cuda();
//}
bool is_distributed(const Tensor& /*self*/) {
    return false;
}
//bool is_complex(const Tensor& self) {
//  return self.is_complex();
//}
bool is_floating_point(const Tensor& self) {
    return self.is_floating_point();
}
bool is_inference(const Tensor& self) {
    return self.is_inference();
}
bool is_signed(const Tensor &self) {
    return self.is_signed();
}
bool _is_zerotensor(const Tensor& self) {
    return self._is_zerotensor();
}
//bool is_conj(const Tensor& self) {
//  return self.is_conj();
//}
bool is_neg(const Tensor& self) {
    return self.is_neg();
}
//bool is_sparse(const Tensor& self) {
//  return self.is_sparse();
//}
//bool is_sparse_csr(const Tensor& self) {
//  return self.is_sparse_csr();
//}
//bool is_quantized(const Tensor& self) {
//  return self.is_quantized();
//}
// True if `self` and `from` have compatible tensor type so that `from`'s
// TensorImpl can be copied to `self`.
//bool _has_compatible_shallow_copy_type(const Tensor& self, const Tensor& from) {
//  return self.unsafeGetTensorNucleus()->has_compatible_shallow_copy_type(
//      from.key_set());
//}
Tensor type_as(const Tensor& self, const Tensor& other) {
    return self.to(other.options());
}
static inline ScalarType promote_skip_undefined(ScalarType a, ScalarType b) {
    if (a == ScalarType::Undefined) {
        return b;
    }
    if (b == ScalarType::Undefined) {
        return a;
    }
    return promoteTypes(a, b);
}
static inline ScalarType combine_categories(ScalarType higher, ScalarType lower) {
    if(isFloatingType(higher)) {
        return higher;
    }
    if (higher == ScalarType::Bool || isFloatingType(lower)) {
        return promote_skip_undefined(higher, lower);
    }
    if (higher != ScalarType::Undefined) {
        return higher;
    }
    return lower;
}
ResultTypeState update_result_type_state(const Tensor& tensor, const ResultTypeState& in_state) {
    if (!tensor.defined()) {
        return in_state;
    }
    ResultTypeState new_state = in_state;
    ScalarType current = tensor.scalar_type();
    if (tensor.unsafeGetTensorNucleus()->is_wrapped_number()) {
        if(isFloatingType(current)) {
            current = typeMetaToScalarType(otter::get_default_dtype());
        }
    }
    if ( tensor.dim() > 0 ) {
        new_state.dimResult = promote_skip_undefined(in_state.dimResult, current);
    } else if (tensor.unsafeGetTensorNucleus()->is_wrapped_number()) {
        new_state.wrappedResult = promote_skip_undefined(in_state.wrappedResult, current);
    } else {
        new_state.zeroResult = promote_skip_undefined(in_state.zeroResult, current);
    }
    return new_state;
}
ResultTypeState update_result_type_state(const Scalar& scalar, const ResultTypeState& in_state) {
    ResultTypeState new_state = in_state;
    ScalarType current = scalar.type();
    if (isFloatingType(current)) {
        current = typeMetaToScalarType(otter::get_default_dtype());
    }
    new_state.wrappedResult = promote_skip_undefined(in_state.wrappedResult, current);
    return new_state;
}
ScalarType result_type(const ResultTypeState& in_state) {
    return combine_categories(in_state.dimResult, combine_categories(in_state.zeroResult, in_state.wrappedResult));
}
ScalarType result_type(TensorList tensors) {
    ResultTypeState state = {};
    for (const Tensor& tensor : tensors) {
        state = update_result_type_state(tensor, state);
    }
    return result_type(state);
}
ScalarType result_type(const Tensor &tensor, const Tensor &other) {
    // NOLINTNEXTLINE(performance-move-const-arg)
    std::vector<Tensor> tensors{std::move(tensor), std::move(other)};
    return native::result_type(tensors);
}
ScalarType result_type(const Tensor &tensor, const Scalar& other) {
    ResultTypeState state = {};
    state = update_result_type_state(tensor, state);
    state = update_result_type_state(other, state);
    return result_type(state);
}
ScalarType result_type(const Scalar& scalar, const Tensor &tensor) {
    return result_type(tensor, scalar);
}
ScalarType result_type(const Scalar& scalar1, const Scalar& scalar2) {
    ResultTypeState state = {};
    state = update_result_type_state(scalar1, state);
    state = update_result_type_state(scalar2, state);
    return result_type(state);
}
bool can_cast(const ScalarType from, const ScalarType to) {
    return otter::canCast(from, to);
}
ScalarType promote_types(ScalarType type1, ScalarType type2) {
    ScalarType ret = promoteTypes(type1, type2);
    OTTER_CHECK(ret != ScalarType::Undefined, "Promotion from ", type1, " and ", type2, " is unsupported.");
    return ret;
}

}   // end namespace native
}   // end namespace otter
