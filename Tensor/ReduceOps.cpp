//
//  ReduceOps.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/18.
//

#include "ReduceOps.hpp"
#include "Tensor.hpp"
#include "TensorFunction.hpp"
#include "Dispatch.hpp"
#include "ReduceOpsUtils.hpp"

namespace otter {

inline ScalarType get_dtype_from_self(
    const Tensor& self,
    const ScalarType& dtype,
    bool promote_integers) {
    if (dtype != ScalarType::Undefined) {
        return dtype;
    }
    ScalarType src_type = self.scalar_type();
    if (promote_integers && otter::isIntegralType(src_type, /*includeBool=*/true)) {
        return otter::ScalarType::Long;
    }
    return src_type;
}

static ScalarType infer_dtype_from_optional(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    const ScalarType& opt_dtype,
    const Tensor& result) {
    // 'opt_dtype' has the priority for both cases.
    if (result.defined()) {
        // Otherwise, get the result type, if defined.
        return opt_dtype != ScalarType::Undefined ? opt_dtype : result.scalar_type();
    } else {
        // Last case is to get the self type.
        // If the self type is an integer, we promote it to kLong.
        return get_dtype_from_self(self, opt_dtype, true);
    }
}

DEFINE_META_FUNCTION_OVERLOAD(sum, dim_IntList)(const Tensor& self, IntArrayRef dim, bool keepdim, ScalarType opt_dtype) {
    auto out_dtype = infer_dtype_from_optional(self, dim, keepdim, opt_dtype, maybe_get_output());
    resize_reduction(*this, self, dim, keepdim, out_dtype);
}

DEFINE_DISPATCH(sum_stub);

DEFINE_IMPL_FUNCTION(sum_out)(const Tensor& self, IntArrayRef dim, bool keepdim, ScalarType opt_dtype,
 const Tensor& result) {
    auto iter = make_reduction_from_out_ty(self, result, dim, keepdim, result.scalar_type());
    if (iter.numel() == 0) {
        result.zero_();
    } else {
        sum_stub(Device::CPU, iter);
    }
}

}   // end namespace otter
