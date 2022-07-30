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
    auto out_dtype = infer_dtype_from_optional(self, opt_dtype, maybe_get_output());
    resize_reduction(*this, self, dim, keepdim, out_dtype);
}

DEFINE_META_FUNCTION_OVERLOAD(prod, dim_int)(const Tensor& self, int64_t dim, bool keepdim, ScalarType dtype) {
    auto out_dtype = infer_dtype_from_optional(self, dtype, maybe_get_output());
    resize_reduction(*this, self, dim, keepdim, out_dtype);
}

DEFINE_META_FUNCTION_OVERLOAD(mean, dim)(const Tensor& self, IntArrayRef opt_dim, bool keepdim, ScalarType opt_dtype) {
  auto in_dtype = get_dtype_from_self(self, opt_dtype, true);
  if (!isFloatingType(in_dtype)) {
      OTTER_CHECK(false, "dtype must be either a floating point or complex dtype.");
  }
  auto out_dtype = infer_dtype_from_optional(self, opt_dtype, maybe_get_output());
  resize_reduction(*this, self, opt_dim, keepdim, out_dtype);
}

DEFINE_DISPATCH(sum_stub);
DEFINE_DISPATCH(prod_stub);
DEFINE_DISPATCH(mean_stub);

DEFINE_IMPL_FUNCTION(sum_out)(const Tensor& self, IntArrayRef dim, bool keepdim, ScalarType /*opt_dtype*/,
 const Tensor& result) {
    auto iter = make_reduction_from_out_ty(self, result, dim, keepdim, result.scalar_type());
    if (iter.numel() == 0) {
        result.zero_();
    } else {
        sum_stub(Device::CPU, iter);
    }
}

void impl_func_prod(
    const Tensor& self,
    IntArrayRef dims,
    bool keepdim,
    ScalarType dtype,
    const Tensor& result) {
    auto iter = make_reduction_from_out_ty(self, result, dims, keepdim, result.scalar_type());
    if (iter.numel() == 0) {
        result.fill_(1);
    } else {
        prod_stub(Device::CPU, iter);
    }
}

DEFINE_IMPL_FUNCTION(prod_out)(const Tensor& self, int64_t dim, bool keepdim, ScalarType dtype, const Tensor& result) {
    impl_func_prod(self, dim, keepdim, dtype, result);
}


DEFINE_IMPL_FUNCTION(mean_out)(const Tensor& self, IntArrayRef opt_dim, bool keepdim, ScalarType opt_dtype, const Tensor& result) {
    ScalarType dtype = result.scalar_type();
    // TODO: the TensorIterator reduction implementation of mean
    // (mean_kernel_impl()) is unvectorized and leads to very poor performance
    // for production workloads. Once that's fixed, the following code can be used
    // in lieu of the sum + divide implementation below.
    int64_t dim_prod = 1;
    if (!opt_dim.empty() || opt_dim.size() == 0 || self.dim() == 0) {
        dim_prod = self.numel();
    } else {
        auto dim = opt_dim;
        for (auto d : dim) {
            dim_prod *= self.size(d);
        }
    }
    auto& result_mut = const_cast<Tensor&>(result);
    otter::native::sum_out(result_mut, self, opt_dim, keepdim, dtype).div_(dim_prod);
}

}   // end namespace otter
