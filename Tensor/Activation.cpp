//
//  Activation.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/1.
//

#include "Activation.hpp"
#include "TensorFunction.hpp"
#include "TensorCompare.hpp"

namespace otter {

DEFINE_DISPATCH(leaky_relu_stub);
DEFINE_DISPATCH(threshold_stub);

DEFINE_META_FUNCTION(leaky_relu) (const Tensor& self, const Scalar& /*negative_slope*/) {
    build_unary_op(maybe_get_output(), self);
}

DEFINE_IMPL_FUNCTION(leaky_relu_out) (const Tensor& /*self*/, const Scalar& negval, const Tensor& /*result*/) {
    leaky_relu_stub(Device::CPU, *this, negval);
}

DEFINE_META_FUNCTION(threshold)(const Tensor& self, const Scalar& /*threshold*/, const Scalar& /*value*/) {
    const Tensor& result = maybe_get_output();
    build(TensorIteratorConfig()
          .set_check_mem_overlap(false)  // threshold is idempotent, so overlap is okay
          .add_output(result)
          .add_input(self)
          .add_input(self) // other
          .allow_cpu_scalars(true)
          .promote_inputs_to_common_dtype(true)
          .cast_common_dtype_to_outputs(true)
          .enforce_safe_casting_to_output(true));
}

DEFINE_IMPL_FUNCTION(threshold_out)(const Tensor& /*self*/, const Scalar& threshold, const Scalar& value, const Tensor& /*result*/) {
    threshold_stub(Device::CPU, *this, threshold, value);
}

Tensor relu(const Tensor & self) {
    return otter::clamp_min(self, 0);
}

Tensor & relu_(Tensor & self) {
    return otter::clamp_min_(self, 0);
}

Tensor relu6(const Tensor& self) {
    return otter::clamp(self, 0, 6);
}

Tensor& reul6_(Tensor& self) {
    return otter::clamp_(self, 0, 6);
}

}   // end namespace otter
