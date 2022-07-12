//
//  FillKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/30.
//

#include "Loop.hpp"
#include "Fill.hpp"
#include "FillKernel.hpp"
#include "Dispatch.hpp"

namespace otter {

void fill_kernel(TensorIterator& iter, const Scalar& value_scalar) {
    OTTER_DISPATCH_ALL_TYPES_AND(otter::ScalarType::Bool, iter.dtype(), "fill_cpu", [&]() {
        scalar_t value = value_scalar.to<scalar_t>();
        cpu_kernel(
            iter,
            [=]() -> scalar_t { return value; });
    });
}

REGISTER_DISPATCH(fill_stub, &fill_kernel);


}
