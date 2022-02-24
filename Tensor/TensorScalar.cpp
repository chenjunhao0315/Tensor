//
//  TensorScalar.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/24.
//

#include "TensorScalar.hpp"
#include "Tensor.hpp"
#include "Dispatch.hpp"

namespace otter {

Scalar item(const Tensor& self) {
    int64_t numel = self.numel();
    OTTER_CHECK(numel == 1, "a Tensor with ", numel, " elements cannot be converted to Scalar");
    return _local_scalar_dense_cpu(self);
}

Scalar _local_scalar_dense_cpu(const Tensor& self) {
    Scalar r;
    OTTER_DISPATCH_ALL_TYPES(self.scalar_type(), "_local_scalar_dense_cpu", [&] {
        scalar_t value = *self.data_ptr<scalar_t>();
        r = Scalar(value);
    });
    return r;
}

}
