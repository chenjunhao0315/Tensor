//
//  ScalarOps.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/5.
//

#include "Dispatch.hpp"
#include "EmptyTensor.hpp"
#include "ScalarOps.hpp"

namespace otter {
namespace {
template <typename scalar_t>
inline void fill_inplace(Tensor& self, const Scalar& value_scalar) {
    auto value = value_scalar.to<scalar_t>();
    scalar_t* dptr = static_cast<scalar_t*>(self.data_ptr());
    *dptr = value;
}
}

Tensor& scalar_fill(Tensor& self, const Scalar& value) {
    OTTER_DISPATCH_ALL_TYPES(self.scalar_type(), "scalar_fill_out", [&]() {
        fill_inplace<scalar_t>(self, value);
    });
    return self;
}

Tensor scalar_tensor(const Scalar& s, ScalarType dtype) {
    Tensor result = empty_cpu({}, dtype);
    scalar_fill(result, s);
    
    return result;
}





}
