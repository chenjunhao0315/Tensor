//
//  ScalarOps.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/5.
//

#ifndef ScalarOps_hpp
#define ScalarOps_hpp

#include "Scalar.hpp"

namespace otter {

class Tensor;

Tensor& scalar_fill(Tensor& self, const Scalar& value);
Tensor scalar_tensor(const Scalar& s, ScalarType dtype);

inline Tensor scalar_to_tensor(const Scalar& scalar, Device device = Device::CPU) {
    if (device == Device::CPU) {
        if (scalar.isFloatingPoint()) {
            return scalar_tensor(scalar, ScalarType::Float);
        } else if (scalar.isBoolean()) {
            return scalar_tensor(scalar, ScalarType::Bool);
        } else {
            return scalar_tensor(scalar, ScalarType::Long);
        }
    }
    return scalar_tensor(scalar, ScalarType::Float);
}

namespace native {
inline Tensor wrapped_scalar_tensor(const Scalar& scalar, const Device device = Device::CPU) {
  auto tensor = scalar_to_tensor(scalar, device);
  tensor.unsafeGetTensorNucleus()->set_wrapped_number(true);
  return tensor;
}

}

}   // end namespace otter

#endif /* ScalarOps_hpp */
