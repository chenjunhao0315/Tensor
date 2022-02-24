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

}

#endif /* ScalarOps_hpp */
