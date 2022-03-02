//
//  Padding.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/2.
//

#ifndef Padding_hpp
#define Padding_hpp

#include "ArrayRef.hpp"

namespace otter {

class Tensor;
class Scalar;

Tensor constant_pad(const Tensor& self, IntArrayRef pad, const Scalar& value);

}

#endif /* Padding_hpp */
