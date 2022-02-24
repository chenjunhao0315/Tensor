//
//  TensorCopy.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/10.
//

#ifndef TensorCopy_hpp
#define TensorCopy_hpp

#include "DispatchStub.hpp"

namespace otter {

class Tensor;
class TensorIterator;

using copy_fn = void(*)(TensorIterator&, bool non_blocking);

Tensor& copy_(Tensor& self, const Tensor& src, bool non_blocking);

DECLARE_DISPATCH(copy_fn, copy_stub);

}

#endif /* TensorCopy_hpp */
