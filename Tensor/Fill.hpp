//
//  Fill.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/30.
//

#ifndef Fill_hpp
#define Fill_hpp

#include "Scalar.hpp"
#include "DispatchStub.hpp"
#include "TensorIterator.hpp"

namespace otter {

DECLARE_DISPATCH(void(*)(TensorIterator&, const Scalar&), fill_stub);

namespace native {
Tensor& fill_out(Tensor& self, const Scalar& value);

Tensor& fill_(Tensor& self, const Scalar& value);

Tensor& zero_cpu_(Tensor& self, int64_t numel);

Tensor& zero_(Tensor& self);
}   // end namespace native
}   // end namespace otter

#endif /* Fill_hpp */
