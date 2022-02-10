//
//  RangeFactory.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/9.
//

#ifndef RangeFactory_hpp
#define RangeFactory_hpp

#include "DispatchStub.hpp"
#include "TensorIterator.hpp"
#include "Scalar.hpp"

namespace otter {

Tensor& linspace_out(const Scalar& start, const Scalar& end, int64_t steps, Tensor& result);

DECLARE_DISPATCH(void(*)(TensorIterator&, const Scalar&, const Scalar&, int64_t), linspace_stub);

}

#endif /* RangeFactory_hpp */
