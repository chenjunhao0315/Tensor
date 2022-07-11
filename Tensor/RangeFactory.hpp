//
//  RangeFactory.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/9.
//

#ifndef RangeFactory_hpp
#define RangeFactory_hpp

#include "DispatchStub.hpp"

namespace otter {

class Tensor;
class Scalar;
class TensorIterator;

Tensor& linspace_out(const Scalar& start, const Scalar& end, int64_t steps, Tensor& result);

Tensor& arange_out(const Scalar& start, const Scalar& end, const Scalar& step, Tensor& result);

Tensor& range_out(const Scalar& start, const Scalar& end, const Scalar& step, Tensor& result);

DECLARE_DISPATCH(void(*)(TensorIterator&, const Scalar&, const Scalar&, const Scalar&), arange_stub);
DECLARE_DISPATCH(void(*)(TensorIterator&, const Scalar&, const Scalar&, int64_t), linspace_stub);

}

#endif /* RangeFactory_hpp */
