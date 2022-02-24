//
//  RangeFactory.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/9.
//

#include "Tensor.hpp"
#include "TensorIterator.hpp"
#include "RangeFactory.hpp"

namespace otter {

Tensor& linspace_out(const Scalar& start, const Scalar& end, int64_t steps, Tensor& result) {
    assert(steps > 0);
    
    if (result.numel() != steps) {
        result.resize_({steps});
    }
    
    if (steps == 0) {
        // skip
    } else if (steps == 1) {
        result.fill_(start);
    } else {
        Tensor r = result;
        auto iter = TensorIterator::borrowing_nullary_op(r);
        linspace_stub(Device::CPU, iter, start, end, steps);
      }
    
    return result;
}

DEFINE_DISPATCH(linspace_stub);

}
