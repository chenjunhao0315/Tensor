//
//  ReduceOps.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/18.
//

#ifndef ReduceOps_hpp
#define ReduceOps_hpp

#include "DispatchStub.hpp"

namespace otter {

class TensorIterator;

using reduce_fn = void(*)(TensorIterator &);

DECLARE_DISPATCH(reduce_fn, sum_stub);
DECLARE_DISPATCH(reduce_fn, prod_stub);
DECLARE_DISPATCH(reduce_fn, mean_stub);
DECLARE_DISPATCH(reduce_fn, min_values_stub);
DECLARE_DISPATCH(reduce_fn, max_values_stub);
DECLARE_DISPATCH(reduce_fn, argmax_stub);
DECLARE_DISPATCH(reduce_fn, argmin_stub);

}   // end namespace otter

#endif /* ReduceOps_hpp */
