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

}   // end namespace otter

#endif /* ReduceOps_hpp */
