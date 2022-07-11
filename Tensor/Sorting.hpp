//
//  Sorting.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/11.
//

#ifndef Sorting_hpp
#define Sorting_hpp

#include "DispatchStub.hpp"

namespace otter {

class TensorBase;

using sort_fn = void(*)(const TensorBase&, const TensorBase&, const TensorBase&, int64_t, bool, bool);
using topk_fn = void(*)(const TensorBase&, const TensorBase&, const TensorBase&, int64_t, int64_t, bool, bool);

DECLARE_DISPATCH(sort_fn, sort_stub);
DECLARE_DISPATCH(topk_fn, topk_stub);

void _fill_indices(const TensorBase &indices, int64_t dim);

}   // end namespace otter

#endif /* Sorting_hpp */
