//
//  TensorCat.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#ifndef TensorCat_hpp
#define TensorCat_hpp

#include "DispatchStub.hpp"
#include "Tensor.hpp"

namespace otter {

using cat_serial_fn = void(*)(Tensor &, TensorList, int64_t);
DECLARE_DISPATCH(cat_serial_fn, cat_serial_stub);

}

#endif /* TensorCat_hpp */
