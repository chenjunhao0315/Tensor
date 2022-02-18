//
//  TensorCatKernel.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#ifndef TensorCatKernel_hpp
#define TensorCatKernel_hpp

#include "TensorCat.hpp"
#include "Tensor.hpp"

namespace otter {

void cat_serial_kernel(Tensor& result, TensorList tensors, int64_t dim);

}

#endif /* TensorCatKernel_hpp */
