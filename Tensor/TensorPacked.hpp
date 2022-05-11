//
//  TensorPacked.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/5/11.
//

#ifndef TensorPacked_hpp
#define TensorPacked_hpp

#include "Tensor.hpp"

namespace otter {

Tensor convertElempack(const Tensor& self, int64_t out_elempack, bool use_padding);

}   // end namespace otter

#endif /* TensorPacked_hpp */
