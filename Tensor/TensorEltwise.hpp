//
//  TensorEltwise.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/20.
//

#ifndef TensorEltwise_hpp
#define TensorEltwise_hpp

#include "Tensor.hpp"

namespace otter {

Tensor eltwise_add_pack4(const Tensor& src1, const Tensor& src2);

Tensor eltwise_add_pack8(const Tensor& src1, const Tensor& src2);

}   // end namespace otter

#endif /* TensorEltwise_hpp */
