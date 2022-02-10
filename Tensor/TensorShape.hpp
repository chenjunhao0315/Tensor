//
//  TensorShape.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/10.
//

#ifndef TensorShape_hpp
#define TensorShape_hpp

#include "Tensor.hpp"
#include "WarpDimUtils.hpp"
#include "Utils.hpp"
#include "TensorResize.hpp"

namespace otter {

Tensor select(const Tensor& self, int64_t dim, int64_t index);

Tensor permute(const Tensor& self, IntArrayRef dims);

namespace native {

Tensor as_strided_tensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride);
Tensor as_strided_tensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride, int64_t memory_offset_);
const Tensor &as_strided_(const Tensor& self, IntArrayRef size, IntArrayRef stride);
const Tensor &as_strided_(const Tensor& self, IntArrayRef size, IntArrayRef stride, int64_t memory_offset_);

}

}

#endif /* TensorShape_hpp */
