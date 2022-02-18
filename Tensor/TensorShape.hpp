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

namespace native {

Tensor select(const Tensor& self, int64_t dim, int64_t index);

Tensor permute(const Tensor& self, IntArrayRef dims);

Tensor& transpose_(Tensor& self, int64_t dim0, int64_t dim1);
Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1);

Tensor expand(const Tensor& self, IntArrayRef sizes);
Tensor expand_as(const Tensor& self, const Tensor& other);

Tensor view(const Tensor& self, IntArrayRef sizes);

Tensor reshape(const Tensor& self, IntArrayRef sizes);
Tensor reshape_as(const Tensor& self, const Tensor& other);

Tensor detach(const Tensor& self);

Tensor as_strided_tensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride);
Tensor as_strided_tensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride, int64_t memory_offset_);
const Tensor &as_strided_(const Tensor& self, IntArrayRef size, IntArrayRef stride);
const Tensor &as_strided_(const Tensor& self, IntArrayRef size, IntArrayRef stride, int64_t memory_offset_);

Tensor slice(const Tensor& self, int64_t dim, int64_t start = INT64_MAX, int64_t end = 0, int64_t step = 1);

Tensor unsqueeze(const Tensor& self, int64_t dim);
Tensor& unsqueeze_(Tensor& self, int64_t dim);

Tensor squeeze(const Tensor& self, int64_t dim);
Tensor& squeeze_(Tensor& self, int64_t dim);

Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length);

Tensor& cat_out(TensorList tensors, int64_t dim, Tensor& out);
Tensor cat(TensorList tensors, int64_t dim);

}   // namespace native

}   // namespace otter

#endif /* TensorShape_hpp */
