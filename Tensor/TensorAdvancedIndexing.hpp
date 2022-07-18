//
//  TensorAdvancedIndexing.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/11.
//

#ifndef TensorAdvancedIndexing_hpp
#define TensorAdvancedIndexing_hpp

#include "Tensor.hpp"
#include "TensorIterator.hpp"
#include "DispatchStub.hpp"

namespace otter {

Tensor nonzero_cpu(const Tensor& self);
Tensor& nonzero_out_cpu(const Tensor& self, Tensor& result);

Tensor & put_(Tensor & self, const Tensor& index, const Tensor & source, const bool accumulate);
Tensor put(const Tensor & self, const Tensor& index, const Tensor & source, const bool accumulate);

Tensor index_put(const Tensor & self, const std::vector<otter::optional<Tensor>>& indices, const Tensor & value, bool accumulate);

Tensor& take_out(const Tensor& self, const Tensor& index, Tensor& out);
Tensor take(const Tensor& self, const Tensor& index);

Tensor & index_put_(Tensor & self, const std::vector<otter::optional<Tensor>>& indices, const Tensor & value, const bool accumulate);

Tensor & index_select_out_cpu_(const Tensor & self, int64_t dim, const Tensor & index, Tensor & result);
Tensor index_select_cpu_(const Tensor & self, int64_t dim, const Tensor & index);

Tensor & index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Scalar& source);
Tensor & index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source);
Tensor index_fill(const Tensor & self, int64_t dim, const Tensor & index, const Scalar& source);
Tensor index_fill(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source);

Tensor masked_select_cpu(const Tensor & self, const Tensor & mask);

Tensor & masked_fill__cpu(Tensor& self, const Tensor & mask, const Scalar& value);
Tensor & masked_fill__cpu(Tensor& self, const Tensor & mask, const Tensor & value);
Tensor masked_fill(const Tensor & self, const Tensor & mask, const Scalar& source);
Tensor masked_fill(const Tensor & self, const Tensor & mask, const Tensor & source);

enum class SCATTER_GATHER_OP: uint8_t {REDUCE_ADD, REDUCE_MULTIPLY, REDUCE_MAXIMUM, REDUCE_MINIMUM, REDUCE_MEAN};
//using index_put_with_sort_fn = void(*)(Tensor &, const otter::List<otter::optional<Tensor>> &, const Tensor &, bool accumulate, bool unsafe);
using gather_fn = void (*)(const Tensor & result, const Tensor & self, int64_t dim, const Tensor & index);
using scatter_fn = void(*)(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);
using scatter_fill_fn = void(*)(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& src);
//using scatter_add_fn = void(*)(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);
using scatter_reduce_fn = void(*)(const Tensor& self, const int64_t dim, const Tensor& index,
                                  const Tensor& src, const SCATTER_GATHER_OP& reduce);
using scatter_scalar_reduce_fn = void(*)(const Tensor& self, const int64_t dim, const Tensor& index,
                                         const Scalar& value, const SCATTER_GATHER_OP& reduce);
//using scatter_reduce_two_fn = void(*)(const Tensor& self, const int64_t dim, const Tensor& index,
//                                      const Tensor& src, const SCATTER_GATHER_OP& reduce);
//DECLARE_DISPATCH(index_put_with_sort_fn, index_put_with_sort_stub);
DECLARE_DISPATCH(gather_fn, gather_stub);
DECLARE_DISPATCH(scatter_fn, scatter_stub);
DECLARE_DISPATCH(scatter_fill_fn, scatter_fill_stub);
//DECLARE_DISPATCH(scatter_add_fn, scatter_add_stub);
DECLARE_DISPATCH(scatter_reduce_fn, scatter_reduce_stub);
DECLARE_DISPATCH(scatter_scalar_reduce_fn, scatter_scalar_reduce_stub);
//DECLARE_DISPATCH(scatter_reduce_two_fn, scatter_reduce_two_stub);
//TORCH_API Tensor& index_out(Tensor& result, const Tensor & self, const otter::List<otter::optional<Tensor>>& indices);

using index_fn = void(*)(TensorIterator &, IntArrayRef indexed_sizes, IntArrayRef indexed_strides);
using index_fill_fn = void(*)(TensorIterator & iter, int64_t dim, int64_t self_dim_size, int64_t self_dim_stride, const Scalar& source);
using index_copy_fn = void(*)(TensorIterator & iter, int64_t dim, int64_t self_dim_size, int64_t self_dim_stride);
using index_put_fn = void(*)(TensorIterator &, IntArrayRef indexed_sizes, IntArrayRef indexed_strides, bool accumulate);
using put_fn = void(*)(TensorIterator & iter, const TensorBase& self, const bool accumulate);
using take_fn = void(*)(TensorIterator & iter, const TensorBase& input);
using flip_fn = void(*)(TensorIterator &, const bool);
using masked_fill_fn = void(*)(TensorIterator &, const Scalar& scalar);
using masked_select_fn = void(*)(TensorIterator &, int64_t orig_stride);
using masked_scatter_fn = void(*)(TensorIterator &, const TensorBase &);
DECLARE_DISPATCH(index_fn, index_stub);
DECLARE_DISPATCH(index_fill_fn, index_fill_stub);
DECLARE_DISPATCH(index_copy_fn, index_copy_stub);
DECLARE_DISPATCH(index_put_fn, index_put_stub);
DECLARE_DISPATCH(put_fn, put_stub);
DECLARE_DISPATCH(take_fn, take_stub);
DECLARE_DISPATCH(flip_fn, flip_stub);
DECLARE_DISPATCH(masked_fill_fn, masked_fill_stub);
DECLARE_DISPATCH(masked_select_fn, masked_select_serial_stub);
DECLARE_DISPATCH(masked_select_fn, masked_select_stub);
DECLARE_DISPATCH(masked_scatter_fn, masked_scatter_stub);

inline int64_t ensure_nonempty_dim(int64_t dim) {
  return std::max<int64_t>(dim, 1);
}
inline int64_t ensure_nonempty_size(const TensorBase &t, int64_t dim) {
  return t.dim() == 0 ? 1 : t.size(dim);
}
inline int64_t ensure_nonempty_stride(const TensorBase &t, int64_t dim) {
  return t.dim() == 0 ? 1 : t.stride(dim);
}
using IdxVec = std::vector<int64_t>;
inline IdxVec ensure_nonempty_vec(IdxVec vec) {
  if (vec.size() == 0) {
    vec.push_back(1);
  }
  return vec;
}

}   // end namespace otter

#endif /* TensorAdvancedIndexing_hpp */
