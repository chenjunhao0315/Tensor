//
//  TensorAdvancedIndexing.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/11.
//

#ifndef TensorAdvancedIndexing_hpp
#define TensorAdvancedIndexing_hpp

#include "Tensor.hpp"
#include "DispatchStub.hpp"

namespace otter {

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
