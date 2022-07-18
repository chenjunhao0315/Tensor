//
//  ExpandUtils.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/12.
//

#ifndef ExpandUtils_hpp
#define ExpandUtils_hpp

#include "Utils.hpp"
#include "MaybeOwned.hpp"
#include "ArrayRef.hpp"
#include "Tensor.hpp"

namespace otter {

// Use for TensorIterator -> compute_shape()
DimVector infer_size_dimvector(IntArrayRef a, IntArrayRef b);

// Use for Tensor -> view
DimVector infer_size_dimvector(IntArrayRef shape, int64_t numel);

std::vector<int64_t> infer_dense_strides(IntArrayRef tensor_sizes, IntArrayRef tensor_strides);

template <typename Container>
struct InferExpandGeometryResult {
    Container sizes;
    Container strides;
    explicit InferExpandGeometryResult(size_t ndim)
        : sizes(ndim), strides(ndim) {}
    explicit InferExpandGeometryResult(IntArrayRef sizes_, size_t ndim)
        : sizes(sizes_.begin(), sizes_.end()), strides(ndim) {}
};

std::tuple<std::vector<int64_t>, std::vector<int64_t>>
inferExpandGeometry(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides,
    IntArrayRef sizes);

InferExpandGeometryResult<DimVector>
inferExpandGeometry_dimvector(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides,
    IntArrayRef sizes);

inline void check_defined(
    std::initializer_list<std::reference_wrapper<const Tensor>> tensors,
    const char* api_name) {
  for (auto& t : tensors) {
    if (!t.get().defined()) {
      fprintf(stderr, "%s (...) called with an undefined Tensor", api_name);
    }
  }
}

inline otter::MaybeOwned<Tensor> expand_inplace(
    const Tensor& tensor,
    const Tensor& to_expand) {
  if (tensor.sizes().equals(to_expand.sizes())) {
    return otter::MaybeOwned<Tensor>::borrowed(to_expand);
  }
  return otter::MaybeOwned<Tensor>::owned(to_expand.expand(tensor.sizes()));
}

inline otter::MaybeOwned<Tensor> expand_inplace(const Tensor& tensor, Tensor&& to_expand) = delete;

inline otter::MaybeOwned<Tensor> expand_inplace(
    const Tensor& tensor,
    const Tensor& to_expand,
    const char* api_name) {
  check_defined({tensor, to_expand}, api_name);
  return expand_inplace(tensor, to_expand);
}

inline otter::MaybeOwned<Tensor> expand_inplace(
    const Tensor& tensor,
    Tensor&& to_expand,
    const char* api_name) = delete;

inline std::tuple<otter::MaybeOwned<Tensor>, otter::MaybeOwned<Tensor>>
expand_outplace(const Tensor& to_expand1, const Tensor& to_expand2) {
  if (to_expand1.sizes().equals(to_expand2.sizes())) {
    return std::make_tuple(
        otter::MaybeOwned<Tensor>::borrowed(to_expand1),
        otter::MaybeOwned<Tensor>::borrowed(to_expand2));
  }
  auto expanded_size =
      infer_size_dimvector(to_expand1.sizes(), to_expand2.sizes());
  return std::make_tuple(
      otter::MaybeOwned<Tensor>::owned(to_expand1.expand(expanded_size)),
      otter::MaybeOwned<Tensor>::owned(to_expand2.expand(expanded_size)));
}
inline std::tuple<otter::MaybeOwned<Tensor>, otter::MaybeOwned<Tensor>>
expand_outplace(Tensor&& to_expand1, const Tensor& to_expand2) = delete;
inline std::tuple<otter::MaybeOwned<Tensor>, otter::MaybeOwned<Tensor>>
expand_outplace(const Tensor& to_expand1, Tensor&& to_expand2) = delete;
inline std::tuple<otter::MaybeOwned<Tensor>, otter::MaybeOwned<Tensor>>
expand_outplace(Tensor&& to_expand1, Tensor&& to_expand2) = delete;

inline std::vector<Tensor> expand_outplace(TensorList to_expand) {
  // expands a list of Tensors; ignores undefined (null) tensors
    bool first = true;
    DimVector sizes;
    for (const auto i : otter::irange(to_expand.size())) {
        if (!to_expand[i].defined()) {
            continue;
        } else if (first) {
            sizes = to_expand[i].sizes();
            first = false;
        } else {
            sizes = infer_size_dimvector(sizes, to_expand[i].sizes());
        }
    }
    std::vector<Tensor> result(to_expand.size());
    for (const auto i : otter::irange(to_expand.size())) {
        if (!to_expand[i].defined()) {
            continue;
        } else if (to_expand[i].sizes().equals(sizes)) {
            result[i] = to_expand[i];
        } else {
            result[i] = to_expand[i].expand(sizes);
        }
    }
    return result;
}

inline MaybeOwned<Tensor> expand_size(const Tensor &to_expand, IntArrayRef sizes) {
    if (to_expand.sizes().equals(sizes)) {
        return MaybeOwned<Tensor>::borrowed(to_expand);
    }

    return MaybeOwned<Tensor>::owned(to_expand.expand(sizes));
}

inline MaybeOwned<Tensor> expand_size(Tensor &&to_expand, IntArrayRef sizes) = delete;

static inline bool is_expandable_to(IntArrayRef shape, IntArrayRef desired) {
    size_t ndim = shape.size();
    size_t target_dim = desired.size();
    if (ndim > target_dim) {
        return false;
    }
    for (const auto i : otter::irange(ndim)) {
        int64_t size = shape[ndim - i - 1];
        int64_t target = desired[target_dim - i - 1];
        if (size != target && size != 1) {
            return false;
        }
    }
    return true;
}

}   // end namespace otter

#endif /* ExpandUtils_hpp */
