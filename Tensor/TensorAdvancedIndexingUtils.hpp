//
//  TensorAdvancedIndexingUtils.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/17.
//

#ifndef TensorAdvancedIndexingUtils_h
#define TensorAdvancedIndexingUtils_h

#include "Tensor.hpp"
#include "ExpandUtils.hpp"

namespace otter {

inline int64_t size_from_dim_(int k, IntArrayRef dims) {
    int64_t r = 1;
    for (const auto i : otter::irange(k, dims.size())) {
        r *= dims[i];
    }
    return r;
}
// Product of all dims up to k (not including dims[k])
inline int64_t size_to_dim_(int k, IntArrayRef dims) {
    OTTER_CHECK((unsigned)k <= dims.size(), "");
    int64_t r = 1;
    for (const auto i : otter::irange(k)) {
        r *= dims[i];
    }
    return r;
}
// Product of all dims between k and l (not including dims[k] and dims[l])
inline int64_t size_between_dim_(int k, int l, IntArrayRef dims) {
    OTTER_CHECK((unsigned)l < dims.size() && (unsigned)k < dims.size(), "");
    int64_t r = 1;
    if (k < l) {
        for (int i = k + 1; i < l; ++i) {
            r *= dims[i];
        }
    } else {
        for (int i = l + 1; i < k; ++i) {
            r *= dims[i];
        }
    }
    return r;
}

struct AdvancedIndex {
  AdvancedIndex(const Tensor& src, TensorList indices);
  Tensor src;
  std::vector<Tensor> indices;
  DimVector indexed_sizes;
  DimVector indexed_strides;
  int64_t dims_before;
  int64_t dims_after;
};

namespace {
static std::string shapes_as_str(TensorList tensors) {
    std::ostringstream os;
    bool first = true;
    for (auto& tensor : tensors) {
        if (tensor.defined()) {
            if (!first) {
                os << ", ";
            }
            os << tensor.sizes();
            first = false;
        }
    }
    return os.str();
}
} // anonymous namespace
static std::tuple<bool, Tensor> canDispatchToMaskedFill(const Tensor& self, const std::vector<otter::optional<Tensor>>& indices, const Tensor& value){
    if (!(value.numel() == 1 && value.device() == Device::CPU)){
        return std::make_tuple(false,Tensor());
    }
    int64_t num_ind = 0;
    Tensor mask;
    auto self_device = self.device();
    for (const otter::optional<Tensor> i: indices) {
    if (!i.has_value() || !(*i).defined()){
        num_ind++;
    } else {
        Tensor index = std::move(*i);
        if ((index.scalar_type() != otter::ScalarType::Byte && index.scalar_type() !=   otter::ScalarType::Bool) || index.device() != self_device || mask.defined()){
            return std::make_tuple(false, Tensor());
        } else {
            mask = index;
            for (const auto j : otter::irange(index.dim())) {
                int64_t srcIdx = num_ind + j;
                OTTER_CHECK(index.size(j) == self.size(srcIdx), "The shape of the mask ", index.sizes(), " at   index ", j, " does not match the shape of the indexed tensor ", self.sizes(), " at index ", srcIdx);
            }
            num_ind += mask.dim();
            }
        }
    }
    for (const auto i : otter::irange(num_ind, self.dim())) {
        (void)i; //Suppress unused variable warning
        mask = mask.unsqueeze(-1);
    }
    return std::make_tuple(true, mask);
}

static OTTER_UNUSED void checkIndexTensorTypes(std::vector<otter::optional<Tensor>>& indices) {
    for (const auto& tensor : indices) {
        if (tensor.has_value() && tensor->defined()) {
            auto scalarType = tensor->scalar_type();
            if (scalarType != otter::ScalarType::Long && scalarType != otter::ScalarType::Byte && scalarType != otter::ScalarType::Bool) {
                OTTER_CHECK(false, "tensors used as indices must be long, byte or bool tensors");
            }
        }
    }
}

[[noreturn]]
static void invalid_mask(const Tensor & self, int64_t idx, const Tensor & mask, int64_t maskIdx) {
    OTTER_CHECK(false, "The shape of the mask ", mask.sizes(), " at index ", maskIdx,
                " does not match the shape of the indexed tensor ", self.sizes(), " at index ", idx);
}

static OTTER_UNUSED std::vector<Tensor> expandTensors(const Tensor & self, std::vector<otter::optional<Tensor>>& indices) {
  // If indices come in as ByteTensor or BoolTensor (masks), expand them into the equivalent indexing by LongTensors
  std::vector<Tensor> result;
  for (const auto& index_opt : indices) {
      if (!index_opt.has_value()) {
          result.emplace_back();
      } else {
          const auto& index = *index_opt;
          if (index.scalar_type() == otter::ScalarType::Byte || index.scalar_type() == otter::ScalarType::Bool) {
              if (index.scalar_type() == otter::ScalarType::Byte) {
                  fprintf(stderr, "indexing with dtype torch.uint8 is now deprecated," \
                             " please use a dtype torch.bool instead.");
              }
              // The sizes of the ByteTensor mask or bool tensor must match the sizes of the
              // corresponding dimensions in self
              for (const auto j : otter::irange(index.dim())) {
                  int64_t srcIdx = result.size() + j;
                  if (index.size(j) != self.size(srcIdx)) {
                      invalid_mask(self, srcIdx, index, j);
                  }
              }
              // Replace with nonzeros
              auto nonzero = index.nonzero();
              for (const auto j : otter::irange(index.dim())) {
                  result.emplace_back(nonzero.select(1, j));
              }
          } else {
              result.emplace_back(std::move(index));
          }
      }
  }
  return result;
}

static OTTER_UNUSED bool hasContiguousSubspace(TensorList tl) {
    // true if all the non-null tensors are adjacent
    auto isDefined = [](const Tensor & tensor){ return tensor.defined(); };
    auto isNull = [](const Tensor & tensor){ return !tensor.defined(); };
    auto start = std::find_if(tl.begin(), tl.end(), isDefined);
    auto stop = std::find_if(tl.rbegin(), tl.rend(), isDefined);
    auto it = std::find_if(start, stop.base(), isNull);
    return it == stop.base();
}

static OTTER_UNUSED std::tuple<Tensor, std::vector<Tensor>>
transposeToFront(Tensor self, TensorList indices) {
    std::vector<int64_t> dims;
    std::vector<Tensor> transposedIndices;
    dims.reserve(self.dim());
    for (const auto i : otter::irange(self.dim())) {
        if (indices[i].defined()) {
            dims.push_back(i);
            transposedIndices.emplace_back(indices[i]);
        }
    }
    for (const auto i : otter::irange(self.dim())) {
        if (!indices[i].defined()) {
            dims.push_back(i);
            transposedIndices.emplace_back();
        }
    }
    return std::make_tuple(self.permute(dims), std::move(transposedIndices));
}

inline std::tuple<Tensor, std::vector<Tensor>, std::vector<int64_t>>
transposeToFrontAndInvPerm(Tensor self, TensorList indices) {
    std::vector<int64_t> dims;
    std::vector<int64_t> invPerm;
    std::vector<Tensor> transposedIndices;
    dims.reserve(self.dim());
    invPerm.resize(self.dim());
    for (const auto i : otter::irange(self.dim())) {
        if (indices[i].defined()) {
            dims.push_back(i);
            transposedIndices.emplace_back(indices[i]);
        }
    }
    for (const auto i : otter::irange(self.dim())) {
        if (!indices[i].defined()) {
            dims.push_back(i);
            transposedIndices.emplace_back();
        }
    }
    for (const auto i : otter::irange(self.dim())) {
        invPerm[dims[i]] = i;
    }
    return std::make_tuple(self.permute(dims), std::move(transposedIndices), std::move(invPerm));
}

static AdvancedIndex make_info(Tensor self, std::vector<otter::optional<Tensor>>& orig) {
  checkIndexTensorTypes(orig);
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more LongTensors
  auto indices = expandTensors(self, orig);
  // next broadcast all index tensors together
  try {
    indices = expand_outplace(indices);
  } catch (std::exception& e) {
    OTTER_CHECK(false, "shape mismatch: indexing tensors could not be broadcast together"
                   " with shapes ", shapes_as_str(indices));
  }
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < (size_t)self.dim()) {
    indices.emplace_back();
  }
  // if the non-null indices are not all adjacent, transpose self and indices
  // together so that they're adjacent at the front
  if (!hasContiguousSubspace(indices)) {
    std::tie(self, indices) = transposeToFront(self, indices);
  }
  // Ensure indices are on the same device as self
//  for (auto & indice : indices) {
//    if (indice.defined() && indice.device() != self.device()) {
//      indice = indice.to(self.device());
//    }
//  }
  return AdvancedIndex(self, indices);
}

}   // end namespace otter

#endif /* TensorAdvancedIndexingUtils_h */
