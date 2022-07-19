//
//  TensorAdvancedIndexing.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/11.
//

#include "TensorAdvancedIndexing.hpp"
#include "TensorFunction.hpp"
#include "MemoryOverlap.hpp"
#include "Dispatch.hpp"
#include "TensorFactory.hpp"
#include "TensorAdvancedIndexingUtils.hpp"
#include "TensorFactory.hpp"
#include "ExpandUtils.hpp"
#include "TensorResize.hpp"
#include "Parallel.hpp"
#include "TensorCopy.hpp"
#include "WarpDimUtils.hpp"


namespace otter {

// checks whether index.dtype == int64
// and self.dtype == src.dtype if src is a Tensor
static void scatter_gather_dtype_check(
                                       const std::string& method_name,
                                       const Tensor& self,
                                       const Tensor& index,
                                       const Tensor& src_opt = Tensor()
                                       ) {
    if (index.numel() != 0) {
        OTTER_CHECK(
                    index.scalar_type() == otter::ScalarType::Long,
                    method_name, "(): Expected dtype int64 for index"
                    );
    }
    if (src_opt.defined()) {
        auto src = src_opt;
        OTTER_CHECK(
                    self.scalar_type() == src.scalar_type(),
                    method_name, "(): Expected self.dtype to be equal to src.dtype"
                    );
    }
}
// Used for `gather`-like methods
// Note: self means the input tensor here
// Test:
// 1. index.size(d) <= self.size(d) for all d != dim
// 2. index.dim() == self.dim()
static OTTER_UNUSED void gather_shape_check(const Tensor& self, int64_t dim,
                                            const Tensor& index
                                            ) {
    auto self_dims = ensure_nonempty_dim(self.dim());
    OTTER_CHECK(self_dims == ensure_nonempty_dim(index.dim()),
                "Index tensor must have the same number of dimensions as input tensor"
                );
    for (const auto i : otter::irange(self_dims)) {
        if (i != dim) {
            OTTER_CHECK(
                        ensure_nonempty_size(index, i) <= ensure_nonempty_size(self, i),
                        "Size does not match at dimension ", i,
                        " expected index ", index.sizes(),
                        " to be smaller than self ", self.sizes(),
                        " apart from dimension ", dim
                        );
        }
    }
}
// Used for `scatter` and `scatter_add`
// Tests:
//  1. index.size(d) <= self.size(d) for all d != dim
//  2. index.size(d) <= src.size(d) for all d if src is a Tensor
//  3. index.dim() == self.dim() == src.dim()
static OTTER_UNUSED void scatter_shape_check(
                                             const Tensor& self, int64_t dim, const Tensor& index,
                                             const Tensor& src_opt = Tensor()
                                             ) {
    if (index.numel() == 0) return;
    OTTER_CHECK(
                ensure_nonempty_dim(self.dim()) == ensure_nonempty_dim(index.dim()),
                "Index tensor must have the same number of dimensions as self tensor"
                );
    bool is_wrong_shape = false;
    int64_t self_dims = ensure_nonempty_dim(self.dim());
    //  Check: index.size(d) <= self.size(d) for all d != dim
    for (const auto d : otter::irange(self_dims)) {
        int64_t index_d_size = ensure_nonempty_size(index, d);
        if (d == dim) continue;
        if (index_d_size > ensure_nonempty_size(self, d)) {
            is_wrong_shape = true;
            break;
        }
    }
    //  Check: index.size(d) <= src.size(d) for all d if src is Tensor
    if (!is_wrong_shape && src_opt.defined()) {
        auto src = src_opt;
        for (const auto d : otter::irange(self_dims)) {
            int64_t index_d_size = ensure_nonempty_size(index, d);
            if (index_d_size > ensure_nonempty_size(src, d)) {
                is_wrong_shape = true;
                break;
            }
        }
    }
    if (src_opt.defined()) {
        auto src = src_opt;
        OTTER_CHECK(
                    ensure_nonempty_dim(src.dim()) == ensure_nonempty_dim(index.dim()),
                    "Index tensor must have the same number of dimensions as src tensor"
                    );
        OTTER_CHECK(!is_wrong_shape,
                    "Expected index ", index.sizes(),
                    " to be smaller than self ", self.sizes(),
                    " apart from dimension ", dim,
                    " and to be smaller size than src ", src.sizes()
                    );
    }
    else {
        OTTER_CHECK(!is_wrong_shape,
                    "Expected index ", index.sizes(),
                    " to be smaller than self ", self.sizes(),
                    " apart from dimension ", dim
                    );
    }
}

SCATTER_GATHER_OP get_operator_enum(const int64_t reduce, bool use_new_options = false) {
    if (use_new_options) {
        if (reduce == 0) {
            return SCATTER_GATHER_OP::REDUCE_ADD;
        } else if (reduce == 1) {
            return SCATTER_GATHER_OP::REDUCE_MULTIPLY;
        } else if (reduce == 2) {
            return SCATTER_GATHER_OP::REDUCE_MEAN;
        } else if (reduce == 3) {
            return SCATTER_GATHER_OP::REDUCE_MAXIMUM;
        } else if (reduce == 4) {
            return SCATTER_GATHER_OP::REDUCE_MINIMUM;
        } else {
            OTTER_CHECK(false, "reduce argument must be either sum, prod, mean, amax or amin.");
        }
    } else {
        if (reduce == 1) {
            return SCATTER_GATHER_OP::REDUCE_ADD;
        } else if (reduce == 2) {
            return SCATTER_GATHER_OP::REDUCE_MULTIPLY;
        } else {
            OTTER_CHECK(false, "reduce argument must be either add or multiply.")
        }
    }
    OTTER_CHECK(false, "reduce argument must be either sum, prod, mean, amax or amin.");
    return SCATTER_GATHER_OP::REDUCE_ADD;
}

DEFINE_META_FUNCTION(gather)(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
    const Tensor& result = maybe_get_output(0);
    int64_t wrapped_dim = otter::maybe_wrap_dim(dim, self.dim());
    // Memory overlap checks need to be done after resizing (if required) is done.
    // But it only makes sense to do these checks when result was defined, hence
    // the boolean variable `check_result` here.
    // For more details, see: https://github.com/pytorch/pytorch/pull/63312#discussion_r694794832
    // and https://github.com/pytorch/pytorch/issues/63837
    bool check_result = result.defined();
    set_output(0, index.sizes(), {}, self.options());
    if (check_result) {
        otter::assert_no_internal_overlap(result);
        otter::assert_no_overlap(result, self);
        otter::assert_no_partial_overlap(result, index);
    }
    auto is_index_empty = index.numel() == 0;
    if (!is_index_empty) {
        OTTER_CHECK(
                    index.scalar_type() == otter::ScalarType::Long,
                    "gather", "(): Expected dtype int64 for index"
                    );
    }
    if (is_index_empty) return;
    otter::gather_shape_check(self, wrapped_dim, index);
}

template <bool use_new_options = false, typename Meta>
void scatter_meta_impl(Meta& meta,
                       const Tensor& self,
                       int64_t dim,
                       const Tensor& index,
                       const Tensor& src = Tensor(),
                       const int64_t reduce = -1) {
    int64_t wrapped_dim = otter::maybe_wrap_dim(dim, self.dim());
    otter::scatter_gather_dtype_check("scatter", self, index, src);
    otter::scatter_shape_check(self, wrapped_dim, index, src);
    auto output = meta.maybe_get_output(0);
    if (output.defined()) {
        otter::assert_no_internal_overlap(output);
        otter::assert_no_overlap(output, index);
        if (src.defined()) {
            otter::assert_no_overlap(output, src);
        }
    }
    meta.set_output(0, self.sizes(), {}, self.options());
    if (reduce != -1) {
        // Check if we have a valid reduce operator.
        get_operator_enum(reduce, use_new_options);
    }
}
DEFINE_META_FUNCTION_OVERLOAD(scatter, src)
(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    scatter_meta_impl(*this, self, dim, index, src);
}
DEFINE_META_FUNCTION_OVERLOAD(scatter, value)
(const Tensor& self, int64_t dim, const Tensor& index, const Scalar& value) {
    scatter_meta_impl(*this, self, dim, index);
}
DEFINE_META_FUNCTION_OVERLOAD(scatter, reduce)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& src,
 const int64_t reduce) {
    scatter_meta_impl(*this, self, dim, index, src, reduce);
}
DEFINE_META_FUNCTION_OVERLOAD(scatter, value_reduce)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Scalar& src,
 const int64_t reduce) {
    scatter_meta_impl(*this, self, dim, index, Tensor(), reduce);
}

static void build_index_op(
    TensorIterator& iter,
    const otter::AdvancedIndex& info,
    const Tensor& result) {
    // 'TensorIterator' needs to own the things comming from 'info', since
    // 'info' will be destroyed after the META function.
    TensorIteratorConfig config;
    // info.src is a restrided view of result
    config.set_check_mem_overlap(false)
        .check_all_same_dtype(false)
        .add_output(result)
        .add_owned_input(info.src);
    for (auto& index : info.indices) {
        config.add_owned_input(index);
    }
    if (!result.defined()) {
        config.declare_static_dtype_and_device(info.src.scalar_type(), info.src.device());
    }
    iter.build(config);
}

structured_index_Tensor::meta_return_ty structured_index_Tensor::meta(const Tensor& self, std::vector<otter::optional<otter::Tensor>> indices) {
    const auto& result = maybe_get_output();
    if (result.defined()) {
        otter::assert_no_internal_overlap(result);
        otter::assert_no_overlap(result, self);
        for (const otter::optional<otter::Tensor>& index : indices) {
            if (index.has_value()) {
                otter::assert_no_overlap(result, *index);
            }
        }
    }
    auto info = make_info(self, indices);
    build_index_op(*this, info, result);
    return structured_index_Tensor::precompute_out<>()
        .set_sizes(std::move(info.indexed_sizes))
        .set_strides(std::move(info.indexed_strides));
}

//DEFINE_META_FUNCTION(scatter_add)
//(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
//    scatter_meta_impl(*this, self, dim, index, src, 0);
//}
//DEFINE_META_FUNCTION_OVERLOAD(scatter_reduce, two)
//(const Tensor& self,
// int64_t dim,
// const Tensor& index,
// const Tensor& src,
// const int64_t reduce,
// bool include_self) {
//    (void) include_self;
//    scatter_meta_impl</*use_new_options=*/true>(*this, self, dim, index, src, reduce);
//}

DEFINE_DISPATCH(index_stub);
DEFINE_DISPATCH(index_fill_stub);
DEFINE_DISPATCH(index_copy_stub);
DEFINE_DISPATCH(index_put_stub);
//DEFINE_DISPATCH(index_put_with_sort_stub);
DEFINE_DISPATCH(put_stub);
DEFINE_DISPATCH(take_stub);
DEFINE_DISPATCH(masked_fill_stub);
//REGISTER_NO_CPU_DISPATCH(index_put_with_sort_stub);
DEFINE_DISPATCH(masked_select_serial_stub);
DEFINE_DISPATCH(masked_select_stub);
DEFINE_DISPATCH(masked_scatter_stub);

DEFINE_DISPATCH(gather_stub);
DEFINE_DISPATCH(scatter_stub);
DEFINE_DISPATCH(scatter_fill_stub);
//DEFINE_DISPATCH(scatter_add_stub);
DEFINE_DISPATCH(scatter_reduce_stub);
DEFINE_DISPATCH(scatter_scalar_reduce_stub);
//DEFINE_DISPATCH(scatter_reduce_two_stub);

DEFINE_IMPL_FUNCTION(gather_out)
(const Tensor& self, int64_t dim, const Tensor& index, bool sparse_grad, const Tensor& result) {
    if (index.numel() == 0) return;
    dim = otter::maybe_wrap_dim(dim, self.dim());
    gather_stub(Device::CPU, result, self, dim, index);
}

static void scatter_reduce_exclude_self_helper(
                                               const Tensor& self,
                                               int64_t dim,
                                               const Tensor& index,
                                               const SCATTER_GATHER_OP& op) {
    OTTER_DISPATCH_ALL_TYPES_AND(otter::ScalarType::Bool, self.scalar_type(), "scatter_reduce_exclude_input_init", [&] {
        scalar_t init_val;
        switch (op) {
            case SCATTER_GATHER_OP::REDUCE_ADD:
                init_val = (scalar_t)0;
                break;
            case SCATTER_GATHER_OP::REDUCE_MULTIPLY:
                init_val = (scalar_t)1;
                break;
            case SCATTER_GATHER_OP::REDUCE_MAXIMUM:
                init_val =std::numeric_limits<scalar_t>::has_infinity ?-std::numeric_limits<scalar_t>::infinity()
                : std::numeric_limits<scalar_t>::lowest();
                break;
            case SCATTER_GATHER_OP::REDUCE_MINIMUM:
                init_val =std::numeric_limits<scalar_t>::has_infinity ?std::numeric_limits<scalar_t>::infinity()
                : std::numeric_limits<scalar_t>::max();
                break;
            case SCATTER_GATHER_OP::REDUCE_MEAN:
                init_val = (scalar_t)0;
                break;
        }
        self.scatter_(dim, index, init_val);
    });
}

template <bool use_new_options = false, typename T, typename ReduceStub, typename FillStub>
void scatter_impl(
                  const Tensor& self,
                  int64_t dim,
                  const Tensor& index,
                  const T& src,
                  const Tensor& out,
                  ReduceStub& reduce_stub,
                  FillStub& fill_stub,
                  const int64_t reduce = -1,
                  bool reduce_includes_self = true) {
    dim = otter::maybe_wrap_dim(dim, self.dim());
    auto mut_out = const_cast<Tensor&>(out);
    if (!self.is_same(mut_out)) {
        mut_out.copy_(self);
    }
    if (index.numel() == 0) return;
    if (reduce != -1) {
        auto op = get_operator_enum(reduce, use_new_options);
        if (!reduce_includes_self) {
            // scatter inits for reduction to appropriate indices (used by scatter_reduce.two)
            scatter_reduce_exclude_self_helper(mut_out, dim, index, op);
        }
        reduce_stub(Device::CPU, mut_out, dim, index, src, op);
    } else {
        fill_stub(Device::CPU, mut_out, dim, index, src);
    }
}
DEFINE_IMPL_FUNCTION(scatter_src_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& src,
 const Tensor& out) {
    scatter_impl(self, dim, index, src, out,
                 scatter_reduce_stub,
                 scatter_stub);
}
DEFINE_IMPL_FUNCTION(scatter_value_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Scalar& value,
 const Tensor& out) {
    scatter_impl(self, dim, index, value, out,
                 scatter_scalar_reduce_stub,
                 scatter_fill_stub);
}
DEFINE_IMPL_FUNCTION(scatter_reduce_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& src,
 const int64_t reduce,
 const Tensor& out) {
    scatter_impl(self, dim, index, src, out,
                 scatter_reduce_stub,
                 scatter_stub,
                 reduce);
}
DEFINE_IMPL_FUNCTION(scatter_value_reduce_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Scalar& value,
 const int64_t reduce,
 const Tensor& out) {
    scatter_impl(self, dim, index, value, out,
                 scatter_scalar_reduce_stub,
                 scatter_fill_stub,
                 reduce);
}

//DEFINE_IMPL_FUNCTION(scatter_add_out)
//(const Tensor& self,
// int64_t dim,
// const Tensor& index,
// const Tensor& src,
// const Tensor& out) {
//    auto mut_out = const_cast<Tensor&>(out);
//    dim = maybe_wrap_dim(dim, self.dim());
//    if (!self.is_same(mut_out)) {
//        mut_out.copy_(self);
//    }
//    if (index.numel() == 0) return;
//    
//    scatter_add_stub(Device::CPU, mut_out, dim, index, src);
//}
//DEFINE_IMPL_FUNCTION(scatter_reduce_two_out)
//(const Tensor& self,
// int64_t dim,
// const Tensor& index,
// const Tensor& src,
// const int64_t reduce,
// bool include_self,
// const Tensor& out) {
//    // See issue https://github.com/pytorch/pytorch/issues/74770
//    scatter_impl</*use_new_options=*/true>(self, dim, index, src, out,
//                                           scatter_reduce_two_stub,
//                                           scatter_stub,
//                                           reduce,
//                                           include_self);
//    if (get_operator_enum(reduce, true) == SCATTER_GATHER_OP::REDUCE_MEAN) {
//        auto ones = otter::ones_like(src);
//        auto count = include_self ? otter::ones_like(out) : otter::zeros_like(out);
//        count.scatter_add_(dim, index, ones);
//        count.masked_fill_(count == 0, 1);
//        if (out.is_floating_point()) {
//            out.div_(count);
//        } else {
//            out.div_(count, "floor");
//        }
//    }
//}

// Replace indexed dimensions in src with stride 0 and the size of the result tensor.
// The offset in these dimensions is computed by the kernel using the index tensor's
// values and the stride of src. The new shape is not meaningful. It's used to make
// the shape compatible with the result tensor.
static Tensor restride_src(const Tensor& src, int64_t dims_before, int64_t dims_indexed, IntArrayRef replacement_shape) {
    auto shape = DimVector(src.sizes());
    auto strides = DimVector(src.strides());
    int64_t end = dims_before + dims_indexed;
    shape.erase(shape.begin() + dims_before, shape.begin() + end);
    strides.erase(strides.begin() + dims_before, strides.begin() + end);
    shape.insert(shape.begin() + dims_before, replacement_shape.begin(), replacement_shape.end());
    strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
    return src.as_strided(shape, strides);
}
// Add dimensions of size 1 to an index tensor so that it can be broadcast to the result
// shape and iterated over element-wise like the result tensor and the restrided src.
static Tensor reshape_indexer(const Tensor& index, int64_t dims_before, int64_t dims_after) {
    auto orig_shape = index.sizes();
    auto shape = DimVector();
    shape.append(dims_before, 1);
    shape.append(orig_shape.begin(), orig_shape.end());
    shape.append(dims_after, 1);
    return index.reshape(shape);
}

AdvancedIndex::AdvancedIndex(const Tensor& src, TensorList indices_list) {
    int64_t element_size_bytes = src.itemsize();
    int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
    IntArrayRef replacement_shape;
    for (const auto dim : otter::irange(indices_list.size())) {
        if (!indices_list[dim].defined()) {
            if (dims_indexed == 0) {
                dims_before++;
            } else {
                dims_after++;
            }
        } else {
            dims_indexed++;
            replacement_shape = indices_list[dim].sizes();
            indexed_sizes.push_back(src.size(dim));
            indexed_strides.push_back(src.stride(dim) * element_size_bytes);
        }
    }
    // Check if the indexed subspace contains a dim of size 0, but the replacement
    // shape does not. This implies that an index is out of bounds, because there
    // is no number that's a valid index for an empty tensor. Normally, out of
    // bounds is handled in the indexing kernel, but this case fails earlier in
    // restride_src with an unhelpful error message.
    if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) != indexed_sizes.end() &&
        std::find(replacement_shape.begin(), replacement_shape.end(), 0) == replacement_shape.end()) {
        OTTER_CHECK(false, "index is out of bounds for dimension with size 0");
    }
    this->dims_before = dims_before;
    this->dims_after = dims_after;
    this->src = restride_src(src, dims_before, dims_indexed, replacement_shape);
    for (auto& index : indices_list) {
        if (index.defined()) {
            indices.push_back(reshape_indexer(index, dims_before, dims_after));
        }
    }
}

static TensorIterator make_index_put_iterator(const AdvancedIndex& info, const Tensor& value) {
  OTTER_CHECK(is_expandable_to(value.sizes(), info.src.sizes()), "shape mismatch: value tensor of shape ", value.sizes(),
             " cannot be broadcast to indexing result of shape ", info.src.sizes());
  OTTER_CHECK(value.scalar_type() == info.src.scalar_type(),
              "Index put requires the source and destination dtypes match, "
              "got ", info.src.scalar_type(), " for the destination "
              "and ", value.scalar_type(), " for the source.");
  TensorIteratorConfig config;
  // info.src is restrided by restride_src with 0 strided dimensions
  config.set_check_mem_overlap(false);
  config.resize_outputs(false);
  config.check_all_same_dtype(false);
  config.add_output(info.src);
  config.add_input(value);
  for (auto& index : info.indices) {
    config.add_input(index);
  }
  return config.build();
}

DEFINE_IMPL_FUNCTION(index_out)(const Tensor& self, DimVector sizes, DimVector strides, const Tensor& result) {
    index_stub(Device::CPU, *this, sizes, strides);
}

Tensor & put_(Tensor & self, const Tensor& index, const Tensor & source, const bool accumulate) {
  // Type and device checks
  OTTER_CHECK(index.scalar_type() == ScalarType::Long, "put_(): Expected a long tensor for index, but got ", index.scalar_type())
  OTTER_CHECK(self.scalar_type() == source.scalar_type(), "put_(): self and source expected to have the same dtype, but got self.dtype = ", self.scalar_type(), " and source.dtype = ", source.scalar_type());
//  OTTER_CHECK(self.device() == source.device() && self.device() == index.device(),
//      "put_(): self, index and source expected to be in the same device, but got self.device = ",
//      self.device(), ", index.device = ", index.device(), ", and source.device = ", source.device());
  // index checks
  OTTER_CHECK(source.numel() == index.numel(), "put_(): Expected source and index to have the same number of elements, but got source.numel() = ", source.numel(), ", index.numel() = ", index.numel());
  OTTER_CHECK(!(self.numel() == 0 && index.numel() != 0), "put_(): Tried to put elements into an empty tensor");
  otter::assert_no_internal_overlap(self);
  otter::assert_no_overlap(self, index);
  otter::assert_no_overlap(self, source);
  // Early return
  if (index.numel() == 0) {
    return self;
  }
  auto index_reshaped = index.reshape(source.sizes());
  // Do not iterate over self, we will compute the offsets manually
  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)
    .check_all_same_dtype(false)
    .add_input(source)
    .add_input(index_reshaped)
    .build();
  put_stub(Device::CPU, iter, self, accumulate);
  return self;
}

Tensor put(const Tensor & self, const Tensor& index, const Tensor & source, const bool accumulate) {
    return self.clone(otter::MemoryFormat::Preserve).put_(index, source, accumulate);
}

Tensor index_put(const Tensor & self, const std::vector<otter::optional<Tensor>>& indices, const Tensor & value, bool accumulate) {
    return self.clone(otter::MemoryFormat::Preserve).index_put_(indices, value, accumulate);
}

Tensor & _index_put_impl_(Tensor & self, const std::vector<otter::optional<Tensor>>& indices, const Tensor & value, const bool accumulate, const bool unsafe) {
  OTTER_CHECK(indices.size() <= (size_t)self.dim(), "too many indices for tensor of dimension ", self.dim(), " (got ", indices.size(), ")");
  if (otter::has_internal_overlap(self) == MemOverlap::YES) {
    fprintf(stderr,
      "Use of index_put_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[indices] = tensor");
  }
  if (!accumulate) {
    auto masked_fill_dispatch = canDispatchToMaskedFill(self, indices, value);
    if (std::get<0>(masked_fill_dispatch)) {
      return self.masked_fill_(std::get<1>(masked_fill_dispatch), value.item());
    }
  }
  auto value_ = value;
//  if (value.device() != self.device() && value.numel() == 1 && value.dim() == 0) {
//    value_ = value.to(self.device());
//  }
  otter::assert_no_overlap(self, value);
  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
  for (const otter::optional<Tensor>& index: indices) {
    if (index.has_value()) {
      otter::assert_no_overlap(self, *index);
    }
  }
  auto info = make_info(self, const_cast<std::vector<otter::optional<Tensor>>&>(indices));
  auto iter = make_index_put_iterator(info, value_);
  index_put_stub(Device::CPU, iter, info.indexed_sizes, info.indexed_strides, accumulate);
  return self;
}

Tensor& take_out(const Tensor& self, const Tensor& index, Tensor& out) {
  // Type and device checks
  OTTER_CHECK(index.scalar_type() == ScalarType::Long, "take(): Expected a long tensor for index, but got ", index.scalar_type())
  OTTER_CHECK(self.scalar_type() == out.scalar_type(), "take(): self and out expected to have the same dtype, but got self.dtype = ", self.scalar_type(), " and out.dtype = ", out.scalar_type());
//  OTTER_CHECK(self.device() == out.device() && self.device() == index.device(),
//      "take(): self, index and out expected to be in the same device, but got self.device = ",
//      self.device(), ", index.device = ", index.device(), ", and out.device = ", out.device());
  // index checks
  OTTER_CHECK(!(self.numel() == 0 && index.numel() != 0), "take(): tried to take from an empty tensor");
  otter::assert_no_internal_overlap(out);
  otter::assert_no_overlap(out, index);
  otter::assert_no_overlap(out, self);
  // Do not iterate over self, we will compute the offsets manually
  // out is resized inside tensor_iterator
  auto iter = TensorIteratorConfig()
    .set_check_mem_overlap(false)
    .check_all_same_dtype(false)
    .add_output(out)
    .add_input(index)
    .build();
  // Early return after out has been resized
  if (index.numel() == 0) {
    return out;
  }
  take_stub(Device::CPU, iter, self);
  return out;
}

Tensor take(const Tensor& self, const Tensor& index) {
    auto out = otter::empty(index.sizes(), self.options());
    otter::take_out(self, index, out);
    return out;
}

Tensor & index_put_(Tensor & self, const std::vector<otter::optional<Tensor>>& indices, const Tensor & value, const bool accumulate) {
    return otter::_index_put_impl_(self, indices, value, accumulate, /*unsafe=*/false);
}

// Check that indices fall within dimension array size
// Avoid redispatch call to min/max
template <typename IndexType>
static void check_indexarray_range(
    const IndexType* indices,
    int64_t n,
    IndexType indexing_axis_dim) {
  for (const auto i : otter::irange(n)) {
    auto idx = indices[i];
    OTTER_CHECK(
        0 <= idx && idx < indexing_axis_dim,
        "INDICES element is out of DATA bounds, id=",
        idx,
        " axis_dim=",
        indexing_axis_dim);
  }
}

Tensor & index_select_out_cpu_dim1_(Tensor & result_contig, const Tensor & self, const Tensor & index_contig) {
  auto self_contig = self.contiguous();
  const otter::TypeMeta dataType = self_contig.dtype();
  size_t item_bytesize = dataType.itemsize();
  auto out = static_cast<char*>(result_contig.data_ptr());
  auto src_base = static_cast<const char*>(self_contig.data_ptr());
  auto self_sizes = self_contig.sizes();
  auto outer_dims_product = otter::size_to_dim_(1, self_sizes);
  auto block_size = otter::size_from_dim_(2, self_sizes);
  auto block_bytesize = block_size * item_bytesize;
  auto src_indexing_axis_dim = self_sizes[1];
  auto src_batch_bytesize = self_sizes[1] * block_bytesize;
  auto N = index_contig.numel();
  auto gathered_batch_bytesize = N * block_bytesize;
  OTTER_DISPATCH_INDEX_TYPES(
    index_contig.scalar_type(), "batch_index_select_compute", [&]() {
      const auto* idxs = index_contig.data_ptr<index_t>();
      check_indexarray_range<index_t>(idxs, N, src_indexing_axis_dim);
      // Special-case single-float copy for efficiency
      if (self.scalar_type() == ScalarType::Float && block_size == 1) {
        for (const auto batch : otter::irange(outer_dims_product)) {
          const float* src_floats =
              (const float*)(src_base + batch * src_batch_bytesize);
          float* dst_floats = (float*)(out + batch * gathered_batch_bytesize);
          for (const auto i : otter::irange(N)) {
            auto idx = idxs[i];
            dst_floats[i] = src_floats[idx];
          }
        }
      } else {
        // outer_dims_product specifies how many times we repeat inner dimensions,
        // so we just iterate over it to cover all outer dimensions.
        for (const auto batch : otter::irange(outer_dims_product)) {
          for (const auto i : otter::irange(N)) {
            auto idx = idxs[i];
            auto src = src_base + batch * src_batch_bytesize + idx * block_bytesize;
            auto dst = out + batch * gathered_batch_bytesize + i * block_bytesize;
            memcpy(dst, src, block_bytesize);
          }
        }
      }
  });
  return result_contig;
}

Tensor & index_select_out_cpu_(const Tensor & self, int64_t dim, const Tensor & index, Tensor & result) {
  dim = maybe_wrap_dim(dim, self.dim());
  auto numel = index.numel();
  OTTER_CHECK(index.dim() <= 1, "index_select(): Index is supposed to be a vector");
  OTTER_CHECK(index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int, "index_select(): Expected dtype int32 or int64 for index");
  OTTER_CHECK(self.scalar_type() == result.scalar_type(), "index_select(): self and result must have the same scalar type");
  OTTER_CHECK(dim == 0 || dim < self.dim(), "index_select(): Indexing dim ", dim, " is out of bounds of tensor");
  otter::assert_no_internal_overlap(result);
  otter::assert_no_overlap(result, self);
  otter::assert_no_overlap(result, index);
  auto result_size = self.sizes().vec();
  if (self.dim() > 0) {
    result_size[dim] = numel;
  }
  otter::native::resize_output(result, result_size);
  auto index_contig = index.contiguous();
  if (self.dim() > 1) {
    if (numel == 0) {
      return result;
    }
    if (self.numel() == 0) {
      auto src_indexing_axis_dim = self.size(dim);
      OTTER_CHECK(src_indexing_axis_dim > 0, "index_select(): self indexing axis dim should be positive");
      OTTER_DISPATCH_INDEX_TYPES(
      index_contig.scalar_type(), "index_select_empty_self_bound_check", [&]() {
        const auto* idxs = index_contig.data_ptr<index_t>();
        check_indexarray_range<index_t>(idxs, numel, src_indexing_axis_dim);
      });
      return result;
    }
    if (dim == 1 && result.is_contiguous()) {
      // fast pass
      return index_select_out_cpu_dim1_(result, self, index_contig);
    }
    auto selfSlice = self.select(dim, 0);
    auto resultSlice = result.select(dim, 0);
    auto selfSlice_data = selfSlice.data_ptr();
    auto resultSlice_data = resultSlice.data_ptr();
    auto self_stride_bytes = self.stride(dim) * elementSize(self.scalar_type());
    auto result_stride_bytes = result.stride(dim) * elementSize(result.scalar_type());
    auto self_dim_size = self.size(dim);
    auto slice_size = selfSlice.numel();
    auto iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(resultSlice)
      .add_input(selfSlice)
      .build();
    auto grain_size = otter::GRAIN_SIZE;
    auto outer_loop =
      // explicitly capture all required variables to work around windows build
      // TODO: fix this when windows can correctly capture variables in nested lambda
      [&index_contig, &iter, &self_dim_size, &selfSlice_data, &self_stride_bytes, &resultSlice_data,
        &result_stride_bytes](int64_t start, int64_t end) {
      auto sub_iter = TensorIterator(iter);
      OTTER_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_select_out_cpu_",
        [&index_contig, &start, &end, &sub_iter, &self_dim_size, &selfSlice_data, &self_stride_bytes,
          &resultSlice_data, &result_stride_bytes] () {
        auto index_data = index_contig.data_ptr<index_t>();
        for (const auto i : otter::irange(start, end)) {
          auto self_i = index_data[i];
          OTTER_CHECK((self_i >= 0) && (self_i < self_dim_size), "index out of range in self");
          auto self_data = static_cast<char*>(selfSlice_data) + self_i * self_stride_bytes;
          auto result_data = static_cast<char*>(resultSlice_data) + i * result_stride_bytes;
          sub_iter.unsafe_replace_operand(0, result_data);
          sub_iter.unsafe_replace_operand(1, self_data);
          copy_stub(Device::CPU, sub_iter, false);
        };
      });
    };
    // parallel on inner loop in case the slice is large enough;
    // otherwise parallel on outer loop
    if (slice_size >= grain_size) {
      outer_loop(0, numel);
    } else {
      // use a fast loop when self and result are contiguous and of the same data type
      if (iter.is_contiguous() && self.scalar_type() == result.scalar_type()) {
        auto slice_size_bytes = slice_size * elementSize(self.scalar_type());
        // explicitly capture all required variables to work around windows build
        // TODO: fix this when windows can correctly capture variables in nested lambda
        otter::parallel_for(0, numel, grain_size / slice_size,
          [&index_contig, &slice_size_bytes, &self_dim_size, &selfSlice_data,
            &self_stride_bytes, &resultSlice_data, &result_stride_bytes](int64_t start, int64_t end) {
          OTTER_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_select_out_cpu_",
            [&index_contig, &slice_size_bytes, &self_dim_size, &selfSlice_data,
              &self_stride_bytes, &resultSlice_data, &result_stride_bytes, &start, &end] () {
            auto index_data = index_contig.data_ptr<index_t>();
            for (const auto i : otter::irange(start, end)) {
              auto self_i = index_data[i];
              OTTER_CHECK((self_i >= 0) && (self_i < self_dim_size), "index out of range in self");
              auto self_data = static_cast<char*>(selfSlice_data) + self_i * self_stride_bytes;
              auto result_data = static_cast<char*>(resultSlice_data) + i * result_stride_bytes;
              memcpy(result_data, self_data, slice_size_bytes);
            }
          });
        });
      } else {
        otter::parallel_for(0, numel, grain_size / slice_size, outer_loop);
      }
    }
  } else {
    OTTER_CHECK(result.dim() <= 1, "result.dim() (", result.dim(), ") must one or zero for given self.dim() (", self.dim(), ")");
    // explicitly capture all required variables to work around windows build
    // TODO: fix this when windows can correctly capture variables in nested lambda
      OTTER_DISPATCH_ALL_TYPES_AND2(ScalarType::HFloat, ScalarType::Bool,
        self.scalar_type(), "index_select", [&index_contig, &self, &result, &dim, &numel] {
        auto self_stride = self.dim() == 0 ? 1 : self.stride(dim);
        auto result_stride = result.dim() == 0 ? 1 : result.stride(dim);
        auto self_data_ptr = self.data_ptr<scalar_t>();
        auto result_data_ptr = result.data_ptr<scalar_t>();
        auto self_numel = self.numel();
        OTTER_DISPATCH_INDEX_TYPES(index_contig.scalar_type(), "index_select_out_cpu_",
          [&index_contig, &numel, &self_numel, &self_data_ptr, &self_stride, &result_data_ptr, &result_stride] {
          auto index_data = index_contig.data_ptr<index_t>();
          for (const auto i : otter::irange(numel)) {
            auto self_i = index_data[i];
            OTTER_CHECK((self_i >= 0) && (self_i < self_numel), "index out of range in self");
            scalar_t *self_ip = self_data_ptr + self_i * self_stride;
            *(result_data_ptr + i * result_stride) = *self_ip;
          }
        });
      });
  }
  return result;
}

Tensor index_select_cpu_(const Tensor & self, int64_t dim, const Tensor & index) {
  Tensor result = otter::empty({0}, self.options());
  return otter::index_select_out_cpu_(self, dim, index, result);
}

Tensor & index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Scalar& source) {
  OTTER_CHECK(index.scalar_type() == ScalarType::Long, "index_fill_(): Expected dtype int64 for index.");
  otter::assert_no_overlap(self, index);
  if (otter::has_internal_overlap(self) == otter::MemOverlap::YES) {
    fprintf(stderr,
      "Use of index_fill_ on expanded tensors is deprecated. "
      "Please clone() the tensor before performing this operation. "
      "This also applies to advanced indexing e.g. tensor[mask] = scalar");
  }
  // Handle the case when `self` is 0-dim
  Tensor self_nonzero_dim = (self.dim() == 0) ? self.unsqueeze(-1) : self;
  dim = otter::maybe_wrap_dim(dim, self_nonzero_dim);
  OTTER_CHECK(index.dim() <= 1, "Index has to be a vector/scalar");
  // Prepare `index` for TensorIterator.
  // It is restrided to be broadcastable over `self` in TensorIterator.
  auto index_sizes = std::vector<int64_t>(self_nonzero_dim.dim(), 1);
  auto index_strides = std::vector<int64_t>(self_nonzero_dim.dim(), 0);
  index_sizes[dim] = index.numel();
  index_strides[dim] = (index.dim() > 0) ? index.stride(0) : 1; // `index` is 1d or scalar
  auto index_restrided = index.as_strided(
    index_sizes, index_strides);
  // Prepare `self` for TensorIterator.
  // Restride `self` to not advance in dimension `dim`.
  // We do not use squash_dim here because `index` will
  // need to advance in this dimension.
  // Note that self_sizes[dim] is set to index.numel().
  // This is done so that self_sizes[dim] and index_sizes[dim]
  // match as required by TensorIterator (input shape should
  // strictly broadcast over output shape, i.e.
  // output.shape[i] >= input.shape[i] for i in range(dims)).
  auto self_sizes = self_nonzero_dim.sizes().vec();
  auto self_strides = self_nonzero_dim.strides().vec();
  self_sizes[dim] = index.numel();
  self_strides[dim] = 0;
  auto self_restrided = self_nonzero_dim.as_strided(self_sizes, self_strides);
  auto iter = TensorIteratorConfig()
    // We do not check for overlap because `self` is restrided
    // with zero stride. Zero strides trigger memory overlap assert
    // within TensorIterator.
    .set_check_mem_overlap(false)
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .add_output(self_restrided)
    .add_input(index_restrided)
    .build();
  auto self_dim_size = (self_nonzero_dim.sizes())[dim];
  auto self_dim_stride = (self_nonzero_dim.strides())[dim];
  index_fill_stub(
    Device::CPU,
    iter,
    dim,
    self_dim_size,
    self_dim_stride,
    source);
  return self;
}

Tensor & index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  OTTER_CHECK(source.dim() == 0, "index_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ", source.dim(), " dimension(s).");
  return self.index_fill_(dim, index, source.item());
}

Tensor index_fill(const Tensor & self, int64_t dim, const Tensor & index, const Scalar& source) {
  return self.clone(otter::MemoryFormat::Preserve).index_fill_(dim, index, source);
}

Tensor index_fill(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return self.clone(otter::MemoryFormat::Preserve).index_fill_(dim, index, source);
}

template <typename scalar_t>
int64_t count_nonzero_impl(TensorIterator& iter, Range range) {
    int64_t num_nonzero = 0;
    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
        constexpr int ilp_factor = 4;
        const char* ptr = data[0];
        const auto stride = strides[0];
        int64_t nonzero[ilp_factor] = {0};
        int64_t i = 0;
        for (; i + (ilp_factor - 1) < n; i += ilp_factor) {
            otter::ForcedUnroll<ilp_factor>{}([&](int k) {
                const auto& val = otter::load<scalar_t>(ptr + k * stride);
                if (val != scalar_t(0)) {
                    ++nonzero[k];
                }
            });
            ptr += ilp_factor * stride;
        }
        for (; i < n; ++i) {
            const auto& val = otter::load<scalar_t>(ptr);
            if (val != scalar_t(0)) {
                ++nonzero[0];
            }
            ptr += stride;
        }
        for (const auto k : otter::irange(1, ilp_factor)) {
            nonzero[0] += nonzero[k];
        }
        num_nonzero += nonzero[0];
    };
    iter.serial_for_each(loop, range);
    return num_nonzero;
}

Tensor& nonzero_out_cpu(const Tensor& self, Tensor& result) {
    OTTER_CHECK(result.scalar_type() == otter::ScalarType::Long,
              "nonzero: Expected out tensor to have scalar type Long "
              "but got scalar type", result.scalar_type());
    otter::assert_no_internal_overlap(result);
    otter::assert_no_overlap(result, self);
    auto iter = TensorIteratorConfig()
        .add_input(self)
        .enforce_linear_iteration()
        .build();
    const auto numel = iter.numel();
    const auto num_threads = otter::get_num_threads();
    DimVector thread_begin(num_threads, -1);
    DimVector thread_count_nonzero(num_threads + 1);
    // Pass 1: Count nonzero element per-thread
    OTTER_DISPATCH_ALL_TYPES_AND2(otter::ScalarType::HFloat, otter::ScalarType::Bool, self.scalar_type(), "nonzero_count_cpu", [&] {
        otter::parallel_for(0, numel, otter::GRAIN_SIZE, [&] (int64_t begin, int64_t end) {
            const auto tid = otter::get_thread_num();
            thread_begin[tid] = begin;
            thread_count_nonzero[tid + 1] = count_nonzero_impl<scalar_t>(iter, {begin, end});
        });
  });
  // Convert thread-local counts to cumulative sum
  for (const auto i : otter::irange(1, thread_count_nonzero.size())) {
    thread_count_nonzero[i] += thread_count_nonzero[i - 1];
  }
  const auto self_sizes = self.sizes();
  const auto total_nonzero = thread_count_nonzero.back();
  const int64_t ndim = self_sizes.size();
  if (otter::native::resize_output(result, {total_nonzero, ndim})) {
    // Default to fortran-contiguous output (see gh-46224)
    result.as_strided_({total_nonzero, ndim}, {1, total_nonzero});
  }
  if (result.numel() == 0) {
    return result;
  }
  // Pass 2: Write indexes
  OTTER_DISPATCH_ALL_TYPES_AND2(otter::ScalarType::HFloat, otter::ScalarType::Bool, self.scalar_type(), "nonzero_cpu", [&] {
      otter::parallel_for(0, numel, otter::GRAIN_SIZE, [&] (int64_t begin, int64_t end) {
          auto tid = otter::get_thread_num();
          // Work needs to be distributed the same on both passes
          OTTER_INTERNAL_ASSERT(begin == thread_begin[tid]);
          // +1 faster than additional condition check inside loop
          otter::SmallVector<int64_t, 33> sizes(ndim + 1, -1);
          std::copy(self_sizes.begin(), self_sizes.end(), sizes.begin() + 1);
          otter::SmallVector<int64_t, 33> current_idx(ndim + 1);
          if (begin > 0) {
              auto idx = begin;
              for (int64_t k = ndim; idx > 0 && k > 0; --k) {
                  current_idx[k] = idx % sizes[k];
                  idx /= sizes[k];
              }
          }
          auto out_accessor = result.accessor<int64_t, 2>();
          auto out_ptr = out_accessor[thread_count_nonzero[tid]].data();
          auto loop = [&](char** data, const int64_t* strides, int64_t n1, int64_t n2) {
              // Copy into local variables to improve compiler alias analysis
              int64_t* OTTER_RESTRICT local_idx = current_idx.data() + 1;
              const int64_t* OTTER_RESTRICT local_sizes = sizes.data() + 1;
              const auto in_stride = strides[0];
              const auto out_stride1 = out_accessor.stride(1);
              const auto out_stride0 = out_accessor.stride(0) - ndim * out_stride1;
              const auto ndim = out_accessor.size(1);
              int64_t* out = out_ptr;
              for (const auto i : otter::irange(n2)) {
                  const char* ptr = data[0] + i * strides[1];
                  for (const auto j : otter::irange(n1)) {
                      (void)j; //Suppress unused variable warning
                      const auto& val = otter::load<scalar_t>(ptr);
                      // If nonzero, write index
                      if (val != scalar_t(0)) {
                          for (const auto k : otter::irange(ndim)) {
                              *out = local_idx[k];
                              out += out_stride1;
                          }
                          out += out_stride0;
                      }
                      ptr += in_stride;
                      // Advance current index
                      int64_t k = ndim - 1;
                      ++local_idx[k];
                      while (OTTER_UNLIKELY(local_idx[k] == local_sizes[k])) {
                          local_idx[k] = 0;
                          --k;
                          ++local_idx[k];
                      }
                  }
              }
              out_ptr = out;
          };
          iter.serial_for_each(loop, {begin, end});
          OTTER_INTERNAL_ASSERT(out_ptr == out_accessor[thread_count_nonzero[tid + 1]].data());
      });
  });
  return result;
}
Tensor nonzero_cpu(const Tensor& self) {
    auto result = otter::empty({0}, self.options().dtype(otter::ScalarType::Long));
    nonzero_out_cpu(self, result);
    return result;
}

static Tensor & masked_fill_impl_cpu(Tensor & self, const Tensor & mask, const Scalar& value) {
    if (mask.scalar_type() == ScalarType::Byte) {
        fprintf(stderr, "masked_fill_ received a mask with dtype torch.uint8, this behavior is now deprecated," \
                "please use a mask with dtype torch.bool instead.");
    }
    if (otter::has_internal_overlap(self) == MemOverlap::YES) {
        fprintf(stderr,
                "Use of masked_fill_ on expanded tensors is deprecated. "
                "Please clone() the tensor before performing this operation. "
                "This also applies to advanced indexing e.g. tensor[mask] = scalar");
    }
    otter::assert_no_partial_overlap(self, mask);
    auto iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)  // deprecated, but not a hard error
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .add_output(self)
        .add_input(mask)
        .build();
    masked_fill_stub(Device::CPU, iter, value);
    return self;
}

Tensor & masked_fill__cpu(Tensor& self, const Tensor & mask, const Scalar& value) {
    masked_fill_impl_cpu(self, mask, value);
    return self;
}

Tensor & masked_fill__cpu(Tensor& self, const Tensor & mask, const Tensor & value) {
    OTTER_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
                "with ", value.dim(), " dimension(s).");
    masked_fill_impl_cpu(self, mask, value.item());
    return self;
}

Tensor masked_fill(const Tensor & self, const Tensor & mask, const Scalar& source) {
    Tensor result;
    {
        otter::MaybeOwned<Tensor> _mask, _self;
        std::tie(_mask, _self) = expand_outplace(mask, self);
        result = _self->clone(otter::MemoryFormat::Contiguous);
        result.masked_fill_(mask, source);
    }
    return result;
}

Tensor masked_fill(const Tensor & self, const Tensor & mask, const Tensor & source) {
    Tensor result;
    {
        otter::MaybeOwned<Tensor> _mask, _self;
        std::tie(_mask, _self) = expand_outplace(mask, self);
        result = _self->clone(otter::MemoryFormat::Contiguous);
        result.masked_fill_(mask, source);
    }
    return result;
}

static Tensor & masked_select_out_impl_cpu(Tensor & result, const Tensor & self, const Tensor & mask) {
    OTTER_CHECK(mask.scalar_type() == ScalarType::Byte || mask.scalar_type() == ScalarType::Bool,
                "masked_select: expected BoolTensor or ByteTensor for mask");
    OTTER_CHECK(self.scalar_type() == result.scalar_type(),
                "masked_select(): self and result must have the same scalar type");
    otter::assert_no_internal_overlap(result);
    otter::assert_no_overlap(result, self);
    otter::assert_no_overlap(result, mask);
    if (mask.scalar_type() == otter::ScalarType::Byte) {
        fprintf(stderr, "masked_select received a mask with dtype torch.uint8, this behavior is now deprecated," \
                   "please use a mask with dtype torch.bool instead.");
    }
    otter::MaybeOwned<Tensor> _mask, _self;
    std::tie(_mask, _self) = expand_outplace(mask, self);
    auto shape = _self->sizes();
    int64_t numel = _mask->sum().item().toLong();
    otter::native::resize_output(result, {numel});
    if (numel == 0) {
        return result;
    }
    // Create strided view of result before feeding into TensorIterator
    auto strides = DimVector(shape.size(), 0);
    auto orig_stride = result.strides()[0];
    auto result_strided = result.as_strided(shape, strides);
    // serial kernel
    // serial kernel requires that src is traversed in its logical order. However, TensorIterator might
    // have reordered dimensions so that src would be traversed in its physical order, producing wrong
    // answers. A sufficient condition that no reorder happened is that both _self and _mask is contiguous.
    // If it is not satisfied, use parallel kernel that handles permutations correctly
    bool use_serial_kernel = (self.numel() < otter::GRAIN_SIZE || otter::get_num_threads() == 1 ) &&
    _self->is_contiguous() && _mask->is_contiguous();
    if (use_serial_kernel) {
        auto iter = TensorIteratorConfig()
            .set_check_mem_overlap(false)  // result is intenionally zero-strided above
            .check_all_same_dtype(false)
            .resize_outputs(false)
            .add_output(result_strided)
            .add_input(*_self)
            .add_input(*_mask)
            .build();
        masked_select_serial_stub(Device::CPU, iter, orig_stride);
        return result;
    }
    // Use a prefix sum to record the output locations of the masked elements,
    // so as to parallel with TensorIterator.
    auto mask_long = otter::empty(shape, self.options().dtype(otter::ScalarType::Long)).copy_(*_mask);
    auto mask_prefix_sum = otter::empty(shape, self.options().dtype(otter::ScalarType::Long));
    auto mask_long_data = mask_long.data_ptr<int64_t>();
    auto mask_prefix_sum_data = mask_prefix_sum.data_ptr<int64_t>();
    // TODO: Here can only use std::partial_sum for C++14,
    // use std::exclusive_scan when PyTorch upgrades to C++17, which have better peformance.
    // std::exclusive_scan(mask_long_data, mask_long_data + mask_long.numel(), mask_prefix_sum_data, 0);
    std::partial_sum(mask_long_data, mask_long_data + mask_long.numel(), mask_prefix_sum_data);
    auto iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)  // result is intenionally zero-strided above
        .check_all_same_dtype(false)
        .resize_outputs(false)
        .add_output(result_strided)
        .add_input(*_self)
        .add_input(*_mask)
        .add_input(mask_prefix_sum)
        .build();
    masked_select_stub(Device::CPU, iter, orig_stride);
    return result;
}

Tensor & masked_select_out_cpu(const Tensor & self, const Tensor & mask, Tensor & result) {
    return masked_select_out_impl_cpu(result, self, mask);
}

Tensor masked_select_cpu(const Tensor & self, const Tensor & mask) {
      Tensor result = otter::empty({0}, self.options());
      return otter::masked_select_out_cpu(self, mask, result);
}

}   // end namespace otter
