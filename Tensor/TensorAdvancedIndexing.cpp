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

}   // end namespace otter
