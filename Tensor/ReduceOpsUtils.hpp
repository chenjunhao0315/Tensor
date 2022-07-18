//
//  ReduceOpsUtils.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/18.
//

#ifndef ReduceOpsUtils_h
#define ReduceOpsUtils_h

#include "Macro.hpp"
#include "TensorIterator.hpp"
#include "WarpDimUtils.hpp"

namespace otter {

// WrapDimUtilsMulti.h
constexpr size_t dim_bitset_size = 64;
static inline std::bitset<dim_bitset_size> dim_list_to_bitset(IntArrayRef dims, int64_t ndims) {
    OTTER_CHECK(ndims <= (int64_t)dim_bitset_size,
                "only tensors with up to ",
                dim_bitset_size,
                " dims are supported");
    std::bitset<dim_bitset_size> seen;
    for (const auto i : otter::irange(dims.size())) {
        size_t dim = maybe_wrap_dim(dims[i], ndims);
        OTTER_CHECK(!seen[dim], "dim ", dim, " appears multiple times in the list of dims");
        seen[dim] = true;
    }
    return seen;
}
//

using DimMask = TensorIterator::DimMask;
static DimMask make_dim_mask(IntArrayRef dims, int64_t ndim) {
    DimMask mask;
    if (dims.empty()) {
        mask = DimMask().flip();
    } else {
        mask = otter::dim_list_to_bitset(dims, ndim);
    }
    return mask;
}

inline DimVector shape_from_dim_mask(const Tensor& self, DimMask mask, bool keepdim) {
    auto shape = DimVector(self.sizes());
    for (int dim = shape.size() - 1; dim >= 0; dim--) {
        if (mask[dim]) {
            if (keepdim) {
                shape[dim] = 1;
            } else {
                shape.erase(shape.begin() + dim);
            }
        }
    }
    return shape;
}

static Tensor review_reduce_result(const Tensor& result, int ndim, DimMask mask, bool keepdim) {
    if (keepdim) {
        return result;
    }
    auto shape = DimVector(result.sizes());
    auto stride = DimVector(result.strides());
    for (const auto dim : otter::irange(ndim)) {
        if (mask[dim]) {
            shape.insert(shape.begin() + dim, 1);
            stride.insert(stride.begin() + dim, 0);
        }
    }
    return result.as_strided(shape, stride);
}

static TensorIterator make_reduction(
    const Tensor& self,
    const Tensor& result,
    IntArrayRef dims,
    bool keepdim,
    ScalarType in_dtype) {
    int64_t ndim = self.dim();
    auto mask = otter::make_dim_mask(dims, ndim);
    auto viewed_result = otter::review_reduce_result(result, ndim, mask, keepdim);
    if (self.scalar_type() == in_dtype) {
        return TensorIterator::reduce_op(viewed_result, self);
    }
    return TensorIterator::reduce_op(viewed_result, self.to(in_dtype));
}

static OTTER_UNUSED DimVector get_reduction_shape(
    const Tensor& self,
    IntArrayRef dims,
    bool keepdim) {
    auto mask = otter::make_dim_mask(dims, self.dim());
    return otter::shape_from_dim_mask(self, mask, keepdim);
}

static void resize_reduction(
    TensorIterator& meta,
    const Tensor& self,
    IntArrayRef dims,
    bool keepdim,
    ScalarType out_dtype) {
    DimVector dims_(dims);
    maybe_wrap_dims(dims_, self.dim());
    auto shape = get_reduction_shape(self, dims_, keepdim);
    meta.set_output(0, shape, {}, self.options().dtype(out_dtype));
}

static OTTER_UNUSED TensorIterator make_reduction_from_out_ty(
    const Tensor& self,
    const Tensor& result,
    IntArrayRef dims,
    bool keepdim,
    ScalarType out_dtype) {
    // special case for type promotion in mixed precision, improves computational
    // efficiency.
    // not generalize this to common mismatched input/output types to avoid cross
    // product of templated kernel launches.
    const bool gpu_lowp_to_f32 = (false && (self.scalar_type() == otter::ScalarType::HFloat) &&
       out_dtype == otter::ScalarType::Float);
    auto in_dtype = gpu_lowp_to_f32 ? self.scalar_type() : out_dtype;
    return make_reduction(self, result, dims, keepdim, in_dtype);
}

}   // end namespace otter

#endif /* ReduceOpsUtils_h */
