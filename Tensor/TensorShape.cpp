//
//  TensorShape.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/10.
//

#include "Tensor.hpp"
#include "WarpDimUtils.hpp"
#include "Utils.hpp"
#include "TensorResize.hpp"
#include "TensorUtils.hpp"
#include "ExpandUtils.hpp"
#include "TensorShape.hpp"
#include "MemoryOverlap.hpp"
#include "Exception.hpp"
#include "TensorIterator.hpp"
#include "Parallel.hpp"
#include "TensorCopy.hpp"
#include "TensorCat.hpp"
#include "TensorFactory.hpp"

namespace otter {

namespace native {

Tensor select(const Tensor& self, int64_t dim, int64_t index) {
    int64_t ndim = self.dim();
    
    dim = maybe_wrap_dim(dim, ndim);
    auto size = self.size(dim);
    if (index < -size || index >= size) {
        OTTER_CHECK(false, "select(): index ", index, " out of range for tensor of size ", self.sizes(), " at dimension ", dim);
    }
    if (index < 0) {
        index += size;
    }
    DimVector sizes(self.sizes().begin(), self.sizes().end());
    DimVector strides(self.strides().begin(), self.strides().end());
    auto memory_offset = self.memory_offset() + index * strides[dim];
    sizes.erase(sizes.begin() + dim);
    strides.erase(strides.begin() + dim);
    auto result = (memory_offset) ? self.as_strided(sizes, strides, memory_offset) : self.as_strided(sizes, strides);
    
    return result;
}

Tensor permute(const Tensor& self, IntArrayRef dims) {
    auto nDims = self.dim();
    // number of dims don't match in permute
    assert(dims.size() == (size_t)nDims);
    auto oldSizes = self.sizes();
    auto oldStrides = self.strides();
    DimVector newSizes(nDims);
    DimVector newStrides(nDims);
    std::vector<bool> seen(nDims);
    for (const auto i : otter::irange(nDims)) {
        auto dim = maybe_wrap_dim(dims[i], nDims);
        OTTER_CHECK(!seen[dim], "repeated dim in permute");
        seen[dim] = true;
        newSizes[i] = oldSizes[dim];
        newStrides[i] = oldStrides[dim];
    }
    
    return self.as_strided(newSizes, newStrides);
}

Tensor& transpose_(Tensor& self, int64_t dim0, int64_t dim1) {
    auto ndims = self.dim();
    dim0 = maybe_wrap_dim(dim0, ndims);
    dim1 = maybe_wrap_dim(dim1, ndims);
    if (dim0 == dim1) {
        return self;
    }
    DimVector sizes(self.sizes().begin(), self.sizes().end());
    DimVector strides(self.strides().begin(), self.strides().end());
    std::swap(strides[dim0], strides[dim1]);
    std::swap(sizes[dim0], sizes[dim1]);
    self.as_strided_(sizes, strides);
    return self;
}

Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
    auto ndims = self.dim();
    dim0 = maybe_wrap_dim(dim0, ndims);
    dim1 = maybe_wrap_dim(dim1, ndims);
    
    if (dim0 == dim1) {
        return self;
    }
    
    DimVector sizes(self.sizes().begin(), self.sizes().end());
    DimVector strides(self.strides().begin(), self.strides().end());
    std::swap(strides[dim0], strides[dim1]);
    std::swap(sizes[dim0], sizes[dim1]);
    auto result = self.as_strided(sizes, strides);
    return result;
}

Tensor expand(const Tensor& self, IntArrayRef sizes) {
    OTTER_CHECK(sizes.size() >= (size_t)self.dim(), "expand(", self.toString(), "{", self.sizes(), "}, size=", sizes, "): the number of sizes provided (", sizes.size(), ") ", "must be greater or equal to the number of dimensions in the tensor (", self.dim(), ")");
    auto expandedPerspectiveView = inferExpandGeometry_dimvector(self.sizes(), self.strides(), sizes);
    
    auto result = self.as_strided(expandedPerspectiveView.sizes, expandedPerspectiveView.strides);
    return result;
}

Tensor expand_as(const Tensor& self, const Tensor& other) {
    return self.expand(other.sizes());
}

template <typename Vec>
Tensor alias_with_sizes_and_strides(const Tensor& self, const Vec& sizes, const Vec& strides) {
    Tensor self_;
    self_ = otter::make_tensor<TensorNucleus>(Memory(self.memory()), self.dtype());
    setStrided(self_, sizes, strides, self.memory_offset());
    
    return self_;
}

Tensor view(const Tensor& self, IntArrayRef sizes) {
    DimVector infer_size = otter::infer_size_dimvector(sizes, self.numel());
    auto stride = otter::computeStride(self.sizes(), self.strides(), infer_size);
    OTTER_CHECK(!stride.empty(), "view size is not compactible with input tensor's size and stride -> trun to use .reshape(...)");
    
    return alias_with_sizes_and_strides(self, infer_size, stride);
}

Tensor view_impl(const Tensor& self, IntArrayRef sizes) {
    DimVector shape = otter::infer_size_dimvector(sizes, self.numel());
    auto stride = otter::computeStride(self.sizes(), self.strides(), shape);
    
    return alias_with_sizes_and_strides(self, shape, stride);
}

Tensor _unsafe_view(const Tensor& self, IntArrayRef sizes) {
    return view_impl(self, sizes);
}

Tensor reshape(const Tensor& self, IntArrayRef sizes) {
    DimVector shape = otter::infer_size_dimvector(sizes, self.numel());
    auto stride = otter::computeStride(self.sizes(), self.strides(), shape);
    
    if (!stride.empty()) {
        return self.view(sizes);
    }
    
    return _unsafe_view(self.clone(MemoryFormat::Contiguous), shape);
}

Tensor reshape_as(const Tensor& self, const Tensor& other) {
    return self.reshape(other.sizes());
}

Tensor detach(const Tensor& self) {
    // autograd
    return Tensor(self.getPtr());
}

Tensor as_strided_tensorimpl(const Tensor& self, IntArrayRef sizes, IntArrayRef strides) {
    return as_strided_tensorimpl(self, sizes, strides, self.memory_offset());
}

Tensor as_strided_tensorimpl(const Tensor& self, IntArrayRef sizes, IntArrayRef strides, int64_t memory_offset_) {
    auto result = otter::make_tensor<TensorNucleus>(Memory(self.memory()), self.dtype());
    
    setStrided(result, sizes, strides, memory_offset_);
    return result;
}

const Tensor &as_strided_(const Tensor& self, IntArrayRef size, IntArrayRef stride) {
    return as_strided_(self, size, stride, self.memory_offset());
}

const Tensor &as_strided_(const Tensor& self, IntArrayRef size, IntArrayRef stride, int64_t memory_offset_) {
    setStrided(self, size, stride, memory_offset_);
    return self;
}

Tensor slice(const Tensor& self, int64_t dim, int64_t start, int64_t end, int64_t step) {
    int64_t ndim = self.dim();
    if (ndim == 0) {
        assert(false);  // "slice() cannot be applied to a 0-dim tensor."
    }
    dim = maybe_wrap_dim(dim, ndim);
    DimVector sizes(self.sizes().begin(), self.sizes().end());
    DimVector strides(self.strides().begin(), self.strides().end());
    
    assert(step > 0); // "slice step must be positive"
    
    if (start == INT64_MAX) start = 0;
    if (start < 0) start += sizes[dim];
    if (end < 0)   end   += sizes[dim];
    
    if (start < 0) {
        start = 0;
    } else if (start >= sizes[dim]) {
        start = sizes[dim];
    }
    
    if (end < start) {
        end = start;
    } else if (end >= sizes[dim]) {
        end = sizes[dim];
    }
    
    auto memory_offset = self.memory_offset() + start * strides[dim];
    auto len = end - start;
    sizes[dim] = (len + step - 1) / step; // round-up
    strides[dim] *= step;
    
    Tensor result;
    result = self.as_strided(sizes, strides, memory_offset);
    
    return result;
}

struct InferUnsqueezeGeometryResult {
    DimVector sizes;
    DimVector strides;
    InferUnsqueezeGeometryResult(IntArrayRef tensor_sizes, IntArrayRef tensor_strides) : sizes(tensor_sizes.begin(), tensor_sizes.end()), strides(tensor_strides.begin(), tensor_strides.end()) {}
};
    
InferUnsqueezeGeometryResult inferUnsqueezeGeometry(const Tensor& tensor, int64_t dim) {
    InferUnsqueezeGeometryResult result(tensor.sizes(), tensor.strides());
    int64_t new_stride = dim >= tensor.dim() ? 1 : result.sizes[dim] * result.strides[dim];
    result.sizes.insert(result.sizes.begin() + dim, 1);
    result.strides.insert(result.strides.begin() + dim, new_stride);

    return result;
}

Tensor unsqueeze(const Tensor& self, int64_t dim) {
    dim = maybe_wrap_dim(dim, self.dim() + 1);
    auto g = inferUnsqueezeGeometry(self, dim);
    
    return self.as_strided(g.sizes, g.strides);
}

Tensor& unsqueeze_(Tensor& self, int64_t dim) {
    dim = maybe_wrap_dim(dim, self.dim() + 1);

    auto g = inferUnsqueezeGeometry(self, dim);
    self.as_strided_(g.sizes, g.strides);
    return self;
}

std::tuple<DimVector, DimVector>
inferSqueezeGeometry(const Tensor &tensor) {
    DimVector sizes;
    DimVector strides;

    for(const auto d : otter::irange(tensor.dim())) {
        if(tensor.sizes()[d] != 1) {
            sizes.push_back(tensor.sizes()[d]);
            strides.push_back(tensor.strides()[d]);
        }
    }

    return std::make_tuple(std::move(sizes), std::move(strides));
}

std::tuple<DimVector, DimVector>
inferSqueezeGeometry(const Tensor& tensor, int64_t dim) {
    DimVector sizes;
    DimVector strides;

    for(const auto d : otter::irange(tensor.dim())) {
        if(d != dim || tensor.sizes()[dim] != 1) {
            sizes.push_back(tensor.sizes()[d]);
            strides.push_back(tensor.strides()[d]);
        }
    }
    return std::make_tuple(std::move(sizes), std::move(strides));
}

Tensor squeeze(const Tensor& self, int64_t dim) {
    int64_t dims = self.dim();
    dim = maybe_wrap_dim(dim, dims);
    if (dims == 0 || self.sizes()[dim] != 1) {
        return self.as_strided(self.sizes(), self.strides());
    }
    auto g = inferSqueezeGeometry(self, dim);
    auto result = self.as_strided(std::get<0>(g), std::get<1>(g));
    return result;
}

Tensor& squeeze_(Tensor& self, int64_t dim) {
    int64_t dims = self.dim();
    dim = maybe_wrap_dim(dim, self.dim());

    if (dims == 0 || self.sizes()[dim] != 1) {
        self.as_strided_(self.sizes(), self.strides());
        return self;
    }
    auto g = inferSqueezeGeometry(self, dim);
    self.as_strided_(std::get<0>(g), std::get<1>(g));
    return self;
}

Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
    assert(self.dim() > 0); // "narrow() cannot be applied to a 0-dim tensor.");
    auto cur_size = self.size(dim);
    if (start != cur_size) {  // start being the end is valid, but not a valid dim specification.
        start = maybe_wrap_dim(start, cur_size);
    }
    assert(length >= 0 && start <= cur_size - length);  // "start (", start, ") + length (", length, ") exceeds dimension size (", cur_size, ").");
    return otter::native::slice(self, dim, start, start + length, 1);
}

inline void check_cat_no_zero_dim(TensorList tensors) {
    for (const auto i : otter::irange(tensors.size())) {
        auto& t = tensors[i];
        assert(t.dim() > 0);
    }
}

inline void check_cat_shape_except_dim(const Tensor & first, const Tensor & second, int64_t dimension, int64_t index) {
    int64_t first_dims = first.dim();
    int64_t second_dims = second.dim();
    OTTER_CHECK(first_dims == second_dims, "Tensors must have same number of dimensions: got ", first_dims, " and ", second_dims);
    for (const auto dim : otter::irange(first_dims)) {
        if (dim == dimension) {
            continue;
        }
        int64_t first_dim_size = first.sizes()[dim];
        int64_t second_dim_size = second.sizes()[dim];
        OTTER_CHECK(first_dim_size == second_dim_size, "Sizes of tensors must match except in dimension ", dimension, ". Expected size ", static_cast<long long>(first_dim_size), " but got size ", static_cast<long long>(second_dim_size), " for tensor number ", index, " in the list.");
    }
}

static bool should_skip(const Tensor& t) {
    return t.numel() == 0 && t.dim() == 1;
}

Tensor& cat_out(TensorList tensors, int64_t dim, Tensor& out) {
    check_cat_no_zero_dim(tensors);
    dim = legacy_cat_wrap_dim(dim, tensors);
    
    bool allContiguous = true;
    
    for (const auto i : otter::irange(tensors.size())) {
        auto lap = get_overlap_status(out, tensors[i]);
        OTTER_CHECK((lap != MemOverlapStatus::PARTIAL && lap != MemOverlapStatus::FULL), "unsupported operation: the input tensors cannot refer to any of the ", "output memory locations. Found overlap in input tensor ", i);
    }
    assert_no_internal_overlap(out);
    
    const Tensor* pnotSkippedTensor = [](const TensorList &tensors) -> const Tensor* {
        for (auto const &tensor : tensors) {
            if (should_skip(tensor)) {
                continue;
            }
            return &tensor;
        }
        return nullptr;
    }(tensors);
    
    if (!pnotSkippedTensor) {
        return out;
    }
    const Tensor& notSkippedTensor = *pnotSkippedTensor;
    
    OTTER_CHECK(tensors.size() > 0, "otter.cat(): expected a non-empty list of Tensors");
    OTTER_CHECK(dim <= notSkippedTensor.dim(), "otter.cat(): dimension ", dim, "out of range");
    
    bool reuse_iterator = true;
    bool no_type_promotion = true;
    // Check the type of the result
    no_type_promotion = out.dtype() == notSkippedTensor.dtype();
    
    int64_t cat_dim_size = 0;
    auto first_tensor_mem_format = tensors[0].suggest_memory_format();
    for (const auto i : otter::irange(tensors.size())) {
        auto const &tensor = tensors[i];
        if (should_skip(tensor)) {
            // don't use fast path for empty tensor
            allContiguous = false;
            continue;
        }
        check_cat_shape_except_dim(notSkippedTensor, tensor, dim, i);
        cat_dim_size += tensor.sizes()[dim];
    
        if (!tensor.is_contiguous(first_tensor_mem_format)) {
            allContiguous = false;
        }
    
        if (tensor.sizes() != notSkippedTensor.sizes() ||
            tensor.strides() != notSkippedTensor.strides()) {
            reuse_iterator = false;
        }
        if (tensor.dtype() != notSkippedTensor.dtype()) {
            no_type_promotion = false;
        }
    }
    // compute the size of the result
    auto result_size = notSkippedTensor.sizes().vec();
    result_size[dim] = cat_dim_size;
    
    if (out.sizes() != result_size) {
        out.resize_(result_size, first_tensor_mem_format);
    }
    
    if (out.numel() == 0) {
        return out;
    }
    
    // fast path for single thread when both inputs and result are contiguous and not empty
    allContiguous = allContiguous && out.is_contiguous(first_tensor_mem_format);
    bool use_serial_kernel = out.numel() < otter::GRAIN_SIZE || otter::get_num_threads() == 1;
    ScalarType dtype = notSkippedTensor.scalar_type();
    bool serial_dtype = (dtype == ScalarType::Double || dtype == ScalarType::Float);
    if (use_serial_kernel && allContiguous && no_type_promotion && serial_dtype) {
        cat_serial_stub(Device::CPU, out, tensors, dim);
        return out;
    }
    
    int64_t offset = 0;
    if (reuse_iterator && out.is_contiguous(first_tensor_mem_format) && no_type_promotion) {
        const auto& source_slice = notSkippedTensor;
        auto slice_dim_size = source_slice.sizes()[dim];
        auto result_slice = out.narrow(dim, 0, slice_dim_size);
        auto result_slice_data = result_slice.data_ptr();
        auto result_stride_bytes = out.stride(dim) * elementSize(out.scalar_type());

        auto iter = TensorIteratorConfig()
            .set_check_mem_overlap(false)  // Already checked above
            .resize_outputs(false)
            .add_output(result_slice)
            .add_input(source_slice)
            .enforce_safe_casting_to_output(true)
            .build();

        for (auto const &tensor : tensors) {
            if (should_skip(tensor)) {
                continue;
            }
            auto source_data = static_cast<char*>(tensor.data_ptr());
            auto result_data = static_cast<char*>(result_slice_data) + offset * result_stride_bytes;
            iter.unsafe_replace_operand(0, result_data);
            iter.unsafe_replace_operand(1, source_data);
            copy_stub(Device::CPU, iter, false);
            offset += slice_dim_size;
        }
    } else {
        for (auto const &tensor: tensors) {
            if (should_skip(tensor)) {
                continue;
            }
            auto slice_dim_size = tensor.sizes()[dim];
            auto result_slice = out.narrow(dim, offset, slice_dim_size);
    
            auto iter = TensorIteratorConfig()
                .set_check_mem_overlap(false)  // Already checked above
                .resize_outputs(false)
                .add_output(result_slice)
                .add_input(tensor)
                .promote_inputs_to_common_dtype(true)
                .cast_common_dtype_to_outputs(true)
                .enforce_safe_casting_to_output(true)
                .build();
            copy_stub(Device::CPU, iter, false);
            offset += slice_dim_size;
        }
    }
    
    return out;
}

Tensor cat(TensorList tensors, int64_t dim) {
    Tensor out = otter::empty({0}, tensors[0].options());
    otter::native::cat_out(tensors, dim, out);
    return out;
}

}   // end namespace native
    
}   // end namespace otter
