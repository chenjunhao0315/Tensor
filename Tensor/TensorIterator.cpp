//
//  TensorIterator.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "TensorIterator.hpp"
#include "TensorResize.hpp"
#include "Parallel.hpp"

namespace otter {

InlineTensorRef::InlineTensorRef() {
    static_assert(alignof(TensorRef) == alignof(TensorBase), "");
    static_assert(sizeof(TensorRef) == sizeof(TensorBase), "");
    new (data_.data()) TensorRef();
}

InlineTensorRef::~InlineTensorRef() {
    get()->~TensorRef();
}

const Tensor& InlineTensorRef::getTensor() const {
    return get()->getTensorRef();
}

namespace {

static TensorRef make_tensor_ref(const TensorBase &tensor) {
    if (tensor.defined()) {
        return TensorRef(tensor);
    } else {
        return TensorRef();
    }
}

inline void get_base_ptrs(char** ptrs, ArrayRef<OperandInfo> operands) {
    std::transform(operands.begin(), operands.end(), ptrs, [](const OperandInfo& op) {
        return static_cast<char*>(op.data);
    });
}

inline void get_strides(int64_t* strides, ArrayRef<OperandInfo> operands, int64_t ndim) {
    for (const auto dim : otter::irange(ndim)) {
        for (const auto arg : otter::irange(operands.size())) {
            *strides++ = operands[arg].stride_bytes[dim];
        }
    }
    // Always at least 2d strides to support 2d for_each loops
    if (ndim < 2) {
        const int64_t ntensors = operands.size();
        std::fill_n(strides, (2 - ndim) * ntensors, 0);
    }
}

}

void OperandInfo::exchange_tensor(MaybeOwned<TensorBase> &&new_tensor) {
    assert(!original_tensor_base_->defined());
    original_tensor_base_ = std::exchange(tensor_base_, new_tensor);
    *original_tensor_storage_ = std::exchange(*tensor_storage_, make_tensor_ref(*tensor_base_));
}

void OperandInfo::tensor(MaybeOwned<TensorBase> &&tensor) {
    tensor_base_ = std::move(tensor);
    *tensor_storage_ = make_tensor_ref(*tensor);
}

int64_t TensorIterator::numel() const {
    int64_t numel = 1;
    for (int64_t size: shape_) {
        numel *= size;
    }
    return numel;
}

void TensorIterator::initialize_operands(TensorIteratorConfig &config) {
    for (auto& tensor : config.tensors_) {
        operands_.emplace_back(std::move(tensor));
    }
    num_outputs_ = config.num_outputs_;
    
    for (const auto i : otter::irange(num_outputs_)) {
        operands_[i].is_output = true;
        const auto& output = this->tensor(i);
        if (output.defined())
            continue;
        
        for (const auto i : otter::irange(num_outputs_, ntensors())) {
            const auto& input = this->tensor(i);
            if (output.is_same(input)) {
                operands_[i].is_read_write = true;
            }
        }
    }
}

void TensorIterator::compute_shape(const TensorIteratorConfig& config) {
    all_ops_same_shape_ = true;
    has_scalars_ = false;
    has_tensors_ = false;
    
    for (auto& op : operands_) {
        if (!op.tensor_base().defined())
            continue;
        if (config.resize_outputs_ && op.is_output)
            continue;
        
        auto shape = op.tensor_base().sizes();
        if (shape.size() == 0) {
            has_scalars_ = true;
        } else {
            has_tensors_ = true;
        }
        
        if (has_scalars_ && has_tensors_) {
            all_ops_same_shape_ = false;
        }
        
        if (shape_.empty()) {
            shape_ = shape;
        } else if (!shape.equals(shape_)) {
            all_ops_same_shape_ = false;
            shape_ = infer_size_dimvector(shape_, shape);
        }
    }
}

void TensorIterator::compute_types(const TensorIteratorConfig &config) {
    Device common_device = Device::CPU;
    
    common_dtype_ = ScalarType::Undefined;
    ScalarType output_dtype = ScalarType::Undefined;
    bool has_different_input_dtypes = false;
    bool has_different_output_dtypes = false;
    bool has_undefined_outputs = false;
    
    for (auto& op : operands_) {
        if (!op.is_type_defined()) {
            has_undefined_outputs = true;
            
            if (has_undefined_outputs) {
                continue;
            }
        }
        
        if (!op.tensor_base().defined()) {
            if (op.is_output)
                fprintf(stderr, "Found undefined input tensor!");
            continue;
        }
        
        assert(op.target_dtype == op.current_dtype);
        
        if (!op.is_output) {
            if (op.target_dtype != common_dtype_) {
                if (common_dtype_ == ScalarType::Undefined) {
                    common_dtype_ = op.target_dtype;
                } else {
                    has_different_input_dtypes = true;
                }
            }
        } else {
            if (op.target_dtype != output_dtype) {
                if (output_dtype == ScalarType::Undefined) {
                    output_dtype = op.target_dtype;
                } else {
                    has_different_output_dtypes = true;
                }
            }
        }
    }
    
    assert(!(has_different_input_dtypes && (has_undefined_outputs)));
    
    if (!has_undefined_outputs) {
        common_dtype_ = has_different_input_dtypes ? ScalarType::Undefined : common_dtype_;
        return;
    }
    
    for (auto& op : operands_) {
        bool is_type_defined = op.is_type_defined();
        
        if (!is_type_defined) {
            op.target_dtype = common_dtype_;
        }
        
        if (!is_type_defined) {
            continue;
        }
        
        if (!op.tensor_base().defined()) {
            continue;
        }
        
        if (common_device == Device::CPU) {
//            if (config.cast_common_dtype_to_outputs_ && op.is_output && op.current_dtype != common_dtype_ && !is_meta_) {
//                TORCH_INTERNAL_ASSERT(op.tensor_base().defined());
//                op.exchange_tensor(c10::MaybeOwned<TensorBase>::owned(
//                                                                      at::empty_like(op.tensor(),
//                                                                                     op.tensor_base().options().dtype(common_dtype_),
//                                                                                     LEGACY_CONTIGUOUS_MEMORY_FORMAT)));
//                if (!names_.empty()) {
//                    namedinference::propagate_names(op.tensor_base(), names_);
//                }
//                op.current_dtype = common_dtype_;
//                op.target_dtype = common_dtype_;
//            }
            
            if (config.promote_inputs_to_common_dtype_ && !op.is_output && op.current_dtype != common_dtype_) {
                op.exchange_tensor(MaybeOwned<TensorBase>::owned(op.tensor().to(common_dtype_)));
                op.current_dtype = common_dtype_;
                op.target_dtype = common_dtype_;
            }
        }
    }
}

void TensorIterator::mark_resize_outputs(const TensorIteratorConfig& config) {
    for (const auto i : otter::irange(num_outputs_)) {
        const auto& output = this->tensor(i);
        if (output.defined() && !output.sizes().equals(shape_)) {
            if (config.resize_outputs_ && !operands_[i].is_read_write) {
                operands_[i].will_resize = true;
            }
        }
    }
}

void TensorIterator::compute_strides(const TensorIteratorConfig &config) {
    for (auto& op : operands_) {
        if (op.tensor_base().defined()) {
            IntArrayRef original_shape = op.tensor_base().sizes();
            auto original_stride = op.tensor_base().strides();
            auto element_size_in_bytes = op.tensor_base().itemsize();
            auto offset = ndim() - original_shape.size();
            if (offset > 0) {
                op.stride_bytes.resize(ndim(), 0);
            } else {
                op.stride_bytes.resize(ndim());
            }
            
            for (const auto i : otter::irange(original_shape.size())) {
                if (original_shape[i] == 1 && shape_[offset + i] != 1) {
                    op.stride_bytes[offset + i] = 0;
                } else {
                    op.stride_bytes[offset + i] = original_stride[i] * element_size_in_bytes;
                }
            }
        }
    }
}

void TensorIterator::reorder_dimensions() {
    permutation_.resize(ndim());
    if (ndim() == 1) {
        permutation_[0] = 0;
        return;
    }
    
    std::iota(permutation_.rbegin(), permutation_.rend(), 0);
    
    auto should_swap = [&](size_t dim0, size_t dim1) {
        for (const auto arg : otter::irange(ntensors())) {
            if (operands_[arg].stride_bytes.empty() || operands_[arg].will_resize) {
                continue;
            }
            int64_t stride0 = operands_[arg].stride_bytes[dim0];
            int64_t stride1 = operands_[arg].stride_bytes[dim1];
            if (operands_[arg].is_output) {
                if ((stride0 == 0) != (stride1 == 0)) {
                    return stride1 == 0 ? 1 : -1;
                }
            }
            if (stride0 == 0 || stride1 == 0) {
                continue;
            } else if (stride0 < stride1) {
                return -1;
            } else  if (stride0 > stride1) {
                return 1;
            } else {
                auto t_dim0 = shape_[dim0];
                auto t_dim1 = shape_[dim1];
                
                if (t_dim0 > t_dim1) {
                    return 1;
                }
            }
        }
        return 0;
    };
    
    for (const auto i : otter::irange(1, ndim())) {
        int dim1 = i;
        for (int dim0 = i - 1; dim0 >= 0; dim0--) {
            int comparison = should_swap(permutation_[dim0], permutation_[dim1]);
            if (comparison > 0) {
                std::swap(permutation_[dim0], permutation_[dim1]);
                dim1 = dim0;
            } else if (comparison < 0) {
                break;
            }
        }
    }
    
    this->permute_dimensions(permutation_);
}

void TensorIterator::permute_dimensions(IntArrayRef permutation) {
    assert(permutation.size() == ndim());
    
    auto reorder = [permutation](IntArrayRef data) {
        auto res = DimVector(data.size(), 0);
        for (const auto i : otter::irange(permutation.size())) {
            res[i] = data[permutation[i]];
        }
        return res;
    };
    
    // Update shape and strides
    shape_ = reorder(shape_);
    for (auto& op : operands_) {
        if (op.stride_bytes.size() > 0) {
            op.stride_bytes = reorder(op.stride_bytes);
        }
    }
}

TensorOptions original_options(const OperandInfo& op) {
    if (op.original_tensor_base().defined()) {
        return op.original_tensor_base().options();
    } else {
        return op.options();
    }
}

void TensorIterator::allocate_or_resize_outputs() {
    for (const auto i : otter::irange(num_outputs_)) {
        auto& op = operands_[i];
        if (!op.tensor_base().defined() || op.will_resize) {
            int element_size = static_cast<int>(elementSize(op.target_dtype));
            op.stride_bytes = compatible_stride(element_size);
            // check if permutation is just an inverted order
            bool inverted = true;
            for (const auto j : otter::irange(ndim())) {
                if (permutation_[j] != ndim() - j - 1) {
                    inverted = false;
                    break;
                }
            }
            auto tensor_shape = invert_permutation(shape_);
            if (inverted) {
                set_output(i, tensor_shape, {}, original_options(op));
            } else {
                auto tensor_stride = invert_permutation(op.stride_bytes);
                for (const auto dim : otter::irange(ndim())) {
                    tensor_stride[dim] /= element_size;
                }
                set_output(i, tensor_shape, tensor_stride, original_options(op));
            }
            op.current_dtype = op.target_dtype;
        } else if (op.tensor_base().defined()) {
            set_output(i, op.tensor_base().sizes(), {}, original_options(op));
        }
    }
}

TensorIterator::StrideVector TensorIterator::compatible_stride(int element_size) const {
    auto stride = StrideVector();
    int64_t next_stride = element_size;
    for (const auto dim : otter::irange(ndim())) {
        stride.push_back(next_stride);
        next_stride *= shape_[dim];
    }
    return stride;
}

DimVector TensorIterator::invert_permutation(IntArrayRef input) const {
    assert(input.size() == permutation_.size());
    auto res = DimVector(input.size());
    for (const auto dim : otter::irange(ndim())) {
        res[permutation_[dim]] = input[dim];
    }
    return res;
}

void TensorIterator::set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) {
    assert(output_idx < num_outputs_);
    auto& op = operands_[output_idx];
    const auto &t = maybe_get_output(output_idx);
    assert(t.defined());
    
    if (!op.tensor_base().defined()) {
        op.tensor(otter::MaybeOwned<TensorBase>::borrowed(t));
        assert(op.target_dtype == t.scalar_type());
    } else if (op.will_resize) {
        if (op.original_tensor_base().defined()) {
            assert(op.original_tensor_base().is_same(t));
            assert(!op.tensor_base().is_same(t));
            TensorRef tensor(op.tensor());
            resize_output(*tensor, sizes);
            if (!strides.empty()) {
                tensor->as_strided_(sizes, strides);
            }
        }
    }
}

const Tensor& TensorIterator::maybe_get_output(int64_t output_idx) {
    return output(static_cast<int>(output_idx));
}

void TensorIterator::build(TensorIteratorConfig &config) {
    // Put all tensor into operands pool
    this->initialize_operands(config);
    // Compute the proper output shape
    this->compute_shape(config);
    // Mark the output tensor which need to be resized
    this->mark_resize_outputs(config);
    
    this->compute_types(config);
    // Compute the proper output strides
    this->compute_strides(config);
    
    this->reorder_dimensions();
    
    this->allocate_or_resize_outputs();
    
    for (auto& op : operands_) {
        assert(op.tensor_base().defined());
        op.data = op.tensor_base().raw_data();
    }
    
    int64_t ndim_offsets = (ndim() ? ndim() : 1);
    view_offsets_ = DimVector(ndim_offsets, 0);
}

void TensorIterator::for_each(loop2d_t loop, int64_t grain_size) {
    int64_t numel = this->numel();
    if (numel == 0) {
        return;
    } else if (numel < grain_size || otter::get_num_threads() == 1) {
        return serial_for_each(loop, {0, numel});
    } else {
        otter::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
            serial_for_each(loop, {begin, end});
        });
    }
}

void TensorIterator::serial_for_each(loop2d_t loop, Range range) const {
    if (range.size() == 0) {
        return;
    }
    
    const auto ntensors = this->ntensors();
    const auto ndim = this->ndim();
    
    otter::SmallBuffer<char*, 4> ptrs(ntensors);
    otter::SmallBuffer<int64_t, 8> strides(ntensors * std::max(ndim, 2));
    
    otter::get_base_ptrs(ptrs.data(), operands_);
    otter::get_strides(strides.data(), operands_, ndim);
    
    otter::internal::serial_for_each(shape_, strides, ptrs.data(), ptrs.size(), loop, range);
}

TensorIterator TensorIterator::binary_op(TensorBase& out, const TensorBase& a, const TensorBase& b) {
    TensorIterator iter;
    iter.build_binary_op(out, a, b);
    return iter;
}

TensorIterator TensorIterator::borrowing_binary_op(
                                                   const TensorBase& out, const TensorBase& a, const TensorBase& b) {
    TensorIterator iter;
    iter.build_borrowing_binary_op(out, a, b);
    return iter;
}

#define BINARY_OP_CONFIG()                              \
TensorIteratorConfig()

void TensorIterator::build_binary_op(const TensorBase &out, const TensorBase &a, const TensorBase &b) {
    this->build(BINARY_OP_CONFIG().add_owned_output(out).add_owned_input(a).add_owned_input(b));
}

void TensorIterator::build_borrowing_binary_op(const TensorBase& out, const TensorBase& a, const TensorBase& b) {
    this->build(BINARY_OP_CONFIG().add_output(out).add_input(a).add_input(b));
}

#define UNARY_FLOAT_OP_CONFIG()                                         \
TensorIteratorConfig()                                              \
.promote_inputs_to_common_dtype(true)

void TensorIterator::build_unary_float_op(const TensorBase &out, const TensorBase &a) {
    this->build(UNARY_FLOAT_OP_CONFIG().add_owned_output(out).add_owned_input(a));
}

void TensorIterator::build_borrowing_unary_float_op(const TensorBase &out, const TensorBase &a) {
    this->build(UNARY_FLOAT_OP_CONFIG().add_output(out).add_input(a));
}

TensorIteratorConfig& TensorIteratorConfig::add_owned_output(const TensorBase& output) {
    assert(num_inputs_ == 0);
    tensors_.push_back(otter::MaybeOwned<TensorBase>::owned(otter::in_place, output));
    num_outputs_++;
    return *this;
}

TensorIteratorConfig& TensorIteratorConfig::add_owned_input(const TensorBase& input) {
    tensors_.push_back(otter::MaybeOwned<TensorBase>::owned(otter::in_place, input));
    num_inputs_++;
    return *this;
}

TensorIteratorConfig& TensorIteratorConfig::add_borrowed_input(const TensorBase &input) {
    tensors_.push_back(MaybeOwned<TensorBase>::borrowed(input));
    num_inputs_++;
    
    return *this;
}

TensorIteratorConfig& TensorIteratorConfig::add_borrowed_output(const TensorBase &output) {
    tensors_.push_back(MaybeOwned<TensorBase>::borrowed(output));
    num_outputs_++;
    
    return *this;
}

DimCounter::DimCounter(IntArrayRef shape, Range range) : shape_(shape), range_(range), values_(shape.size()), offset_(range.begin) {
    
    std::fill(values_.begin(), values_.end(), 0);
    if (range.begin == 0) return;
    
    int64_t linear_offset = range.begin;
    int64_t ndim = values_.size();
    for (const auto dim : otter::irange(ndim)) {
        int64_t size = shape[dim];
        if (size > 0) {
            values_[dim] = linear_offset % size;
            linear_offset /= size;
        }
    }
    assert(linear_offset == 0);
}

std::array<int64_t, 2> DimCounter::max_2d_step() const {
    int64_t step0 = std::min(shape_[0] - values_[0], range_.end - offset_);
    int64_t step1 = 1;
    if (step0 == shape_[0] && shape_.size() >= 1) {
        step1 = std::min(shape_[1] - values_[1], (range_.end - offset_) / shape_[0]);
    }
    return {step0, step1};
}

bool DimCounter::is_done() const {
    return offset_ >= range_.end;
}

void DimCounter::increment(const std::array<int64_t, 2>& step) {
    offset_ += step[0] * step[1];
    int64_t ndim = values_.size();
    int64_t overflow = step[0];
    int i = 0;
    if (step[1] != 1) {
        assert(step[0] == shape_[0] && values_[0] == 0);
        i = 1;
        overflow = step[1];
    }
    for (; i < ndim && overflow > 0; i++) {
        auto size = shape_[i];
        auto prev = values_[i];
        auto value = prev + overflow;
        if (value >= size) {
            overflow = 1;
            value -= size;
            assert(value < size);
        } else {
            overflow = 0;
        }
        values_[i] = value;
    }
    assert(overflow == 0 || overflow == 1);
}


}   // end namespace otter
