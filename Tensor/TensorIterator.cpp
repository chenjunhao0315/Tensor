//
//  TensorIterator.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "TensorFactory.hpp"
#include "TensorIterator.hpp"
#include "TensorResize.hpp"
#include "Parallel.hpp"
#include "ExpandUtils.hpp"
#include "MemoryOverlap.hpp"
#include "DefaultDtype.hpp"
#include "TypeProperties.hpp"

namespace otter {

using DimMask = TensorIterator::DimMask;
using PtrVector = TensorIterator::PtrVector;
using loop2d_t = TensorIterator::loop2d_t;
using StrideVector = TensorIterator::StrideVector;

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
    *tensor_storage_ = make_tensor_ref(*tensor_base_);
}

void OperandInfo::restore_original_tensor() {
    assert(original_tensor_base_->defined());
    tensor_base_ = std::move(original_tensor_base_);
    *tensor_storage_ = std::exchange(*original_tensor_storage_, TensorRef{});
}

void* TensorIterator::data_ptr(int arg) const {
    return operands_[arg].data;
}

bool TensorIterator::is_contiguous() const {
    if (numel() == 1) {
        return true;
    }
    if (ndim() != 1) {
        return false;
    }
    return has_contiguous_first_dim();
}

bool TensorIterator::is_dim_reduced(int dim) const {
    for (auto& op : operands_) {
        if (op.is_output && op.stride_bytes[dim] == 0 && shape_[dim] > 1) {
            return true;
        }
    }
    return false;
}

StrideVector TensorIterator::get_strides() const {
    const auto dim = ndim();
    StrideVector strides(std::max(dim, 2) * ntensors());
    otter::get_strides(strides.data(), operands_, dim);
    return strides;
}

StrideVector TensorIterator::get_dim_strides(int dim) const {
    auto dims = ndim();
    auto inner_strides = StrideVector();
    for (auto& op : operands_) {
        inner_strides.push_back(dims == 0 ? 0 : op.stride_bytes[dim]);
    }
    return inner_strides;
}


void TensorIterator::coalesce_dimensions() {
    if (ndim() <= 1) {
        return;
    }
    // We can coalesce two adjacent dimensions if either dim has size 1 or if:
    // shape[n] * stride[n] == shape[n + 1].
    auto can_coalesce = [&](int dim0, int dim1) {
        auto shape0 = shape_[dim0];
        auto shape1 = shape_[dim1];
        if (shape0 == 1 || shape1 == 1) {
            return true;
        }
        for (const auto i : otter::irange(ntensors())) {
            auto& stride = operands_[i].stride_bytes;
            if (shape0 * stride[dim0] != stride[dim1]) {
                return false;
            }
        }
        return true;
    };
    // replace each operands stride at dim0 with its stride at dim1
    auto replace_stride = [&](int dim0, int dim1) {
        for (const auto i : otter::irange(ntensors())) {
            auto& stride = operands_[i].stride_bytes;
            stride[dim0] = stride[dim1];
        }
    };
    int prev_dim = 0;
    for (const auto dim : otter::irange(1, ndim())) {
        if (can_coalesce(prev_dim, dim)) {
            if (shape_[prev_dim] == 1) {
                replace_stride(prev_dim, dim);
            }
            shape_[prev_dim] *= shape_[dim];
        } else {
            prev_dim++;
            if (prev_dim != dim) {
                replace_stride(prev_dim, dim);
                shape_[prev_dim] = shape_[dim];
            }
        }
    }
    shape_.resize(prev_dim + 1);
    for (const auto i : otter::irange(ntensors())) {
        operands_[i].stride_bytes.resize(ndim());
    }
    has_coalesced_dimensions_ = true;
}

int TensorIterator::num_reduce_dims() const {
    int count = 0;
    for (const auto dim : otter::irange(ndim())) {
        if (operands_[0].stride_bytes[dim] == 0) {
            count++;
        }
    }
    return count;
}

int64_t TensorIterator::num_output_elements() const {
    int64_t elem = 1;
    for (const auto dim : otter::irange(ndim())) {
        if (operands_[0].stride_bytes[dim] != 0 || shape_[dim] == 0)  {
            elem *= shape_[dim];
        }
    }
    return elem;
}

SmallVector<char*, 4> TensorIterator::get_base_ptrs() const {
    auto ptrs = SmallVector<char*, 4>(ntensors());
    otter::get_base_ptrs(ptrs.data(), operands_);
    return ptrs;
}

int64_t TensorIterator::numel() const {
    int64_t numel = 1;
    for (int64_t size: shape_) {
        numel *= size;
    }
    return numel;
}

void TensorIterator::narrow(int dim, int64_t start, int64_t size) {
    OTTER_INTERNAL_ASSERT(dim < ndim() && size >= 1);
    shape_[dim] = size;
    view_offsets_[dim] += start;
    for (auto& op : operands_) {
        op.data = ((char*)op.data) + op.stride_bytes[dim] * start;
    }
    if (size == 1 && !is_reduction_) {
        coalesce_dimensions();
    }
}
void TensorIterator::select_all_keeping_dim(int start_dim, IntArrayRef indices) {
    OTTER_INTERNAL_ASSERT(start_dim <= ndim());
    for (const auto i : otter::irange(start_dim, ndim())) {
        for (auto& op : operands_) {
            op.data = ((char*)op.data) + op.stride_bytes[i] * indices[i - start_dim];
        }
        shape_[i] = 1;
    }
}

void TensorIterator::unsafe_replace_operand(int arg, void* data) {
    operands_[arg].data = data;
}

void TensorIterator::initialize_operands(TensorIteratorConfig &config) {
    for (auto& tensor : config.tensors_) {
        operands_.emplace_back(std::move(tensor));
    }
    num_outputs_ = config.num_outputs_;
    
    for (const auto i : otter::irange(num_outputs_)) {
        operands_[i].is_output = true;
        const auto& output = this->tensor(i);
        if (!output.defined())
            continue;
        
        for (const auto i : otter::irange(num_outputs_, ntensors())) {
            const auto& input = this->tensor(i);
            if (output.is_same(input)) {
                operands_[i].is_read_write = true;
            }
        }
    }
}

void TensorIterator::compute_mem_overlaps(const TensorIteratorConfig& config) {
    if (!config.check_mem_overlap_) {
        return;
    }
    for (const auto i : otter::irange(num_outputs_)) {
        const auto& output = tensor_base(i);
        if (!output.defined()) continue;
        assert_no_internal_overlap(output);
        for (const auto j : otter::irange(num_outputs_, ntensors())) {
            const auto& input = tensor_base(j);
            if (!input.is_same(output)) {
                assert_no_partial_overlap(output, input);
            }
        }
    }
}

void TensorIterator::compute_shape(const TensorIteratorConfig& config) {
    if (!config.static_shape_.empty()) {
        shape_ = config.static_shape_;
        return;
    }
    
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

ScalarType TensorIterator::compute_common_dtype() {
    otter::native::ResultTypeState state = {};
    for (const auto& op : operands_) {
        if (op.is_output) {
            continue;
        }
        state = otter::native::update_result_type_state(op.tensor(), state);
    }
    common_dtype_ = otter::native::result_type(state);
    OTTER_INTERNAL_ASSERT(common_dtype_ != ScalarType::Undefined);
    return common_dtype_;
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
            if (config.static_dtype_ != ScalarType::Undefined) {
                op.target_dtype = config.static_dtype_;
            } else {
                has_undefined_outputs = true;
            }
            
            if (has_undefined_outputs) {
                continue;
            }
        }
        
        if (!op.tensor_base().defined()) {
            OTTER_INTERNAL_ASSERT(op.is_output, "Found undefined input tensor!");
            continue;
        }
        
        OTTER_INTERNAL_ASSERT(op.target_dtype == op.current_dtype);
        
        if (common_device == Device::CPU) {
            common_device = op.tensor_base().device();
        }
        
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
    
    OTTER_INTERNAL_ASSERT(!(has_different_input_dtypes && !config.promote_inputs_to_common_dtype_ &&
                            (has_undefined_outputs || config.enforce_safe_casting_to_output_ ||
                             config.cast_common_dtype_to_outputs_)));
    
    if (config.check_all_same_dtype_ &&
        (has_different_input_dtypes || has_different_output_dtypes ||
         (common_dtype_ != output_dtype && output_dtype != ScalarType::Undefined))) {
        for (auto& op : operands_) {
            if (!op.tensor_base().defined()) {
                continue;
            }
            OTTER_INTERNAL_ASSERT(op.target_dtype == common_dtype_);
        }
    }
    
    if (!has_undefined_outputs && !config.check_all_same_device_ &&
        !config.promote_inputs_to_common_dtype_ && !config.cast_common_dtype_to_outputs_ &&
        !config.enforce_safe_casting_to_output_) {
        common_dtype_ = has_different_input_dtypes ? ScalarType::Undefined : common_dtype_;
        return;
    }
    
    if (has_different_input_dtypes && config.promote_inputs_to_common_dtype_) {
        common_dtype_ = compute_common_dtype();
    }
    
    if (config.promote_integer_inputs_to_float_ && otter::isIntegralType(common_dtype_, true)) {
        common_dtype_ = otter::get_default_dtype_as_scalartype();
    }
    
    common_device_ = common_device;
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
            if (config.cast_common_dtype_to_outputs_ && op.is_output && op.current_dtype != common_dtype_) {
                OTTER_INTERNAL_ASSERT(op.tensor_base().defined());
                op.exchange_tensor(MaybeOwned<TensorBase>::owned(otter::empty_like(op.tensor(), op.tensor_base().options().dtype(common_dtype_))));
                op.current_dtype = common_dtype_;
                op.target_dtype = common_dtype_;
            }
            
            if (config.promote_inputs_to_common_dtype_ && !op.is_output && op.current_dtype != common_dtype_) {
                op.exchange_tensor(MaybeOwned<TensorBase>::owned(op.tensor().to(common_dtype_)));
                op.current_dtype = common_dtype_;
                op.target_dtype = common_dtype_;
            }
        }
    }
}

void TensorIterator::mark_resize_outputs(const TensorIteratorConfig& config) {
    if (!config.static_shape_.empty()) {
        return;
    }
    for (const auto i : otter::irange(num_outputs_)) {
        const auto& output = this->tensor(i);
        if (output.defined() && !output.sizes().equals(shape_)) {
            if (config.resize_outputs_ && !operands_[i].is_read_write) {
                operands_[i].will_resize = true;
                continue;
            }
            // for reduction, output size does not match shape_, as output is reduced size, and shape_ is size of the input
            OTTER_CHECK(is_reduction_,  "output with shape ", output.sizes(), " doesn't match the broadcast shape ", shape_);
        }
    }
}

void TensorIterator::compute_strides(const TensorIteratorConfig &/*config*/) {
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
    
    if (enforce_linear_iteration_) {
        permute_dimensions(permutation_);
        return;
    }
    
    auto should_swap = [&](size_t dim0, size_t dim1) {
        for (const auto arg : otter::irange(ntensors())) {
            if (operands_[arg].stride_bytes.empty() || operands_[arg].will_resize) {
                continue;
            }
            int64_t stride0 = operands_[arg].stride_bytes[dim0];
            int64_t stride1 = operands_[arg].stride_bytes[dim1];
            if (is_reduction_ && operands_[arg].is_output) {
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

void TensorIterator::cast_outputs() {
    for (auto& op : operands_) {
        if (op.is_output && op.original_tensor_base().defined() && op.original_tensor_base().scalar_type() != op.current_dtype) {
            const auto &original_tensor = op.original_tensor();
            const auto &tensor = op.tensor();
            if (original_tensor.sizes() != tensor.sizes()){
                original_tensor.resize_as_(tensor).as_strided_(tensor.sizes(), tensor.strides());
            }
            original_tensor.copy_(tensor);
            op.restore_original_tensor();
        }
    }
}

void TensorIterator::set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions /*options*/) {
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
            otter::native::resize_output(*tensor, sizes);
            if (!strides.empty()) {
                tensor->as_strided_(sizes, strides);
            }
        }
    }
    op.current_dtype = op.tensor_base().scalar_type();
}

const Tensor& TensorIterator::maybe_get_output(int64_t output_idx) {
    return output(static_cast<int>(output_idx));
}

void TensorIterator::build(TensorIteratorConfig &config) {
    is_reduction_ = config.is_reduction_;
    enforce_linear_iteration_ = config.enforce_linear_iteration_;
    
    // Put all tensor into operands pool
    this->initialize_operands(config);
    // Check memory overlap
    this->compute_mem_overlaps(config);
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

#define BINARY_FLOAT_OP_CONFIG()                \
TensorIteratorConfig()                          \
.set_check_mem_overlap(true)                    \
.promote_inputs_to_common_dtype(true)           \
.cast_common_dtype_to_outputs(true)             \
.enforce_safe_casting_to_output(true)           \
.promote_integer_inputs_to_float(true)

void TensorIterator::build_binary_float_op(const TensorBase& out, const TensorBase& a, const TensorBase& b) {
    this->build(BINARY_FLOAT_OP_CONFIG()
                .add_owned_output(out)
                .add_owned_input(a)
                .add_owned_input(b));
}

void TensorIterator::build_borrowing_binary_float_op(const TensorBase& out, const TensorBase& a, const TensorBase& b) {
    this->build(BINARY_FLOAT_OP_CONFIG()
                .add_output(out)
                .add_input(a)
                .add_input(b));
}

#define BINARY_OP_CONFIG()                          \
TensorIteratorConfig()                              \
.set_check_mem_overlap(true)                        \
.promote_inputs_to_common_dtype(true)               \
.cast_common_dtype_to_outputs(true)                 \
.enforce_safe_casting_to_output(true)               \

void TensorIterator::build_binary_op(const TensorBase &out, const TensorBase &a, const TensorBase &b) {
    this->build(BINARY_OP_CONFIG().add_owned_output(out).add_owned_input(a).add_owned_input(b));
}

void TensorIterator::build_borrowing_binary_op(const TensorBase& out, const TensorBase& a, const TensorBase& b) {
    this->build(BINARY_OP_CONFIG().add_output(out).add_input(a).add_input(b));
}

#define UNARY_FLOAT_OP_CONFIG()                                         \
TensorIteratorConfig()                                                  \
.set_check_mem_overlap(true)                                            \
.promote_inputs_to_common_dtype(true)                                   \
.promote_inputs_to_common_dtype(true)                                   \
.cast_common_dtype_to_outputs(true)                                     \
.enforce_safe_casting_to_output(true)                                   \
.promote_integer_inputs_to_float(true)

void TensorIterator::build_unary_float_op(const TensorBase &out, const TensorBase &a) {
    this->build(UNARY_FLOAT_OP_CONFIG().add_owned_output(out).add_owned_input(a));
}

void TensorIterator::build_borrowing_unary_float_op(const TensorBase &out, const TensorBase &a) {
    this->build(UNARY_FLOAT_OP_CONFIG().add_output(out).add_input(a));
}

#define UNARY_OP_CONFIG()                                \
TensorIteratorConfig()                                 \
.set_check_mem_overlap(true)                         \
.cast_common_dtype_to_outputs(false)                 \
.enforce_safe_casting_to_output(false)               \
.check_all_same_dtype(true)

void TensorIterator::build_unary_op(const TensorBase& out, const TensorBase& a) {
    build(UNARY_OP_CONFIG()
          .add_owned_output(out)
          .add_owned_input(a));
}

void TensorIterator::build_borrowing_unary_op(const TensorBase& out, const TensorBase& a) {
    build(UNARY_OP_CONFIG()
          .add_output(out)
          .add_input(a));
}

#define NULLARY_OP_CONFIG()                                     \
TensorIteratorConfig()                                        \
.check_all_same_dtype(false)                                \
.resize_outputs(false)

TensorIterator TensorIterator::nullary_op(TensorBase& out) {
    return NULLARY_OP_CONFIG()
        .add_owned_output(out)
        .build();
}

TensorIterator TensorIterator::borrowing_nullary_op(const TensorBase& out) {
    return NULLARY_OP_CONFIG()
        .add_output(out)
        .build();
}

TensorIterator TensorIterator::reduce_op(TensorBase& out, const TensorBase& a) {
    OTTER_INTERNAL_ASSERT(out.defined());
    return TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .add_owned_output(out)
        .add_owned_input(a)
        .resize_outputs(false)
        .is_reduction(true)
        // TODO: not supporting casting to outputs is only really necessary for arg{min,max}
        .promote_inputs_to_common_dtype(true)
        .build();
}
TensorIterator TensorIterator::reduce_op(TensorBase& out1, TensorBase& out2, const TensorBase& a) {
    OTTER_INTERNAL_ASSERT(out1.defined());
    OTTER_INTERNAL_ASSERT(out2.defined());
//    OTTER_CHECK(a.device() == out1.device() && out1.device() == out2.device(),
//        "reduce_op(): expected input and both outputs to be on same device, but input is on ", a.device(),
//        ", output1 is on ", out1.device(), " and output2 is on", out2.device());
    OTTER_CHECK(out1.dim() == out2.dim(), "reduce_op(): expected both outputs to have same number of dims, but  output1 has ", out1.dim(),
        " and output2 has ", out2.dim());
    OTTER_CHECK(out1.sizes() == out2.sizes(), "reduce_op(): expected both outputs to have same sizes, but output1   has ", out1.sizes(),
        " and output2 has ", out2.sizes());
    OTTER_CHECK(out1.strides() == out2.strides(), "reduce_op(): expected both outputs to have same strides, but     output1 has ", out1.strides(),
        " and output2 has ", out2.strides());
    return TensorIteratorConfig()
        .set_check_mem_overlap(false)
        .add_owned_output(out1)
        .add_owned_output(out2)
        .add_owned_input(a)
        .resize_outputs(false)
        .is_reduction(true)
        .check_all_same_dtype(false)
        .build();
}

static void set_up_comparison_op_config(TensorIteratorConfig& config, const TensorBase& out) {
    config.set_check_mem_overlap(true);
    config.allow_cpu_scalars(true);
    config.promote_inputs_to_common_dtype(true);
    
    // When 'out' isn't defined (e.g. for the functional operator 'a == b'), we
    // want the output to be bool. Otherwise (e.g. 'torch.eq(a, b, out=c)') we
    // don't coerce the output.
    if (!out.defined()) {
        config.declare_static_dtype(otter::ScalarType::Bool);
    }
    
    // Note [special-case bool outputs]
    // We explicitly don't call `cast_common_dtype_to_outputs` when the output tensor
    // has `bool` dtype. This is a performance optimization: the functional
    // version of all comparison/logical ops uses a bool output tensor, and we'd like to
    // avoid creating a temporary copy of the output.
    // However, note that all kernels using this TensorIterator will need to special-case when
    // the output tensor has bool dtype, and provide a lambda of type (scalar_t, scalar_t -> bool).
    if (out.defined() && out.scalar_type() != otter::ScalarType::Bool) {
        config.cast_common_dtype_to_outputs(true);
    }
}

void TensorIterator::build_comparison_op(const TensorBase& out, const TensorBase& a, const TensorBase& b) {
    TensorIteratorConfig config;
    set_up_comparison_op_config(config, out);
    
    config.add_owned_output(out);
    config.add_owned_input(a);
    config.add_owned_input(b);
    build(config);
}

void TensorIterator::build_borrowing_comparison_op(const TensorBase& out, const TensorBase& a, const TensorBase& b) {
    TensorIteratorConfig config;
    set_up_comparison_op_config(config, out);
    
    config.add_borrowed_output(out);
    config.add_borrowed_input(a);
    config.add_borrowed_input(b);
    build(config);
}

void TensorIterator::build_borrowing_except_last_argument_comparison_op(const TensorBase& out, const TensorBase& a, const TensorBase& b) {
    TensorIteratorConfig config;
    set_up_comparison_op_config(config, out);
    
    config.add_borrowed_output(out);
    config.add_borrowed_input(a);
    config.add_owned_input(b);
    build(config);
}

TensorIterator TensorIterator::binary_op(TensorBase& out, const TensorBase& a, const TensorBase& b) {
    TensorIterator iter;
    iter.build_binary_op(out, a, b);
    return iter;
}

TensorIterator TensorIterator::borrowing_binary_op(const TensorBase& out, const TensorBase& a, const TensorBase& b) {
    TensorIterator iter;
    iter.build_borrowing_binary_op(out, a, b);
    return iter;
}

TensorIterator TensorIterator::unary_op(TensorBase& out, const TensorBase& a) {
    TensorIterator iter;
    iter.build_unary_op(out, a);
    return iter;
}

// Reduce
static bool use_two_pass_reduction(TensorIterator& iter);
static void two_pass_reduction(TensorIterator& iter, loop2d_t loop);
static void parallel_dim_reduction(TensorIterator& iter, loop2d_t loop);

void TensorIterator::parallel_reduce(loop2d_t loop) {
    OTTER_CHECK(ntensors() == 2, "parallel_reduce only supports one input and one output");
    int64_t numel = this->numel();
    if (numel < otter::GRAIN_SIZE || otter::get_num_threads() == 1 ||
        otter::in_parallel_region()) {
        serial_for_each(loop, {0, numel});
    } else if (use_two_pass_reduction(*this)) {
        two_pass_reduction(*this, loop);
    } else {
        parallel_dim_reduction(*this, loop);
    }
}
static bool use_two_pass_reduction(TensorIterator& iter) {
    return iter.output(0).numel() == 1;
}
static void two_pass_reduction(TensorIterator& iter, loop2d_t loop) {
    const int max_threads = otter::get_num_threads();
    auto dst = iter.output(0);
    auto unsqueezed = dst.unsqueeze(0);
    auto buffer_shape = DimVector(unsqueezed.sizes());
    buffer_shape[0] = max_threads;
    auto buffer = otter::empty(buffer_shape, dst.options());
    // Fill with the identity
    buffer.copy_(unsqueezed);
    auto buffer_stride = buffer.strides()[0] * buffer.itemsize();
    auto buffer_0 = buffer[0];
    auto first_reduce = TensorIterator::reduce_op(buffer_0, iter.input(0));
    OTTER_INTERNAL_ASSERT(first_reduce.output(0).is_alias_of(buffer_0));
    otter::parallel_for(0, iter.numel(), otter::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        const auto thread_num = otter::get_thread_num();
        auto shape = first_reduce.shape();
        auto strides = first_reduce.get_strides();
        // Bump output ptr so each thread has its own ouput slice
        auto base_ptrs = first_reduce.get_base_ptrs();
        base_ptrs[0] += buffer_stride * thread_num;
        internal::serial_for_each(shape, strides, base_ptrs.data(),
                                      base_ptrs.size(), loop, {begin, end});
    });
    auto final_reduce = TensorIterator::reduce_op(unsqueezed, buffer);
    final_reduce.for_each(loop);
}
/// Chooses a dimension over which to parallelize. Prefers the outer-most
/// dimension thats larger than the number of available threads.
static int find_split_dim(TensorIterator& iter) {
    int num_threads = otter::get_num_threads();
    auto shape = iter.shape();
    // start with the outer-most dimension
    int best_dim = iter.ndim() - 1;
    for (int dim = best_dim; dim >= 0 && !iter.is_dim_reduced(dim); dim--) {
        if (shape[dim] >= num_threads) {
            return dim;
        } else if (shape[dim] > shape[best_dim]) {
            best_dim = dim;
        }
    }
    assert(!iter.is_dim_reduced(best_dim));
    return best_dim;
}
static std::tuple<int64_t, int64_t>
round_columns(TensorIterator& iter, int dim, int multiple, int64_t begin, int64_t end) {
    begin = begin - (begin % multiple);
    if (end != iter.shape()[dim]) {
        // only round the 'end' column down if it's not the final column
        end = end - (end % multiple);
    }
    return std::make_tuple(begin, end);
}
static void parallel_dim_reduction(TensorIterator& iter, loop2d_t loop) {
    assert(iter.ndim() >= 1);
    int dim = find_split_dim(iter);
    int64_t cols = iter.shape()[dim];
    int element_size = iter.element_size(/*arg=*/1);
    bool should_round_columns = iter.strides(1)[dim] == element_size;
    otter::parallel_for(0, cols, 1, [&](int64_t begin, int64_t end) {
        if (should_round_columns) {
            // round columns to multiples of 128 bytes if adjacent columns are
            // contiguous in memory.
            int64_t cols_per_128_bytes = 128 / element_size;
            std::tie(begin, end) = round_columns(iter, dim, cols_per_128_bytes, begin, end);
        }
        if (begin == end) {
            return;
        }
        auto sub_iter = TensorIterator(iter);
        sub_iter.narrow(dim, begin, end - begin);
        sub_iter.for_each(loop);
    });
}
void TensorIterator::foreach_reduced_elt(loop_subiter_t loop, bool parallelize) {
    assert(ninputs() == 1);
    assert(noutputs() >= 1);
    auto shape = this->shape();
    if (output(0).numel() == 0) {
        return;
    }
    if (output(0).numel() == 1) {
        loop(*this);
    }
    else if (numel() < otter::GRAIN_SIZE || otter::get_num_threads() == 1 ||
             otter::in_parallel_region() || !parallelize) {
        auto reduce_dims = num_reduce_dims();
        auto non_reduced_shape = shape.slice(reduce_dims, shape.size() - reduce_dims);
        int64_t non_reduced_numel = 1;
        for (const auto i : non_reduced_shape) {
            non_reduced_numel *= i;
        }
        DimCounter dims {non_reduced_shape, {0, non_reduced_numel}};
        while (!dims.is_done()) {
            TensorIterator reduced = *this;
            reduced.select_all_keeping_dim(reduce_dims, dims.values_);
            loop(reduced);
            dims.increment({1, 1});
        }
    }
    else {
        int dim = find_split_dim(*this);
        int64_t cols = shape[dim];
        otter::parallel_for(0, cols, 1, [&](int64_t begin, int64_t end) {
            if (begin == end) {
                return;
            }
            TensorIterator sub_iter(*this);
            sub_iter.narrow(dim, begin, end - begin);
            // On some broken setups, `#ifdef _OPENMP` is true,
            // and `get_num_threads` returns > 1, but
            // `#pragma omp parallel` is ignored.
            // There is no API to check for this, so we need to explicitly
            // stop trying to parallelize if we've already gotten here.
            //
            // (If we are on one of those broken setups, we will
            //  only have one thread here, and end - begin == cols.)
            sub_iter.foreach_reduced_elt(loop, false);
        });
    }
}
//

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

TensorIteratorConfig& TensorIteratorConfig::declare_static_dtype_and_device(ScalarType dtype, Device device) {
    OTTER_CHECK(!check_all_same_dtype_, "check_all_same_dtype(false) must be called before declare_static_dtype(...)");
    static_dtype_ = dtype;
    static_device_ = device;
    return *this;
}

TensorIteratorConfig& TensorIteratorConfig::declare_static_dtype(ScalarType dtype) {
    OTTER_CHECK(!check_all_same_dtype_, "check_all_same_dtype(false) must be called before declare_static_dtype(...)");
    static_dtype_ = dtype;
    return *this;
}

TensorIteratorConfig& TensorIteratorConfig::declare_static_device(Device device) {
    static_device_ = device;
    return *this;
}

TensorIteratorConfig& TensorIteratorConfig::declare_static_shape(IntArrayRef shape) {
    // WARNING:
    //   This will bypass all shape checking in the TensorIterator. Kernels which call this method
    //   are expected to check shapes before calling `add_owned_input` or `add_owned_output`.
    OTTER_CHECK(!resize_outputs_, "resize_outputs() must be called before declare_static_shape(...)")
    static_shape_ = DimVector(shape);
    return *this;
}

TensorIteratorConfig& TensorIteratorConfig::declare_static_shape(IntArrayRef shape, IntArrayRef squash_dims) {
    declare_static_shape(shape);
    if (!static_shape_.size()) return *this;
    for (const auto& squash_dim : squash_dims) {
        OTTER_CHECK(squash_dim >= 0 && squash_dim < static_cast<int64_t>(static_shape_.size()),
                    "squash_dim ", squash_dim, " must be in [0, ", static_shape_.size(), ").");
        (static_shape_)[squash_dim] = 1;
    }
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
