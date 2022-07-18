//
//  TensorIterator.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef TensorIterator_hpp
#define TensorIterator_hpp

#include <array>
#include <numeric>

#include "Tensor.hpp"
#include "Utils.hpp"
#include "SmallBuffer.hpp"
#include "FunctionRef.hpp"

namespace otter {

constexpr int64_t GRAIN_SIZE = 32768;

struct DimCounter;
class TensorIteratorConfig;

class TensorRef {
public:
    TensorRef() = default;
    
    ~TensorRef() {
        ref_.unsafeReleaseTensorNucleus();
    }
    
    TensorRef(const TensorBase& src) : ref_(Tensor::unsafe_borrow_t{}, src) {
        assert(src.defined());
    }
    
    TensorRef(const TensorRef& rhs) : ref_(Tensor::unsafe_borrow_t{}, rhs.ref_) {}
    
    TensorRef& operator=(TensorRef rhs) {
        std::swap(ref_, rhs.ref_);
        return *this;
    }
    
    bool has_value() const {
        return ref_.defined();
    }
    
    const Tensor& getTensorRef() const & {
        return ref_;
    }
    
    const Tensor& operator*() const & {
        return ref_;
    }
    
    const Tensor* operator->() const & {
        return &ref_;
    }
    
    operator bool() const {
        return ref_.defined();
    }
    
private:
    Tensor ref_;
};

class InlineTensorRef {
    alignas(alignof(TensorBase)) std::array<char, sizeof(TensorBase)> data_;
public:
    InlineTensorRef();
    ~InlineTensorRef();
    
    TensorRef* get() {
        return reinterpret_cast<TensorRef*>(data_.data());
    }
    const TensorRef* get() const {
        return reinterpret_cast<const TensorRef*>(data_.data());
    }
    
    TensorRef& operator*() { return *get(); }
    const TensorRef& operator*() const { return *get(); }
    TensorRef* operator->() { return get(); }
    const TensorRef* operator->() const { return get(); }
    
    const Tensor& getTensor() const;
};

struct OperandInfo {
    using StrideVector = SmallVector<int64_t, 6>;
    OperandInfo() = default;
    
    explicit OperandInfo(MaybeOwned<TensorBase> &&t) {
        if (t->defined()) {
            device_ = t->device();
            target_dtype = t->scalar_type();
            current_dtype = target_dtype;
        }
        tensor(std::move(t));
    }
    
    void exchange_tensor(MaybeOwned<TensorBase> &&new_tensor);
    
    void tensor(MaybeOwned<TensorBase> &&t);
    
    const Tensor& tensor() const {
        return tensor_storage_.getTensor();
    }
    const TensorBase& tensor_base() const {
        return *tensor_base_;
    }
    
    const Tensor& original_tensor() const {
        return original_tensor_storage_.getTensor();
    }
    
    const TensorBase& original_tensor_base() const {
        return *original_tensor_base_;
    }
    
    void restore_original_tensor();
    
    TensorOptions options() const {
        return TensorOptions(target_dtype).device(device_);
    }
    
    bool is_type_defined() const { return target_dtype != ScalarType::Undefined; }
    
    Device device_ = Device::CPU;
    ScalarType target_dtype = ScalarType::Undefined;
    ScalarType current_dtype = ScalarType::Undefined;
    bool is_output = false;
    bool is_read_write = false;
    bool will_resize = false;
    
    void* data = nullptr;
    StrideVector stride_bytes;
    
    MaybeOwned<TensorBase> tensor_base_;
    MaybeOwned<TensorBase> original_tensor_base_ = MaybeOwned<TensorBase>::owned(otter::in_place);
    InlineTensorRef tensor_storage_;
    InlineTensorRef original_tensor_storage_;
};

class TensorIterator {
public:
    using DimMask = std::bitset<64>;
    using StrideVector = SmallVector<int64_t, 6>;
    using PtrVector = SmallVector<char*, 4>;
    using loop2d_t = otter::FunctionRef<void(char** data, const int64_t* strides, int64_t size0, int64_t size1)>;
    using loop_subiter_t = otter::FunctionRef<void(TensorIterator& subiter)>;
    
    void build(TensorIteratorConfig& config);
    
    int ndim() const { return static_cast<int>(shape_.size()); }
    int64_t numel() const;
    int ntensors() const { return static_cast<int>(operands_.size()); }
    int noutputs() const { return num_outputs_; }
    int ninputs() const { return ntensors() - noutputs(); }
    IntArrayRef shape() const { return shape_; }
    IntArrayRef view_offsets() const { return view_offsets_; }
    
    bool is_contiguous() const;
    bool is_dim_reduced(int dim) const;
    
    int64_t element_size(int arg) const {
        return elementSize(dtype(arg));
    }
    
    IntArrayRef strides(int arg) const {
        return operands_[arg].stride_bytes;
    }
    
    bool has_contiguous_first_dim() const {
        int num_tensors = ntensors();
        for (const auto i : otter::irange(num_tensors)) {
            if (strides(i)[0] != element_size(i)) {
                return false;
            }
        }
        return true;
    }
    
    void coalesce_dimensions();
    int num_reduce_dims() const;
    
    void* data_ptr(int arg) const;
    ScalarType dtype(int arg = 0) const { return operands_[arg].current_dtype; }
    ScalarType input_dtype(int arg=0) const { return operands_[num_outputs_ + arg].current_dtype; }
    ScalarType common_dtype() const { return common_dtype_; }
    ScalarType compute_common_dtype();
    
    void initialize_operands(TensorIteratorConfig& config);
    void compute_mem_overlaps(const TensorIteratorConfig& config);
    void compute_shape(const TensorIteratorConfig& config);
    void compute_types(const TensorIteratorConfig& config);
    void mark_resize_outputs(const TensorIteratorConfig& config);
    void compute_strides(const TensorIteratorConfig& config);
    void reorder_dimensions();
    void permute_dimensions(IntArrayRef permutation);
    void allocate_or_resize_outputs();
    StrideVector compatible_stride(int element_size) const;
    DimVector invert_permutation(IntArrayRef input) const;
    void cast_outputs();
    
    virtual void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options);
    const Tensor& maybe_get_output() { return maybe_get_output(0); }
    virtual const Tensor& maybe_get_output(int64_t output_idx);
    
    const Tensor& tensor(int arg) {
        return operands_[arg].tensor();
    }
    
    const TensorBase& tensor_base(int arg) {
        return operands_[arg].tensor_base();
    }
    
    const TensorBase& input_base(int arg = 0) {
        assert(arg >= 0 && arg < ntensors() - num_outputs_);
        return tensor_base(num_outputs_ + arg);
    }
    const Tensor& input(int arg = 0) {
        assert(arg >= 0 && arg < ntensors() - num_outputs_);
        return tensor(num_outputs_ + arg);
    }
    
    const Tensor& output(int arg = 0) {
        assert(arg < num_outputs_);
        return tensor(arg);
    }
    
    const TensorBase& output_base(int arg = 0) {
        assert(arg < num_outputs_);
        return tensor_base(arg);
    }
    
    /// Shrinks an iterated dimension
    void narrow(int dim, int64_t start, int64_t size);
    /// Narrows every dim after and including `start_dim` to size one.
    void select_all_keeping_dim(int start_dim, IntArrayRef starts);
    /// Replaces the data pointer for the operand at index `arg`.
    /// The new pointer should have the same sizes, strides and dtype as the
    /// original
    void unsafe_replace_operand(int arg, void* data);
    
    StrideVector get_dim_strides(int dim) const;
    StrideVector get_strides() const;
    StrideVector get_inner_strides() const {
        return get_dim_strides(0);
    }
    PtrVector get_base_ptrs() const;
    
    template <typename loop1d_t, std::enable_if_t<std::is_convertible<loop1d_t, otter::FunctionRef<void(char**, const int64_t* strides, int64_t size)>>::value, int> = 0>
    void for_each(loop1d_t loop, int64_t grain_size = otter::GRAIN_SIZE) {
        for_each(loop_2d_from_1d(loop), grain_size);
    }
    
    void for_each(loop2d_t loop, int64_t grain_size = otter::GRAIN_SIZE);
    
    void foreach_reduced_elt(loop_subiter_t loop, bool parallelize = true);
    
    void parallel_reduce(loop2d_t loop);
    
    template <typename loop1d_t, std::enable_if_t<std::is_convertible<loop1d_t,  otter::FunctionRef<void(char**, const int64_t* strides, int64_t size)>>::value, int> = 0>
    void serial_for_each(loop1d_t loop, Range range) {
        serial_for_each(loop_2d_from_1d(loop), range);
    }
    
    void serial_for_each(loop2d_t loop, Range range) const;
    
    template <typename loop1d_t>
    auto loop_2d_from_1d(const loop1d_t& loop) {
        return [loop, ntensor=ntensors()](char** base, const int64_t* strides, int64_t size0, int64_t size1) {
            PtrVector data(base, base + ntensor);
            const int64_t* outer_strides = &strides[ntensor];
            for (const auto i : otter::irange(size1)) {
                if (i > 0) {
                    for (const auto arg : otter::irange(ntensor)) {
                        data[arg] += outer_strides[arg];
                    }
                }
                loop(data.data(), strides, size0);
            }
        };
    }
    
    void build_binary_float_op(const TensorBase& out, const TensorBase& a, const TensorBase& b);
    
    void build_borrowing_binary_float_op(const TensorBase& out, const TensorBase& a, const TensorBase& b);
    
    void build_binary_op(const TensorBase& out, const TensorBase& a, const TensorBase& b);
    void build_borrowing_binary_op(const TensorBase& out, const TensorBase& a, const TensorBase& b);
    
    void build_unary_float_op(const TensorBase& out, const TensorBase& a);
    void build_borrowing_unary_float_op(const TensorBase& out, const TensorBase& a);
    
    void build_unary_op(const TensorBase& out, const TensorBase& a);
    void build_borrowing_unary_op(const TensorBase& out, const TensorBase& a);
    
    void build_comparison_op(const TensorBase& out, const TensorBase& a, const TensorBase& b);
    void build_borrowing_comparison_op(const TensorBase& out, const TensorBase& a, const TensorBase& b);
    void build_borrowing_except_last_argument_comparison_op(const TensorBase& out, const TensorBase& a, const TensorBase& b);
    
    static TensorIterator nullary_op(TensorBase& out);
    static TensorIterator unary_op(TensorBase& out, const TensorBase& a);
    static TensorIterator binary_op(TensorBase& out, const TensorBase& a, const TensorBase& b);
    static TensorIterator borrowing_nullary_op(const TensorBase& out);
    static TensorIterator borrowing_nullary_op(TensorBase&& out) = delete;
    static TensorIterator borrowing_binary_op(const TensorBase& out, const TensorBase& a, const TensorBase& b);
    static TensorIterator reduce_op(TensorBase& out, const TensorBase& a);
    static TensorIterator reduce_op(
        TensorBase& out1,
        TensorBase& out2,
        const TensorBase& a);
    
private:
    DimVector shape_;
    DimVector permutation_;
    DimVector view_offsets_;
    SmallVector<OperandInfo, 4> operands_;
    
    int num_outputs_ = 0;
    bool all_ops_same_shape_ = false;
    bool has_scalars_ = false;
    bool has_tensors_ = false;
    bool is_reduction_ = false;
    bool enforce_linear_iteration_ = false;
    bool has_coalesced_dimensions_ = false;
    
    Device common_device_;
    ScalarType common_dtype_;
};

class TensorIteratorConfig {
public:
    friend class TensorIterator;
    TensorIteratorConfig() {}
    
    TensorIteratorConfig(const TensorIteratorConfig&) = delete;
    TensorIteratorConfig(const TensorIteratorConfig&&) = delete;
    
    TensorIteratorConfig& add_input(const TensorBase& input) {
        return add_borrowed_input(input);
    }
    
    TensorIteratorConfig& add_output(const TensorBase& output) {
        return add_borrowed_output(output);
    }
    
    TensorIteratorConfig& add_owned_output(const TensorBase& output);
    TensorIteratorConfig& add_owned_input(const TensorBase& input);
    
    TensorIteratorConfig& add_borrowed_input(const TensorBase& input);
    TensorIteratorConfig& add_borrowed_output(const TensorBase& output);
    
    // Ensure that cannot borrow from temporaries
    TensorIteratorConfig& add_borrowed_input(const TensorBase&& input) = delete;
    TensorIteratorConfig& add_borrowed_output(const TensorBase&& output) = delete;
    
    TensorIteratorConfig& declare_static_dtype_and_device(ScalarType dtype, Device device);
    TensorIteratorConfig& declare_static_dtype(ScalarType dtype);
    TensorIteratorConfig& declare_static_device(Device device);
    TensorIteratorConfig& declare_static_shape(IntArrayRef shape);
    TensorIteratorConfig& declare_static_shape(IntArrayRef shape, IntArrayRef squash_dims);
    
    TensorIterator build() {
        TensorIterator iter;
        iter.build(*this);
        
        return iter;
    }
    
    TensorIteratorConfig& set_check_mem_overlap(bool check_mem_overlap) {
        check_mem_overlap_ = check_mem_overlap;
        return *this;
    }
    
    TensorIteratorConfig& allow_cpu_scalars(const bool _allow_cpu_scalars) {
        allow_cpu_scalars_ = _allow_cpu_scalars;
        return *this;
    }
    
    TensorIteratorConfig& check_all_same_dtype(const bool _check_all_same_dtype) {
        check_all_same_dtype_ = _check_all_same_dtype;
        return *this;
    }
    
    TensorIteratorConfig& check_all_same_device(const bool _check_all_same_device) {
        check_all_same_device_ = _check_all_same_device;
        return *this;
    }
    
    TensorIteratorConfig& enforce_safe_casting_to_output(const bool _enforce_safe_casting_to_output) {
        enforce_safe_casting_to_output_ = _enforce_safe_casting_to_output;
        return *this;
    }
    
    TensorIteratorConfig& promote_inputs_to_common_dtype(const bool _promote_inputs_to_common_dtype) {
        promote_inputs_to_common_dtype_ = _promote_inputs_to_common_dtype;
        if (_promote_inputs_to_common_dtype) {
            check_all_same_dtype_ = false;
        }
        return *this;
    }
    
    TensorIteratorConfig& promote_integer_inputs_to_float(const bool _promote_integer_inputs_to_float) {
        promote_integer_inputs_to_float_ = _promote_integer_inputs_to_float;
        assert(!promote_integer_inputs_to_float_ || promote_inputs_to_common_dtype_);
        return *this;
    }
    
    TensorIteratorConfig& cast_common_dtype_to_outputs(const bool _cast_common_dtype_to_outputs) {
        cast_common_dtype_to_outputs_ = _cast_common_dtype_to_outputs;
        if (_cast_common_dtype_to_outputs) {
            check_all_same_dtype_ = false;
        }
        return *this;
    }
    
    TensorIteratorConfig& resize_outputs(bool resize_outputs) {
        resize_outputs_ = resize_outputs;
        return *this;
    }
    
    TensorIteratorConfig& enforce_linear_iteration(const bool _enforce_linear_iteration = true) {
        enforce_linear_iteration_ = _enforce_linear_iteration;
        return *this;
    }
    
    TensorIteratorConfig& is_reduction(const bool _is_reduction) {
        is_reduction_ = _is_reduction;
        return *this;
    }
    
private:
    SmallVector<MaybeOwned<TensorBase>, 4> tensors_;
    
    int num_inputs_ = 0;
    int num_outputs_ = 0;
    
    ScalarType static_dtype_ = ScalarType::Undefined;
    Device static_device_ = Device::Undefined;
    DimVector static_shape_ = {};
    
    bool resize_outputs_ = true;
    bool check_mem_overlap_ = true;
    bool allow_cpu_scalars_ = false;
    bool check_all_same_dtype_ = false;
    bool check_all_same_device_ = false;
    bool enforce_safe_casting_to_output_ = false;
    bool promote_inputs_to_common_dtype_ = false;
    bool cast_common_dtype_to_outputs_ = false;
    bool promote_integer_inputs_to_float_ = false;
    bool is_reduction_ = false;
    bool enforce_linear_iteration_ = false;
};

struct DimCounter {
    DimCounter(IntArrayRef shape, Range range);
    
    void increment(const std::array<int64_t, 2>& step);
    bool is_done() const;
    std::array<int64_t, 2> max_2d_step() const;
    
    IntArrayRef shape_;
    Range range_;
    SmallBuffer<int64_t, 4> values_;
    int64_t offset_;
};

namespace internal {

inline void get_data_ptrs(char** ptrs, ArrayRef<char*> base, IntArrayRef strides, IntArrayRef counter) {
    const int64_t ntensors = base.size();
    const int64_t ndim = counter.size();
    std::copy(base.begin(), base.end(), ptrs);
    for (const auto dim : otter::irange(ndim)) {
        int64_t value = counter[dim];
        for (const auto arg : otter::irange(ntensors)) {
            ptrs[arg] += value * strides[dim * ntensors + arg];
        }
    }
}

inline void serial_for_each(IntArrayRef shape, IntArrayRef strides, char** base_ptrs, size_t ntensors, typename TensorIterator::loop2d_t loop, Range range) {
    const auto ndim = shape.size();
    assert(strides.size() == ntensors * std::max(size_t{2}, ndim));
    
    if (ndim <= 1) {
        if (range.begin == 0) {
            loop(base_ptrs, strides.data(), range.size(), 1);
        } else {
            otter::SmallBuffer<char*, 4> ptrs(ntensors);
            get_data_ptrs(ptrs.data(), {base_ptrs, ntensors}, strides, {range.begin});
            loop(ptrs.data(), strides.data(), range.size(), 1);
        }
    } else {
        otter::SmallBuffer<char*, 4> ptrs(ntensors);
        auto counter = DimCounter(shape, range);
        while (!counter.is_done()) {
            get_data_ptrs(ptrs.data(), {base_ptrs, ntensors}, strides, counter.values_);
            auto step = counter.max_2d_step();
            loop(ptrs.data(), strides.data(), step[0], step[1]);
            counter.increment(step);
        }
    }
    
}


}


}

#endif /* TensorIterator_hpp */
