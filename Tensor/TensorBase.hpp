//
//  TensorBase.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/1.
//

#ifndef TensorBase_hpp
#define TensorBase_hpp

#include "Memory.hpp"
#include "DType.hpp"
#include "PerspectiveView.hpp"
#include "MaybeOwned.hpp"
#include "TensorOptions.hpp"
#include "Scalar.hpp"
#include "ExclusivelyOwned.hpp"
#include "WarpDimMinimal.hpp"
#include "MemoryFormat.hpp"

#define NOT_IMPLEMENTED fprintf(stderr, "NOT_IMPLEMENTED!")

namespace otter {
struct TensorNucleus : public Ptr_quantum {
public:
    TensorNucleus() = delete;
    TensorNucleus(const TensorNucleus&) = delete;
    TensorNucleus& operator=(const TensorNucleus&) = delete;
    TensorNucleus(TensorNucleus&&) = delete;
    TensorNucleus& operator=(TensorNucleus&&) = delete;
    
    TensorNucleus(Memory&& memory, const TypeMeta data_type, Device device);
    
    TensorNucleus(Memory&& memory, const TypeMeta data_type);
    
    void release_resources() override {
        memory_ = {};
    }
    
    int64_t dim() const;
    
    virtual int64_t size(size_t idx) const;
    
    IntArrayRef sizes() const {
        return perspective_view_.sizes_arrayref();
    }
    
    virtual int64_t stride(size_t idx) const;
    
    virtual IntArrayRef strides() const {
        return perspective_view_.strides_arrayref();
    }
    
    virtual void set_memory_offset(int64_t memory_offset) {
        memory_offset_ = memory_offset;
    }
    
    void refresh_contiguous() {
        is_contiguous_ = compute_contiguous();
        
        switch (dim()) {
            case 4:
                is_channels_last_contiguous_ = compute_channels_last_contiguous_2d();
                is_channels_last_ = compute_strides_like_channels_last_2d();
                is_non_overlapping_and_dense_ = is_contiguous_ || is_channels_last_contiguous_ || compute_non_overlapping_and_dense();
                break;
            case 5:
                is_channels_last_contiguous_ = compute_channels_last_contiguous_2d();
                is_channels_last_ = compute_strides_like_channels_last_2d();
                is_non_overlapping_and_dense_ = is_contiguous_ || is_channels_last_contiguous_ || compute_non_overlapping_and_dense();
                break;
            default:
                is_channels_last_contiguous_ = false;
                is_channels_last_ = false;
                is_non_overlapping_and_dense_ = is_contiguous_ || compute_non_overlapping_and_dense();
        }
    }
    
    bool is_contiguous(MemoryFormat memory_format = MemoryFormat::Contiguous) const {
        return is_contiguous_;
    }
    
    bool is_non_overlapping_and_dense() const {
        return is_non_overlapping_and_dense_;
    }
    
    bool is_strides_like_channels_last() const {
        return is_channels_last_;
    }
    
    bool compute_contiguous() const;
    bool compute_non_overlapping_and_dense() const;
    bool compute_channels_last_contiguous_2d() const;
    bool compute_strides_like_channels_last_2d() const;
    
    void set_sizes_and_strides(IntArrayRef newSizes, IntArrayRef newStrides) {
        if (newSizes.size() != newStrides.size())
            fprintf(stderr, "[TensorNucleus] Dimensionality of sizes (%zu) must match dimensionality of strides(%zu)!\n", newSizes.size(), newStrides.size());
        const int64_t new_dim = newSizes.size();
        
        perspective_view_.set_sizes(newSizes);
        if (new_dim > 0) {
            for (size_t dim = new_dim; dim--; ) {
                if (newStrides[dim] >= 0) {
                    perspective_view_.stride_at(dim) = newStrides[dim];
                } else {
                    if (dim == new_dim - 1) {
                        perspective_view_.stride_at(dim) = 1;
                    } else {
                        perspective_view_.stride_at(dim) = (std::max<int64_t>(perspective_view_.size_at(dim + 1), 1)) * perspective_view_.stride_at(dim + 1);
                    }
                }
            }
        }
        this->update_numel();
        this->refresh_contiguous();
    }
    
    int64_t numel() const {
        return numel_;
    }
    
    int64_t compute_numel() const {
        int64_t n = 1;
        for (auto s : this->sizes()) {
            n *= s;
        }
        return n;
    }
    
    void update_numel() {
        numel_ = compute_numel();
    }
    
    bool is_empty() const {
        return numel_ == 0;
    }
    
    bool memory_initialized() {
        return memory_.data() || numel_ == 0;
    }
    
    bool dtype_initialized() {
        return data_type_ != TypeMeta();
    }
    
    template <typename T>
    inline T* data_ptr_nucleus() const {
        return memory_.unsafe_data<T>() + memory_offset_;
    }
    
    inline void* raw_data() const {
        if (this->is_empty()) {
            return nullptr;
        }
        return static_cast<void*>(static_cast<char*>(memory_.data()) + data_type_.itemsize() * memory_offset_);
    }
    
    template<typename T>
    inline T* mutable_data() {
        if (memory_initialized() && data_type_.Match<T>()) {
            return static_cast<T*>(memory_.data()) + memory_offset_;
        }
        return static_cast<T*>(raw_mutable_data(TypeMeta::Make<T>()));
    }
    
    inline void* raw_mutable_data(const TypeMeta type) {
        if (data_type_ == type && memory_initialized()) {
            return static_cast<void*>(static_cast<char*>(memory_.data()) + memory_offset_ * type.itemsize());
        } else {
            memory_offset_ = 0;
            data_type_ = type;
            
            const Allocator* allocator = memory_.allocator();
            if (allocator == nullptr) {
                allocator = GetAllocator(memory_.device());
            }
            if (type.placementNew()) {
                NOT_IMPLEMENTED;
            } else {
                memory_.set_data_ptr_noswap(allocator->allocate(numel_ * type.itemsize()));
            }
            memory_.set_nbytes(numel_ * type.itemsize());
            return memory_.data();
        }
    }
    
    TypeMeta dtype() const {
        return data_type_;
    }
    
    ScalarType scalar_type() const {
        return data_type_.toScalarType();
    }
    
    Device device() const {
        return device_;
    }
    
    size_t itemsize() const {
        return data_type_.itemsize();
    }
    
    int64_t memory_offset() const {
        return memory_offset_;
    }
    
    bool has_memory() const {
        return memory_;
    }
    
    const Memory& memory() const {
        // Do some check
        return memory_;
    }
    
    inline const Memory& unsafe_memory() const {
        return memory_;
    }
    
    inline void FreeMemory() {
        memory_ = Memory::create_empty(memory_.device());
        memory_offset_ = 0;
    }
    
    void set_sizes_contiguous(IntArrayRef newSizes) {
        perspective_view_.set_sizes(newSizes);
        this->update_numel();
        this->empty_tensor_restride(MemoryFormat::Contiguous);
    }
    
    void empty_tensor_restride(MemoryFormat memory_format) {
        switch (memory_format) {
            case MemoryFormat::Contiguous: {
                const int64_t dim_ = dim();
                perspective_view_.resize(dim_);
                if (dim_ > 0) {
                    const int64_t last_idx = dim_ - 1;
                    perspective_view_.stride_at(last_idx) = 1;
                    for (int64_t i = last_idx; i--; ) {
                        perspective_view_.stride_at(i) = perspective_view_.stride_at(i + 1) * std::max<int64_t>(perspective_view_.size_at(i + 1), 1);
                    }
                }
                break;
            }
            case MemoryFormat::ChannelsLast: {
                NOT_IMPLEMENTED;
            }
            case MemoryFormat::ChannelsLast3d: {
                NOT_IMPLEMENTED;
            }
            case MemoryFormat::Preserve: {
                assert(false);  // Unsupport
            }
        }
        this->refresh_contiguous();
    }
    
    void set_storage_keep_dtype(Memory memory) {
        memory_ = std::move(memory);
        device_ = memory_.device();
    }
    
    void set_storage_and_dtype(Memory memory, const TypeMeta data_type) {
        set_storage_keep_dtype(memory);
        data_type_ = data_type;
    }
    
    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
    bool SetDimsTemplate(ArrayRef<T> src) {
        int64_t old_numel = numel_;
        perspective_view_.resize(src.size());
        int64_t new_numel = 1;
        for (size_t i = 0; i < src.size(); ++i) {
            new_numel *= src[i];
            perspective_view_.size_at(i) = src[i];
        }
        numel_ = new_numel;
        this->empty_tensor_restride(MemoryFormat::Contiguous);
        return numel_ != old_numel;
    }
    
    bool SetDims(ArrayRef<int64_t> d) {
        return SetDimsTemplate(d);
    }
    
    bool SetDims(ArrayRef<int> d) {
        return SetDimsTemplate(d);
    }
    
    bool SetDims(const int64_t d0) {
        return SetDimsTemplate(IntArrayRef{d0});
    }
    
    bool SetDims(const int64_t d0, const int64_t d1) {
        return SetDimsTemplate(IntArrayRef{d0, d1});
    }
    
    bool SetDims(const int64_t d0, const int64_t d1, const int64_t d2) {
        return SetDimsTemplate(IntArrayRef{d0, d1, d2});
    }
    
    bool SetDims(const int64_t d0, const int64_t d1, const int64_t d2, const int64_t d3) {
        return SetDimsTemplate(IntArrayRef{d0, d1, d2, d3});
    }
    
    template <typename... Ts>
    void Resize(Ts... dim_source) {
        bool size_changed = SetDims(dim_source...);
        if (size_changed) {
            this->HandleResize();
        }
    }
    
    template <typename T>
    void Resize(const std::vector<T>& dim_source) {
        Resize(ArrayRef<T>(dim_source));
    }
    
    void HandleResize() {
        bool reset_tensor = false;
        
        reset_tensor = memory_.nbytes() < (memory_offset_ + numel_) * data_type_.itemsize();
        
        if (reset_tensor && memory_initialized()) {
            this->FreeMemory();
        }
    }
    
    // retain for autograd
    
    //
private:
    Device device_;
    
    Memory memory_;
    int64_t memory_offset_ = 0;
    int64_t numel_ = 1;
    
    TypeMeta data_type_;
    PerspectiveView perspective_view_;
    
    inline void init_bitfields() {
        is_contiguous_ = true;
        is_channels_last_ = false;
        is_channels_last_contiguous_ = false;
        is_non_overlapping_and_dense_ = true;
    }
    
    bool is_contiguous_ : 1;
    bool is_channels_last_ : 1;
    bool is_channels_last_contiguous_ : 1;
    bool is_non_overlapping_and_dense_ : 1;
};

struct UndefinedTensorNucleus : public TensorNucleus {
public:
    static constexpr inline TensorNucleus* singleton() {
        return &_singleton;
    }
//    IntArrayRef strides() const override;
//    int64_t size(int64_t d) const override;
//    int64_t stride(int64_t d) const override;
private:
    UndefinedTensorNucleus();
    static UndefinedTensorNucleus _singleton;
};

class TensorBase {
public:
    struct unsafe_borrow_t { explicit unsafe_borrow_t() = default; };
protected:
    explicit TensorBase(unsafe_borrow_t, const TensorBase& rhs) : tensor_nucleus_(Ptr<TensorNucleus, UndefinedTensorNucleus>::reclaim(rhs.tensor_nucleus_.get())) {}
    friend MaybeOwnedTraits<TensorBase>;
public:
    TensorBase() = default;
    TensorBase(const TensorBase&) = default;
    TensorBase(TensorBase&&) = default;
    
    TensorBase& operator=(const TensorBase& t) & {
        tensor_nucleus_ = t.tensor_nucleus_;
        return *this;
    }
    
    TensorBase& operator=(TensorBase&& t) & {
        tensor_nucleus_ = std::move(t.tensor_nucleus_);
        return *this;
    }
    
    explicit TensorBase(Ptr<TensorNucleus, UndefinedTensorNucleus> tensor_nucleus) : tensor_nucleus_(std::move(tensor_nucleus)) {
        if (tensor_nucleus_.get() == nullptr) {
            fprintf(stderr, "[TensorBase] Initialization failed!\n");
        }
    }
    
    template <typename T>
    T* data_ptr() const;
    
    void* data_ptr() const {
        return this->unsafeGetTensorNucleus()->raw_data();
    }
    
    inline void* raw_data() const {
        return tensor_nucleus_->raw_data();
    }
    
    inline void* raw_mutable_data(const TypeMeta meta) const {
        return tensor_nucleus_.get()->raw_mutable_data(meta);
    }
    
    inline void* raw_mutable_data() const {
        const auto& data_type = tensor_nucleus_->dtype();
        return raw_mutable_data(data_type);
    }
    
    template <typename T>
    inline T* mutable_data() const {
        return tensor_nucleus_.get()->mutable_data<T>();
    }
    
    int64_t dim() const {
        return tensor_nucleus_->dim();
    }
    
    int64_t size(int64_t dim) const {
        dim = maybe_wrap_dim(dim, this->dim(), false);
        return sizes()[dim];
    }
    
    int64_t stride(int64_t dim) const {
        dim = maybe_wrap_dim(dim, this->dim(), false);
        return strides()[dim];
    }
    
    TensorNucleus* unsafeGetTensorNucleus() const {
        return tensor_nucleus_.get();
    }
    
    TensorNucleus* unsafeReleaseTensorNucleus() {
        return tensor_nucleus_.release();
    }
    
    Ptr<TensorNucleus, UndefinedTensorNucleus> getPtr() const {
        return tensor_nucleus_;
    }
    
    Ptr<TensorNucleus, UndefinedTensorNucleus> unsafeReleasePtr() {
        return std::move(tensor_nucleus_);
    }
    
    int64_t memory_offset() const {
        return tensor_nucleus_->memory_offset();
    }
    
    bool defined() const {
        return tensor_nucleus_;
    }
    
    void reset() {
        tensor_nucleus_.reset();
    }
    
    bool is_same(const TensorBase& other) const noexcept {
        return tensor_nucleus_ == other.tensor_nucleus_;
    }
    
    size_t use_count() const noexcept {
        return tensor_nucleus_.use_count();
    }
    
    bool has_memory() const {
        return tensor_nucleus_->has_memory();
    }
    
    const Memory& memory() const {
        return tensor_nucleus_->memory();
    }
    
    TypeMeta dtype() const noexcept {
        return tensor_nucleus_->dtype();
    }
    
    IntArrayRef sizes() const {
        return tensor_nucleus_->sizes();
    }
    
    IntArrayRef strides() const {
        return tensor_nucleus_->strides();
    }
    
    size_t nbytes() const {
        return tensor_nucleus_->numel() * tensor_nucleus_->itemsize();
    }
    
    int64_t numel() const {
        return tensor_nucleus_->numel();
    }
    
    size_t itemsize() const {
        return tensor_nucleus_->itemsize();
    }
    
    TensorOptions options() const {
        return TensorOptions().dtype(dtype()).device(device());
    }
    
    void print() const;
    
    std::string toString() const;
    
    ScalarType scalar_type() const {
        return tensor_nucleus_->scalar_type();
    }
    
    Device device() const {
        return tensor_nucleus_->device();
    }
    
    void as_strided_(IntArrayRef newSizes, IntArrayRef newStrides) const {
        tensor_nucleus_.get()->set_sizes_and_strides(newSizes, newStrides);
    }
    
    bool is_contiguous(MemoryFormat memory_format = MemoryFormat::Contiguous) const {
        return tensor_nucleus_->is_contiguous(memory_format);
    }
    
    bool is_non_overlapping_and_dense() const {
        return tensor_nucleus_->is_non_overlapping_and_dense();
    }
    
    MemoryFormat suggest_memory_format(bool channels_last_strides_exact_match = false) const {
//        if (!is_mkldnn() && !is_sparse()) {
        if (true) {
            if (tensor_nucleus_->is_strides_like_channels_last()) {
                if (!channels_last_strides_exact_match || get_channels_last_strides_2d(sizes()) == strides()) {
                    return MemoryFormat::ChannelsLast;
                }
            }
        }
        return MemoryFormat::Contiguous;
    }
    
protected:
    Ptr<TensorNucleus, UndefinedTensorNucleus> tensor_nucleus_;
};

template <typename T, typename... Args>
TensorBase make_tensor_base(Args&&... args) {
    return TensorBase(make_otterptr<T>(std::forward<Args>(args)...));
}

template <>
struct MaybeOwnedTraits<TensorBase> {
    using owned_type = TensorBase;
    using borrow_type = TensorBase;
    
    static borrow_type create_borrow(const owned_type& from) {
        return borrow_type(borrow_type::unsafe_borrow_t{}, from);
    }
    
    static void assign_borrow(borrow_type& lhs, const borrow_type& rhs) {
        lhs.unsafeReleaseTensorNucleus();
        lhs = borrow_type(borrow_type::unsafe_borrow_t{}, rhs);
    }
    
    static void destroy_borrow(borrow_type& target) {
        target.unsafeReleaseTensorNucleus();
    }
    
    static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
        return borrow;
    }
    
    static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
        return &borrow;
    }
};

template <>
struct ExclusivelyOwnedTraits<TensorBase> {
    using repr_type = TensorBase;
    using pointer_type = TensorBase*;
    using const_pointer_type = const TensorBase*;
    
    static repr_type nullRepr() {
        return TensorBase();
    }
    
    template <class... Args>
    static repr_type createInPlace(Args&&... args) {
        return TensorBase(std::forward<Args>(args)...);
    }
    
    static repr_type moveToRepr(TensorBase&& x) {
        return std::move(x);
    }
    
    static void destroyOwned(TensorBase& x) {
        TensorNucleus* const toDestroy = x.unsafeReleaseTensorNucleus();
        assert(toDestroy != nullptr);
        const bool isUndefined = toDestroy == UndefinedTensorNucleus::singleton();
        assert(toDestroy->refCount_ == 1 || (toDestroy->refCount_ == 0 && isUndefined));
        if (!isUndefined) {
            toDestroy->refCount_ = 0;
            
            toDestroy->release_resources();
            delete toDestroy;
        }
    }
    
    static TensorBase take(TensorBase& x) {
        return std::move(x);
    }
    
    static pointer_type getImpl(repr_type& x) {
        return &x;
    }
    
    static const_pointer_type getImpl(const repr_type& x) {
        return &x;
    }
};


}   // end namespace otter

#endif /* TensorBase_hpp */
