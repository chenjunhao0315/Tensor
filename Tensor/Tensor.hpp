//
//  Tensor.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef Tensor_hpp
#define Tensor_hpp

#include <climits>

#include "TensorBase.hpp"
#include "TensorAccessor.hpp"
#if __ARM_NEON
#include <arm_neon.h>
#endif
#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif
#endif

namespace otter {

struct Generator;

class TensorRef;

class Tensor : public TensorBase {
protected:
    explicit Tensor(unsafe_borrow_t, const TensorBase& rhs): TensorBase(unsafe_borrow_t{}, rhs) {}
    friend MaybeOwnedTraits<Tensor>;
    friend TensorRef;
public:
    Tensor() = default;
    
    explicit Tensor(Ptr<TensorNucleus> tensor_nucleus) : TensorBase(std::move(tensor_nucleus)) {}
    
    Tensor(const Tensor &tensor) = default;
    Tensor(Tensor &&tensor) = default;
    
    explicit Tensor(const TensorBase &base): TensorBase(base) {}
    Tensor(TensorBase &&base): TensorBase(std::move(base)) {}
    
    template<typename T, size_t N, int64_t P = 1>
    TensorAccessor<T, N, P> accessor() const& {
        OTTER_CHECK(N > 0, "accessor is used for indexing tensor, for scalars use *data_ptr<T>()");
        OTTER_CHECK(dim() == N, "TensorAccessor expected ", N, " dims but tensor has ", dim());
        OTTER_CHECK(elempack() == P, "Elempack should be same as tensor");
        return TensorAccessor<T, N, P>((T*)raw_data(), sizes().data(), strides().data());
    }
    template<typename T, size_t N, int64_t P>
    TensorAccessor<T, N, P> accessor() && = delete;
    
    template<typename T, size_t N>
    TensorRawAccessor<T, N> raw_accessor() const& {
        OTTER_CHECK(N > 0, "accessor is used for indexing tensor, for scalars use *data_ptr<T>()");
        OTTER_CHECK(dim() == N, "TensorAccessor expected ", N, " dims but tensor has ", dim());
        return TensorRawAccessor<T, N>(raw_data(), sizes().data(), strides().data(), itemsize());
    }
    template<typename T, size_t N>
    TensorRawAccessor<T, N> raw_accessor() && = delete;
    
    Tensor& operator=(const TensorBase& x) & {
        tensor_nucleus_ = x.getPtr();
        return *this;
    }
    Tensor& operator=(TensorBase&& x) & {
        tensor_nucleus_ = x.unsafeReleasePtr();
        return *this;
    }
    
    Tensor& operator=(const Tensor &x) & {
        return operator=(static_cast<const TensorBase&>(x));
    }
    Tensor& operator=(Tensor &&x) & {
        return operator=(static_cast<TensorBase&&>(x));
    }
    
    Tensor& operator=(Scalar value) {
        return fill_(value);
    }
    
    Tensor& operator=(const Tensor &rhs) && {
        return copy_(rhs);
    }
    
    Tensor& operator=(Tensor&& rhs) && {
        return copy_(rhs);
    }
    
    bool is_floating_point() const {
        return otter::isFloatingType(this->scalar_type());
    }
    bool is_signed() const {
        return otter::isSignedType(this->scalar_type());
    }
    bool is_inference() const {
        return true;
    }
    bool _is_zerotensor() const {
        return false;
    }
    bool is_neg() const {
        return false;
    }
    
    template <typename T>
    T item() const;
    
    Scalar item() const;
    
    Tensor packing(int64_t elempack) const;
    
    Tensor operator[](int64_t index) const;
    Tensor select(int64_t dim, int64_t index) const;
    
    Tensor& copy_(const Tensor& src, bool non_blocking = false) const;
    
    Tensor clone(MemoryFormat memory_format = MemoryFormat::Preserve) const;
    
    Tensor contiguous(MemoryFormat memory_format = MemoryFormat::Contiguous) const;
    
    Tensor permute(IntArrayRef dims) const;
    
    Tensor& transpose_(int64_t dim0, int64_t dim1) const;
    Tensor transpose(int64_t dim0, int64_t dim1) const;
    
    Tensor expand(IntArrayRef sizes) const;
    Tensor expand_as(const Tensor& other) const;
    
    const Tensor& resize_(IntArrayRef shape) const;
    const Tensor& resize_as_(const Tensor& the_template) const;
    const Tensor& resize_(IntArrayRef shape, MemoryFormat memory_format) const;
    const Tensor& resize_as_(const Tensor& the_template, MemoryFormat memory_format) const;
    Tensor as_strided(IntArrayRef sizes, IntArrayRef strides) const;
    Tensor as_strided(IntArrayRef sizes, IntArrayRef strides, int64_t memory_offset) const;
    const Tensor& as_strided_(IntArrayRef sizes, IntArrayRef strides) const;
    const Tensor& as_strided_(IntArrayRef sizes, IntArrayRef strides, int64_t memory_offset) const;
    
    Tensor view(IntArrayRef sizes) const;
    
    Tensor reshape(IntArrayRef sizes) const;
    Tensor reshape_as(const Tensor& other) const;
    
    Tensor slice(int64_t dim = 0, int64_t start = INT64_MAX, int64_t end = 0, int64_t step = 1) const;
    
    Tensor unsqueeze(int64_t dim) const;
    Tensor& unsqueeze_(int64_t dim) const;
    
    Tensor squeeze(int64_t dim) const;
    Tensor& squeeze_(int64_t dim) const;
    
    Tensor narrow(int64_t dim, int64_t start, int64_t length) const;
    
    Tensor unfold(int64_t dim, int64_t size, int64_t step) const;
    
    Tensor detach() const;
    
    Tensor operator~() const {
        return bitwise_not();
    }
    
    Tensor operator-() const {
        return neg();
    }
    
    Tensor& operator+=(const Tensor& other) {
        return add_(other);
    }
    
    Tensor& operator+=(Scalar other) {
        return add_(other);
    }
    
    Tensor& operator-=(const Tensor& other) {
        return sub_(other);
    }
    
    Tensor& operator-=(Scalar other) {
        return sub_(other);
    }
    
    Tensor& operator*=(const Tensor& other) {
        return mul_(other);
    }
    
    Tensor& operator*=(Scalar other) {
        return mul_(other);
    }
    
    Tensor& operator/=(const Tensor& other) {
        return div_(other);
    }
    
    Tensor& operator/=(Scalar other) {
        return div_(other);
    }
    
    Tensor& operator&=(const Tensor& other) {
        return bitwise_and_(other);
    }
    
    Tensor& operator|=(const Tensor& other) {
        return bitwise_or_(other);
    }
    
    Tensor& operator^=(const Tensor& other) {
        return bitwise_xor_(other);
    }
    
    Tensor& zero_();
    Tensor& fill_(const Scalar& value);
    Tensor& fill_(const Tensor& value);
    
#if __ARM_NEON
    Tensor& fill_(float32x4_t _v);
    Tensor& fill_(uint16x4_t _v);
    Tensor& fill_(int32x4_t _v);
    Tensor& fill_(int32x4_t _v0, int32x4_t _v1);
#endif // __ARM_NEON
#if __SSE2__
#if __AVX__
    Tensor& fill_(__m256 _v);
#endif // __AVX__
    Tensor& fill_(__m128 _v);
    Tensor& fill_(__m128i _v);
#endif // __SSE2__
    
    template <typename scalar_t>
    Tensor& type_fill_(scalar_t value);
    
    Tensor to(ScalarType dtype, bool non_blocking = false, bool copy = false, MemoryFormat memory_format = MemoryFormat::Preserve) const;
    Tensor to(TensorOptions options, bool non_blocking = false, bool copy = false, MemoryFormat memory_format = MemoryFormat::Preserve) const;
    
    Tensor& uniform_(double from, double to) const;
    Tensor& uniform_(double from, double to, Generator generator) const;
    
    Tensor& normal_(double mean = 0, double std = 1) const;
    Tensor& normal_(double mean, double std, Generator generator) const;
    
    Tensor& random_(int64_t from, int64_t to = INT_MAX) const;
    Tensor& random_(int64_t from, int64_t to, Generator generator) const;
    Tensor& random_(int64_t to) const;
    Tensor& random_(int64_t to, Generator generator) const;
    Tensor& random_() const;
    Tensor& random_(Generator generator) const;
    
    Tensor& add_(const Tensor& other, const Scalar& alpha = 1) const;
    Tensor add(const Tensor& other, const Scalar& alpha = 1) const;
    Tensor& add_(const Scalar& other, const Scalar& alpha = 1) const;
    Tensor add(const Scalar& other, const Scalar& alpha = 1) const;
    
    Tensor& sub_(const Tensor& other, const Scalar& alpha = 1) const;
    Tensor sub(const Tensor& other, const Scalar& alpha = 1) const;
    Tensor& sub_(const Scalar& other, const Scalar& alpha = 1) const;
    Tensor sub(const Scalar& other, const Scalar& alpha = 1) const;
    
    Tensor& mul_(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor& mul_(const Scalar& other) const;
    Tensor mul(const Scalar& other) const;
    
    Tensor& div_(const Tensor& other) const;
    Tensor div(const Tensor& other) const;
    Tensor& div_(const Scalar& other) const;
    Tensor div(const Scalar& other) const;
    
    Tensor& remainder_(const Tensor& other) const;
    Tensor remainder(const Tensor& other) const;
    Tensor& remainder_(const Scalar& other) const;
    Tensor remainder(const Scalar& other) const;
    
    Tensor& bitwise_and_(const Tensor& other) const;
    Tensor bitwise_and(const Tensor& other) const;
    Tensor& bitwise_and_(const Scalar& other) const;
    Tensor bitwise_and(const Scalar& other) const;
    
    Tensor& bitwise_or_(const Tensor& other) const;
    Tensor bitwise_or(const Tensor& other) const;
    Tensor& bitwise_or_(const Scalar& other) const;
    Tensor bitwise_or(const Scalar& other) const;
    
    Tensor& bitwise_xor_(const Tensor& other) const;
    Tensor bitwise_xor(const Tensor& other) const;
    Tensor& bitwise_xor_(const Scalar& other) const;
    Tensor bitwise_xor(const Scalar& other) const;
    
    Tensor& bitwise_not_() const;
    Tensor bitwise_not() const;
    
    Tensor eq(const Scalar & other) const;
    Tensor eq(const Tensor & other) const;
    Tensor & eq_(const Scalar & other) const;
    Tensor & eq_(const Tensor & other) const;
    
    Tensor ne(const Scalar & other) const;
    Tensor ne(const Tensor & other) const;
    Tensor & ne_(const Scalar & other) const;
    Tensor & ne_(const Tensor & other) const;
    
    Tensor ge(const Scalar & other) const;
    Tensor ge(const Tensor & other) const;
    Tensor & ge_(const Scalar & other) const;
    Tensor & ge_(const Tensor & other) const;
    
    Tensor le(const Scalar & other) const;
    Tensor le(const Tensor & other) const;
    Tensor & le_(const Scalar & other) const;
    Tensor & le_(const Tensor & other) const;
    
    Tensor gt(const Scalar & other) const;
    Tensor gt(const Tensor & other) const;
    Tensor & gt_(const Scalar & other) const;
    Tensor & gt_(const Tensor & other) const;
    
    Tensor lt(const Scalar & other) const;
    Tensor lt(const Tensor & other) const;
    Tensor & lt_(const Scalar & other) const;
    Tensor & lt_(const Tensor & other) const;
    
    Tensor& neg_() const;
    Tensor neg() const;
    
    Tensor& abs_() const;
    Tensor abs() const;
    
    Tensor& sin_() const;
    Tensor sin() const;
    
    Tensor& cos_() const;
    Tensor cos() const;
    
    Tensor& tan_() const;
    Tensor tan() const;
    
    Tensor& exp_() const;
    Tensor exp() const;
    
    Tensor& sqrt_() const;
    Tensor sqrt() const;
    
    Tensor dot(const Tensor& other) const;
    
    Tensor addmm(const Tensor& mat1, const Tensor& mat2, const Scalar& beta = 1, const Scalar& alpha = 1) const;
    Tensor& addmm_(const Tensor& mat1, const Tensor& mat2, const Scalar& beta = 1, const Scalar& alpha = 1) const;
    
    Tensor mm(const Tensor& other) const;
    
    Tensor softmax(int64_t dim, ScalarType dtype = ScalarType::Undefined) const;
    
};

template <typename T, typename... Args>
Tensor make_tensor(Args&&... args) {
    return Tensor(make_otterptr<T>(std::forward<Args>(args)...));
}

#if __ARM_NEON
OTTER_ALWAYS_INLINE Tensor& Tensor::fill_(float32x4_t _v) {
    int size = (int)numel();
    float* ptr = (float*)raw_data();
    for (int i = 0; i < size; i++) {
        vst1q_f32(ptr, _v);
        ptr += 4;
    }
    
    return *this;
}

OTTER_ALWAYS_INLINE Tensor& Tensor::fill_(uint16x4_t _v) {
    int size = (int)numel();
    unsigned short* ptr = (unsigned short*)raw_data();
    for (int i = 0; i < size; i++) {
        vst1_u16(ptr, _v);
        ptr += 4;
    }
    
    return *this;
}

OTTER_ALWAYS_INLINE Tensor& Tensor::fill_(int32x4_t _v) {
    int size = (int)numel();
    int* ptr = (int*)raw_data();
    for (int i = 0; i < size; i++) {
        vst1q_s32(ptr, _v);
        ptr += 4;
    }
    
    return *this;
}

OTTER_ALWAYS_INLINE Tensor& Tensor::fill_(int32x4_t _v0, int32x4_t _v1) {
    int size = (int)numel();
    int* ptr = (int*)raw_data();
    for (int i = 0; i < size; i++) {
        vst1q_s32(ptr, _v0);
        vst1q_s32(ptr + 4, _v1);
        ptr += 8;
    }
    
    return *this;
}
#endif // __ARM_NEON

#if __SSE2__
#if __AVX__
OTTER_ALWAYS_INLINE Tensor& Tensor::fill_(__m256 _v) {
    int size = (int)numel();
    float* ptr = (float*)raw_data();
    for (int i = 0; i < size; i++) {
        _mm256_storeu_ps(ptr, _v);
        ptr += 8;
    }
    
    return *this;
}
#endif // __AVX__
OTTER_ALWAYS_INLINE Tensor& Tensor::fill_(__m128 _v) {
    int size = (int)numel();
    float* ptr = (float*)raw_data();
    for (int i = 0; i < size; i++) {
        _mm_storeu_ps(ptr, _v);
        ptr += 4;
    }
    
    return *this;
}

OTTER_ALWAYS_INLINE Tensor& Tensor::fill_(__m128i _v) {
    int size = (int)numel();
    unsigned short* ptr = (unsigned short*)raw_data();
    for (int i = 0; i < size; i++) {
        _mm_store_si128((__m128i*)ptr, _v);
        ptr += 8;
    }
    
    return *this;
}
#endif // __SSE2__

template <typename scalar_t>
Tensor& Tensor::type_fill_(scalar_t value) {
    int size = (int)numel();
    scalar_t* ptr = (scalar_t*)raw_data();
    for (int i = 0; i < size; i++) {
        ptr[i] = value;
    }
    
    return *this;
}

template <>
struct MaybeOwnedTraits<Tensor> {
    using owned_type = Tensor;
    using borrow_type = Tensor;
    
    static borrow_type create_borrow(const owned_type& from) {
        return borrow_type(borrow_type::unsafe_borrow_t{}, from);
    }
    
    static void assign_borrow(borrow_type& lhs, const borrow_type& rhs) {
        lhs.unsafeReleaseTensorNucleus();
        lhs = borrow_type(borrow_type::unsafe_borrow_t{}, rhs);
    }
    
    static void destroy_borrow(borrow_type& toDestroy) {
        toDestroy.unsafeReleaseTensorNucleus();
    }
    
    static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
        return borrow;
    }
    
    static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
        return &borrow;
    }
};

template <>
struct ExclusivelyOwnedTraits<Tensor> {
    using repr_type = Tensor;
    using pointer_type = Tensor*;
    using const_pointer_type = const Tensor*;
    
    static repr_type nullRepr() {
        return Tensor();
    }
    
    template <class... Args>
    static repr_type createInPlace(Args&&... args) {
        return Tensor(std::forward<Args>(args)...);
    }
    
    static repr_type moveToRepr(Tensor&& x) {
        return std::move(x);
    }
    
    static void destroyOwned(Tensor& x) {
        return ExclusivelyOwnedTraits<TensorBase>::destroyOwned(x);
    }
    
    static Tensor take(Tensor& x) {
        return std::move(x);
    }
    
    static pointer_type getImpl(repr_type& x) {
        return &x;
    }
    
    static const_pointer_type getImpl(const repr_type& x) {
        return &x;
    }
};

class TensorPrinter {
public:
    TensorPrinter(int limit) : limit_(limit) {}
    
    template <class T>
    void print(const Tensor& tensor);
private:
    int limit_;
};

template <class T>
void TensorPrinter::print(const Tensor& tensor) {
    int max_length = static_cast<int>(std::min(tensor.numel(), int64_t(limit_)));
    int elempack = tensor.elempack();
    const T* tensor_data = (const T*)tensor.raw_data();
    
    for (int i = 0; i < max_length; ++i) {
        if (i)
            std::cout << ", ";
        if (elempack > 1) {
            std::cout << "[";
            for (int j = 0; j < elempack; ++j) {
                if (j)
                    std::cout << ", ";
                std::cout << *tensor_data;
                tensor_data++;
            }
            std::cout << "]";
        } else {
            std::cout << *tensor_data;
            tensor_data++;
        }
    }
    std::cout << std::endl;
}

}   // namespace otter

#endif /* Tensor_hpp */

