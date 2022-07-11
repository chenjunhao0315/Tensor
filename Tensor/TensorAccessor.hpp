//
//  TensorAccessor.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/15.
//

#ifndef TensorAccessor_hpp
#define TensorAccessor_hpp

#include "ArrayRef.hpp"

namespace otter {

template <typename T>
struct DefaultPtrTraits {
    typedef T* PtrType;
};

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class TensorAccessorBase {
public:
    typedef typename PtrTraits<T>::PtrType PtrType;
    
    TensorAccessorBase(PtrType data_, const index_t* sizes_, const index_t* strides_) : data_(data_), sizes_(sizes_), strides_(strides_) {}
    IntArrayRef sizes() const {
        return IntArrayRef(sizes_, N);
    }
    IntArrayRef strides() const {
        return IntArrayRef(strides_, N);
    }
    index_t stride(index_t i) const {
        return strides_[i];
    }
    index_t size(index_t i) const {
        return sizes_[i];
    }
    PtrType data() {
        return data_;
    }
    const PtrType data() const {
        return data_;
    }
protected:
    PtrType data_;
    const index_t* sizes_;
    const index_t* strides_;
};

template<typename T, size_t N, int64_t P = 1, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class TensorAccessor : public TensorAccessorBase<T, N, PtrTraits, index_t> {
public:
    typedef typename PtrTraits<T>::PtrType PtrType;
    
    TensorAccessor(PtrType data_, const index_t* sizes_, const index_t* strides_) : TensorAccessorBase<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}
    
    TensorAccessor<T, N - 1, P, PtrTraits, index_t> operator[](index_t i) {
        return TensorAccessor<T, N - 1, P, PtrTraits, index_t>(this->data_ + this->strides_[0] * i * P, this->sizes_ + 1, this->strides_ + 1);
    }
    
    const TensorAccessor<T, N - 1, P, PtrTraits, index_t> operator[](index_t i) const {
        return TensorAccessor<T, N - 1, P, PtrTraits, index_t>(this->data_ + this->strides_[0] * i * P, this->sizes_ + 1, this->strides_ + 1);
    }
};

template<typename T, template <typename U> class PtrTraits, typename index_t>
class TensorAccessor<T, 1, 1, PtrTraits, index_t> : public TensorAccessorBase<T, 1, PtrTraits,index_t> {
public:
    typedef typename PtrTraits<T>::PtrType PtrType;
    
    TensorAccessor(PtrType data_, const index_t* sizes_, const index_t* strides_) : TensorAccessorBase<T, 1, PtrTraits, index_t>(data_, sizes_, strides_) {}
    T & operator[](index_t i) {
        // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
        return this->data_[this->strides_[0] * i];
    }
    const T & operator[](index_t i) const {
        return this->data_[this->strides_[0] * i];
    }
};

template<typename T, template <typename U> class PtrTraits, typename index_t>
class TensorAccessor<T, 1, 4, PtrTraits, index_t> : public TensorAccessorBase<T, 1, PtrTraits,index_t> {
public:
    typedef typename PtrTraits<T>::PtrType PtrType;
    
    TensorAccessor(PtrType data_, const index_t* sizes_, const index_t* strides_) : TensorAccessorBase<T, 1, PtrTraits, index_t>(data_, sizes_, strides_) {}
    T* operator[](index_t i) {
        // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
        return this->data_ + this->strides_[0] * i * 4;
    }
    const T* operator[](index_t i) const {
        return this->data_ + this->strides_[0] * i * 4;
    }
};

template<typename T, template <typename U> class PtrTraits, typename index_t>
class TensorAccessor<T, 1, 8, PtrTraits, index_t> : public TensorAccessorBase<T, 1, PtrTraits,index_t> {
public:
    typedef typename PtrTraits<T>::PtrType PtrType;
    
    TensorAccessor(PtrType data_, const index_t* sizes_, const index_t* strides_) : TensorAccessorBase<T, 1, PtrTraits, index_t>(data_, sizes_, strides_) {}
    T* operator[](index_t i) {
        // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
        return this->data_ + this->strides_[0] * i * 8;
    }
    const T* operator[](index_t i) const {
        return this->data_ + this->strides_[0] * i * 8;
    }
};

template<typename T, size_t N, typename index_t = int64_t>
class TensorRawAccessorBase {
public:
    TensorRawAccessorBase(void* data_, const index_t* sizes_, const index_t* strides_, const index_t elemsize_) : data_(data_), sizes_(sizes_), strides_(strides_), elemsize_(elemsize_) {}
    IntArrayRef sizes() const {
        return IntArrayRef(sizes_, N);
    }
    IntArrayRef strides() const {
        return IntArrayRef(strides_, N);
    }
    index_t stride(index_t i) const {
        return strides_[i];
    }
    index_t size(index_t i) const {
        return sizes_[i];
    }
    T* data() {
        return (T*)data_;
    }
    const T* data() const {
        return (T*)data_;
    }
protected:
    void* data_;
    const index_t* sizes_;
    const index_t* strides_;
    const index_t elemsize_;
};

template<typename T, size_t N, typename index_t = int64_t>
class TensorRawAccessor : public TensorRawAccessorBase<T, N, index_t> {
public:
    TensorRawAccessor(void* data_, const index_t* sizes_, const index_t* strides_, const index_t elemsize_) : TensorRawAccessorBase<T, N, index_t>(data_, sizes_, strides_, elemsize_) {}
    
    TensorRawAccessor<T, N - 1, index_t> operator[](index_t i) {
        return TensorRawAccessor<T, N - 1, index_t>((unsigned char*)this->data_ + this->strides_[0] * i * this->elemsize_, this->sizes_ + 1, this->strides_ + 1, this->elemsize_);
    }
    
    const TensorRawAccessor<T, N - 1, index_t> operator[](index_t i) const {
        return TensorRawAccessor<T, N - 1, index_t>((unsigned char*)this->data_ + this->strides_[0] * i * this->elemsize_, this->sizes_ + 1, this->strides_ + 1, this->elemsize_);
    }
};

template<typename T, typename index_t>
class TensorRawAccessor<T, 1, index_t> : public TensorRawAccessorBase<T, 1, index_t> {
public:
    TensorRawAccessor(void* data_, const index_t* sizes_, const index_t* strides_, const index_t elemsize_) : TensorRawAccessorBase<T, 1, index_t>(data_, sizes_, strides_, elemsize_) {}
    T* operator[](index_t i) {
        // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
        return (T*)this->data_[this->strides_[0] * i * this->elemsize_];
    }
    const T* operator[](index_t i) const {
        return (T*)this->data_[this->strides_[0] * i * this->elemsize_];
    }
};

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

template <typename T>
struct RestrictPtrTraits {
    using PtrType = T* RESTRICT;
};

template <typename T, typename index_t = int64_t, template <typename U> class PtrTraits = DefaultPtrTraits>
class ConstStridedRandomAccessor {
public:
    using difference_type = index_t;
    using value_type = const T;
    using pointer = const typename PtrTraits<T>::PtrType;
    using reference = const value_type&;
    using iterator_category = std::random_access_iterator_tag;
    
    using PtrType = typename PtrTraits<T>::PtrType;
    using index_type = index_t;
    
    // Constructors {
    
    ConstStridedRandomAccessor(PtrType ptr, index_t stride)
    : ptr{ptr}, stride{stride}
    {}
    
    explicit ConstStridedRandomAccessor(PtrType ptr)
    : ptr{ptr}, stride{static_cast<index_t>(1)}
    {}
    
    ConstStridedRandomAccessor()
    : ptr{nullptr}, stride{static_cast<index_t>(1)}
    {}
    // }
    
    // Pointer-like operations {
    
    reference operator*() const {
        return *ptr;
    }
    
    const value_type* operator->() const {
        return reinterpret_cast<const value_type*>(ptr);
    }
    
    reference operator[](index_t idx) const {
        return ptr[idx * stride];
    }
    // }
    
    // Prefix/postfix increment/decrement {
    
    ConstStridedRandomAccessor& operator++() {
        ptr += stride;
        return *this;
    }
    
    ConstStridedRandomAccessor operator++(int) {
        ConstStridedRandomAccessor copy(*this);
        ++*this;
        return copy;
    }
    
    ConstStridedRandomAccessor& operator--() {
        ptr -= stride;
        return *this;
    }
    
    ConstStridedRandomAccessor operator--(int) {
        ConstStridedRandomAccessor copy(*this);
        --*this;
        return copy;
    }
    // }
    
    // Arithmetic operations {
    
    ConstStridedRandomAccessor& operator+=(index_t offset) {
        ptr += offset * stride;
        return *this;
    }
    
    ConstStridedRandomAccessor operator+(index_t offset) const {
        return ConstStridedRandomAccessor(ptr + offset * stride, stride);
    }
    
    friend ConstStridedRandomAccessor operator+(
                                                index_t offset,
                                                const ConstStridedRandomAccessor& accessor
                                                ) {
        return accessor + offset;
    }
    
    ConstStridedRandomAccessor& operator-=(index_t offset) {
        ptr -= offset * stride;
        return *this;
    }
    
    ConstStridedRandomAccessor operator-(index_t offset) const {
        return ConstStridedRandomAccessor(ptr - offset * stride, stride);
    }
    
    // Note that this operator is well-defined when `this` and `other`
    // represent the same sequences, i.e. when
    // 1. this.stride == other.stride,
    // 2. |other - this| / this.stride is an Integer.
    
    difference_type operator-(const ConstStridedRandomAccessor& other) const {
        return (ptr - other.ptr) / stride;
    }
    // }
    
    // Comparison operators {
    
    bool operator==(const ConstStridedRandomAccessor& other) const {
        return (ptr == other.ptr) && (stride == other.stride);
    }
    
    bool operator!=(const ConstStridedRandomAccessor& other) const {
        return !(*this == other);
    }
    
    bool operator<(const ConstStridedRandomAccessor& other) const {
        return ptr < other.ptr;
    }
    
    bool operator<=(const ConstStridedRandomAccessor& other) const {
        return (*this < other) || (*this == other);
    }
    
    bool operator>(const ConstStridedRandomAccessor& other) const {
        return !(*this <= other);
    }
    
    bool operator>=(const ConstStridedRandomAccessor& other) const {
        return !(*this < other);
    }
    // }
    
protected:
    PtrType ptr;
    index_t stride;
};

template <typename T, typename index_t = int64_t, template <typename U> class PtrTraits = DefaultPtrTraits>
class StridedRandomAccessor
: public ConstStridedRandomAccessor<T, index_t, PtrTraits> {
public:
    using difference_type = index_t;
    using value_type = T;
    using pointer = typename PtrTraits<T>::PtrType;
    using reference = value_type&;
    
    using BaseType = ConstStridedRandomAccessor<T, index_t, PtrTraits>;
    using PtrType = typename PtrTraits<T>::PtrType;
    
    // Constructors {
    
    StridedRandomAccessor(PtrType ptr, index_t stride)
    : BaseType(ptr, stride)
    {}
    
    explicit StridedRandomAccessor(PtrType ptr)
    : BaseType(ptr)
    {}
    
    StridedRandomAccessor()
    : BaseType()
    {}
    // }
    
    // Pointer-like operations {
    
    reference operator*() const {
        return *this->ptr;
    }
    
    value_type* operator->() const {
        return reinterpret_cast<value_type*>(this->ptr);
    }
    
    reference operator[](index_t idx) const {
        return this->ptr[idx * this->stride];
    }
    // }
    
    // Prefix/postfix increment/decrement {
    
    StridedRandomAccessor& operator++() {
        this->ptr += this->stride;
        return *this;
    }
    
    StridedRandomAccessor operator++(int) {
        StridedRandomAccessor copy(*this);
        ++*this;
        return copy;
    }
    
    StridedRandomAccessor& operator--() {
        this->ptr -= this->stride;
        return *this;
    }
    
    StridedRandomAccessor operator--(int) {
        StridedRandomAccessor copy(*this);
        --*this;
        return copy;
    }
    // }
    
    // Arithmetic operations {
    
    StridedRandomAccessor& operator+=(index_t offset) {
        this->ptr += offset * this->stride;
        return *this;
    }
    
    StridedRandomAccessor operator+(index_t offset) const {
        return StridedRandomAccessor(this->ptr + offset * this->stride, this->stride);
    }
    
    friend StridedRandomAccessor operator+(
                                           index_t offset,
                                           const StridedRandomAccessor& accessor
                                           ) {
        return accessor + offset;
    }
    
    StridedRandomAccessor& operator-=(index_t offset) {
        this->ptr -= offset * this->stride;
        return *this;
    }
    
    StridedRandomAccessor operator-(index_t offset) const {
        return StridedRandomAccessor(this->ptr - offset * this->stride, this->stride);
    }
    
    // Note that here we call BaseType::operator- version
    
    difference_type operator-(const BaseType& other) const {
        return (static_cast<const BaseType&>(*this) - other);
    }
    // }
};

// operator_brackets_proxy is used in
// CompositeRandomAccessor in place of operator[].
// For some iterators, references returned by operator[]
// could become invalid, operator_brackets_proxy tries to
// resolve that by making accessor[n] to be equivalent to
// *(accessor + n).
template <typename Accessor>
class operator_brackets_proxy {
    using reference = typename std::iterator_traits<Accessor>::reference;
    using value_type = typename std::iterator_traits<Accessor>::value_type;
    
public:
    
    operator_brackets_proxy(Accessor const& accessor)
    : accessor(accessor)
    {}
    
    
    operator reference() {
        return *accessor;
    }
    
    
    reference operator*() {
        return *accessor;
    }
    
    
    operator_brackets_proxy& operator=(value_type const& val) {
        *accessor = val;
        return *this;
    }
    
private:
    Accessor accessor;
};

// references_holder is used as a surrogate for the
// references type from std::iterator_traits in CompositeRandomAccessor.
// It is assumed in CompositeRandomAccessor that
// References = tuple<Types&...>,
// Values = tuple<Types...> by default,
// but they could be anything as long as References could be
// cast to Values.
// If you plan to use it with STL, for example, you will need to
// define 'swap` and `get`(aka std::get) methods.
template <typename Values, typename References>
class references_holder {
public:
    using values = Values;
    using references = References;
    
    
    references_holder(references refs)
    : refs{refs}
    {}
    
    
    operator references() {
        return refs;
    }
    
    
    operator values() {
        return refs;
    }
    
    
    references_holder& operator=(values vals) {
        refs = vals;
        return *this;
    }
    
    
    references& data() {
        return refs;
    }
    
protected:
    references refs;
};

// CompositeRandomAccessor is essentially a simplified version of
// a random access iterator over two random access iterators.
// TupleInfo should contain a variadic type `tuple`, and a method `tie`,
// which constructs a tuple of references from a variadic list of arguments.
template <typename KeyAccessor, typename ValueAccessor, typename TupleInfo>
class CompositeRandomAccessor {
    using self_type = CompositeRandomAccessor<KeyAccessor, ValueAccessor, TupleInfo>;
    
    using key_accessor_value_type =
    typename std::iterator_traits<KeyAccessor>::value_type;
    using value_accessor_value_type =
    typename std::iterator_traits<ValueAccessor>::value_type;
    using key_accessor_reference_type =
    typename std::iterator_traits<KeyAccessor>::reference;
    using value_accessor_reference_type =
    typename std::iterator_traits<ValueAccessor>::reference;
    
    using composite_value_type = typename TupleInfo::template tuple<
    key_accessor_value_type,
    value_accessor_value_type>;
    using composite_reference = typename TupleInfo::template tuple<
    key_accessor_reference_type,
    value_accessor_reference_type>;
    
public:
    using value_type = composite_value_type;
    using reference = references_holder<composite_value_type, composite_reference>;
    // Note that CompositeRandomAccessor does not hold key and values
    // in a specific datastrcture, which means that a pointer to a (key, value)
    // is not defined. Hence we just use a pointer type of the KeyAccessor.
    using pointer = typename std::iterator_traits<KeyAccessor>::pointer;
    using difference_type = typename std::iterator_traits<KeyAccessor>::difference_type;
    using iterator_category = std::random_access_iterator_tag;
    
    CompositeRandomAccessor() = default;
    
    CompositeRandomAccessor(KeyAccessor keys, ValueAccessor values)
    : keys(keys), values(values)
    {}
    
    // Pointer-like operations {
    
    reference operator*() const {
        return TupleInfo::tie(*keys, *values);
    }
    
    // operator->() is supposed to return a pointer type.
    // Since CompositeRandomAccessor does not hold pointers to pairs,
    // we just return a pointer to a key.
    
    auto* operator->() const {
        return keys.operator->();
    }
    
    reference operator[](difference_type idx) {
        return operator_brackets_proxy<self_type>(
                                                  CompositeRandomAccessor(keys + idx, values + idx)
                                                  );
    }
    // }
    
    // Prefix/postfix increment/decrement {
    
    CompositeRandomAccessor& operator++() {
        ++keys;
        ++values;
        return *this;
    }
    
    CompositeRandomAccessor operator++(int) {
        CompositeRandomAccessor copy(*this);
        ++*this;
        return copy;
    }
    
    CompositeRandomAccessor& operator--() {
        --keys;
        --values;
        return *this;
    }
    
    CompositeRandomAccessor operator--(int) {
        CompositeRandomAccessor copy(*this);
        --*this;
        return copy;
    }
    // }
    
    // Arithmetic operations {
    
    CompositeRandomAccessor& operator+=(difference_type offset) {
        keys += offset;
        values += offset;
        return *this;
    }
    
    CompositeRandomAccessor operator+(difference_type offset) const {
        return CompositeRandomAccessor(keys + offset, values + offset);
    }
    
    friend CompositeRandomAccessor operator+(
                                             difference_type offset,
                                             const CompositeRandomAccessor& accessor
                                             ) {
        return accessor + offset;
    }
    
    CompositeRandomAccessor& operator-=(difference_type offset) {
        keys -= offset;
        values -= offset;
        return *this;
    }
    
    CompositeRandomAccessor operator-(difference_type offset) const {
        return CompositeRandomAccessor(keys - offset, values - offset);
    }
    
    difference_type operator-(const CompositeRandomAccessor& other) const {
        return keys - other.keys;
    }
    // }
    
    // Comparison operators {
    
    bool operator==(const CompositeRandomAccessor& other) const {
        return keys == other.keys;
    }
    
    bool operator!=(const CompositeRandomAccessor& other) const {
        return keys != other.keys;
    }
    
    bool operator<(const CompositeRandomAccessor& other) const {
        return keys < other.keys;
    }
    
    bool operator<=(const CompositeRandomAccessor& other) const {
        return keys <= other.keys;
    }

    bool operator>(const CompositeRandomAccessor& other) const {
        return keys > other.keys;
    }
    
    bool operator>=(const CompositeRandomAccessor& other) const {
        return keys >= other.keys;
    }
    // }
    
protected:
    KeyAccessor keys;
    ValueAccessor values;
};

struct TupleInfoCPU {
    template <typename ...Types>
    using tuple = std::tuple<Types...>;
    
    template <typename ...Types>
    static constexpr auto tie(Types&... args) noexcept {
        return std::tie(args...);
    }
};

template <typename KeyAccessor, typename ValueAccessor>
using CompositeRandomAccessorCPU =
CompositeRandomAccessor<KeyAccessor, ValueAccessor, TupleInfoCPU>;

template <typename Values, typename References>
void swap(
          references_holder<Values, References> rh1,
          references_holder<Values, References> rh2
          ) {
    return std::swap(rh1.data(), rh2.data());
}

template <int N, typename Values, typename References>
auto get(references_holder<Values, References> rh) -> decltype(std::get<N>(rh.data())) {
    return std::get<N>(rh.data());
}

}

#endif /* TensorAccessor_hpp */
