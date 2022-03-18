//
//  Tensor.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef Tensor_hpp
#define Tensor_hpp

#include "TensorBase.hpp"
#include "TensorAccessor.hpp"

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
    
    template<typename T, size_t N>
    TensorAccessor<T,N> accessor() const& {
        static_assert(N > 0, "accessor is used for indexing tensor, for scalars use *data_ptr<T>()");
        OTTER_CHECK(dim() == N, "TensorAccessor expected ", N, " dims but tensor has ", dim());
        return TensorAccessor<T,N>(data_ptr<T>(), sizes().data(), strides().data());
    }
    template<typename T, size_t N>
    TensorAccessor<T,N> accessor() && = delete;
    
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
    
    template <typename T>
    T item() const;
    
    Scalar item() const;
    
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
    
    Tensor to(ScalarType dtype, bool non_blocking = false, bool copy = false, MemoryFormat memory_format = MemoryFormat::Preserve) const;
    
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
    
};

template <typename T, typename... Args>
Tensor make_tensor(Args&&... args) {
    return Tensor(make_otterptr<T>(std::forward<Args>(args)...));
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
    const T* tensor_data = tensor.data_ptr<T>();
    
    for (int i = 0; i < max_length - 1; ++i) {
        std::cout << tensor_data[i] << ",";
    }
    if (max_length) {
        std::cout << tensor_data[max_length - 1];
    }
    std::cout << std::endl;
}

}   // namespace otter

#endif /* Tensor_hpp */

