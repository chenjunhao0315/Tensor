//
//  Tensor.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef Tensor_hpp
#define Tensor_hpp

#include "TensorBase.hpp"

namespace otter {

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
    
    template <typename T>
    Tensor& resize_(ArrayRef<T> shape) {
        tensor_nucleus_->Resize(shape);
        return *this;
    }
    
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
    
    Tensor to(ScalarType dtype) const;
    
    Tensor& add_(const Tensor& other, const Scalar& alpha = 1);
    Tensor add(const Tensor& other, const Scalar& alpha = 1) const;
    Tensor& add_(const Scalar& other, const Scalar& alpha = 1);
    Tensor add(const Scalar& other, const Scalar& alpha = 1) const;
    
    Tensor& sub_(const Tensor& other, const Scalar& alpha = 1);
    Tensor sub(const Tensor& other, const Scalar& alpha = 1) const;
    Tensor& sub_(const Scalar& other, const Scalar& alpha = 1);
    Tensor sub(const Scalar& other, const Scalar& alpha = 1) const;
    
    Tensor& mul_(const Tensor& other);
    Tensor mul(const Tensor& other) const;
    Tensor& mul_(const Scalar& other);
    Tensor mul(const Scalar& other) const;
    
    Tensor& div_(const Tensor& other);
    Tensor div(const Tensor& other) const;
    Tensor& div_(const Scalar& other);
    Tensor div(const Scalar& other) const;
    
    Tensor& remainder_(const Tensor& other);
    Tensor remainder(const Tensor& other) const;
    Tensor& remainder_(const Scalar& other);
    Tensor remainder(const Scalar& other) const;
    
    Tensor& bitwise_and_(const Tensor& other);
    Tensor bitwise_and(const Tensor& other) const;
    Tensor& bitwise_and_(const Scalar& other);
    Tensor bitwise_and(const Scalar& other) const;
    
    Tensor& bitwise_or_(const Tensor& other);
    Tensor bitwise_or(const Tensor& other) const;
    Tensor& bitwise_or_(const Scalar& other);
    Tensor bitwise_or(const Scalar& other) const;
    
    Tensor& bitwise_xor_(const Tensor& other);
    Tensor bitwise_xor(const Tensor& other) const;
    Tensor& bitwise_xor_(const Scalar& other);
    Tensor bitwise_xor(const Scalar& other) const;
    
    Tensor& bitwise_not_();
    Tensor bitwise_not() const;
    
    Tensor& neg_();
    Tensor neg() const;
    
    Tensor& abs_();
    Tensor abs() const;
    
    Tensor& sin_();
    Tensor sin() const;
    
    Tensor& cos_();
    Tensor cos() const;
    
    Tensor& tan_();
    Tensor tan() const;
    
    Tensor& exp_();
    Tensor exp() const;
    
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
    const T* tensor_data = tensor.data<T>();
    
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

