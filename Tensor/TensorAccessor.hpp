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

template<typename T, size_t N, int64_t P, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
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

}

#endif /* TensorAccessor_hpp */
