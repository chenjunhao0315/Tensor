//
//  ArrayRef.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef ArrayRef_hpp
#define ArrayRef_hpp

#include <vector>
#include <iterator>
#include <initializer_list>

#include "SmallVector.hpp"

namespace otter {

template <typename T>
class ArrayRef {
public:
    using iterator = const T*;
    using const_iterator = const T*;
    using value_type = T;
    
    using reverse_iterator = std::reverse_iterator<iterator>;
    
    constexpr ArrayRef() : data_(nullptr), length_(0) {}
    constexpr ArrayRef(T& single) : data_(&single), length_(1) {}
    ArrayRef(const T* data, size_t length) : data_(data), length_(length) {}
    ArrayRef(const T* begin, const T* end) : data_(begin), length_(end - begin) {}
    
    template <typename U>
    ArrayRef(const SmallVectorTemplateCommon<T, U>& Vec) : data_(Vec.data()), length_(Vec.size()) {
//        debugCheckNullptrInvariant();
    }
    
    template <typename A>
    ArrayRef(const std::vector<T, A>& vec) : data_(vec.data()), length_(vec.size()) {
        static_assert(
            !std::is_same<T, bool>::value,
            "ArrayRef<bool> cannot be constructed from a std::vector<bool> bitfield.");
    }
    
    constexpr ArrayRef(const std::initializer_list<T>& vec) : data_((std::begin(vec) == std::end(vec)) ? static_cast<T*>(nullptr) : std::begin(vec)), length_(vec.size()) {}
    
    template <typename Container, typename = std::enable_if_t<std::is_same<std::remove_const_t<decltype(std::declval<Container>().data())>, T*>::value>>
    ArrayRef(const Container& container) : data_(container.data()), length_(container.size()) {
//        debugCheckNullptrInvariant();
    }
    
    template <size_t N>
    constexpr ArrayRef(const T (&Arr)[N]) : data_(Arr), length_(N) {}
    
    constexpr iterator begin() const {
        return data_;
    }
    
    constexpr iterator end() const {
        return data_ + length_;
    }
    
    // These are actually the same as iterator, since ArrayRef only
    // gives you const iterators.
    constexpr const_iterator cbegin() const {
      return data_;
    }
    
    constexpr const_iterator cend() const {
      return data_ + length_;
    }
    
    constexpr reverse_iterator rbegin() const {
      return reverse_iterator(end());
    }
    
    constexpr reverse_iterator rend() const {
      return reverse_iterator(begin());
    }
    
    constexpr const_iterator const_begin() const {
        return data_;
    }
    
    constexpr const_iterator const_end() const {
        return data_ + length_;
    }
    
    constexpr bool empty() const {
        return length_ == 0;
    }
    
    constexpr const T* data() const {
        return data_;
    }
    
    constexpr size_t size() const {
        return length_;
    }
    
    const T& front() const {
        if (empty()) fprintf(stderr, "[ArrayRef] Empty array!\n");
        return data_[0];
    }
    
    const T& back() const {
        if (empty()) fprintf(stderr, "[ArrayRef] Empty array!\n");
        return data_[length_ - 1];;
    }
    
    ArrayRef<T> slice(size_t N, size_t M) const {
        return ArrayRef<T>(data() + N, M);
    }
    
    ArrayRef<T> slice(size_t N) const {
        return slice(N, size() - N);
    }
    
    constexpr const T& operator[](size_t idx) const {
        return data_[idx];
    }
    
    std::vector<T> vec() {
        return std::vector<T>(data_, data_ + length_);
    }
    
    constexpr bool equals(ArrayRef RHS) const {
        return length_ == RHS.length_ && std::equal(begin(), end(), RHS.begin());
    }
private:
    const T* data_;
    size_t length_;
};

template <typename T>
std::ostream& operator<<(std::ostream& out, ArrayRef<T> list) {
    int i = 0;
    out << "[";
    for (auto e : list) {
        if (i++ > 0)
            out << ", ";
        out << e;
    }
    out << "]";
    return out;
}

template <typename T>
bool operator==(otter::ArrayRef<T> a1, otter::ArrayRef<T> a2) {
    return a1.equals(a2);
}

template <typename T>
bool operator!=(otter::ArrayRef<T> a1, otter::ArrayRef<T> a2) {
    return !a1.equals(a2);
}

template <typename T>
bool operator==(const std::vector<T>& a1, otter::ArrayRef<T> a2) {
    return otter::ArrayRef<T>(a1).equals(a2);
}

template <typename T>
bool operator!=(const std::vector<T>& a1, otter::ArrayRef<T> a2) {
    return !otter::ArrayRef<T>(a1).equals(a2);
}

template <typename T>
bool operator==(otter::ArrayRef<T> a1, const std::vector<T>& a2) {
    return a1.equals(otter::ArrayRef<T>(a2));
}

template <typename T>
bool operator!=(otter::ArrayRef<T> a1, const std::vector<T>& a2) {
    return !a1.equals(otter::ArrayRef<T>(a2));
}

using IntArrayRef = ArrayRef<int64_t>;

class Tensor;
using TensorList = ArrayRef<Tensor>;

}   // end namespace otter

#endif /* ArrayRef_hpp */
