//
//  Accumulator.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef Accumulator_hpp
#define Accumulator_hpp

#include <iterator>
#include <numeric>
#include <type_traits>

namespace otter {

template <typename C, typename std::enable_if<std::is_integral<typename C::value_type>::value, int>::type = 0>
inline int64_t sum_integers(const C& container) {
    return std::accumulate(container.begin(), container.end(), static_cast<int64_t>(0));
}

template <typename Iter, typename std::enable_if<std::is_integral<typename std::iterator_traits<Iter>::value_type>::value, int>::type = 0>
inline int64_t sum_integers(Iter begin, Iter end) {
    return std::accumulate(begin, end, static_cast<int64_t>(0));
}

template <typename C, typename std::enable_if<std::is_integral<typename C::value_type>::value, int>::type = 0>
inline int64_t multiply_integers(const C& container) {
    return std::accumulate(
                           container.begin(),
                           container.end(),
                           static_cast<int64_t>(1),
                           std::multiplies<int64_t>());
}

template <typename Iter, typename std::enable_if<std::is_integral<typename std::iterator_traits<Iter>::value_type>::value, int>::type = 0>
inline int64_t multiply_integers(Iter begin, Iter end) {
    return std::accumulate(
                           begin, end, static_cast<int64_t>(1), std::multiplies<int64_t>());
}

template <typename C, typename std::enable_if<std::is_integral<typename C::value_type>::value, int>::type = 0>
inline int64_t numelements_from_dim(const int k, const C& dims) {
    if (k > dims.size()) {
        return 1;
    } else {
        auto cbegin = dims.cbegin();
        std::advance(cbegin, k);
        return multiply_integers(cbegin, dims.cend());
    }
}

template <typename C, typename std::enable_if<std::is_integral<typename C::value_type>::value, int>::type = 0>
inline int64_t numelements_to_dim(const int k, const C& dims) {
    auto cend = dims.cbegin();
    std::advance(cend, k);
    return multiply_integers(dims.cbegin(), cend);
}


template <typename C, typename std::enable_if<std::is_integral<typename C::value_type>::value, int>::type = 0>
inline int64_t numelements_between_dim(int k, int l, const C& dims) {
    if (k > l) {
        std::swap(k, l);
    }
    
    auto cbegin = dims.cbegin();
    auto cend = dims.cbegin();
    std::advance(cbegin, k);
    std::advance(cend, l);
    return multiply_integers(cbegin, cend);
}

}   // end namespace oter

#endif /* Accumulator_hpp */
