//
//  C++17.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef C__17_hpp
#define C__17_hpp

#include <type_traits>
#include <utility>
#include <memory>
#include <sstream>
#include <string>
#include <cstdlib>
#include <functional>

namespace otter {


#ifdef __cpp_lib_apply

template <class F, class Tuple>
inline constexpr decltype(auto) apply(F&& f, Tuple&& t) {
    return std::apply(std::forward<F>(f), std::forward<Tuple>(t));
}

#else

namespace detail {
template <class F, class Tuple, std::size_t... INDEX>
#if defined(_MSC_VER)
constexpr auto apply_impl(F&& f, Tuple&& t, std::index_sequence<INDEX...>)
#else
constexpr decltype(auto) apply_impl(F&& f, Tuple&& t, std::index_sequence<INDEX...>)
#endif
{
    return std::forward<F>(f)(std::get<INDEX>(std::forward<Tuple>(t))...);
}
}  // namespace detail

template <class F, class Tuple>
constexpr decltype(auto) apply(F&& f, Tuple&& t) {
    return detail::apply_impl(
        std::forward<F>(f), std::forward<Tuple>(t),
        std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

#endif





}

#endif /* C__17_hpp */
