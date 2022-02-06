//
//  in_place.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/31.
//

#ifndef in_place_h
#define in_place_h

namespace otter {

struct in_place_t {
    explicit in_place_t() = default;
};

template <std::size_t I>
struct in_place_index_t {
    explicit in_place_index_t() = default;
};

template <typename T>
struct in_place_type_t {
    explicit in_place_type_t() = default;
};

constexpr in_place_t in_place{};

}


#endif /* in_place_h */
