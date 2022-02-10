//
//  TypeCast.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/10.
//

#ifndef TypeCast_h
#define TypeCast_h

namespace otter {

template <typename dest_t, typename src_t>
struct static_cast_with_inter_type {
    static inline dest_t apply(src_t src) {
        return static_cast<dest_t>(src);
    }
};

template <typename src_t>
struct static_cast_with_inter_type<uint8_t, src_t> {
    static inline uint8_t apply(src_t src) {
        return static_cast<uint8_t>(static_cast<int64_t>(src));
    }
};




}

#endif /* TypeCast_h */
