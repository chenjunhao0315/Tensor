//
//  MemoryFormat.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/11.
//

#ifndef MemoryFormat_hpp
#define MemoryFormat_hpp

#include <cstdint>

#include "Utils.hpp"
#include "Exception.hpp"

namespace otter {

enum class MemoryFormat : int8_t {
    Contiguous,
    Preserve,
    ChannelsLast,
    ChannelsLast3d
};

inline std::vector<int64_t> get_channels_last_strides_2d(IntArrayRef sizes) {
    std::vector<int64_t> strides(sizes.size());
    switch (sizes.size()) {
        case 4:
            strides[1] = 1;
            strides[3] = sizes[1];
            strides[2] = strides[3] * sizes[3];
            strides[0] = strides[2] * sizes[2];
            return strides;
        case 3:
            strides[0] = 1;
            strides[2] = sizes[0];
            strides[1] = strides[2] * sizes[2];
            return strides;
        default:
            assert(false);  // "ChannelsLast2d doesn't support size ", sizes.size());
    }
}

inline bool is_channels_last_strides_2d_s4(const IntArrayRef sizes, const IntArrayRef strides) {
    int64_t min = 0;
    // special case for trivial C dimension. default to NCHW
    if (strides[1] == 0) {
        return false;
    }
    // loop strides indices
    for (auto& d : {1, 3, 2, 0}) {
        if (sizes[d] == 0) {
            return false;
        }
        if (strides[d] < min) {
            return false;
        }
        // Fallback to NCHW as default layout for ambiguous cases
        // This is the flaw of implicit memory_format from strides.
        // N111 tensor with identical strides for size 1 dimension;
        // Two cases could lead us here:
        // a. N111 contiguous Tensor ([N,1,1,1]@[1,1,1,1])
        // b. N11W contiguous Tensor sliced on the W-dimension.
        // ([N,1,1,1]@[W,W,W,W])
        if (d == 0 && min == strides[1]) {
            return false;
        }
        // This is necessary to:
        // 1. distinguish the memory_format of N1H1;
        //     [H, 1, 1, 1] channels_last stride
        //     [H, H, 1, 1] contiguous stride
        // 2. permutation of 1C1W:
        //     [1, C, 1, H]@[HC, H, H, 1] transpose(1, 3)
        //     [1, H, 1, C]@[HC, 1, H, H] shouldn't be identified as channels_last
        min = strides[d];
        if (sizes[d] > 1) {
            min *= sizes[d];
        }
    }
    return true;
}

inline bool is_channels_last_strides_2d(const IntArrayRef sizes, const IntArrayRef strides) {
    switch (sizes.size()) {
        case 4:
            return is_channels_last_strides_2d_s4(sizes, strides);
        case 3:
            // TODO dim == 3 case will be enabled once it is fully tested
            return false;
        default:
            return false;
    }
}


}

#endif /* MemoryFormat_hpp */
