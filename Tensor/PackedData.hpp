//
//  PackedData.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/5/11.
//

#ifndef PackedData_h
#define PackedData_h

#include <cstdint>

namespace otter {

template <typename T, size_t size>
struct alignas(sizeof(T) * size) pdata {
    using value_type = T;
    using size_type = int;
    
    T values[size];
    
    static constexpr size_type elempack() {
        return size;
    }
    
    pdata() : values{0} {}
    
    pdata(T val) {
        for (int i = 0; i < elempack(); ++i) {
            values[i] = val;
        }
    }
};

typedef pdata<float, 4> floatp4;

}   // end namespace otter


#endif /* PackedData_h */
