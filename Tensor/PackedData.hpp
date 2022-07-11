//
//  PackedData.h
//  Tensor
//
//  Created by 陳均豪 on 2022/6/9.
//

#ifndef PackedData_h
#define PackedData_h

namespace otter {

template <typename T>
struct alignas(sizeof(T) * 4) elempack4 {
    using value_type = T;
    
    T values[4] = {0};
};

template <typename T>
struct alignas(sizeof(T) * 8) elempack8 {
    using value_type = T;
    
    T values[8] = {0};
};

template <typename T>
struct alignas(sizeof(T) * 16) elempack16 {
    using value_type = T;
    
    T values[16] = {0};
};

template <typename T>
struct alignas(sizeof(T) * 32) elempack32 {
    using value_type = T;
    
    T values[32] = {0};
};

template <typename T>
struct alignas(sizeof(T) * 64) elempack64 {
    using value_type = T;
    
    T values[64] = {0};
};

}   // end namespace otter


#endif /* PackedData_h */
