//
//  Device.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/30.
//

#ifndef Device_hpp
#define Device_hpp

#include <iostream>
#include <cstdint>
#include "Macro.hpp"

enum class Device : int16_t {
    CPU,
    Undefined
};

static OTTER_UNUSED std::ostream& operator<<(std::ostream& os, Device& d) {
    switch (d) {
        case Device::CPU:
            os << "CPU"; break;
        default:
            os << "Undefined"; break;
    }
    return os;
}

#endif /* Device_hpp */
