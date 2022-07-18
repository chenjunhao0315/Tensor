//
//  Device.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/30.
//

#include "Device.hpp"

std::ostream& operator<<(std::ostream& os, Device& d) {
    switch (d) {
        case Device::CPU:
            os << "CPU"; break;
        default:
            os << "Undefined"; break;
    }
    return os;
}
