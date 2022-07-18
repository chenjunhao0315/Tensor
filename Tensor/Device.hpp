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

std::ostream& operator<<(std::ostream& os, Device& d);

#endif /* Device_hpp */
