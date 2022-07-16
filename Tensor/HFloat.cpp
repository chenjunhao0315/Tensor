//
//  HFloat.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/16.
//

#include "HFloat.hpp"

#include <iostream>

namespace otter {

std::ostream& operator<<(std::ostream& out, const HFloat& value) {
    out << (float)value;
    return out;
}

}   // end namespace otte
