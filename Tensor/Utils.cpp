//
//  Utils.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "Utils.hpp"

namespace otter {

std::ostream& operator<<(std::ostream& out, const Range& range) {
    out << "Range[" << range.begin << ", " << range.end << "]";
    return out;
}

}   // end namespace otter
