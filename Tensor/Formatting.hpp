//
//  Formatting.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/11.
//

#ifndef Formatting_hpp
#define Formatting_hpp

#include <ostream>
#include "Tensor.hpp"

namespace otter {

void print(const Tensor & t, int64_t linesize=80);
std::ostream& print(std::ostream& out, const Tensor& t, int64_t linesize);

static inline std::ostream& operator<<(std::ostream& out, const Tensor& t) {
    return print(out, t, 80);
}

static inline std::ostream& operator<<(std::ostream& out, Scalar s) {
    if (s.isFloatingPoint()) {
        return out << s.toDouble();
    }
    if (s.isBoolean()) {
        return out << (s.toBool() ? "true" : "false");
    }
    if (s.isIntegral(false)) {
        return out << s.toLong();
    }
    throw std::logic_error("Unknown type in Scalar");
    return out;
}

}

#endif /* Formatting_hpp */
