//
//  ScalarType.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include <sstream>
#include "ScalarType.hpp"


std::string toString(ScalarType type) {
    switch(type) {
#define DEFINE_STR(_1, n) case ScalarType::n: return std::string(#n); break;
        OTTER_ALL_SCALAR_TYPES(DEFINE_STR)
#undef DEFINE_STR
        default:
            return std::string("Undefined");
    }
}

void report_overflow(const char* name) {
    std::ostringstream oss;
    oss << "value cannot be converted to type " << name << " without overflow";
    throw std::runtime_error(oss.str());
}
