//
//  GeneratorNucleus.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/18.
//

#include "GeneratorNucleus.hpp"

namespace otter {

GeneratorNucleus::GeneratorNucleus(Device device) : device_(device) {}

Ptr<GeneratorNucleus> GeneratorNucleus::clone() const {
    auto res = this->clone_impl();
    otter::raw::ptr::incref(res);
    return Ptr<GeneratorNucleus>::reclaim(res);
}

Device GeneratorNucleus::device() const {
    return device_;
}

namespace detail {

uint64_t getNonDeterministicRandom(bool is_cuda) {
    uint64_t s;
    s = (uint64_t)std::chrono::high_resolution_clock::now()
        .time_since_epoch()
        .count();
    return s;
}

}

}   // end namespace otter
