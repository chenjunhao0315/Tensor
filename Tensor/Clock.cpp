//
//  Clock.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/16.
//

#include "Clock.hpp"

namespace otter {

Clock::Clock() {
    time_start = high_resolution_clock::now();
    time_stop = high_resolution_clock::now();
}

void Clock::start() {
    time_start = high_resolution_clock::now();
}

void Clock::stop() {
    time_stop = high_resolution_clock::now();
}

}
