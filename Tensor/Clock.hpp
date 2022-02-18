//
//  Clock.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/16.
//

#ifndef Clock_hpp
#define Clock_hpp

#include <chrono>

using namespace std::chrono;

namespace otter {

class Clock {
public:
    Clock();
    void start();
    void stop();
    
    template <typename T = milliseconds>
    long long int getElapsed();
    template <typename T = milliseconds>
    void stop_and_show(const char *postfix = "ms");
    template <typename T = milliseconds>
    void lap_and_show(const char *postfix = "ms");
    template <typename T = milliseconds>
    void show(const char *postfix = "ms");
private:
    high_resolution_clock::time_point time_start;
    high_resolution_clock::time_point time_stop;
};

template <typename T>
void Clock::stop_and_show(const char *postfix) {
    time_stop = high_resolution_clock::now();
    auto duration = duration_cast<T>(time_stop - time_start);
    printf("Elapsed time: %lld %s\n", duration.count(), postfix);
}

template <typename T>
void Clock::lap_and_show(const char *postfix) {
    auto time_lap = high_resolution_clock::now();
    auto duration = duration_cast<T>(time_lap - time_start);
    printf("Elapsed time: %lld %s\n", duration.count(), postfix);
}

template <typename T>
void Clock::show(const char *postfix) {
    auto duration = duration_cast<T>(time_stop - time_start);
    printf("Elapsed time: %lld %s\n", duration.count(), postfix);
}

template <typename T>
long long int Clock::getElapsed() {
    auto time_lap = high_resolution_clock::now();
    auto duration = duration_cast<T>(time_lap - time_start);
    return duration.count();
}

}

#endif /* Clock_hpp */
