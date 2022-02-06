//
//  ParallelOpenMP.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "Config.hpp"

#if OTTER_OPENMP
#include "Parallel.hpp"

#include <atomic>
#include <cassert>

namespace otter {

namespace {

std::atomic<int> num_threads{-1};
thread_local int this_thread_id{0};

} // namespace

void set_thread_num(int id) {
    this_thread_id = id;
}

void init_num_threads() {
    auto nthreads = num_threads.load();
    if (nthreads > 0) {
        set_num_threads(nthreads);
    } else {
#ifdef _OPENMP
        omp_set_num_threads(intraop_default_num_threads());
#endif
    }
}

void set_num_threads(int nthreads) {
    assert(nthreads > 0);
    num_threads.store(nthreads);
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif
}

int get_num_threads() {
#ifdef _OPENMP
    lazy_init_num_threads();
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int get_thread_num() {
    return this_thread_id;
}

bool in_parallel_region() {
#ifdef _OPENMP
    return omp_in_parallel();
#else
    return false;
#endif
}

}   // end namespace otter

#endif
