//
//  ParallelNative.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/5/31.
//

#include "ParallelNative.hpp"

#if OTTER_PARALLEL_NATIVE

#ifdef _OPENMP
#include <omp.h>
#endif

namespace otter {

namespace {
thread_local bool in_parallel_region_ = false;
thread_local int thread_num_ = 0;
}  // namespace (anonymous)

void set_thread_num(int thread_num) {
  thread_num_ = thread_num;
}

void set_num_threads(int nthreads) {
    (void)nthreads;
}

int get_num_threads() {
    return 1;
}

int get_kmp_blocktime() {
#if defined(_OPENMP) && __clang__
    return kmp_get_blocktime();
#else
    return 0;
#endif
}

void set_kmp_blocktime(int time_ms) {
#if defined(_OPENMP) && __clang__
    kmp_set_blocktime(time_ms);
#else
    (void)time_ms;
#endif
}

}   // end namespace otter

#endif // OTTER_PARALLEL_NATIVE
