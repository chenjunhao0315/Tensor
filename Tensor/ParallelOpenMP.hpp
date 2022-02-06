//
//  ParallelOpenMP.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef ParallelOpenMP_hpp
#define ParallelOpenMP_hpp

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>

#ifdef _OPENMP
#define INTRA_OP_PARALLEL

#include <omp.h>
#endif  // _OPENMP

namespace otter {

#ifdef _OPENMP
template <typename F>
inline void invoke_parallel(int64_t begin, int64_t end, int64_t grain_size, const F& f) {
    std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
    std::exception_ptr eptr;
    
#pragma omp parallel
    {
        int64_t num_threads = omp_get_num_threads();
        if (grain_size > 0) {
            num_threads = std::min(num_threads, divup((end - begin), grain_size));
        }
        
        int64_t tid = omp_get_thread_num();
        int64_t chunk_size = divup((end - begin), num_threads);
        int64_t begin_tid = begin + tid * chunk_size;
        if (begin_tid < end) {
            try {
                ThreadIdGuard tid_guard((int)tid);
                f(begin_tid, std::min(end, chunk_size + begin_tid));
            } catch (...) {
                if (!err_flag.test_and_set()) {
                    eptr = std::current_exception();
                }
            }
        }
    }
    if (eptr) {
        std::rethrow_exception(eptr);
    }
}

#endif  // _OPENMP

}

#endif /* ParallelOpenMP_hpp */
