//
//  Parallel-inline.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef Parallel_inline_h
#define Parallel_inline_h

namespace otter {

template <class F>
inline void parallel_for(const int64_t begin, const int64_t end, const int64_t grain_size, const F& f) {
    if (begin >= end) {
        return;
    }
    
#ifdef INTRA_OP_PARALLEL
    otter::lazy_init_num_threads();
    const auto numiter = end - begin;
    const bool use_parallel = (numiter > grain_size && numiter > 1 && !otter::in_parallel_region() && otter::get_num_threads() > 1);
    if (!use_parallel) {
        ThreadIdGuard tid_guard(0);
        f(begin, end);
        return;
    }
    
    invoke_parallel(begin, end, grain_size, f);
#else
    ThreadIdGuard tid_guard(0);
    f(begin, end);
#endif
}







}

#endif /* Parallel_inline_h */
