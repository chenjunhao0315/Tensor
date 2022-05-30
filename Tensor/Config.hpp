//
//  Config.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef Config_h
#define Config_h

#ifndef OTTER_CONFIG
#include "platform.hpp"
#endif

#ifndef OTTER_MOBILE
#define OTTER_MOBILE 0
#endif

#ifndef OTTER_OPENMP
#define OTTER_OPENMP 1
#endif

#ifndef OTTER_AVX
#define OTTER_AVX 1
#endif

#ifndef OTTER_OPENCV_DRAW
#define OTTER_OPENCV_DRAW 1
#endif

#ifndef OTTER_BENCHMARK
#define OTTER_BENCHMARK 1
#endif

#if OTTER_MOBILE

#else
#if OTTER_AVX
#define CPU_CAPABILITY_AVX2 1
#else

#endif
#endif

#if OTTER_OPENMP

#else
#define OTTER_PARALLEL_NATIVE 1
#endif


#endif /* Config_h */
