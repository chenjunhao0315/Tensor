//
//  Config.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef Config_h
#define Config_h

#define OTTER_MOBILE 0

#define OTTER_OPENMP 1

#define OTTER_AVX 1

#define OTTER_OPENCV_DRAW 1

#define OTTER_BENCHMARK 1

#if OTTER_MOBILE

#else
#if OTTER_AVX
#define CPU_CAPABILITY_AVX2 1
#else

#endif
#endif


#endif /* Config_h */
