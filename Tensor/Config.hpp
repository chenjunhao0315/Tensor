//
//  Config.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef Config_h
#define Config_h

#define OTTER_MOBLE 0

#define OTTER_OPENMP 1

#define OTTER_AVX 1

#if OTTER_MOBLE
#if OTTER_AVX

#define CPU_CAPABILITY_AVX2 1
#else
#define CPU_CAPABILITY_AVX2 0
#endif
#else
#define CPU_CAPABILITY_AVX2 0
#endif


#endif /* Config_h */
