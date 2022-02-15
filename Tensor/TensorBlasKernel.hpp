//
//  TensorBlasKernel.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/12.
//

#ifndef TensorBlasKernel_hpp
#define TensorBlasKernel_hpp

#include <functional>

namespace otter {

template <typename scalar_t>
scalar_t dot_impl(int64_t n, scalar_t* x, int64_t incx, scalar_t* y, int64_t incy);

}   // end namespace otter

#endif /* TensorBlasKernel_hpp */
