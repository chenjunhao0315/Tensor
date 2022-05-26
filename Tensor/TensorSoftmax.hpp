//
//  TensorSoftmax.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/5/26.
//

#ifndef TensorSoftmax_hpp
#define TensorSoftmax_hpp

#include "DispatchStub.hpp"

namespace otter {

class Tensor;

using forward_fn = void (*)(const Tensor&, const Tensor&);
DECLARE_DISPATCH(forward_fn, softmax_lastdim_kernel);

using forward_fn_with_dim = void(*)(const Tensor &, const Tensor &, const int64_t);
DECLARE_DISPATCH(forward_fn_with_dim, softmax_kernel);

Tensor softmax_out(Tensor& output, const Tensor& input, int64_t dim);

Tensor softmax(const Tensor& input, int64_t dim);

}

#endif /* TensorSoftmax_hpp */
