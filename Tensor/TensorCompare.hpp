//
//  TensorCompare.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/6.
//

#ifndef TensorCompare_hpp
#define TensorCompare_hpp

#include "DispatchStub.hpp"

namespace otter {

class Tensor;
class Scalar;
class TensorIterator;

using clamp_fn = void (*)(TensorIterator &);
DECLARE_DISPATCH(clamp_fn, clamp_stub);
DECLARE_DISPATCH(clamp_fn, clamp_min_stub);
DECLARE_DISPATCH(clamp_fn, clamp_max_stub);

namespace detail {
enum class ClampLimits {Min, Max, MinMax};
}

DECLARE_DISPATCH(void (*)(TensorIterator &, const Scalar&, const Scalar&), clamp_scalar_stub);
DECLARE_DISPATCH(void (*)(TensorIterator &, Scalar), clamp_min_scalar_stub);
DECLARE_DISPATCH(void (*)(TensorIterator &, Scalar), clamp_max_scalar_stub);

Tensor clamp(const Tensor& self, const Scalar& min, const Scalar& max);
Tensor clamp(const Tensor& self, const Tensor& min, const Tensor& max);
Tensor& clamp_(Tensor& self, const Scalar& min, const Scalar& max);
Tensor& clamp_(Tensor& self, const Tensor& min, const Tensor& max);

Tensor& clamp_max_out(const Tensor& self, const Scalar& max, Tensor& result);
Tensor& clamp_max_out(const Tensor& self, const Tensor& max, Tensor& result);
Tensor clamp_max(const Tensor& self, const Scalar& max);
Tensor clamp_max(const Tensor& self, const Tensor& max);
Tensor& clamp_max_(Tensor& self, const Scalar& max);
Tensor& clamp_max_(Tensor& self, const Tensor& max);

Tensor& clamp_min_out(const Tensor& self, const Scalar& min, Tensor& result);
Tensor& clamp_min_out(const Tensor& self, const Tensor& min, Tensor& result);
Tensor clamp_min(const Tensor& self, const Scalar& min);
Tensor clamp_min(const Tensor& self, const Tensor& min);
Tensor& clamp_min_(Tensor& self, const Scalar& min);
Tensor& clamp_min_(Tensor& self, const Tensor& min);

}   // end namespace otter

#endif /* TensorCompare_hpp */
