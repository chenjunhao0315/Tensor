//
//  BatchNormalization.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/21.
//

#ifndef BatchNormalization_hpp
#define BatchNormalization_hpp

#include "DispatchStub.hpp"
#include "Tensor.hpp"

namespace otter {

using batchnorm_fn = void (*)(Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, bool, double);
using batchnorm_alpha_beta_fn = void (*)(Tensor&, const Tensor&, const Tensor&, const Tensor&);

DECLARE_DISPATCH(batchnorm_fn, batchnorm_cpu_stub);
DECLARE_DISPATCH(batchnorm_alpha_beta_fn, batchnorm_cpu_alpha_beta_stub);

}

#endif /* BatchNormalization_hpp */
