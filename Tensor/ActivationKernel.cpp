//
//  ActivationKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/1.
//

#include "Activation.hpp"
#include "ActivationKernel.hpp"
#include "TensorIterator.hpp"
#include "Dispatch.hpp"
#include "Loop.hpp"

namespace otter {

void leaky_relu_kernel(TensorIterator& iter, const Scalar& negval_) {
    OTTER_DISPATCH_ALL_TYPES(iter.dtype(), "leaky_relu_cpu", [&] {
        using Vec = vec::Vectorized<scalar_t>;
        auto zero_vec = Vec((scalar_t)(0));
        auto one_vec = Vec((scalar_t)(1));
        scalar_t negval = negval_.to<scalar_t>();
        Vec negval_v = Vec(negval);
        cpu_kernel_vec(iter,
            [&](scalar_t a) -> scalar_t {
                return a > scalar_t(0) ? a : a * negval;
            },
            [&](Vec a) -> Vec {
                auto r = Vec::blendv(negval_v, one_vec, a > zero_vec);
                return a * r;
            }
        );
    });
}

REGISTER_DISPATCH(leaky_relu_stub, &leaky_relu_kernel);

}   // end namesapce otter
