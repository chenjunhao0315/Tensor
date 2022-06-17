//
//  BatchNormalizationKernel.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/21.
//

#ifndef BatchNormalizationKernel_hpp
#define BatchNormalizationKernel_hpp

#include "Tensor.hpp"

namespace otter {

void batchnorm_cpu_kernel(Tensor& output, const Tensor& input, const Tensor& weight, const Tensor& bias, const Tensor& save_mean, const Tensor& save_invstd, const Tensor& running_mean, const Tensor& runing_var, bool train, double eps);

void batchnorm_cpu_alpha_beta_kernel(Tensor& output, const Tensor& input, const Tensor& alpha, const Tensor& beta);

template <typename scalar_t>
static TensorAccessor<scalar_t, 1, 1> conditional_accessor_1d(const Tensor& self) {
    if (!self.defined()) {
        return TensorAccessor<scalar_t, 1, 1>(nullptr, nullptr, nullptr);
    }
    return self.accessor<scalar_t, 1>();
}

template <typename scalar_t>
static scalar_t* conditional_data_ptr(const Tensor& self) {
    return (self.defined()) ? self.data_ptr<scalar_t>() : nullptr;
}

}

#endif /* BatchNormalizationKernel_hpp */
