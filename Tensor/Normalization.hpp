//
//  Normalization.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/21.
//

#ifndef Normalization_hpp
#define Normalization_hpp

#include "Tensor.hpp"

namespace otter {

std::tuple<Tensor, Tensor, Tensor> batchnorm_cpu(const Tensor& self, const Tensor& weight, const Tensor& bias, const Tensor& running_mean, const Tensor& running_var, bool train, double momentum, double eps);

Tensor batchnorm(const Tensor& self, const Tensor& weight, const Tensor& bias, const Tensor& running_mean, const Tensor& running_var, bool train, double momentum, double eps);

Tensor batchnorm_alpha_beta(const Tensor& self, const Tensor& alpha, const Tensor& beta);

}

#endif /* Normalization_hpp */
