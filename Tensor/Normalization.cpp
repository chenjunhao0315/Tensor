//
//  Normalization.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/21.
//

#include "Normalization.hpp"

#include "Dispatch.hpp"
#include "TensorFactory.hpp"
#include "TensorIterator.hpp"
#include "Loop.hpp"
#include "BatchNormalization.hpp"
#include "TensorOperator.hpp"
#include "ScalarOps.hpp"

namespace otter {

static bool is_contiguous(const Tensor& t) {
    return t.is_contiguous() || t.is_contiguous(MemoryFormat::ChannelsLast);
}

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> batchnorm_cpu_transform_input_template(const Tensor& input, const Tensor& weight, const Tensor& bias, const Tensor& save_mean, const Tensor& save_invstd, const Tensor& running_mean, const Tensor& running_var, bool train, double eps) {
    
    bool all_contiguous = is_contiguous(input)
        && (!weight.defined() || weight.is_contiguous())
        && (!bias.defined() || bias.is_contiguous())
        && running_mean.is_contiguous()
        && running_var.is_contiguous();
    
    if (all_contiguous) {
        Tensor output = otter::empty_like(input, input.options());
        batchnorm_cpu_stub(Device::CPU, output, input, weight, bias, save_mean, save_invstd, running_mean, running_var, train, eps);
        return std::make_tuple(output, save_mean, save_invstd);
    }
    
    const int64_t ndim = input.dim();
    DimVector sizes(ndim, 1);
    DimVector strides(ndim, 0);
    
    auto as_nd = [&](const Tensor& t) {
        assert(t.defined() && t.dim() == 1);
        sizes[1] = t.sizes()[0];
        strides[1] = t.strides()[0];
        return t.as_strided(sizes, strides);
    };
    
    auto mean = as_nd(train ? save_mean : running_mean);
    auto invstd = as_nd([&]{
        if (train) {
            return save_invstd;
        } else {
            return 1 / otter::sqrt(running_var + eps);
        }
    }());
    
    auto w = (weight.defined()) ? as_nd(weight) : native::wrapped_scalar_tensor(1, Device::CPU);
    auto b = (bias.defined()) ? as_nd(bias) : native::wrapped_scalar_tensor(0, Device::CPU);
    
    Tensor output = otter::empty_like(input);
    auto iter = TensorIteratorConfig()
                .add_output(output)
                .add_input(input)
                .add_input(mean)
                .add_input(invstd)
                .add_input(w)
                .add_input(b)
                .build();

    cpu_kernel(iter, [=](scalar_t input, scalar_t mean, scalar_t invstd, scalar_t weight, scalar_t bias) {
        return ((input - mean) * invstd) * weight + bias;
    });
    
    return std::make_tuple(output, save_mean, save_invstd);
}

std::tuple<Tensor, Tensor, Tensor> batchnorm_cpu(const Tensor& self, const Tensor& weight, const Tensor& bias, const Tensor& running_mean, const Tensor& running_var, bool train, double momentum, double eps) {
    return OTTER_DISPATCH_FLOATING_TYPES(self.scalar_type(), "batchnorm", [&] {
        if (!train) {
            auto save_mean = otter::empty({0} ,self.options());
            auto save_var  = otter::empty({0}, self.options());
            return batchnorm_cpu_transform_input_template<scalar_t>(self, weight, bias, save_mean, save_var, running_mean, running_var, false, eps);
        } else {
            // TODO: Skip for now
            return batchnorm_cpu_transform_input_template<scalar_t>(self, weight, bias, Tensor(), Tensor(), running_mean, running_var, true, eps);
        }
    });
}

Tensor batchnorm(const Tensor& self, const Tensor& weight, const Tensor& bias, const Tensor& running_mean, const Tensor& running_var, bool train, double momentum, double eps) {
    return std::get<0>(batchnorm_cpu(self, weight, bias, running_mean, running_var, train, momentum, eps));
}

Tensor batchnorm_alpha_beta(const Tensor& self, const Tensor& alpha, const Tensor& beta) {
    Tensor out = otter::empty_like(self, self.options());
    batchnorm_cpu_alpha_beta_stub(Device::CPU, out, self, alpha, beta);
    return out;
}

}   // end namespace otter
