//
//  BatchNormalizationKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/21.
//

#include "Vec.hpp"
#include "TensorFactory.hpp"
#include "Dispatch.hpp"
#include "BatchNormalization.hpp"
#include "BatchNormalizationKernel.hpp"
#include "Parallel.hpp"

namespace otter {

template <typename scalar_t>
void batchnorm_cpu_collect_linear_and_constant_terms(
    scalar_t* alpha, scalar_t* beta, int64_t n_channel,
    const Tensor& weight /* optional */, const Tensor& bias /* optional */,
    const Tensor& save_mean, const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var, bool train, double eps) {
    
    const scalar_t* weight_data = (weight.defined()) ? weight.data_ptr<scalar_t>() : nullptr;
    const scalar_t* bias_data   = (bias.defined()) ? bias.data_ptr<scalar_t>() : nullptr;
    
    auto save_mean_a    = conditional_accessor_1d<scalar_t>(save_mean);
    auto save_invstd_a  = conditional_accessor_1d<scalar_t>(save_invstd);
    auto running_mean_a = conditional_accessor_1d<scalar_t>(running_mean);
    auto running_var_a  = conditional_accessor_1d<scalar_t>(running_var);
    
    for (const auto c : otter::irange(n_channel)) {
        scalar_t mean, invstd;
        if (train) {
            mean   = save_mean_a[c];
            invstd = save_invstd_a[c];
        } else {
            mean   = running_mean_a[c];
            invstd = std::sqrt(running_var_a[c] + static_cast<scalar_t>(eps));
        }
        scalar_t weight_v = weight_data ? weight_data[c] : 1;
        scalar_t bias_v = bias_data ? bias_data[c] : 0;
        alpha[c] = invstd * weight_v;
        beta[c] = bias_v - mean * alpha[c];
    }
}

template <typename scalar_t>
void batchnorm_cpu_contiguous_impl(Tensor& output, const Tensor& input, const Tensor& weight, const Tensor& bias, const Tensor& save_mean, const Tensor& save_invstd, const Tensor& running_mean, const Tensor& running_var, bool train, double eps) {
    
    using Vec = vec::Vectorized<scalar_t>;
    int64_t n_batch    = input.size(0);
    int64_t n_channel  = input.size(1);
    int64_t image_size = input.numel() / n_batch / n_channel;
    
    Tensor alpha = otter::empty({n_channel}, input.options());
    Tensor beta  = otter::empty({n_channel}, input.options());
    scalar_t* alpha_data = alpha.data_ptr<scalar_t>();
    scalar_t* beta_data  = beta.data_ptr<scalar_t>();
    
    batchnorm_cpu_collect_linear_and_constant_terms(alpha_data, beta_data, n_channel, weight, bias, save_mean, save_invstd, running_mean, running_var, train, eps);
    
    scalar_t* output_data = output.data_ptr<scalar_t>();
    const scalar_t* input_data = input.data_ptr<scalar_t>();
    
    if (image_size != 1) {
        const int64_t loop_size = image_size - (image_size % Vec::size());
        otter::parallel_for(0, n_batch * n_channel, 1, [&](int64_t begin, int64_t end) {
            int64_t n = 0;
            int64_t c = 0;
            data_index_init(begin, n, n_batch, c, n_channel);
            
            for (const auto i : otter::irange(begin, end)) {
                const Vec alpha_vec(alpha_data[c]);
                const Vec beta_vec(beta_data[c]);
                int64_t offset = i * image_size;
                int64_t d = 0;
                for (; d < loop_size; d += Vec::size()) {
                    Vec data_vec = Vec::loadu(input_data + offset + d);
                    Vec output_vec = data_vec * alpha_vec + beta_vec;
                    output_vec.store(output_data + offset + d);
                }
                if (image_size - d > 0) {
                    Vec data_vec = Vec::loadu(input_data + offset + d, image_size - d);
                    Vec output_vec = data_vec * alpha_vec + beta_vec;
                    output_vec.store(output_data + offset + d, static_cast<int>(image_size - d));
                }
            }
            data_index_step(n, n_batch, c, n_channel);
        });
    } else {
        const int64_t loop_size = image_size - (image_size % Vec::size());
        otter::parallel_for(0, n_batch, 1, [&](int64_t begin, int64_t end) {
            for (const auto n : otter::irange(begin, end)) {
                int64_t offset = n * n_channel;
                int64_t d = 0;
                for (; d < loop_size; d += Vec::size()) {
                    Vec alpha_vec = Vec::loadu(alpha_data + d);
                    Vec beta_vec = Vec::loadu(beta_data + d);
                    Vec data_vec = Vec::loadu(input_data + offset + d);
                    Vec output_vec = data_vec * alpha_vec + beta_vec;
                    output_vec.store(output_data + offset + d);
                }
                if (n_channel - d > 0) {
                    Vec alpha_vec = Vec::loadu(alpha_data + d, n_channel - d);
                    Vec beta_vec = Vec::loadu(beta_data + d, n_channel - d);
                    Vec data_vec = Vec::loadu(input_data + offset + d, n_channel - d);
                    Vec output_vec = data_vec * alpha_vec + beta_vec;
                    output_vec.store(output_data + offset + d, static_cast<int>(n_channel - d));
                }
            }
        });
    }
}

template <typename scalar_t>
void batchnorm_cpu_channel_last_impl(Tensor& output, const Tensor& input, const Tensor& weight, const Tensor& bias, const Tensor& save_mean, const Tensor& save_invstd, const Tensor& running_mean, const Tensor& running_var, bool train, double eps) {
    
    using Vec = vec::Vectorized<scalar_t>;
    int64_t n_batch = input.size(0);
    int64_t n_channel = input.size(1);
    int64_t image_size = input.numel() / n_batch / n_channel;

    Tensor alpha = otter::empty({n_channel}, input.options());
    Tensor beta = otter::empty({n_channel}, input.options());
    scalar_t* alpha_data = alpha.data_ptr<scalar_t>();
    scalar_t* beta_data = beta.data_ptr<scalar_t>();

    batchnorm_cpu_collect_linear_and_constant_terms<scalar_t>(
          alpha_data, beta_data, n_channel, weight, bias,
          save_mean, save_invstd, running_mean, running_var, train, eps);

    scalar_t* output_data = output.data_ptr<scalar_t>();
    const scalar_t* input_data = input.data_ptr<scalar_t>();

    // Apply the linear terms to the input,
    // output(n, c, h, w) = input(n, c, h, w) * alpha(c) + beta(c)
    const int64_t loop_size = n_channel - (n_channel % Vec::size());
    otter::parallel_for(0, n_batch * image_size, 1, [&](int64_t begin, int64_t end) {
        for (const auto i : otter::irange(begin, end)) {
            int64_t offset = i * n_channel;
            int64_t d = 0;
            // vectorize on channel dimension, for normal batch_norm input size,
            // alpha/beta should fit in L1 cache, otherwise consider blocking.
            for (; d < loop_size; d += Vec::size()) {
                Vec alpha_vec = Vec::loadu(alpha_data + d);
                Vec beta_vec = Vec::loadu(beta_data + d);
                Vec data_vec = Vec::loadu(input_data + offset + d);
                Vec output_vec = data_vec * alpha_vec + beta_vec;
                output_vec.store(output_data + offset + d);
            }
            if (n_channel - d > 0) {
                Vec alpha_vec = Vec::loadu(alpha_data + d, n_channel - d);
                Vec beta_vec = Vec::loadu(beta_data + d, n_channel - d);
                Vec data_vec = Vec::loadu(input_data + offset + d, n_channel - d);
                Vec output_vec = data_vec * alpha_vec + beta_vec;
                output_vec.store(output_data + offset + d, static_cast<int>(n_channel - d));
            }
        }
    });
}

void batchnorm_cpu_kernel(Tensor& output, const Tensor& input, const Tensor& weight, const Tensor& bias, const Tensor& save_mean, const Tensor& save_invstd, const Tensor& running_mean, const Tensor& running_var, bool train, double eps) {
    if (input.is_contiguous()) {
        OTTER_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batchnorm_cpu_contiguous", [&] {
            batchnorm_cpu_contiguous_impl<scalar_t>(output, input, weight, bias, save_mean, save_invstd, running_mean, running_var, train, eps);
        });
    } else if (input.is_contiguous(MemoryFormat::ChannelsLast)) {
        OTTER_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batchnorm_cpu_channel_last", [&] {
            batchnorm_cpu_channel_last_impl<scalar_t>(output, input, weight, bias, save_mean, save_invstd, running_mean, running_var, train, eps);
        });
    } else {
        OTTER_CHECK(false, "Batchnorm expect contiguous input");
    }
}

template <typename scalar_t>
void batchnorm_cpu_alpha_beta_contiguous_impl(Tensor& output, const Tensor& input, const Tensor& alpha, const Tensor& beta) {
    
    using Vec = vec::Vectorized<scalar_t>;
    int64_t n_batch    = input.size(0);
    int64_t n_channel  = input.size(1);
    int64_t image_size = input.numel() / n_batch / n_channel;
    
    scalar_t* alpha_data = alpha.data_ptr<scalar_t>();
    scalar_t* beta_data  = beta.data_ptr<scalar_t>();
    
    scalar_t* output_data = output.data_ptr<scalar_t>();
    const scalar_t* input_data = input.data_ptr<scalar_t>();
    
    if (image_size != 1) {
        const int64_t loop_size = image_size - (image_size % Vec::size());
        otter::parallel_for(0, n_batch * n_channel, 1, [&](int64_t begin, int64_t end) {
            int64_t n = 0;
            int64_t c = 0;
            data_index_init(begin, n, n_batch, c, n_channel);
            
            for (const auto i : otter::irange(begin, end)) {
                const Vec alpha_vec(alpha_data[c]);
                const Vec beta_vec(beta_data[c]);
                int64_t offset = i * image_size;
                int64_t d = 0;
                for (; d < loop_size; d += Vec::size()) {
                    Vec data_vec = Vec::loadu(input_data + offset + d);
                    Vec output_vec = data_vec * alpha_vec + beta_vec;
                    output_vec.store(output_data + offset + d);
                }
                if (image_size - d > 0) {
                    Vec data_vec = Vec::loadu(input_data + offset + d, image_size - d);
                    Vec output_vec = data_vec * alpha_vec + beta_vec;
                    output_vec.store(output_data + offset + d, static_cast<int>(image_size - d));
                }
            }
            data_index_step(n, n_batch, c, n_channel);
        });
    } else {
        const int64_t loop_size = image_size - (image_size % Vec::size());
        otter::parallel_for(0, n_batch, 1, [&](int64_t begin, int64_t end) {
            for (const auto n : otter::irange(begin, end)) {
                int64_t offset = n * n_channel;
                int64_t d = 0;
                for (; d < loop_size; d += Vec::size()) {
                    Vec alpha_vec = Vec::loadu(alpha_data + d);
                    Vec beta_vec = Vec::loadu(beta_data + d);
                    Vec data_vec = Vec::loadu(input_data + offset + d);
                    Vec output_vec = data_vec * alpha_vec + beta_vec;
                    output_vec.store(output_data + offset + d);
                }
                if (n_channel - d > 0) {
                    Vec alpha_vec = Vec::loadu(alpha_data + d, n_channel - d);
                    Vec beta_vec = Vec::loadu(beta_data + d, n_channel - d);
                    Vec data_vec = Vec::loadu(input_data + offset + d, n_channel - d);
                    Vec output_vec = data_vec * alpha_vec + beta_vec;
                    output_vec.store(output_data + offset + d, static_cast<int>(n_channel - d));
                }
            }
        });
    }
}

template <typename scalar_t>
void batchnorm_cpu_alpha_beta_channel_last_impl(Tensor& output, const Tensor& input, const Tensor& alpha, const Tensor& beta) {
    
    using Vec = vec::Vectorized<scalar_t>;
    int64_t n_batch = input.size(0);
    int64_t n_channel = input.size(1);
    int64_t image_size = input.numel() / n_batch / n_channel;

    scalar_t* alpha_data = alpha.data_ptr<scalar_t>();
    scalar_t* beta_data = beta.data_ptr<scalar_t>();

    scalar_t* output_data = output.data_ptr<scalar_t>();
    const scalar_t* input_data = input.data_ptr<scalar_t>();

    // Apply the linear terms to the input,
    // output(n, c, h, w) = input(n, c, h, w) * alpha(c) + beta(c)
    const int64_t loop_size = n_channel - (n_channel % Vec::size());
    otter::parallel_for(0, n_batch * image_size, 1, [&](int64_t begin, int64_t end) {
        for (const auto i : otter::irange(begin, end)) {
            int64_t offset = i * n_channel;
            int64_t d = 0;
            // vectorize on channel dimension, for normal batch_norm input size,
            // alpha/beta should fit in L1 cache, otherwise consider blocking.
            for (; d < loop_size; d += Vec::size()) {
                Vec alpha_vec = Vec::loadu(alpha_data + d);
                Vec beta_vec = Vec::loadu(beta_data + d);
                Vec data_vec = Vec::loadu(input_data + offset + d);
                Vec output_vec = data_vec * alpha_vec + beta_vec;
                output_vec.store(output_data + offset + d);
            }
            if (n_channel - d > 0) {
                Vec alpha_vec = Vec::loadu(alpha_data + d, n_channel - d);
                Vec beta_vec = Vec::loadu(beta_data + d, n_channel - d);
                Vec data_vec = Vec::loadu(input_data + offset + d, n_channel - d);
                Vec output_vec = data_vec * alpha_vec + beta_vec;
                output_vec.store(output_data + offset + d, static_cast<int>(n_channel - d));
            }
        }
    });
}

void batchnorm_cpu_alpha_beta_kernel(Tensor& output, const Tensor& input, const Tensor& alpha, const Tensor& beta) {
    if (input.is_contiguous()) {
        OTTER_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batchnorm_cpu_alpha_beta_contiguous", [&] {
            batchnorm_cpu_alpha_beta_contiguous_impl<scalar_t>(output, input, alpha, beta);
        });
    } else if (input.is_contiguous(MemoryFormat::ChannelsLast)) {
        OTTER_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batchnorm_cpu_alpha_beta_channel_last", [&] {
            batchnorm_cpu_alpha_beta_channel_last_impl<scalar_t>(output, input, alpha, beta);
        });
    }
}

REGISTER_DISPATCH(batchnorm_cpu_stub, &batchnorm_cpu_kernel);
REGISTER_DISPATCH(batchnorm_cpu_alpha_beta_stub, &batchnorm_cpu_alpha_beta_kernel);

}   // end namesapce otter
