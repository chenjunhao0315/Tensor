//
//  ConvolutionMM2D.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/15.
//

#include "Tensor.hpp"
#include "ConvolutionMM2D.hpp"
#include "TensorUtils.hpp"
#include "TensorFactory.hpp"
#include "Math.hpp"
#include "Dispatch.hpp"
#include "Parallel.hpp"
#include "Unfold2D.hpp"
#include "TensorBlas.hpp"

namespace otter {

static Tensor view_weight_2d(const Tensor& weight_) {
    Tensor weight = weight_.contiguous();
    if (weight.dim() == 4) {
        const int64_t s1 = weight.size(0);
        const int64_t s2 = weight.size(1) * weight.size(2) * weight.size(3);
        return weight.view({s1, s2});
    } else {
        return weight;
    }
    return weight;
}

static inline void slow_conv2d_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    const Tensor& bias,
    int64_t kernel_height, int64_t kernel_width,
    int64_t stride_height, int64_t stride_width,
    int64_t pad_height, int64_t pad_width,
    bool weight_optional) {
    
    assert(kernel_height > 0 && kernel_width > 0);
    assert(stride_height > 0 && stride_width > 0);
    
    if (weight.defined()) {
        assert(weight.numel() > 0 && (weight.dim() == 2 || weight.dim() == 4));
        if (bias.defined()) {
            check_dim_size(bias, 1, 0, weight.size(0));
        }
    } else {
        assert(weight_optional);
    }
    
    int64_t ndim = input.dim();
    int64_t dim_planes = 1;
    int64_t dim_height = 2;
    int64_t dim_width  = 3;
    
    assert(ndim == 4);  // Expect 4D input tensor
    for (const auto dim : otter::irange(2, ndim)) {
        assert(input.size(dim) != 0);
    }
    
    const int64_t input_height = input.size(dim_height);
    const int64_t input_width  = input.size(dim_width);
    
    const int64_t exact_input_height = input_height + 2 * pad_height;
    const int64_t exact_input_width  = input_width  + 2 * pad_width;
    
    assert(exact_input_height > kernel_height && exact_input_width > kernel_width);
    
    const int64_t output_height = div_round_up<int64_t>(exact_input_height - kernel_height, stride_height) + 1;
    const int64_t output_width  = div_round_up<int64_t>(exact_input_width - kernel_width, stride_width) + 1;
    
    assert(output_height >= 1 && output_width >= 1);
    
    if (weight.defined()) {
        int64_t n_input_plane = weight.size(1);
        if (weight.dim() == 2) {
            n_input_plane /= (kernel_height * kernel_width);
        }
        if (input.size(1) != 0) {
            check_dim_size(input, ndim, dim_planes, n_input_plane);
        }
    }

    if (grad_output.defined()) {
        if (weight.defined()) {
            int64_t n_output_plane = weight.size(0);
            check_dim_size(grad_output, ndim, dim_planes, n_output_plane);
        } else if (bias.defined()) {
            assert(bias.numel() > 0);   // "non-empty bias tensor expected"
            const int64_t n_output_plane = bias.dim() == 0 ? 1 : bias.size(0);
            check_dim_size(grad_output, ndim, dim_planes, n_output_plane);
        }
        check_dim_size(grad_output, ndim, dim_height, output_height);
        check_dim_size(grad_output, ndim, dim_width, output_width);
    }
}

static Tensor compute_columns2d(
    const Tensor& input,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef kernel_size) {
    // TODO: Maybe pass by value
    const int64_t kernel_height = kernel_size[0];
    const int64_t kernel_width  = kernel_size[1];
    const int64_t pad_height    = padding[0];
    const int64_t pad_width     = padding[1];
    const int64_t stride_height = stride[0];
    const int64_t stride_width  = stride[1];
    const int64_t dim_batch = 0;
    const int64_t dim_planes = 1;
    const int64_t dim_height = 2;
    const int64_t dim_width  = 3;
    const int64_t n_input_planes  = input.size(dim_planes);
    const int64_t input_height    = input.size(dim_height);
    const int64_t input_width     = input.size(dim_width);
    const int64_t output_height   = (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
    const int64_t output_width    = (input_width  + 2 * pad_width  - kernel_width ) / stride_width  + 1;
    const int64_t batch_size      = input.size(dim_batch);
    //
    
    Tensor columns;
    if ((kernel_height == 1) && (stride_height == 1) && (pad_height == 0) && (kernel_width == 1) && (stride_width == 1) && (pad_width == 0)) {
        columns = input.view({batch_size, n_input_planes, output_height * output_width}).detach();
    } else {
        columns = otter::empty({batch_size, n_input_planes * kernel_height * kernel_width, output_height * output_width}, input.options());
        OTTER_DISPATCH_ALL_TYPES(input.scalar_type(), "slow_conv2d_cpu", [&] {
            auto input_a   = input.accessor<scalar_t, 4>();
            auto columns_a = columns.accessor<scalar_t, 3>();
            
            otter::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
                for (const auto t : otter::irange(start, end)) {
                    auto input_t   = input_a[t];
                    auto columns_t = columns_a[t];
                    
                    unfold2d_copy_stub(
                        Device::CPU,
                        otter::CppTypeToScalarType<scalar_t>::value,
                        columns_t.data(),
                        input_t.data(),
                        kernel_height, kernel_width,
                        stride_height, stride_width,
                        pad_height, pad_width,
                        n_input_planes, input_height, input_width,
                        output_height, output_width);
                }
            });
        });
    }
    
    return columns.contiguous();
}

template <typename scalar_t>
static void slow_conv2d_update_output_frame(
    TensorAccessor<scalar_t, 3> input,
    TensorAccessor<scalar_t, 3> output,
    TensorAccessor<scalar_t, 2> weight,
    bool has_bias,
    TensorAccessor<scalar_t, 2> finput,
    int64_t kernel_height, int64_t kernel_width,
    int64_t stride_height, int64_t stride_width,
    int64_t pad_height, int64_t pad_width,
    int64_t n_input_planes, int64_t input_height, int64_t input_width,
    int64_t n_output_planes, int64_t output_height, int64_t output_width) {
    
    const int64_t beta = (has_bias) ? 1 : 0;
    
    const int64_t m = output_height * output_width;
    const int64_t n = n_output_planes;
    const int64_t k = n_input_planes * kernel_height * kernel_width;
    
    const int64_t lda = m;
    const int64_t ldb = k;
    const int64_t ldc = m;
    
    otter::gemm(
        TransposeType::NoTranspose, TransposeType::NoTranspose,
        m, n, k,
        static_cast<scalar_t>(1),
        finput.data(), lda,
        weight.data(), ldb,
        static_cast<scalar_t>(beta),
        output.data(), ldc);

}

Tensor& slow_conv2d_forward_out_cpu(
    const Tensor& self,
    const Tensor& weight_,
    const Tensor& bias_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    const int64_t kernel_height = kernel_size[0];
    const int64_t kernel_width  = kernel_size[1];
    const int64_t pad_height    = padding[0];
    const int64_t pad_width     = padding[1];
    const int64_t stride_height = stride[0];
    const int64_t stride_width  = stride[1];
    
    const Tensor weight_2d = view_weight_2d(weight_);
    
    slow_conv2d_shape_check(
        self,                           // input
        Tensor(),                       // grad_output
        weight_2d,                      // weight
        bias_,                          // bias
        kernel_height, kernel_width,    // kernel_size
        stride_height, stride_width,    // stride_size
        pad_height, pad_width,          // padding_size
        false);                         // false: defined weight, true: undefined weight
    
    const Tensor input = self.contiguous();
    const int64_t dim_batch = 0;
    const int64_t dim_planes = 1;
    const int64_t dim_height = 2;
    const int64_t dim_width  = 3;
    
    const int64_t n_input_planes  = input.size(dim_planes);
    const int64_t input_height    = input.size(dim_height);
    const int64_t input_width     = input.size(dim_width);
    const int64_t n_output_planes = weight_.size(0);
    const int64_t output_height   = (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
    const int64_t output_width    = (input_width  + 2 * pad_width  - kernel_width ) / stride_width  + 1;
    
    const int64_t batch_size      = input.size(dim_batch);
    
    Tensor finput = compute_columns2d(input, padding, stride, kernel_size);
    output.resize_({batch_size, n_output_planes, output_height, output_width});
    if (bias_.defined()) {
        output.copy_(bias_.reshape({-1, 1, 1}));
    }
    
    assert(output.is_contiguous());
    
    OTTER_DISPATCH_ALL_TYPES(input.scalar_type(), "slow_conv2d_cpu", [&] {
        auto input_a     = input.accessor<scalar_t, 4>();
        auto output_a    = output.accessor<scalar_t, 4>();
        auto finput_a    = finput.accessor<scalar_t, 3>();
        auto weight_2d_a = weight_2d.accessor<scalar_t, 2>();
        
        otter::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
            for (const auto t : otter::irange(start, end)) {
                auto input_t  = input_a[t];
                auto output_t = output_a[t];
                auto finput_t = finput_a[t];
                
                slow_conv2d_update_output_frame(
                    input_t,
                    output_t,
                    weight_2d_a,
                    bias_.defined(),
                    finput_t,
                    kernel_height, kernel_width,
                    stride_height, stride_width,
                    pad_height, pad_width,
                    n_input_planes, input_height, input_width,
                    n_output_planes, output_height, output_width);
            }
        });
    });
    
    return output;
}

Tensor slow_conv2d(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {
    
    auto out = otter::empty({}, self.options());
    otter::slow_conv2d_forward_out_cpu(self, weight, bias, kernel_size, stride, padding, out);
    
    return out;
}

Tensor slow_conv2d_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    return otter::slow_conv2d_forward_out_cpu(self, weight, bias, kernel_size, stride, padding, output);
}

}   // end namespace otter
