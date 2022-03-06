//
//  DilatedConvolution.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/23.
//

#include "DilatedConvolutionUtils.hpp"
#include "DilatedConvolution.hpp"

#include "Tensor.hpp"
#include "TensorFactory.hpp"
#include "Dispatch.hpp"
#include "Accumulator.hpp"
#include "TensorBlas.hpp"
#include "im2col.hpp"
#include "vol2col.hpp"

namespace otter {

template <typename Dtype, int64_t dim>
void hvol2col(
    const Dtype* data_hvol,
    const int channels,
    const IntArrayRef input_size,
    const IntArrayRef output_size,
    const IntArrayRef kernel_size,
    const IntArrayRef stride_size,
    const IntArrayRef pad_size,
    const IntArrayRef dilation_size,
    Dtype* data_col) {
    
    if (dim == 3) {
        vol2col<Dtype>(
            data_hvol,
            channels,
            input_size[0],
            input_size[1],
            input_size[2],
            output_size[0],
            output_size[1],
            output_size[2],
            kernel_size[0],
            kernel_size[1],
            kernel_size[2],
            pad_size[0],
            pad_size[1],
            pad_size[2],
            stride_size[0],
            stride_size[1],
            stride_size[2],
            dilation_size[0],
            dilation_size[1],
            dilation_size[2],
            data_col);
    }
    if (dim == 2) {
        im2col<Dtype>(
            data_hvol,
            channels,
            input_size[0],
            input_size[1],
            output_size[0],
            output_size[1],
            kernel_size[0],
            kernel_size[1],
            pad_size[0],
            pad_size[1],
            stride_size[0],
            stride_size[1],
            dilation_size[0],
            dilation_size[1],
            data_col);
    }
}

// column to hyper-volume, CPU
template <typename Dtype, int64_t dim>
void col2hvol(
    const Dtype* data_col,
    const int channels,
    const IntArrayRef input_size,
    const IntArrayRef output_size,
    const IntArrayRef kernel_size,
    const IntArrayRef stride_size,
    const IntArrayRef pad_size,
    const IntArrayRef dilation_size,
    Dtype* data_hvol) {
    
    if (dim == 3) {
        col2vol<Dtype>(
            data_col,
            channels,
            input_size[0],
            input_size[1],
            input_size[2],
            output_size[0],
            output_size[1],
            output_size[2],
            kernel_size[0],
            kernel_size[1],
            kernel_size[2],
            pad_size[0],
            pad_size[1],
            pad_size[2],
            stride_size[0],
            stride_size[1],
            stride_size[2],
            dilation_size[0],
            dilation_size[1],
            dilation_size[2],
            data_hvol);
    }
    if (dim == 2) {
        col2im<Dtype>(
            data_col,
            channels,
            input_size[0],
            input_size[1],
            output_size[0],
            output_size[1],
            kernel_size[0],
            kernel_size[1],
            pad_size[0],
            pad_size[1],
            stride_size[0],
            stride_size[1],
            dilation_size[0],
            dilation_size[1],
            data_hvol);
    }
}

template <int64_t dim>
void slow_conv_dilated_all_cpu_template(
    Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& grad_output,
    Tensor& grad_input,
    Tensor& grad_weight,
    Tensor& grad_bias,
    IntArrayRef kernel_size,
    IntArrayRef stride_size,
    IntArrayRef pad_size,
    IntArrayRef dilation_size) {
    
    //    slow_conv_dilated_location_check(input, weight, bias, grad_output);   // TODO: Not need to check now
    auto options = input.options();
    // The rear part of input tensor sizes:
    auto input_size = input.sizes().slice(2);
    // The rear part of output tensor sizes:
    auto output_size = otter::get_output_size<dim>( input, kernel_size, stride_size, pad_size, dilation_size);
    int64_t batchSize = input.size(0);
    int64_t n_input_planes = weight.size(1);
    int64_t nOutputPlane = weight.size(0);
    // Temporary buffer:
    Tensor columns = otter::empty({0}, options);
    if (output.defined() || grad_weight.defined() || grad_input.defined()) {
        const int64_t m = otter::multiply_integers(kernel_size);
        const int64_t n = otter::multiply_integers(output_size);
        columns.resize_({n_input_planes * m, n});
    }
    // Initialize
    if (grad_weight.defined()) {
        grad_weight.zero_();
    }
    if (grad_bias.defined()) {
        grad_bias.zero_();
    }
    if (output.defined() && !bias.defined()) {
        output.zero_();
    }
    // Helpers
    Tensor grad_output_n;
    std::vector<int64_t> dims(dim);
    std::iota(dims.begin(), dims.end(), 1);
    
    OTTER_DISPATCH_FLOATING_TYPES(input.scalar_type(), "slow_conv_dilated<>", [&] {
        // For each elt in batch, do:
        for (const auto elt : otter::irange(batchSize)) {
            // Matrix multiply per output:
            Tensor input_n = input.select(0, elt);
            
            // Output
            if (output.defined()) {
                Tensor output_n = output.select(0, elt);
                if (bias.defined()) {
                    for (const auto n : otter::irange(nOutputPlane)) {
                        output_n.select(0, n).fill_(bias[n]);
                    }
                }
                // Extract columns:
                hvol2col<scalar_t, dim>(
                    input_n.data_ptr<scalar_t>(),
                    (int)n_input_planes,
                    input_size,
                    output_size,
                    kernel_size,
                    stride_size,
                    pad_size,
                    dilation_size,
                    columns.data_ptr<scalar_t>());
                
                otter::gemm(
                /*transa=*/TransposeType::NoTranspose,
                /*transb=*/TransposeType::NoTranspose,
                /*     m=*/columns.size(1),
                /*     n=*/nOutputPlane,
                /*     k=*/columns.size(0),
                /* alpha=*/(scalar_t)1,
                /*     A=*/columns.data_ptr<scalar_t>(),
                /*   lda=*/columns.size(1),
                /*     B=*/weight.data_ptr<scalar_t>(),
                /*   ldb=*/columns.size(0),
                /*  beta=*/(scalar_t)1,
                /*     C=*/output_n.data_ptr<scalar_t>(),
                /*   ldc=*/columns.size(1));
                
            } else {
                // All gradients
                grad_output_n = grad_output.select(0, elt);
            }
            
            // Gradient of input:
            if (grad_input.defined()) {
                otter::gemm(
                /*transa=*/TransposeType::NoTranspose,
                /*transb=*/TransposeType::Transpose,
                /*     m=*/columns.size(1),
                /*     n=*/columns.size(0),
                /*     k=*/nOutputPlane,
                /* alpha=*/(scalar_t)1,
                /*     A=*/grad_output_n.data_ptr<scalar_t>(),
                /*   lda=*/columns.size(1),
                /*     B=*/weight.data_ptr<scalar_t>(),
                /*   ldb=*/columns.size(0),
                /*  beta=*/(scalar_t)0,
                /*     C=*/columns.data_ptr<scalar_t>(),
                /*   ldc=*/columns.size(1));
                // Unpack columns back into input:
                Tensor grad_input_n = grad_input.select(0, elt);
                
                col2hvol<scalar_t, dim>(
                    columns.data_ptr<scalar_t>(),
                    (int)n_input_planes,
                    input_size,
                    output_size,
                    kernel_size,
                    stride_size,
                    pad_size,
                    dilation_size,
                    grad_input_n.data_ptr<scalar_t>());
            }
            
            // Gradient of weight:
            if (grad_weight.defined()) {
                // Extract columns:
                hvol2col<scalar_t, dim>(
                    input_n.data_ptr<scalar_t>(),
                    (int)n_input_planes,
                    input_size,
                    output_size,
                    kernel_size,
                    stride_size,
                    pad_size,
                    dilation_size,
                    columns.data_ptr<scalar_t>());
                
                scalar_t scale = 1; // TODO: expose as argument?
                otter::gemm(
                /*transa=*/TransposeType::Transpose,
                /*transb=*/TransposeType::NoTranspose,
                /*     m=*/columns.size(0),
                /*     n=*/nOutputPlane,
                /*     k=*/columns.size(1),
                /* alpha=*/scale,
                /*     A=*/columns.data_ptr<scalar_t>(),
                /*   lda=*/columns.size(1),
                /*     B=*/grad_output_n.data_ptr<scalar_t>(),
                /*   ldb=*/columns.size(1),
                /*  beta=*/(scalar_t)1,
                /*     C=*/grad_weight.data_ptr<scalar_t>(),
                /*   ldc=*/columns.size(0));
            }
            
            // Gradient of bias:
            if (grad_bias.defined()) {
                
                
// TODO: Need to be implemented
                
                
                
                
//                grad_bias += grad_output_n.sum(dims);
            }
        }
    });
    
} // slow_conv_dilated_all_cpu_template

Tensor slow_conv_dilated2d_forward_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias, IntArrayRef kernel_size, IntArrayRef stride_size, IntArrayRef pad_size, IntArrayRef dilation_size) {
    
    Tensor undefined;
    
    slow_conv_dilated_shape_check<2>(input, weight, bias, undefined, kernel_size, stride_size, pad_size, dilation_size);
    
    auto is_batch = input.dim() == 4;
    auto options = input.options();
    
    auto output_size = get_output_size<2>(input, weight, kernel_size, stride_size, pad_size, dilation_size);
    
    const Tensor input_ = (is_batch ? input.contiguous() : input.contiguous().unsqueeze(0));
    const Tensor weight_ = weight.contiguous();
    const Tensor bias_ = (bias.defined() ? bias.contiguous() : undefined);
    Tensor output = otter::empty(output_size, options);
    Tensor output_ = (is_batch ? output : output.unsqueeze(0));
    
    slow_conv_dilated_all_cpu_template<2>(output_, input_, weight_, bias_, undefined, undefined, undefined, undefined, kernel_size, stride_size, pad_size, dilation_size);
    
    return output;
}

Tensor slow_conv_dilated2d(const Tensor& input, const Tensor& weight, const Tensor& bias, IntArrayRef kernel_size, IntArrayRef stride_size, IntArrayRef pad_size, IntArrayRef dilation_size) {
    return slow_conv_dilated2d_forward_cpu(input, weight, bias, kernel_size, stride_size, pad_size, dilation_size);
}

}   // end namespace otter
