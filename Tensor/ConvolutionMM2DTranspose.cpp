//
//  ConvolutionMM2DTranspose.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/3.
//

#include "ConvolutionMM2DTranspose.hpp"
#include "TensorUtils.hpp"
#include "TensorFactory.hpp"
#include "Dispatch.hpp"
#include "TensorBlas.hpp"
#include "im2col.hpp"
#include "Parallel.hpp"
#include "Dispatch.hpp"
#include "TensorTransform.hpp"

namespace otter {

static inline void slow_conv_transpose2d_shape_check(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    const Tensor& bias,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int pad_height,
    int pad_width,
    int output_padding_height,
    int output_padding_width,
    int dilation_height,
    int dilation_width,
    bool weight_nullable) {
    OTTER_CHECK(
        kernel_width > 0 && kernel_height > 0,
        "kernel size should be greater than zero, but got kernel_height: ",
        kernel_height,
        " kernel_width: ",
        kernel_width);
    OTTER_CHECK(
        stride_width > 0 && stride_height > 0,
        "stride should be greater than zero, but got stride_height: ",
        stride_height,
        " stride_width: ",
        stride_width);
    OTTER_CHECK(
        dilation_width > 0 && dilation_height > 0,
        "dilation should be greater than zero, but got dilation_height: ",
        dilation_height,
        ", dilation_width: ",
        dilation_width);
    OTTER_CHECK(
        (output_padding_width < stride_width || output_padding_width < dilation_width) &&
        (output_padding_height < stride_height || output_padding_height < dilation_height),
        "output padding must be smaller than either stride or dilation, but got output_padding_height: ",
        output_padding_height,
        " output_padding_width: ",
        output_padding_width,
        " stride_height: ",
        stride_height,
        " stride_width: ",
        stride_width,
        " dilation_height: ",
        dilation_height,
        " dilation_width: ",
        dilation_width);

    if (weight.defined()) {
        OTTER_CHECK(
            weight.numel() != 0 && (weight.dim() == 2 || weight.dim() == 4),
            "non-empty 2D or 4D weight tensor expected, but got: ",
            weight.sizes());
        if (bias.defined()) {
            check_dim_size(bias, 1, 0, weight.size(1));
        }
    } else if (!weight_nullable) {
        OTTER_CHECK(false, "weight tensor is expected to be non-nullable");
    }

    int ndim = input.dim();
    int dimf = 0;
    int dimh = 1;
    int dimw = 2;

    if (ndim == 4) {
        dimf++;
        dimh++;
        dimw++;
    }

    OTTER_CHECK(
        input.numel() != 0 && (ndim == 3 || ndim == 4),
        "non-empty 3D or 4D input tensor expected but got a tensor with size ",
        input.sizes());

    int64_t input_height = input.size(dimh);
    int64_t input_width = input.size(dimw);
    int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
        (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
    int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
        (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

    if (output_width < 1 || output_height < 1) {
        OTTER_CHECK(false,
            "Given input size per channel: (",
            input_height,
            " x ",
            input_width,
            "). "
            "Calculated output size per channel: (",
            output_height,
            " x ",
            output_width,
            "). Output size is too small");
    }

    if (weight.defined()) {
        int64_t n_input_plane = weight.size(0);
        check_dim_size(input, ndim, dimf, n_input_plane);
    }

    if (grad_output.defined()) {
        if (weight.defined()) {
            int64_t n_output_plane = weight.size(1);
            check_dim_size(grad_output, ndim, dimf, n_output_plane);
        } else if (bias.defined()) {
            int64_t n_output_plane = bias.size(0);
            check_dim_size(grad_output, ndim, dimf, n_output_plane);
        }
        check_dim_size(grad_output, ndim, dimh, output_height);
        check_dim_size(grad_output, ndim, dimw, output_width);
    }
}

void slow_conv_transpose2d_out_cpu_template(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
    int64_t kernel_height = kernel_size[0];
    int64_t kernel_width = kernel_size[1];
    int64_t dilation_height = dilation[0];
    int64_t dilation_width = dilation[1];
    int64_t pad_height = padding[0];
    int64_t pad_width = padding[1];
    int64_t stride_height = stride[0];
    int64_t stride_width = stride[1];
    int64_t output_padding_height = output_padding[0];
    int64_t output_padding_width = output_padding[1];

    int n_input_plane = weight.size(0);
    int n_output_plane = weight.size(1);

    Tensor input_ = input.contiguous();
    Tensor weight_ = weight.contiguous();

    Tensor bias_ = Tensor();

    if (bias.defined()) {
        bias_ = bias.contiguous();
    }

    bool is_batch = false;
    if (input_.dim() == 3) {
        // Force batch
        is_batch = true;
    }

    int64_t input_height = input_.size(2);
    int64_t input_width = input_.size(3);
    int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
        (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
    int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
        (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

    // Batch size + input planes
    int64_t batch_size = input_.size(0);

    // Create temporary columns
    Tensor columns = otter::zeros({n_output_plane * kernel_width * kernel_height, input_height * input_width}, input_.options());

    // Define a buffer of ones, for bias accumulation
    Tensor ones = bias.defined() ? otter::ones({output_height, output_width}, input_.options()) : Tensor();

    OTTER_DISPATCH_FLOATING_TYPES_AND(otter::ScalarType::Long,
        input.scalar_type(), "slow_conv_transpose2d_out_cpu", [&] {
        // For each elt in batch, do:
        for (const auto elt : otter::irange(batch_size)) {
            // Helpers
            Tensor input_n;
            Tensor output_n;

            // Matrix mulitply per output:
            input_n = input_.select(0, elt);
            output_n = output.select(0, elt);

            // M,N,K are dims of matrix A and B
            // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
            int64_t m = weight_.size(1) * weight_.size(2) * weight_.size(3);
            int64_t n = input_height * input_width;
            int64_t k = weight_.size(0);

            // Do GEMM (note: this is a bit confusing because gemm assumes
            // column-major matrices)
            otter::gemm(
                TransposeType::NoTranspose,
                TransposeType::Transpose,
                n,
                m,
                k,
                static_cast<scalar_t>(1),
                input_n.data_ptr<scalar_t>(),
                n,
                weight_.data_ptr<scalar_t>(),
                m,
                static_cast<scalar_t>(0),
                columns.data_ptr<scalar_t>(),
                n);

            // Unpack columns back into input:
            col2im<scalar_t>(
                columns.data_ptr<scalar_t>(),
                n_output_plane,
                output_height,
                output_width,
                input_height,
                input_width,
                kernel_height,
                kernel_width,
                pad_height,
                pad_width,
                stride_height,
                stride_width,
                dilation_height,
                dilation_width,
                output_n.data_ptr<scalar_t>());

            // Do Bias after:
            // M,N,K are dims of matrix A and B
            // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
            int64_t m_ = n_output_plane;
            int64_t n_ = output_height * output_width;
            int64_t k_ = 1;

            // Do GEMM (note: this is a bit confusing because gemm assumes
            // column-major matrices)
            if (bias.defined()) {
                otter::gemm(
                    TransposeType::Transpose,
                    TransposeType::NoTranspose,
                    n_,
                    m_,
                    k_,
                    static_cast<scalar_t>(1),
                    ones.data_ptr<scalar_t>(),
                    k_,
                    bias_.data_ptr<scalar_t>(),
                    k_,
                    static_cast<scalar_t>(1),
                    output_n.data_ptr<scalar_t>(),
                    n_);
            }
        }

        // Resize output
        if (is_batch) {
            output.resize_({n_output_plane, output_height, output_width});
            input_.resize_({n_input_plane, input_height, input_width});
        }
    });
}

void slow_conv_transpose2d_forward_out_cpu(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    OTTER_CHECK(
          kernel_size.size() == 2,
          "It is expected kernel_size equals to 2, but got size ",
          kernel_size.size());

    OTTER_CHECK(
          dilation.size() == 2,
          "It is expected dilation equals to 2, but got size ",
          dilation.size());

    OTTER_CHECK(
          padding.size() == 2,
          "It is expected padding equals to 2, but got size ",
          padding.size());

    OTTER_CHECK(
          stride.size() == 2,
          "It is expected stride equals to 2, but got size ",
          stride.size());

    OTTER_CHECK(
          output_padding.size() == 2,
          "It is expected stride equals to 2, but got size ",
          output_padding.size());

    int64_t kernel_height = kernel_size[0];
    int64_t kernel_width = kernel_size[1];
    int64_t dilation_height = dilation[0];
    int64_t dilation_width = dilation[1];
    int64_t pad_height = padding[0];
    int64_t pad_width = padding[1];
    int64_t stride_height = stride[0];
    int64_t stride_width = stride[1];
    int64_t output_padding_height = output_padding[0];
    int64_t output_padding_width = output_padding[1];

    slow_conv_transpose2d_shape_check(
        input,
        Tensor(),
        weight,
        bias,
        kernel_height,
        kernel_width,
        stride_height,
        stride_width,
        pad_height,
        pad_width,
        output_padding_height,
        output_padding_width,
        dilation_height,
        dilation_width,
        false);

    int n_output_plane = weight.size(1);

    Tensor input_ = input.contiguous();

    if (input_.dim() == 3) {
        input_.resize_({1, input_.size(0), input_.size(1), input_.size(2)});
    }

    int64_t input_height = input_.size(2);
    int64_t input_width = input_.size(3);
    int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
        (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
    int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
        (dilation_width * (kernel_width - 1) + 1) + output_padding_width;

    // Batch size + input planes
    int64_t batch_size = input_.size(0);

    // Resize output
    output.resize_({batch_size, n_output_plane, output_height, output_width}, MemoryFormat::Contiguous);
    
    slow_conv_transpose2d_out_cpu_template(output, input, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}

Tensor slow_conv_transpose2d(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, input.options());
    
    return slow_conv_transpose2d_out(input, weight, kernel_size, bias, stride, padding, output_padding, dilation, output);
}

Tensor& slow_conv_transpose2d_out(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    otter::slow_conv_transpose2d_forward_out_cpu(input, weight, kernel_size, bias, stride, padding, output_padding, dilation, output);
    
    return output;
}

Tensor slide_win_conv_transpose2d(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, input.options());
    
    return slide_win_conv_transpose2d_out(input, weight, kernel_size, bias, stride, padding, output_padding, dilation, output);
}

Tensor& slide_win_conv_transpose2d_out(
    const Tensor& self,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef output_padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    const int64_t kernel_height   = kernel_size[0];
    const int64_t kernel_width    = kernel_size[1];
    const int64_t stride_height   = stride[0];
    const int64_t stride_width    = stride[1];
    const int64_t dilation_height = dilation[0];
    const int64_t dilation_width  = dilation[1];
    const int64_t output_padding_height = output_padding[0];
    const int64_t output_padding_width  = output_padding[1];
    
    const int64_t dim_planes = 1;
    const int64_t dim_height = 2;
    const int64_t dim_width  = 3;
    
    const int64_t input_channels  = self.size(dim_planes);
    const int64_t input_height    = self.size(dim_height);
    const int64_t input_width     = self.size(dim_width);
    const int64_t output_channels = weight.size(1);
    
    const int64_t batch_size      = self.size(0);
    
    int64_t output_height = (input_height - 1) * stride_height +
        (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
    int64_t output_width = (input_width - 1) * stride_width +
        (dilation_width * (kernel_width - 1) + 1) + output_padding_width;
    
    auto output_pad = otter::empty({batch_size, output_channels, output_height, output_width}, self.options());
    
    const int bias_term = (bias.defined()) ? 1 : 0;
    
    const int64_t kernelSize = kernel_height * kernel_width;
    
    // kernel offsets
    std::vector<int> _space_ofs(kernelSize);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = int(output_width * dilation_height - kernel_width * dilation_width);
        for (int i = 0; i < kernel_height; i++) {
            for (int j = 0; j < kernel_width; j++) {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_width;
            }
            p2 += gap;
        }
    }
    
    OTTER_DISPATCH_FLOATING_TYPES_AND(ScalarType::Long, self.scalar_type(), "slide_win_deconv", [&] {
        auto input_a = self.accessor<scalar_t, 4>()[0];
        auto weight_a = weight.accessor<scalar_t, 4>()[0];
        auto bias_data = (bias_term) ? bias.data_ptr<scalar_t>() : nullptr;
        
        otter::parallel_for(0, output_channels, 0, [&](int64_t begin, int64_t end) {
            for (const auto p : otter::irange(begin, end)) {
                auto out = output_pad[0][p];

                const scalar_t bias = bias_term ? bias_data[p] : static_cast<scalar_t>(0);

                out.fill_(bias);
                
                auto out_a = out.accessor<scalar_t, 2>();

                // shadowed variable for less openmp task args
                const int w = (int)self.size(3);
                const int h = (int)self.size(2);

                for (int i = 0; i < h; i++) {
                    for (int j = 0; j < w; j++) {
                        scalar_t* outptr = out_a[i * stride_height].data() + j * stride_width;

                        const scalar_t* kptr = (const scalar_t*)weight_a.data() + kernelSize * input_channels * p;

                        for (int q = 0; q < input_channels; q++) {
                            const scalar_t val = input_a[q][i][j];

                            for (int k = 0; k < kernelSize; k++) {
                                float w = kptr[k];
                                outptr[space_ofs[k]] += val * w;
                            }

                            kptr += kernelSize;
                        }
                    }
                }
            }
        });
    });
    
    if (padding[0] > 0 || padding[1] > 0) {
        output = otter::crop(output_pad, {padding[1], padding[1], padding[0], padding[0]});
    } else {
        output = output_pad;
    }
    
    return output;
}

}   // end namespace otter
