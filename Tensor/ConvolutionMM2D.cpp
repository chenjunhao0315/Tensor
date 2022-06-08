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
#include "Padding.hpp"
#include "Quantize.hpp"

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
    
    if (weight.defined()) {
        OTTER_CHECK(
                weight.numel() > 0 && (weight.dim() == 2 || weight.dim() == 4),
                "non-empty 2D or 4D weight tensor expected, but got: ",
                weight.sizes());
        if (bias.defined()) {
            check_dim_size(bias, 1, 0, weight.size(0));
        }
    } else {
        OTTER_CHECK(weight_optional, "weight tensor is undefined");
    }
    
    int64_t ndim = input.dim();
    int64_t dim_planes = 1;
    int64_t dim_height = 2;
    int64_t dim_width  = 3;
    
    OTTER_CHECK(ndim == 4, "Expected 4D input tensor, but got: ", input.sizes());
    for (const auto dim : otter::irange(2, ndim)) {
        OTTER_CHECK(input.size(dim) != 0,
                        "Expected non-zero size for input dimension ", dim,
                        ", but got input shape: ", input.sizes(), ". Only the batch and channel dimensions support size 0.");
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
        int64_t input_channels = weight.size(1);
        if (weight.dim() == 2) {
            input_channels /= (kernel_height * kernel_width);
        }
        if (input.size(1) != 0) {
            check_dim_size(input, ndim, dim_planes, input_channels);
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
    const int64_t input_channels  = input.size(dim_planes);
    const int64_t input_height    = input.size(dim_height);
    const int64_t input_width     = input.size(dim_width);
    const int64_t output_height   = (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
    const int64_t output_width    = (input_width  + 2 * pad_width  - kernel_width ) / stride_width  + 1;
    const int64_t batch_size      = input.size(dim_batch);
    
    Tensor columns;
    if ((kernel_height == 1) && (stride_height == 1) && (pad_height == 0) && (kernel_width == 1) && (stride_width == 1) && (pad_width == 0)) {
        columns = input.view({batch_size, input_channels, output_height * output_width}).detach();
    } else {
        columns = otter::empty({batch_size, input_channels * kernel_height * kernel_width, output_height * output_width}, input.options());
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
                        input_channels, input_height, input_width,
                        output_height, output_width);
                }
            });
        });
    }
    
    return columns.contiguous();
}

template <typename scalar_t>
static void slow_conv2d_update_output_frame(
    TensorAccessor<scalar_t, 3> /*input*/,
    TensorAccessor<scalar_t, 3> output,
    TensorAccessor<scalar_t, 2> weight,
    bool has_bias,
    TensorAccessor<scalar_t, 2> finput,
    int64_t kernel_height, int64_t kernel_width,
    int64_t /*stride_height*/, int64_t /*stride_width*/,
    int64_t /*pad_height*/, int64_t /*pad_width*/,
    int64_t input_channels, int64_t /*input_height*/, int64_t /*input_width*/,
    int64_t output_channels, int64_t output_height, int64_t output_width) {
    
    const int64_t beta = (has_bias) ? 1 : 0;
    
    const int64_t m = output_height * output_width;
    const int64_t n = output_channels;
    const int64_t k = input_channels * kernel_height * kernel_width;
    
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
    
    const int64_t input_channels  = input.size(dim_planes);
    const int64_t input_height    = input.size(dim_height);
    const int64_t input_width     = input.size(dim_width);
    const int64_t output_channels = weight_.size(0);
    const int64_t output_height   = (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
    const int64_t output_width    = (input_width  + 2 * pad_width  - kernel_width ) / stride_width  + 1;
    
    const int64_t batch_size      = input.size(dim_batch);
    
    Tensor finput = compute_columns2d(input, padding, stride, kernel_size);
    output.resize_({batch_size, output_channels, output_height, output_width});
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
                    input_channels, input_height, input_width,
                    output_channels, output_height, output_width);
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

Tensor& slow_conv2d_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    return otter::slow_conv2d_forward_out_cpu(self, weight, bias, kernel_size, stride, padding, output);
}

Tensor& slide_win_conv2d_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0);
    
    auto output_shape = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_(output_shape);
    
    const int kernel_w = kernel_size[1];
    const int kernel_h = kernel_size[0];
    const int stride_w = stride[1];
    const int stride_h = stride[0];
    
    const int w = (int)input.size(3);
    const int inch = (int)input.size(1);

    const int outw  = (int)output_shape[3];
    const int outh  = (int)output_shape[2];
    const int outch = (int)output_shape[1];

    const int bias_term = bias.defined() ? 1 : 0;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_w;
        for (int i = 0; i < kernel_h; i++) {
            for (int j = 0; j < kernel_w; j++) {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

    OTTER_DISPATCH_ALL_TYPES(self.scalar_type(), "conv2d", [&] {
        auto input_a = input.accessor<scalar_t, 4>()[0];
        auto output_a = output.accessor<scalar_t, 4>()[0];
        auto bias_data = (bias_term) ? bias.data_ptr<scalar_t>() : nullptr;
        auto weight_data = weight.data_ptr<scalar_t>();
        
        otter::parallel_for(0, outch, 0, [&](int64_t begin, int end) {
            for (const auto p : otter::irange(begin, end)) {
                scalar_t* outptr = output_a[p].data();

                for (int i = 0; i < outh; i++) {
                    for (int j = 0; j < outw; j++) {
                        scalar_t sum = 0.f;

                        if (bias_term)
                            sum = bias_data[p];

                        const scalar_t* kptr = (const scalar_t*)weight_data + maxk * inch * p;

                        for (int q = 0; q < inch; q++)
                        {
                            const auto m = input_a[q];
                            const scalar_t* sptr = m[i * stride_h].data() + j * stride_w;

                            for (int k = 0; k < maxk; k++) // 29.23
                            {
                                scalar_t val = sptr[space_ofs[k]]; // 20.72
                                scalar_t wt = kptr[k];
                                sum += val * wt; // 41.45
                            }

                            kptr += maxk;
                        }
                        
                        outptr[j] = sum;
                    }

                    outptr += outw;
                }
            }
        });
    });
    
    return output;
}
    
Tensor slide_win_conv2d(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {
    
    auto out = otter::empty({}, self.options());
    slide_win_conv2d_out(self, weight, bias, kernel_size, stride, padding, out);
    
    return out;
}

Tensor& slide_win_conv2d_int8_fp32_out(
    const Tensor& self,
    const Tensor& input_int8_scales,
    const Tensor& weight,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    otter::Tensor input_q;
    if (self.scalar_type() != ScalarType::Byte)
        input_q = otter::quantize_to_int8(self, input_int8_scales);
    else
        input_q = self;
    auto input = otter::constant_pad(input_q, {padding[1], padding[1], padding[0], padding[0]}, 0);
    
    auto output_shape = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_(output_shape);
    
    const int kernel_w = kernel_size[1];
    const int kernel_h = kernel_size[0];
    const int stride_w = stride[1];
    const int stride_h = stride[0];
    const int dilation_w = dilation[1];
    const int dilation_h = dilation[0];
    
    const int w = (int)input.size(3);
    const int inch = (int)input.size(1);

    const int outw  = (int)output_shape[3];
    const int outh  = (int)output_shape[2];
    const int outch = (int)output_shape[1];

    const int bias_term = bias.defined() ? 1 : 0;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }
    
    float* bias_data = bias_term ? bias.data_ptr<float>() : nullptr;
    
    auto output_a = output.accessor<float, 4>()[0];
    const signed char* weight_data = (const signed char*)weight.data_ptr<unsigned char>();
    auto weight_data_int8_scales_a = weight_int8_scales.accessor<float, 1>();
    auto input_int8_scales_a = input_int8_scales.accessor<float, 1>();
    auto input_a = input.accessor<unsigned char, 4>()[0];
    
    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            signed char* outptr = (signed char*)output_a[p].data();

            for (int i = 0; i < outh; i++) {
                for (int j = 0; j < outw; j++) {
                    int sum = 0;

                    const signed char* kptr = (signed char*)weight_data + maxk * inch * p;

                    // channels
                    for (int q = 0; q < inch; q++)
                    {
                        auto m = input_a[q];
                        const signed char* sptr = (signed char*)m[i * stride_h].data() + j * stride_w;

                        for (int k = 0; k < maxk; k++)
                        {
                            int val = sptr[space_ofs[k]];
                            int wt = kptr[k];
                            sum += val * wt;
                        }

                        kptr += maxk;
                    }

                    float scale_in;
                    if (weight_data_int8_scales_a[p] == 0)
                        scale_in = 0;
                    else
                        scale_in = 1.f / (input_int8_scales_a[0] * weight_data_int8_scales_a[p]);

                    float sumfp32 = sum * scale_in;

                    if (bias_term)
                        sumfp32 += bias_data[p];

                    // dequantize
                    ((float*)outptr)[0] = sumfp32;
                    outptr += 4;
                }
            }
        }
    });
    
    return output;
}
    
Tensor slide_win_conv2d_int8_fp32(
    const Tensor& self,
    const Tensor& input_scale_data,
    const Tensor& weight,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto out = otter::empty({}, otter::ScalarType::Float);
    slide_win_conv2d_int8_fp32_out(self, input_scale_data, weight, weight_int8_scales, bias, kernel_size, stride, padding, dilation, out);
    
    return out;
}

Tensor& slide_win_conv2d_int8_out(
    const Tensor& self,
    const Tensor& input_int8_scales,
    const Tensor& weight,
    const Tensor& weight_int8_scales,
    const Tensor& /*bias*/,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    otter::Tensor input_q;
    if (self.scalar_type() != ScalarType::Byte)
        input_q = otter::quantize_to_int8(self, input_int8_scales);
    else
        input_q = self;
    auto input = otter::constant_pad(input_q, {padding[1], padding[1], padding[0], padding[0]}, 0);
    
    auto output_shape = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_(output_shape);
    
    const int kernel_w = kernel_size[1];
    const int kernel_h = kernel_size[0];
    const int stride_w = stride[1];
    const int stride_h = stride[0];
    const int dilation_w = dilation[1];
    const int dilation_h = dilation[0];
    
    const int w = (int)input.size(3);
    const int inch = (int)input.size(1);

    const int outw  = (int)output_shape[3];
    const int outh  = (int)output_shape[2];
    const int outch = (int)output_shape[1];

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }
    
    auto output_a = output.accessor<int, 4>()[0];
    const signed char* weight_data_int8 = (const signed char*)weight.data_ptr<unsigned char>();
    auto input_a = input.accessor<unsigned char, 4>()[0];
    
    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            int* outptr = (int*)output_a[p].data();

            for (int i = 0; i < outh; i++) {
                for (int j = 0; j < outw; j++) {
                    int sum = 0;

                    const signed char* kptr = (signed char*)weight_data_int8 + maxk * inch * p;

                    // channels
                    for (int q = 0; q < inch; q++)
                    {
                        auto m = input_a[q];
                        const signed char* sptr = (signed char*)m[i * stride_h].data() + j * stride_w;

                        for (int k = 0; k < maxk; k++)
                        {
                            signed char val = sptr[space_ofs[k]];
                            signed char wt = kptr[k];
                            sum += val * wt;
                        }

                        kptr += maxk;
                    }

                    outptr[j] = sum;
                }
                
                outptr += outw;
            }
        }
    });
    
    return output;
}
    
Tensor slide_win_conv2d_int8(
    const Tensor& self,
    const Tensor& input_scale_data,
    const Tensor& weight,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto out = otter::empty({}, otter::ScalarType::Int);
    slide_win_conv2d_int8_out(self, input_scale_data, weight, weight_int8_scales, bias, kernel_size, stride, padding, dilation, out);
    
    return out;
}

}   // end namespace otter
