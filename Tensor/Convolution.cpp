//
//  Convolution.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/15.
//

#include "Tensor.hpp"
#include "TensorShape.hpp"
#include "Convolution.hpp"
#include "DepthwiseConvKernel.hpp"
#include "DilatedConvolution.hpp"
#include "Convolution1x1s1.hpp"

namespace otter {

DEFINE_DISPATCH(convolution_depthwise3x3_winograd_stub);

std::ostream& operator<<(std::ostream & out, const ConvParams& params) {
    out << "ConvParams {"
        << "  stride = " << IntArrayRef{params.stride}
        << "  padding = " << IntArrayRef{params.padding}
        << "  dilation = " << IntArrayRef{params.dilation}
        << "  transposed = " << params.transposed
        << "  output_padding = " << IntArrayRef{params.output_padding}
        << "  groups = " << params.groups
        << "  benchmark = " << params.benchmark
        << "}";
    return out;
}

static void check_shape_forward(const Tensor& input, const IntArrayRef& weight_sizes, const Tensor& bias, const ConvParams& params) {
    int64_t k = input.dim();
    int64_t weight_dim = weight_sizes.size();
    int64_t groups = params.groups;
    const auto& padding = params.padding;
    const auto& dilation = params.dilation;
    bool transposed = params.transposed;
    
    OTTER_CHECK(!params.is_padding_neg(), "negative padding is not supported");
    OTTER_CHECK(!params.is_output_padding_neg(), "negative output_padding is not supported");
    OTTER_CHECK(!params.is_stride_nonpos(), "non-positive stride is not supported");
    
    OTTER_CHECK(weight_dim == k,
                "Expected ", weight_dim, "-dimensional input for ", weight_dim,
                "-dimensional weight ", weight_sizes, ", but got ", k, "-dimensional input of size ",
                input.sizes(), " instead");
    OTTER_CHECK(weight_sizes[0] >= groups,
                "Given groups=", groups, ", expected weight to be at least ", groups,
                " at dimension 0, but got weight of size ", weight_sizes, " instead");
    OTTER_CHECK(weight_sizes[0] % groups == 0,
                "Given groups=", groups, ", expected weight to be divisible by ",
                groups, " at dimension 0, but got weight of size [", weight_sizes,
                "] instead");
    
    if (!transposed) {
        std::vector<int64_t> input_shape;
        std::vector<int64_t> kernel_shape;
        bool kernel_size_correct = true;
        
        OTTER_CHECK(input.size(1) == (weight_sizes[1] * groups),
                    "Given groups=", groups, ", weight of size ", weight_sizes,
                    ", expected input", input.sizes(), " to have ",
                    (weight_sizes[1] * groups), " channels, but got ", input.size(1),
                    " channels instead");

        OTTER_CHECK(!bias.defined() || (bias.dim() == 1 && bias.size(0) == weight_sizes[0]),
                    "Given weight of size ", weight_sizes,
                    ", expected bias to be 1-dimensional with ", weight_sizes[0], " elements",
                    ", but got bias of size ", bias.sizes(), " instead");
        
        for (const auto i : otter::irange(2, k)) {
            input_shape.push_back(input.size(i) + 2 * padding[i - 2]);
            // log new kernel size considering dilation
            kernel_shape.push_back(dilation[i - 2] * (weight_sizes[i] - 1) + 1);
            if (input_shape.back() < kernel_shape.back()) {
                kernel_size_correct = false;
            }
        }
        
        assert(input_shape.size() == kernel_shape.size());
        
        if (!kernel_size_correct) {
            throw "Kernel size can't be greater than actual input size";
        }
        
    } else {
        assert(input.size(1) == weight_sizes[0]);
        assert(!bias.defined() || (bias.dim() == 1 && bias.size(0) == weight_sizes[1] * groups));
    }
}

ConvBackend select_proper_conv_backend(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const bool need_backward,
    const ConvParams& params) {
    
    if (false) {    // is depthwise
        
    } else if (!need_backward && params.use_cpu_depthwise3x3_winograd(input, weight)) {
        return ConvBackend::Winograd3x3Depthwise;
    } else if (input.device() == Device::CPU) { // or input.is_cuda()
        if (params.transposed) {
            
        } else {
            if (input.dim() == 4) {
                if (params.is_dilated()) {
                    return ConvBackend::SlowDilated2d;
                } else {
                    if (false) {    // NNpack
                        
                    } else if (params.use_cpu_1x1s1_optimization(input, weight)) {
                        if (weight.size(0) >= 64 && weight.size(1) >= 64)
                            return ConvBackend::Slow_gemm_1x1s1;
                        else
                            return ConvBackend::Slow1x1s1;
                    } else {
                        return ConvBackend::Slow2d;
                    }
                }
            } else if (input.dim() == 5 && (params.is_dilated())) {
//            } else if (input.dim() == 5 && (input.is_cuda() || params.is_dilated())) {
                return ConvBackend::SlowDilated3d;
            } else if (input.dim() == 5) { /* dim == 5, CPU, non-dilated */
                /* CPU implementation has specialized MM kernels
                 for non-dilated case here */
                return ConvBackend::Slow3d;
            } else {
                // unsupported
            }
        }
    } else {
        return ConvBackend::Overrideable;
    }
    throw "unsupported ConvNd parameters";
    return ConvBackend::Slow2d;
}

static auto view3d(const Tensor& tensor) -> Tensor {
    assert(tensor.dim() == 4);   // "expected 4D tensor, got tensor with ", tensor.ndimension(), " dimensions instead");
    return tensor.squeeze(2);
}

static auto view4d(const Tensor& tensor) -> Tensor {
    assert(tensor.dim() == 3);   // "expected 3D tensor, got tensor with ", tensor.ndimension(), " dimensions instead");
    return tensor.unsqueeze(2);
}

static Tensor subtensor(Tensor& tensor, int dim, int groups, int g) {
    if (!tensor.defined()) {
        return Tensor();
    }
    int64_t n = tensor.sizes()[dim] / groups;
    return tensor.narrow(dim, n * g, n).contiguous();
}

Tensor convolution(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    bool benchmark) {
    
    auto input = input_r;
    auto weight = weight_r;
    auto bias = bias_r;
    auto k = weight.dim();
    auto dim = k - 2;
    
    assert(k > 0);  // "weight should have at least three dimensions"
    
    auto weight_sizes = weight.sizes();
    
    ConvParams params;
    params.stride    = expand_param_if_needed(stride_, "stride", dim);
    params.padding   = expand_param_if_needed(padding_, "padding", dim);
    params.dilation  = expand_param_if_needed(dilation_, "dilation", dim);
    params.output_padding = expand_param_if_needed(output_padding_, "output_padding", dim);
    params.transposed = transposed_;
    params.benchmark = benchmark;
    params.groups    = groups_;
    
    check_shape_forward(input, weight_sizes, bias, params);
    
    if (k == 3) {
        // avoid accidentally going through NHWC for permuted 3d input.
        input = input.contiguous();
        params.view_1d_as_2d();
        input  = view4d(input);
        weight = view4d(weight);
    }
    
    bool need_backward = false; // TODO: haha
    ConvBackend backend = select_proper_conv_backend(input, weight, bias, need_backward, params);
    
    Tensor output;
    
    switch (backend) {
        case ConvBackend::Winograd3x3Depthwise:
            output = convolution_depthwise3x3_winograd_stub(Device::CPU, input, weight, bias, params.stride, params.padding, params.groups);
            break;
        case ConvBackend::Slow2d:
        case ConvBackend::SlowDilated2d:
        case ConvBackend::Slow1x1s1:
        case ConvBackend::Slow_gemm_1x1s1:
            if (params.groups == 1) {
                output = otter::convolution_nogroup_backend(input.contiguous(), weight, bias, backend, params);
            } else {
                std::vector<Tensor> outputs(params.groups);
                input = input.contiguous();
                for (const auto g : otter::irange(params.groups)) {
                    auto input_g = subtensor(input, 1, params.groups, g);
                    auto weight_g = subtensor(weight, 0, params.groups, g);
                    auto bias_g = subtensor(bias, 0, params.groups, g);
                    outputs[g] = otter::convolution_nogroup_backend(input_g, weight_g, bias_g, backend, params);
                }
                output = otter::native::cat(outputs, 1);
            }
            
        default:
            break;
    }
    
    if (k == 3) {
        output = view3d(output);
    }
    
    return output;
}

Tensor convolution_nogroup_backend(const Tensor& self, const Tensor& weight, const Tensor& bias, ConvBackend backend, ConvParams& params) {
    auto kernel_size = weight.sizes().slice(2);
    switch (backend) {
        case ConvBackend::Slow2d:
            return otter::slow_conv2d(self, weight, bias, kernel_size, params.stride, params.padding);
        case ConvBackend::SlowDilated2d:
            return otter::slow_conv_dilated2d(self, weight, bias, kernel_size, params.stride, params.padding, params.dilation);
        case ConvBackend::Slow1x1s1:
            return Tensor();
        case ConvBackend::Slow_gemm_1x1s1:
            return otter::conv_gemm_1x1s1(self, weight, bias, kernel_size, params.stride, params.padding);
        default:
            assert(false);  // Unsupported nogroup conv backend
    }
    
    return Tensor();
}

}
