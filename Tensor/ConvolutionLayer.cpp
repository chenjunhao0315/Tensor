//
//  ConvolutionLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/21.
//

#include "ConvolutionLayer.hpp"
#include "Convolution.hpp"
#include "ActivationLayer.hpp"

#include "TensorFactory.hpp"
#include "TensorMaker.hpp"

#include "ConvolutionMM2DNeon.hpp"
#include "ConvolutionMM2DX86.hpp"
#include "ConvolutionMM2DInt8X86.hpp"
#include "ConvolutionMM2DInt8Neon.hpp"

#if __SSE2__
#include "ConvolutionMM2DX86Pack.hpp"
#include "DepthwiseConvKernelX86Pack.hpp"

#include "ConvolutionMM2DInt8X86Pack.hpp"
#endif

#if __ARM_NEON__
#include "ConvolutionMM2DNeonPack.hpp"
#include "DepthwiseConvKernelNeonPack.hpp"

#include "ConvolutionMM2DInt8NeonPack.hpp"
#endif

#include "Quantize.hpp"
 
namespace otter {

ConvolutionLayer::ConvolutionLayer() {
    one_blob_only = true;
    support_inplace = false;
    
#if __SSE2__
    support_packing = true;
#elif __ARM_NEON__
    support_packing = true;
#endif
    
    activation = nullptr;
}

ConvolutionLayer::~ConvolutionLayer() {
    if (activation) {
        delete activation;
        activation = nullptr;
    }
}

int ConvolutionLayer::parse_param(LayerOption& option, ParamDict& pd) {
    pd.clear();
    int in_channels   = opt_find_int(option, "in_channels", 1);
    int out_channels  = opt_find_int(option, "out_channels", 1);
    int kernel_height = opt_find_int(option, "kernel_h", -1);
    int kernel_width  = opt_find_int(option, "kernel_w", -1);
    int kernel        = opt_find_int(option, "kernel", 3);
    if (kernel_height < 1 || kernel_width < 1) {
        if (kernel_height < 0) kernel_height = kernel;
        if (kernel_width < 0)  kernel_width  = kernel;
    }
    int stride_height = opt_find_int(option, "stride_h", -1);
    int stride_width  = opt_find_int(option, "stride_w", -1);
    int stride        = opt_find_int(option, "stride", 1);
    if (stride_height < 1 || stride_width < 1) {
        if (stride_height < 0) stride_height = stride;
        if (stride_width < 0)  stride_width  = stride;
    }
    int padding_height = opt_find_int(option, "padding_h", -1);
    int padding_width  = opt_find_int(option, "padding_w", -1);
    int padding        = opt_find_int(option, "padding", 0);
    if (padding_height < 0 || padding_width < 0) {
        if (padding_height < 0) padding_height = padding;
        if (padding_width < 0)  padding_width  = padding;
    }
    int dilation_height = opt_find_int(option, "dilation_h", -1);
    int dilation_width  = opt_find_int(option, "dilation_w", -1);
    int dilation        = opt_find_int(option, "dilation", 1);
    if (dilation_height < 1 || dilation_width < 1) {
        if (dilation_height < 0) dilation_height = dilation;
        if (dilation_width < 0)  dilation_width  = dilation;
    }
    int output_padding_height = opt_find_int(option, "output_padding_h", -1);
    int output_padding_width  = opt_find_int(option, "output_padding_w", -1);
    int output_padding        = opt_find_int(option, "output_padding", 0);
    if (output_padding_height < 0 || output_padding_width < 0) {
        if (output_padding_height < 0) output_padding_height = output_padding;
        if (output_padding_width < 0)  output_padding_width  = output_padding;
    }
    int groups = opt_find_int(option, "groups", 1);
    int bias_term = (!opt_find(option, "batchnorm")) ? 1 : 0;
    if (opt_find(option, "bias_term")) {
        if (option["bias_term"] == "false")
            bias_term = 0;
        else
            bias_term = 1;
    }
    int int8_scale_term = opt_find_int(option, "int8_scale_term", 0);
    
    std::string activation = opt_find_string(option, "activation", "");
    
    int activation_type = 0;
    if (activation == "Relu") {
        activation_type = 1;
    } else if (activation == "LRelu") {
        activation_type = 2;
    } else if (activation == "Relu6") {
        activation_type = 3;
    } else if (activation == "Sigmoid") {
        activation_type = 4;
    }
    
    Tensor activation_params;
    if (opt_check_string(option, "activation_params")) {
        int num_params = (int)std::count(option["activation_params"].begin(), option["activation_params"].end(), ',') + 1;
        activation_params = otter::empty({num_params}, otter::ScalarType::Float);
        auto activation_params_a = activation_params.accessor<float, 1>();
        std::stringstream ss;
        ss << option["activation_params"];
        float n; char c;
        for (const auto i : otter::irange(num_params)) {
            ss >> n >> c;
            activation_params_a[i] = n;
        }
    }
    
    if (opt_find(option, "batchnorm"))
        activation_type = 0;
    
    pd.set((int)ConvParam::In_channels, in_channels);
    pd.set((int)ConvParam::Out_channels, out_channels);
    pd.set((int)ConvParam::Kernel_height, kernel_height);
    pd.set((int)ConvParam::Kernel_width, kernel_width);
    pd.set((int)ConvParam::Stride_height, stride_height);
    pd.set((int)ConvParam::Stride_width,  stride_width);
    pd.set((int)ConvParam::Padding_height, padding_height);
    pd.set((int)ConvParam::Padding_width,  padding_width);
    pd.set((int)ConvParam::Dilation_height, dilation_height);
    pd.set((int)ConvParam::Dilation_width,  dilation_width);
    pd.set((int)ConvParam::Output_padding_height, output_padding_height);
    pd.set((int)ConvParam::Output_padding_height, output_padding_height);
    pd.set((int)ConvParam::Group, groups);
    pd.set((int)ConvParam::Bias_term, bias_term);
    pd.set((int)ConvParam::Int8_scale_term, int8_scale_term);
    pd.set((int)ConvParam::Activation_type, activation_type);
    pd.set((int)ConvParam::Activation_params, activation_params);
    
    return 0;
}

int ConvolutionLayer::compute_output_shape(ParamDict &pd) {
    auto shape_a = bottom_shapes[0].accessor<int, 2>()[0];
    int input_batch    = shape_a[0];
    int input_channels = shape_a[1];
    int input_height   = shape_a[2];
    int input_width    = shape_a[3];
    int out_channels    = pd.get((int)ConvParam::Out_channels, 1);
    int kernel_height   = pd.get((int)ConvParam::Kernel_height, 3);
    int kernel_width    = pd.get((int)ConvParam::Kernel_width, 3);
    int stride_height   = pd.get((int)ConvParam::Stride_height, 1);
    int stride_width    = pd.get((int)ConvParam::Stride_width,  1);
    int padding_height  = pd.get((int)ConvParam::Padding_height, 0);
    int padding_width   = pd.get((int)ConvParam::Padding_width,  0);
    int dilation_height = pd.get((int)ConvParam::Dilation_height, 1);
    int dilation_width  = pd.get((int)ConvParam::Dilation_width,  1);
    int groups          = pd.get((int)ConvParam::Group, 1);
    int out_width  = (input_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) / stride_width + 1;
    int out_height = (input_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1) / stride_height + 1;
    int weight_data_size = input_channels / groups * kernel_height * kernel_width * out_channels;
    
    pd.set((int)ConvParam::In_channels, input_channels);
    pd.set((int)ConvParam::Weight_data_size, weight_data_size);
    pd.set(OUTPUT_SHAPE_HINT, otter::tensor({input_batch, out_channels, out_height, out_width}, ScalarType::Int).view({1, -1}));
    
    return 0;
}

int ConvolutionLayer::load_param(const ParamDict &pd) {
    in_channels     = pd.get((int)ConvParam::In_channels, 1);
    out_channels    = pd.get((int)ConvParam::Out_channels, 1);
    kernel_height   = pd.get((int)ConvParam::Kernel_height, 3);
    kernel_width    = pd.get((int)ConvParam::Kernel_width, 3);
    stride_height   = pd.get((int)ConvParam::Stride_height, 1);
    stride_width    = pd.get((int)ConvParam::Stride_width,  1);
    padding_height  = pd.get((int)ConvParam::Padding_height, 0);
    padding_width   = pd.get((int)ConvParam::Padding_width,  0);
    dilation_height = pd.get((int)ConvParam::Dilation_height, 1);
    dilation_width  = pd.get((int)ConvParam::Dilation_width,  1);
    output_padding_height = pd.get((int)ConvParam::Output_padding_height, 0);
    output_padding_width  = pd.get((int)ConvParam::Output_padding_height, 0);
    groups = pd.get((int)ConvParam::Group, 1);
    bias_term = pd.get((int)ConvParam::Bias_term, 0);
    weight_data_size = pd.get((int)ConvParam::Weight_data_size, 0);
    int8_scale_term = pd.get((int)ConvParam::Int8_scale_term, 0);
    activation_type = pd.get((int)ConvParam::Activation_type, 0);
    activation_params = pd.get((int)ConvParam::Activation_params, Tensor());
    
    return 0;
}

int ConvolutionLayer::init_model() {
    if (int8_scale_term) {
        weight_data = (otter::rand({out_channels, in_channels / groups, kernel_height, kernel_width}, ScalarType::Float)).mul_(100).to(ScalarType::Byte);
        if (bias_term)
            bias_data = otter::rand({out_channels}, ScalarType::Float);
        
        if (in_channels == groups && groups == out_channels) {
            if (int8_scale_term == 1 || int8_scale_term == 101) {
                weight_data_int8_scales = otter::rand({groups}, ScalarType::Float);
                bottom_blob_int8_scales = otter::rand({groups}, ScalarType::Float);
            } else if (int8_scale_term == 2 || int8_scale_term == 102) {
                weight_data_int8_scales = otter::rand({groups}, ScalarType::Float);
                bottom_blob_int8_scales = otter::rand({groups}, ScalarType::Float);
            }
            
            if (int8_scale_term > 100) {
                top_blob_int8_scales = otter::rand({groups}, ScalarType::Float);
            }
        } else {
            if (int8_scale_term) {
                weight_data_int8_scales = otter::rand({out_channels}, ScalarType::Float);
                bottom_blob_int8_scales = otter::rand({1}, ScalarType::Float);
            }
            if (int8_scale_term > 100) {
                top_blob_int8_scales = otter::rand({1}, ScalarType::Float);
            }
        }
        return 0;
    }
    
    weight_data = otter::rand({out_channels, in_channels / groups, kernel_height, kernel_width}, ScalarType::Float);
    if (bias_term)
        bias_data = otter::rand({out_channels}, ScalarType::Float);
    
    return 0;
}

int ConvolutionLayer::load_model(const Initializer& initializer) {
    if (initializer.type != InitializerType::Ncnn) {
        if (bias_term) {
            bias_data = initializer.load({out_channels}, 1);
        }
    }
    
    weight_data = initializer.load({out_channels, in_channels / groups, kernel_height, kernel_width}, 0);
    
    if (initializer.type == InitializerType::Ncnn) {
        if (bias_term) {
            bias_data = initializer.load({out_channels}, 1);
        }
        // depthwise
        if (in_channels == groups && groups == out_channels) {
            if (int8_scale_term == 1 || int8_scale_term == 101) {
                weight_data_int8_scales = initializer.load({groups}, 1);
                bottom_blob_int8_scales = initializer.load({1}, 1);

                float bottom_blob_int8_scale = bottom_blob_int8_scales.item().toFloat();
                bottom_blob_int8_scales = otter::full({groups}, bottom_blob_int8_scale, otter::ScalarType::Float);
            } else if (int8_scale_term == 2 || int8_scale_term == 102) {
                weight_data_int8_scales = initializer.load({1}, 1);
                bottom_blob_int8_scales = initializer.load({1}, 1);

                // extend group if only one provided
                float weight_data_int8_scale = weight_data_int8_scales.item().toFloat();
                weight_data_int8_scales = otter::full({groups}, weight_data_int8_scale, otter::ScalarType::Float);

                float bottom_blob_int8_scale = bottom_blob_int8_scales.item().toFloat();
                bottom_blob_int8_scales = otter::full({groups}, bottom_blob_int8_scale, otter::ScalarType::Float);
            }
            
            if (int8_scale_term > 100) {
                top_blob_int8_scales = initializer.load({1}, 1);

                float top_blob_int8_scale = top_blob_int8_scales.item().toFloat();
                top_blob_int8_scales = otter::full({groups}, top_blob_int8_scale, otter::ScalarType::Float);
            }
        } else {
            if (int8_scale_term) {
                weight_data_int8_scales = initializer.load({out_channels}, 1);
                bottom_blob_int8_scales = initializer.load({1}, 1);
            }
            if (int8_scale_term > 100) {
                top_blob_int8_scales = initializer.load({1}, 1);
            }
        }
    }
    
    return 0;
}

int ConvolutionLayer::create_pipeline(const NetOption& opt) {
    
    activation = create_activation_layer(activation_type, activation_params);
    
    if (weight_data.scalar_type() == otter::ScalarType::Byte) {
        return create_pipeline_int8(opt);
    }
    
    int elempack = 1;
    int out_elempack = 1;
    
#if defined(__ARM_NEON__)
    if (opt.use_packing_layout) {
        elempack = in_channels % 4 == 0 ? 4 : 1;
        out_elempack = out_channels % 4 == 0 ? 4 : 1;
    }
    
    if (elempack == 4 && out_elempack == 4) {
        if (in_channels == groups && groups == out_channels) {
            int maxk = kernel_width * kernel_height;
            
            weight_data_tf = weight_data.view({groups, maxk}).packing(4);
        }
        
        if (kernel_width == 1 && kernel_height == 1 && stride_width == 1 && stride_height == 1) {
            otter::convolution_im2col_sgemm_transform_kernel_pack4_neon(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        } else if (kernel_width == 1 && kernel_height == 1 && stride_width == 2 && stride_height == 2) {
            otter::convolution_im2col_sgemm_transform_kernel_pack4_neon(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        }
    }
    
    if (elempack == 4 && out_elempack == 1) {
        if (in_channels == groups && groups == out_channels) {
            int maxk = kernel_width * kernel_height;
            
            weight_data_tf = weight_data.view({groups * 4, maxk}).packing(4);
        }
        
        if (kernel_width == 1 && kernel_height == 1 && stride_width == 1 && stride_height == 1) {
            otter::convolution_im2col_sgemm_transform_kernel_pack4to1_neon(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        } else if (kernel_width == 1 && kernel_height == 1 && stride_width == 2 && stride_height == 2) {
            otter::convolution_im2col_sgemm_transform_kernel_pack4to1_neon(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        }
    }
    
    if (elempack == 1 && out_elempack == 4) {
        if (in_channels == groups && groups == out_channels) {
            int maxk = kernel_width * kernel_height;
            
            weight_data_tf = weight_data.view({groups * 4, maxk}).packing(4);
        }
        
        if (kernel_width == 1 && kernel_height == 1 && stride_width == 1 && stride_height == 1) {
            otter::convolution_im2col_sgemm_transform_kernel_pack1to4_neon(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        } else if (kernel_width == 1 && kernel_height == 1 && stride_width == 2 && stride_height == 2) {
            otter::convolution_im2col_sgemm_transform_kernel_pack1to4_neon(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        } else if (kernel_width == 3 && kernel_height == 3 && stride_width == 2 && stride_height == 2) {
            otter::convolution_transform_kernel_pack1to4_neon(weight_data, weight_data_tf, in_channels, out_channels, kernel_width, kernel_height);
        }
    }
    
    if (elempack == 1 && out_elempack == 1) {
        if (in_channels == groups && groups == out_channels) {
            return 0;
        }
        if (kernel_width == 1 && kernel_height == 1 && stride_width == 1 && stride_height == 1) {
            if (in_channels >= 64 && out_channels >= 64) {
                otter::convolution_im2col_sgemm_transform_kernel_neon(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
            }
        } else if (kernel_width == 1 && kernel_height == 1 && stride_width == 2 && stride_height == 2) {
            otter::convolution_im2col_sgemm_transform_kernel_neon(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        } else if (kernel_width == 3 && kernel_height == 3 && stride_width == 1 && stride_height == 1) {
            if (in_channels >= 16 && weight_data.size(0) >= 16) {
                otter::conv3x3s1_winograd64_transform_kernel_neon5(weight_data, weight_3x3_winograd63_data, in_channels, out_channels);
            }
        } else if (kernel_width == 3 && kernel_height == 3 && stride_width == 2 && stride_height == 2) {
            otter::convolution_im2col_sgemm_transform_kernel_neon(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
            otter::conv3x3s2_transform_kernel_neon(weight_data, weight_3x3s2_data, in_channels, out_channels);
        } else {
            bool prefer_sgemm = (stride_width == 1 && stride_height == 1 && (in_channels >= 12 || out_channels >= 12)) || ((stride_width >= 2 || stride_height >= 2) && (in_channels >= 16 || out_channels >= 16));
            
            if (prefer_sgemm) {
                otter::convolution_im2col_sgemm_transform_kernel_neon(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
            }
        }
    }
#endif  // __ARM_NEON__
    
#if __SSE2__
    if (opt.use_packing_layout) {
#if __AVX__
        elempack = in_channels % 8 == 0 ? 8 : in_channels % 4 == 0 ? 4 : 1;
        out_elempack = out_channels % 8 == 0 ? 8 : out_channels % 4 == 0 ? 4 : 1;
#else
        elempack = in_channels % 4 == 0 ? 4 : 1;
        out_elempack = out_channels % 4 == 0 ? 4 : 1;
#endif
    }
    
#if __SSE2__
#if __AVX__
    if (elempack == 1 && out_elempack == 8) {
        if (kernel_width == 1 && kernel_height == 1 && stride_width == 1 && stride_height == 1) {
            otter::convolution_im2col_sgemm_transform_kernel_pack1to8_avx(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        } else {
            otter::convolution_im2col_sgemm_transform_kernel_pack1to8_avx(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        }
    }
    
    if (elempack == 4 && out_elempack == 8) {
        if (kernel_width == 1 && kernel_height == 1 && stride_width == 1 && stride_height == 1) {
            otter::convolution_im2col_sgemm_transform_kernel_pack4to8_avx(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        } else {
            otter::convolution_im2col_sgemm_transform_kernel_pack4to8_avx(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        }
    }
    
    if (elempack == 8 && out_elempack == 8) {
        if (in_channels == groups && groups == out_channels) {
            int maxk = kernel_width * kernel_height;
            
            weight_data_tf = weight_data.view({groups, maxk}).packing(8);
            return 0;
        }
        
        if (kernel_width == 1 && kernel_height == 1 && stride_width == 1 && stride_height == 1) {
            otter::convolution_im2col_sgemm_transform_kernel_pack8_avx(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        } else if (kernel_width == 1 && kernel_height == 1 && stride_width == 2 && stride_height == 2) {
            otter::convolution_im2col_sgemm_transform_kernel_pack8_avx(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        } else if (kernel_height == 3 && kernel_width == 3 && stride_height== 1 && stride_width == 1) {
            if (in_channels >= 8 && out_channels >= 8 && in_channels <= 32 && out_channels <= 32) {
                otter::conv3x3s1_winograd63_transform_kernel_pack8_avx(weight_data, weight_3x3_winograd63_data, in_channels, out_channels);
            } else if (in_channels >= 8 && out_channels >= 8) {
                otter::conv3x3s1_winograd43_transform_kernel_pack8_avx(weight_data, weight_3x3_winograd43_data, in_channels, out_channels);
            } else {
                otter::conv3x3s1_winograd23_transform_kernel_pack8_avx(weight_data, weight_3x3_winograd23_data, in_channels, out_channels);
            }
        } else {
            otter::convolution_im2col_sgemm_transform_kernel_pack8_avx(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        }
    }
    
    if (elempack == 8 && out_elempack == 1) {
        if (kernel_width == 1 && kernel_height == 1 && stride_width == 1 && stride_height == 1) {
            otter::convolution_im2col_sgemm_transform_kernel_pack8to1_avx(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        } else {
            otter::convolution_im2col_sgemm_transform_kernel_pack8to1_avx(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        }
    }
    
    if (elempack == 8 && out_elempack == 4) {
        if (kernel_width == 1 && kernel_height == 1 && stride_width == 1 && stride_height == 1) {
            otter::convolution_im2col_sgemm_transform_kernel_pack8to4_avx(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        } else {
            otter::convolution_im2col_sgemm_transform_kernel_pack8to4_avx(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        }
    }
#endif  // __AVX__
#endif  // __SSE2__
    
    if (elempack == 4 && out_elempack == 4) {
        if (in_channels == groups && groups == out_channels) {
            int maxk = kernel_width * kernel_height;
            
            weight_data_tf = weight_data.view({groups, maxk}).packing(4);
            return 0;
        }
        
        if (kernel_width == 1 && kernel_height == 1 && stride_width == 1 && stride_height == 1) {
            otter::convolution_im2col_sgemm_transform_kernel_pack4_sse(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        } else if (kernel_width == 1 && kernel_height == 1 && stride_width == 2 && stride_height == 2) {
            otter::convolution_im2col_sgemm_transform_kernel_pack4_sse(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        }
    }
    
    if (elempack == 4 && out_elempack == 1) {
        if (kernel_width == 1 && kernel_height == 1 && stride_width == 1 && stride_height == 1) {
            otter::convolution_im2col_sgemm_transform_kernel_pack4to1_sse(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        } else if (kernel_width == 1 && kernel_height == 1 && stride_width == 2 && stride_height == 2) {
            otter::convolution_im2col_sgemm_transform_kernel_pack4to1_sse(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        }
    }
    
    if (elempack == 1 && out_elempack == 4) {
        if (kernel_width == 1 && kernel_height == 1 && stride_width == 1 && stride_height == 1) {
            otter::convolution_im2col_sgemm_transform_kernel_pack1to4_sse(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        } else if (kernel_width == 1 && kernel_height == 1 && stride_width == 2 && stride_height == 2) {
            otter::convolution_im2col_sgemm_transform_kernel_pack1to4_sse(weight_data, weight_sgemm_data, in_channels, out_channels, kernel_width, kernel_height);
        }
    }
    
    if (elempack == 1 && out_elempack == 1) {
        if (in_channels == groups && groups == out_channels) {
            return 0;
        }
        if (kernel_width == 3 && kernel_height == 3 && stride_width == 1 && stride_height == 1) {
            if (in_channels >= 16 && out_channels >= 16) {
                otter::conv3x3s1_winograd43_transform_kernel_sse(weight_data, weight_3x3_winograd43_data, in_channels, out_channels);
            } else {
                otter::conv3x3s1_winograd23_transform_kernel_sse(weight_data, weight_3x3_winograd23_data, in_channels, out_channels);
            }
        }
        otter::convolution_im2col_sgemm_transform_kernel_x86(weight_data, weight_sgemm_data, in_channels, out_channels, weight_data.size(3), weight_data.size(2));
    }
#endif
    
    return 0;
}

int ConvolutionLayer::forward(const Tensor &bottom_blob, Tensor &top_blob, const NetOption& opt) const {
    
    if (int8_scale_term) {
        return forward_int8(bottom_blob, top_blob, opt);
    }
    
    Tensor optimize_kernel;
    
    int64_t elempack = bottom_blob.elempack();
    int64_t out_elempack = 1;
    
    if (opt.use_packing_layout) {
#if __SSE2__
#if __AVX__
        out_elempack = out_channels % 8 == 0 ? 8 : out_channels % 4 == 0 ? 4 : 1;
#else
        out_elempack = out_channels % 4 == 0 ? 4 : 1;
#endif  // __AVX__
#elif __ARM_NEON__  // __SSE2__
        out_elempack = out_channels % 4 == 0 ? 4 : 1;
#endif  // __ARM_NEON__
    }
    
#if __SSE2__
#if __AVX__
    if (elempack == 1 && out_elempack == 8) {
        optimize_kernel = weight_sgemm_data;
    }
    
    if (elempack == 4 && out_elempack == 8) {
        optimize_kernel = weight_sgemm_data;
    }
    
    if (elempack == 8 && out_elempack == 8) {
        if (kernel_width == 3 && kernel_height == 3 && stride_width == 1 && stride_height == 1) {
            if (in_channels >= 8 && out_channels >= 8 && in_channels <= 32 && out_channels <= 32) {
                optimize_kernel = weight_3x3_winograd63_data;
            } else if (in_channels >= 8 && out_channels >= 8) {
                optimize_kernel = weight_3x3_winograd43_data;
            } else {
                optimize_kernel = weight_3x3_winograd23_data;
            }
        } else {
            optimize_kernel = weight_sgemm_data;
        }
    }
    
    if (elempack == 8 && out_elempack == 1) {
        optimize_kernel = weight_sgemm_data;
    }
    
    if (elempack == 8 && out_elempack == 4) {
        optimize_kernel = weight_sgemm_data;
    }
#endif  // __AVX__
#endif  // __SSE2__
    
    if (elempack == 4 && out_elempack == 4) {
        if (in_channels == groups && groups == out_channels) {
            optimize_kernel = weight_data_tf;
        } else if (kernel_width == 1 && kernel_height == 1 && stride_width == 1 && stride_height == 1) {
            optimize_kernel = weight_sgemm_data;
        } else if (kernel_width == 1 && kernel_height == 1 && stride_width == 2 && stride_height == 2) {
            optimize_kernel = weight_sgemm_data;
        }
    }
    
    if (elempack == 1 && out_elempack == 4) {
        if (in_channels == groups && groups == out_channels) {
            optimize_kernel = weight_data_tf;
        } else if (kernel_width == 1 && kernel_height == 1 && stride_width == 1 && stride_height == 1) {
            optimize_kernel = weight_sgemm_data;
        } else if (kernel_width == 1 && kernel_height == 1 && stride_width == 2 && stride_height == 2) {
#if __SSE2__
            optimize_kernel = weight_sgemm_data;
#endif
        } else if (kernel_width == 3 && kernel_height == 3 && stride_width == 2 && stride_height == 2) {
#if __ARM_NEON__
            optimize_kernel = weight_data_tf;
#endif
        }
    }
    
    if (elempack == 4 && out_elempack == 1) {
        if (in_channels == groups && groups == out_channels) {
            optimize_kernel = weight_data_tf;
        } else if (kernel_width == 1 && kernel_height == 1 && stride_width == 1 && stride_height == 1) {
            optimize_kernel = weight_sgemm_data;
        } else if (kernel_width == 1 && kernel_height == 1 && stride_width == 2 && stride_height == 2) {
#if __SSE2__
            optimize_kernel = weight_sgemm_data;
#endif
        } else if (kernel_width == 3 && kernel_height == 3 && stride_width == 2 && stride_height == 2) {
#if __ARM_NEON__
            optimize_kernel = weight_data_tf;
#endif
        }
    }
    
    if (elempack == 1 && out_elempack == 1) {
            // General
#if __ARM_NEON__
            if (kernel_width == 1 && kernel_height == 1 && stride_width == 1 && stride_height == 1) {
                if (in_channels >= 64 && out_channels >= 64) {
                    optimize_kernel = weight_sgemm_data;
                }
            } else if (kernel_width == 1 && kernel_height == 1 && stride_width == 2 && stride_height == 2) {
                optimize_kernel = weight_sgemm_data;
            } else if (kernel_height == 3 && kernel_width == 3 && stride_width == 1 && stride_height == 1) {
                if (in_channels >= 16 && out_channels >= 16 && bottom_blob.size(3) <= 120 && bottom_blob.size(2) <= 120) {
                    optimize_kernel = weight_3x3_winograd63_data;
                }
            } else if (kernel_width == 3 && kernel_height == 3 && stride_width == 2 && stride_height == 2) {
                auto output_shape = otter::calculate_conv_output_size(bottom_blob.sizes(), weight_data.sizes(), {stride_height, stride_width}, {padding_height, padding_width});
                if (!(output_shape[2] >= 8 && output_shape[3] >= 8)) {
                    optimize_kernel = weight_sgemm_data;
                } else {
                    optimize_kernel = weight_3x3s2_data;
                }
            } else {
                bool prefer_sgemm = (stride_width == 1 && stride_height == 1 && (in_channels >= 12 || out_channels >= 12)) || ((stride_width >= 2 || stride_height >= 2) && (in_channels >= 16 || out_channels >= 16));
                
                if (prefer_sgemm) {
                    optimize_kernel = weight_sgemm_data;
                }
            }
#elif __SSE2__
            if (kernel_width == 3 && kernel_height == 3 && stride_width == 1 && stride_height == 1) {
                if (in_channels >= 16 && out_channels >= 16) {
                    optimize_kernel = weight_3x3_winograd43_data;
                } else {
                    optimize_kernel = weight_3x3_winograd23_data;
                }
            } else {
                optimize_kernel = weight_sgemm_data;
            }
#endif
    }
    
    top_blob = otter::convolution(
        bottom_blob, weight_data, optimize_kernel, bias_data,
        {stride_height, stride_width},
        {padding_height, padding_width},
        {dilation_height, dilation_width},
        false,      // transpose
        {output_padding_height, output_padding_width},
        groups,
        opt.use_packing_layout,
        Tensor(),   // bottom_blob_int8_scales
        Tensor()    // weight_data_int8_scales
    );
    
    if (activation) {
        activation->forward_inplace(top_blob, opt);
    }
    
    return 0;
}

int ConvolutionLayer::create_pipeline_int8(const NetOption& opt) {
    scale_in_data = otter::empty({out_channels}, otter::ScalarType::Float);
    auto scale_in_data_a = scale_in_data.accessor<float, 1>();
    auto input_int8_scales_a = bottom_blob_int8_scales.accessor<float, 1>();
    auto weight_data_int8_scales_a = weight_data_int8_scales.accessor<float, 1>();
    for (const auto p : otter::irange(0, out_channels)) {
        float scale_in;
        if (weight_data_int8_scales_a[p] == 0)
            scale_in = 0;
        else
            scale_in = 1.f / (input_int8_scales_a[0] * weight_data_int8_scales_a[p]);

        scale_in_data_a[p] = scale_in;
    }
    
#if __SSE2__
    int elempack = 1;
    int out_elempack = 1;
    if (opt.use_packing_layout) {
        elempack = in_channels % 8 == 0 ? 8 : 1;
        out_elempack = out_channels % 4 == 0 ? 4 : 1;
    }
    
    if (in_channels == groups && groups == out_channels) {
        if (elempack == 8) {
            int maxk = kernel_width * kernel_height;
            weight_data_tf = weight_data.view({groups, maxk}).packing(8);
        } else if (elempack == 1) {
            weight_data_tf = weight_data;
        }
        
        return 0;
    }
    
    if (elempack == 8 && out_elempack == 4) {
        otter::convolution_im2col_sgemm_transform_kernel_pack8to4_int8_x86(weight_data, weight_sgemm_int8_data, in_channels, out_channels, kernel_width, kernel_height);
    }
    
    if (elempack == 8 && out_elempack == 1) {
        otter::convolution_im2col_sgemm_transform_kernel_pack8to1_int8_x86(weight_data, weight_sgemm_int8_data, in_channels, out_channels, kernel_width, kernel_height);
    }
    
    if (elempack == 1 && out_elempack == 4) {
        otter::convolution_im2col_sgemm_transform_kernel_pack1to4_int8_x86(weight_data, weight_sgemm_int8_data, in_channels, out_channels, kernel_width, kernel_height);
    }
    
    if (elempack == 1 && out_elempack == 1) {
        otter::convolution_im2col_sgemm_transform_kernel_int8_sse(weight_data, weight_sgemm_int8_data, in_channels, out_channels, kernel_width, kernel_height);
    }
#elif __ARM_NEON__
    int elempack = 1;
    int out_elempack = 1;
    if (opt.use_packing_layout) {
        elempack = in_channels % 8 == 0 ? 8 : 1;
        out_elempack = out_channels % 4 == 0 ? 4 : 1;
    }
    
    if (in_channels == groups && groups == out_channels) {
        if (elempack == 8) {
            int maxk = kernel_width * kernel_height;
            weight_data_tf = weight_data.view({groups, maxk}).packing(8);
        } else if (elempack == 1) {
            weight_data_tf = weight_data;
        }
        
        return 0;
    }
    
    if (elempack == 8 && out_elempack == 4) {
        otter::convolution_im2col_sgemm_transform_kernel_pack8to4_int8_neon(weight_data, weight_sgemm_int8_data, in_channels, out_channels, kernel_width, kernel_height);
    }
    
    if (elempack == 8 && out_elempack == 1) {
        otter::convolution_im2col_sgemm_transform_kernel_pack8to1_int8_neon(weight_data, weight_sgemm_int8_data, in_channels, out_channels, kernel_width, kernel_height);
    }
    
    if (elempack == 1 && out_elempack == 4) {
        otter::convolution_im2col_sgemm_transform_kernel_pack1to4_int8_neon(weight_data, weight_sgemm_int8_data, in_channels, out_channels, kernel_width, kernel_height);
    }
    
    if (elempack == 1 && out_elempack == 1) {
        otter::convolution_im2col_sgemm_transform_kernel_int8_neon(weight_data, weight_sgemm_int8_data, in_channels, out_channels, kernel_width, kernel_height);
    }
#endif
    
    return 0;
}

int ConvolutionLayer::forward_int8(const Tensor &bottom_blob, Tensor &top_blob, const NetOption &opt) const {
    
    Tensor optimize_kernel;
    
    auto k = weight_data.dim();
    auto dim = k - 2;
    
    ConvParams params;
    params.stride    = expand_param_if_needed({stride_height, stride_width}, "stride", dim);
    params.padding   = expand_param_if_needed({padding_height, padding_width}, "padding", dim);
    params.dilation  = expand_param_if_needed({dilation_height, dilation_width}, "dilation", dim);
    params.output_padding = expand_param_if_needed({output_padding_height, output_padding_width}, "output_padding", dim);
    params.transposed = false;
    params.benchmark = false;
    params.groups    = groups;
        
    auto dtype = bottom_blob.scalar_type();
    Tensor bottom_blob_int8;
    if (dtype != ScalarType::Byte && dtype != ScalarType::Byte4 && dtype != ScalarType::Byte8)
        bottom_blob_int8 = otter::quantize_to_int8(bottom_blob, bottom_blob_int8_scales, opt.use_packing_layout);
    else
        bottom_blob_int8 = bottom_blob;
    
    int elempack = bottom_blob_int8.elempack();
    int out_elempack_int32 = 1;
    
#if __SSE2__
    if (opt.use_packing_layout) {
        out_elempack_int32 = out_channels % 4 == 0 ? 4 : 1;
    }
#elif __ARM_NEON__
    if (opt.use_packing_layout) {
        out_elempack_int32 = out_channels % 4 == 0 ? 4 : 1;
    }
#endif
    
    if (params.use_cpu_x86(bottom_blob, weight_data)) {
        // depthwise
        if (params.is_depthwise(bottom_blob, weight_data)) {
            optimize_kernel = weight_data_tf;
        } else {
            if (elempack == 8 && out_elempack_int32 == 4) {
                optimize_kernel = weight_sgemm_int8_data;
            } else if (elempack == 8 && out_elempack_int32 == 1) {
                optimize_kernel = weight_sgemm_int8_data;
            } else if (elempack == 1 && out_elempack_int32 == 4) {
                optimize_kernel = weight_sgemm_int8_data;
            } else if (elempack == 1 && out_elempack_int32 == 1) { // general
                optimize_kernel = weight_sgemm_int8_data;
            }
        }
    } else if (params.use_cpu_neon(bottom_blob, weight_data)) {
        if (params.is_depthwise(bottom_blob, weight_data)) {
            optimize_kernel = weight_data_tf;
        } else {
            if (elempack == 8 && out_elempack_int32 == 4) {
                optimize_kernel = weight_sgemm_int8_data;
            } else if (elempack == 8 && out_elempack_int32 == 1) {
                optimize_kernel = weight_sgemm_int8_data;
            } else if (elempack == 1 && out_elempack_int32 == 4) {
                optimize_kernel = weight_sgemm_int8_data;
            } else if (elempack == 1 && out_elempack_int32 == 1) { // general
                optimize_kernel = weight_sgemm_int8_data;
            }
        }
    }
    
    auto top_blob_int32 = otter::convolution(
        bottom_blob_int8, weight_data, optimize_kernel, bias_data,
        params.stride,
        params.padding,
        params.dilation,
        false,      // transpose
        params.output_padding,
        groups,
        opt.use_packing_layout,
        bottom_blob_int8_scales,
        weight_data_int8_scales
    );
    
    bool use_int8_requantize = int8_scale_term > 100;
    
    if (use_int8_requantize) {
        top_blob = requantize_from_int32_to_int8(top_blob_int32, scale_in_data, top_blob_int8_scales, bias_data, activation_type, activation_params, opt.use_packing_layout);
    } else {
        top_blob = dequantize_from_int32(top_blob_int32, scale_in_data, bias_data, opt.use_packing_layout);

        if (activation) {
            activation->forward_inplace(top_blob, opt);
        }
    }
    
    return 0;
}

}   // end namespace otter
