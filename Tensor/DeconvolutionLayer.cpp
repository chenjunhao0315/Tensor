//
//  DeconvolutionLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/5.
//

#include "DeconvolutionLayer.hpp"
#include "Tensor.hpp"
#include "TensorFactory.hpp"
#include "TensorMaker.hpp"
#include "Convolution.hpp"

namespace otter {

DeconvolutionLayer::DeconvolutionLayer() {
    one_blob_only = true;
    support_inplace = false;
}

int DeconvolutionLayer::parse_param(LayerOption& option, ParamDict& pd) {
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
    
    pd.set((int)DeconvParam::In_channels, in_channels);
    pd.set((int)DeconvParam::Out_channels, out_channels);
    pd.set((int)DeconvParam::Kernel_height, kernel_height);
    pd.set((int)DeconvParam::Kernel_width, kernel_width);
    pd.set((int)DeconvParam::Stride_height, stride_height);
    pd.set((int)DeconvParam::Stride_width,  stride_width);
    pd.set((int)DeconvParam::Padding_height, padding_height);
    pd.set((int)DeconvParam::Padding_width,  padding_width);
    pd.set((int)DeconvParam::Dilation_height, dilation_height);
    pd.set((int)DeconvParam::Dilation_width,  dilation_width);
    pd.set((int)DeconvParam::Output_padding_height, output_padding_height);
    pd.set((int)DeconvParam::Output_padding_height, output_padding_height);
    pd.set((int)DeconvParam::Group, groups);
    pd.set((int)DeconvParam::Bias_term, bias_term);
    
    return 0;
}

int DeconvolutionLayer::compute_output_shape(ParamDict &pd) {
    auto shape_a = bottom_shapes[0].accessor<int, 1>();
    int input_batch    = shape_a[0];
    int input_channels = shape_a[1];
    int input_height   = shape_a[2];
    int input_width    = shape_a[3];
    int out_channels    = pd.get((int)DeconvParam::Out_channels, 1);
    int kernel_height   = pd.get((int)DeconvParam::Kernel_height, 3);
    int kernel_width    = pd.get((int)DeconvParam::Kernel_width, 3);
    int stride_height   = pd.get((int)DeconvParam::Stride_height, 1);
    int stride_width    = pd.get((int)DeconvParam::Stride_width,  1);
    int padding_height  = pd.get((int)DeconvParam::Padding_height, 0);
    int padding_width   = pd.get((int)DeconvParam::Padding_width,  0);
    int dilation_height = pd.get((int)DeconvParam::Dilation_height, 1);
    int dilation_width  = pd.get((int)DeconvParam::Dilation_width,  1);
    int groups          = pd.get((int)DeconvParam::Group, 1);
    int output_padding_height = pd.get((int)DeconvParam::Output_padding_height, 0);
    int output_padding_width  = pd.get((int)DeconvParam::Output_padding_width, 0);
    int output_height = (input_height - 1) * stride_height - 2 * padding_height +
        (dilation_height * (kernel_height - 1) + 1) + output_padding_height;
    int output_width = (input_width - 1) * stride_width - 2 * padding_width +
        (dilation_width * (kernel_width - 1) + 1) + output_padding_width;
    int weight_data_size = input_channels * out_channels / groups * kernel_height * kernel_width;
    
    pd.set((int)DeconvParam::In_channels, input_channels);
    pd.set((int)DeconvParam::Weight_data_size, weight_data_size);
    pd.set(OUTPUT_SHAPE_HINT, otter::tensor({input_batch, out_channels, output_height, output_width}, ScalarType::Int));
    
    return 0;
}

int DeconvolutionLayer::load_param(const ParamDict &pd) {
    in_channels     = pd.get((int)DeconvParam::In_channels, 1);
    out_channels    = pd.get((int)DeconvParam::Out_channels, 1);
    kernel_height   = pd.get((int)DeconvParam::Kernel_height, 3);
    kernel_width    = pd.get((int)DeconvParam::Kernel_width, 3);
    stride_height   = pd.get((int)DeconvParam::Stride_height, 1);
    stride_width    = pd.get((int)DeconvParam::Stride_width,  1);
    padding_height  = pd.get((int)DeconvParam::Padding_height, 0);
    padding_width   = pd.get((int)DeconvParam::Padding_width,  0);
    dilation_height = pd.get((int)DeconvParam::Dilation_height, 1);
    dilation_width  = pd.get((int)DeconvParam::Dilation_width,  1);
    output_padding_height = pd.get((int)DeconvParam::Output_padding_height, 0);
    output_padding_width  = pd.get((int)DeconvParam::Output_padding_height, 0);
    groups = pd.get((int)DeconvParam::Group, 1);
    bias_term = pd.get((int)DeconvParam::Bias_term, 0);
    weight_data_size = pd.get((int)DeconvParam::Weight_data_size, 0);
    
    return 0;
}

int DeconvolutionLayer::init_model() {
    weight_data = otter::rand({in_channels, out_channels / groups, kernel_height, kernel_width}, ScalarType::Float);
    if (bias_term)
        bias_data = otter::rand({out_channels}, ScalarType::Float);
    
    return 0;
}

int DeconvolutionLayer::load_model(const Initializer& initializer) {
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
    }
    
    return 0;
}

int DeconvolutionLayer::forward(const Tensor &bottom_blob, Tensor &top_blob, const NetOption& /*opt*/) const {
    
    top_blob = otter::convolution(
        bottom_blob, weight_data, bias_data,
        {stride_height, stride_width},
        {padding_height, padding_width},
        {dilation_height, dilation_width},
        true,      // transpose
        {output_padding_height, output_padding_width},
        groups,
        false       // benchmark
    );
    
    return 0;
}

REGISTER_LAYER_CLASS(Deconvolution);

}   // end namespace otter
