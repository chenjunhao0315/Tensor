//
//  MaxPoolLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/2.
//

#include "MaxPoolLayer.hpp"
#include "Pool.hpp"
#include "Padding.hpp"

#include "TensorMaker.hpp"

namespace otter {

MaxPoolLayer::MaxPoolLayer() {
    one_blob_only = true;
    support_inplace = false;
}

int MaxPoolLayer::parse_param(LayerOption& option, ParamDict& pd) {
    pd.clear();
    int stride_height = opt_find_int(option, "stride_h", -1);
    int stride_width  = opt_find_int(option, "stride_w", -1);
    int stride        = opt_find_int(option, "stride", 1);
    if (stride_height < 1 || stride_width < 1) {
        if (stride_height < 0) stride_height = stride;
        if (stride_width < 0)  stride_width  = stride;
    }
    int kernel_height = opt_find_int(option, "kernel_h", -1);
    int kernel_width  = opt_find_int(option, "kernel_w", -1);
    int kernel        = opt_find_int(option, "kernel", stride);
    if (kernel_height < 1 || kernel_width < 1) {
        if (kernel_height < 0) kernel_height = kernel;
        if (kernel_width < 0)  kernel_width  = kernel;
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
    int ceil_mode = (opt_check_string(option, "ceil_mode")) ? 1 : 0;
    int darknet_mode = (opt_check_string(option, "darknet_mode")) ? 1 : 0;
    
    pd.set((int)MaxPoolParam::Kernel_height, kernel_height);
    pd.set((int)MaxPoolParam::Kernel_width, kernel_width);
    pd.set((int)MaxPoolParam::Stride_height, stride_height);
    pd.set((int)MaxPoolParam::Stride_width,  stride_width);
    pd.set((int)MaxPoolParam::Padding_height, padding_height);
    pd.set((int)MaxPoolParam::Padding_width,  padding_width);
    pd.set((int)MaxPoolParam::Dilation_height, dilation_height);
    pd.set((int)MaxPoolParam::Dilation_width,  dilation_width);
    pd.set((int)MaxPoolParam::Ceil_mode, ceil_mode);
    pd.set((int)MaxPoolParam::Darknet_mode, darknet_mode);
    
    return 0;
}

int MaxPoolLayer::compute_output_shape(ParamDict &pd) {
    auto shape_a = bottom_shapes[0].accessor<int, 1>();
    int input_batch = shape_a[0];
    int input_channels = shape_a[1];
    int input_height = shape_a[2];
    int input_width = shape_a[3];
    
    int kernel_height   = pd.get((int)MaxPoolParam::Kernel_height, 3);
    int kernel_width    = pd.get((int)MaxPoolParam::Kernel_width, 3);
    int stride_height   = pd.get((int)MaxPoolParam::Stride_height, 1);
    int stride_width    = pd.get((int)MaxPoolParam::Stride_width,  1);
    int padding_height  = pd.get((int)MaxPoolParam::Padding_height, 0);
    int padding_width   = pd.get((int)MaxPoolParam::Padding_width,  0);
    int dilation_height = pd.get((int)MaxPoolParam::Dilation_height, 1);
    int dilation_width  = pd.get((int)MaxPoolParam::Dilation_width,  1);
    int ceil_mode       = pd.get((int)MaxPoolParam::Ceil_mode, 0);
    int darknet_mode    = pd.get((int)MaxPoolParam::Darknet_mode, 0);
    
    int out_height;
    int out_width;
    
    if (darknet_mode) {
        out_height = (input_height + padding_height - kernel_height) / stride_height + 1;
        out_width = (input_width + padding_width - kernel_width) / stride_height + 1;
    } else {
        out_height = pooling_output_shape(input_height, kernel_height, padding_height, stride_height, dilation_height, ceil_mode);
        out_width = pooling_output_shape(input_width, kernel_width, padding_width, stride_width, dilation_width, ceil_mode);
    }
    
    pd.set(OUTPUT_SHAPE_HINT, otter::tensor({input_batch, input_channels, out_height, out_width}, ScalarType::Int));
    
    return 0;
}

int MaxPoolLayer::load_param(const ParamDict &pd) {
    stride_height   = pd.get((int)MaxPoolParam::Stride_height, 1);
    stride_width    = pd.get((int)MaxPoolParam::Stride_width,  1);
    kernel_height   = pd.get((int)MaxPoolParam::Kernel_height, stride_height);
    kernel_width    = pd.get((int)MaxPoolParam::Kernel_width, stride_width);
    padding_height  = pd.get((int)MaxPoolParam::Padding_height, kernel_height - 1);
    padding_width   = pd.get((int)MaxPoolParam::Padding_width,  kernel_width - 1);
    dilation_height = pd.get((int)MaxPoolParam::Dilation_height, 1);
    dilation_width  = pd.get((int)MaxPoolParam::Dilation_width,  1);
    ceil_mode       = pd.get((int)MaxPoolParam::Ceil_mode, 0);
    darknet_mode    = pd.get((int)MaxPoolParam::Darknet_mode, 0);
    
    return 0;
}

int MaxPoolLayer::forward(const Tensor& bottom_blob, Tensor& top_blob, const NetOption& opt) const {
    if (darknet_mode) {
        int height_offset = (kernel_height - 1) / 2;
        int width_offset = (kernel_width - 1) / 2;
            
        auto bottom_blob_pad = otter::constant_pad(bottom_blob, {height_offset, kernel_height - height_offset - 1, width_offset, kernel_width - width_offset - 1}, -10000000);
        top_blob = otter::max_pool2d(bottom_blob_pad, {kernel_height, kernel_width}, {stride_height, stride_width}, {0, 0}, {1, 1}, false);
    } else {
        top_blob = otter::max_pool2d(bottom_blob, {kernel_height, kernel_width}, {stride_height, stride_width}, {padding_height, padding_width}, {dilation_height, dilation_width}, ceil_mode);
    }
    return 0;
}

}   // end namespace otter
