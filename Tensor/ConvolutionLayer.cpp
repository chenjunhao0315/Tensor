//
//  ConvolutionLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/21.
//

#include "ConvolutionLayer.hpp"
#include "LayerRegistry.hpp"
#include "Convolution.hpp"

namespace otter {

ConvolutionLayer::ConvolutionLayer() {
    one_blob_only = true;
    support_inplace = false;
}

int ConvolutionLayer::prase_param(LayerOption& option, ParamDict& pd) {
    pd.clear();
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
    
    pd.set((int)ConvParam::Stride_height, stride_height);
    pd.set((int)ConvParam::Stride_width,  stride_width);
    pd.set((int)ConvParam::Padding_height, padding_height);
    pd.set((int)ConvParam::Padding_width,  padding_width);
    pd.set((int)ConvParam::Dilation_height, dilation_height);
    pd.set((int)ConvParam::Dilation_width,  dilation_height);
    pd.set((int)ConvParam::Output_padding_height, output_padding_height);
    pd.set((int)ConvParam::Output_padding_height, output_padding_height);
    pd.set((int)ConvParam::Group, groups);
    
    
    return 0;
}

int ConvolutionLayer::load_param(const ParamDict &pd) {
    stride_height   = pd.get((int)ConvParam::Stride_height, 1);
    stride_width    = pd.get((int)ConvParam::Stride_width,  1);
    padding_height  = pd.get((int)ConvParam::Padding_height, 0);
    padding_width   = pd.get((int)ConvParam::Padding_width,  0);
    dilation_height = pd.get((int)ConvParam::Dilation_height, 1);
    dilation_width  = pd.get((int)ConvParam::Dilation_width,  1);
    output_padding_height = pd.get((int)ConvParam::Output_padding_height, 0);
    output_padding_width  = pd.get((int)ConvParam::Output_padding_height, 0);
    groups = pd.get((int)ConvParam::Group, 1);
    
    return 0;
}

int ConvolutionLayer::forward(const Tensor &bottom_blob, Tensor &top_blob, const NetOption &opt) const {
    
    top_blob = otter::convolution(
        bottom_blob, weight_data, bias_data,
        {stride_height, stride_width},
        {padding_height, padding_width},
        {dilation_height, dilation_width},
        false,      // transpose
        {output_padding_height, output_padding_width},
        groups,
        false       // benchmark
    );
    
    return 0;
}

REGISTER_LAYER_CLASS(Convolution);

}   // end namespace otter
