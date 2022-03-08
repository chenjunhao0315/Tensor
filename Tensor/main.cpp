//
//  main.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "OTensor.hpp"
#include "Clock.hpp"
#include "Net.hpp"
#include "Normalization.hpp"
#include "Padding.hpp"
#include "Pool.hpp"

using namespace std;

int main(int argc, const char * argv[]) {
    otter::Net net;
    net.addLayer(otter::LayerOption{{"type", "Input"}, {"name", "data"}, {"output", "data"}, {"channel", "3"}, {"height", "320"}, {"width", "320"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_1"}, {"out_channels", "16"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "2"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_2"}, {"out_channels", "16"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_3"}, {"out_channels", "16"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "16"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_4"}, {"out_channels", "8"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_5"}, {"out_channels", "16"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_6"}, {"out_channels", "16"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "16"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_7"}, {"out_channels", "8"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_1"}, {"input", "bn_conv_7, bn_conv_4"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_8"}, {"out_channels", "48"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_9"}, {"out_channels", "48"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "2"}, {"groups", "48"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_10"}, {"out_channels", "16"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_11"}, {"out_channels", "64"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_12"}, {"out_channels", "64"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "64"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_13"}, {"out_channels", "16"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_2"}, {"input", "bn_conv_13, bn_conv_10"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_14"}, {"out_channels", "64"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_15"}, {"out_channels", "64"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "64"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_16"}, {"out_channels", "16"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_3"}, {"input", "bn_conv_16, sc_2"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_17"}, {"out_channels", "64"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_18"}, {"out_channels", "64"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "2"}, {"groups", "64"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_19"}, {"out_channels", "16"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_20"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_21"}, {"out_channels", "96"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "96"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_22"}, {"out_channels", "16"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_4"}, {"input", "bn_conv_22, bn_conv_19"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_23"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_24"}, {"out_channels", "96"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "96"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_25"}, {"out_channels", "16"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_5"}, {"input", "bn_conv_25, sc_4"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_26"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_27"}, {"out_channels", "96"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "96"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_28"}, {"out_channels", "32"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_29"}, {"out_channels", "192"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_30"}, {"out_channels", "192"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "192"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_31"}, {"out_channels", "32"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_6"}, {"input", "bn_conv_31, bn_conv_28"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_32"}, {"out_channels", "192"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_33"}, {"out_channels", "192"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "192"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_34"}, {"out_channels", "32"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_7"}, {"input", "bn_conv_34, sc_6"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_35"}, {"out_channels", "192"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_36"}, {"out_channels", "192"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "192"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_37"}, {"out_channels", "32"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_8"}, {"input", "bn_conv_37, sc_7"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_38"}, {"out_channels", "192"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_39"}, {"out_channels", "192"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "192"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_40"}, {"out_channels", "32"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_9"}, {"input", "bn_conv_40, sc_8"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_41"}, {"out_channels", "192"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_42"}, {"out_channels", "192"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "2"}, {"groups", "192"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_43"}, {"out_channels", "48"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_44"}, {"out_channels", "272"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_45"}, {"out_channels", "272"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "272"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_46"}, {"out_channels", "48"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_10"}, {"input", "bn_conv_46, bn_conv_43"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_47"}, {"out_channels", "272"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_48"}, {"out_channels", "272"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "272"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_49"}, {"out_channels", "48"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_11"}, {"input", "bn_conv_49, sc_10"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_50"}, {"out_channels", "272"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_51"}, {"out_channels", "272"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "272"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_52"}, {"out_channels", "48"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_12"}, {"input", "bn_conv_52, sc_11"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_53"}, {"out_channels", "272"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_54"}, {"out_channels", "272"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "272"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_55"}, {"out_channels", "48"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_13"}, {"input", "bn_conv_55, sc_12"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_56"}, {"out_channels", "272"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_57"}, {"out_channels", "272"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "2"}, {"groups", "272"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_58"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_59"}, {"out_channels", "448"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_60"}, {"out_channels", "448"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "448"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_61"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_14"}, {"input", "bn_conv_61, bn_conv_58"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_62"}, {"out_channels", "448"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_63"}, {"out_channels", "448"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "448"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_64"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_15"}, {"input", "bn_conv_64, sc_14"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_65"}, {"out_channels", "448"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_66"}, {"out_channels", "448"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "448"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_67"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_16"}, {"input", "bn_conv_67, sc_15"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_68"}, {"out_channels", "448"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_69"}, {"out_channels", "448"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "448"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_70"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_17"}, {"input", "bn_conv_70, sc_16"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_71"}, {"out_channels", "448"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_72"}, {"out_channels", "448"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "448"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_73"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_18"}, {"input", "bn_conv_73, sc_17"}});
    net.addLayer(otter::LayerOption{{"type", "MaxPool"}, {"name", "mp_1"}, {"kernel", "3"}, {"padding", "2"}, {"darknet_mode", "true"}});
    net.addLayer(otter::LayerOption{{"type", "MaxPool"}, {"name", "mp_2"}, {"kernel", "5"}, {"padding", "4"}, {"input", "sc_18"}, {"darknet_mode", "true"}});
    net.addLayer(otter::LayerOption{{"type", "MaxPool"}, {"name", "mp_3"}, {"kernel", "9"}, {"padding", "8"}, {"input", "sc_18"}, {"darknet_mode", "true"}});
    net.addLayer(otter::LayerOption{{"type", "Concat"}, {"name", "concat_1"}, {"input", {"mp_3, mp_2, mp_1, sc_18"}}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_74"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_75"}, {"out_channels", "96"}, {"kernel", "5"}, {"padding", "2"}, {"stride", "1"}, {"groups", "96"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_76"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_77"}, {"out_channels", "96"}, {"kernel", "5"}, {"padding", "2"}, {"stride", "1"}, {"groups", "96"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_78"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_79"}, {"out_channels", "255"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}});
    net.addLayer(otter::LayerOption{{"type", "Upsample"}, {"name", "upsample_1"}, {"stride", "2"}, {"darknet_mode", "true"}, {"input", "lr_conv_74"}});
    net.addLayer(otter::LayerOption{{"type", "Concat"}, {"name", "concat_2"}, {"input", {"upsample_1, sc_13"}}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_80"}, {"out_channels", "144"}, {"kernel", "5"}, {"padding", "2"}, {"stride", "1"}, {"groups", "144"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_81"}, {"out_channels", "144"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_82"}, {"out_channels", "144"}, {"kernel", "5"}, {"padding", "2"}, {"stride", "1"}, {"groups", "144"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_83"}, {"out_channels", "144"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_84"}, {"out_channels", "255"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}});
    net.addLayer(otter::LayerOption{{"type", "Yolov3DetectionOutput"}, {"name", "yolo"}, {"input", "conv_79, conv_84"}});
    net.compile();
//    net.summary();

//    net.load_weight("yolo-fastest-1.1-xl.dam");
//
//    auto in = otter::full({1, 3, 320, 320}, 0.5, otter::ScalarType::Float);
//    otter::Clock clock;
//    auto ex = net.create_extractor();
//    ex.input("data", in);
//    otter::Tensor out;
//    ex.extract("conv_84", out, 0);
//    clock.stop_and_show();
//    out.print();
//
//    cout << out << endl;
    
//    auto t1 = otter::range(1, 27, 1, otter::ScalarType::Float).view({1, 3, 3, 3});
//    auto t2 = t1[0];
//    cout << t2.slice(0, 1, 3);
    
    return 0;
}
