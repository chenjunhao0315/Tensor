//
//  main.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "OTensor.hpp"
#include "Clock.hpp"
#include "Net.hpp"
#include "TensorFunction.hpp"
#include "Pool.hpp"
#include "Padding.hpp"

using namespace std;

int main(int argc, const char * argv[]) {
    
//    auto plane = otter::full({1, 3, 32, 32}, 1, otter::ScalarType::Float);
//    auto kernel = otter::full({3, 1, 3, 3}, 1, otter::ScalarType::Float);
//    otter::Tensor bias = otter::full({3}, 2, otter::ScalarType::Float);
//
//    otter::Tensor out;
//    otter::Clock a;
//    out = otter::convolution(plane, kernel, bias, {1, 1}, {1, 1}, {1, 1}, false, {0, 0}, 3, false);
//    a.stop_and_show();
//
//    auto input = otter::full({1, 3, 5, 5}, -1, otter::ScalarType::Float);
//
//    otter::Net net;
//    net.addLayer(otter::LayerOption{{"type", "Input"}, {"name", "data"}, {"output", "data"}, {"channel", "3"}, {"height", "320"}, {"width", "320"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_1"}, {"out_channels", "16"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "2"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_2"}, {"out_channels", "16"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_3"}, {"out_channels", "16"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "16"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_1"}, {"input", "bn_conv_3, bn_conv_2"}});
//    net.compile();
//    net.summary();

//    otter::Clock c;
//    auto extractor = net.create_extractor();
//    extractor.input("data", input);
//    otter::Tensor result;
//    extractor.extract("bn_conv_3", result, 0);
//    c.stop_and_show();
//    cout << result << endl;
    
//    auto plane = otter::tensor({1, 2, 3, 4, 5, 6, 7, 8, 9}, otter::ScalarType::Float).view({1, 1, 3, 3});
//    auto cube = otter::native::cat({plane, plane * 10, plane * 100}, 1);
//    auto weight = otter::tensor({1, 2, 3, 4}, otter::ScalarType::Float).view({1, 1, 2, 2});
//    weight = otter::native::cat({weight, weight, weight}, 1);
//    weight = otter::native::cat({weight, weight, weight, weight, weight, weight, weight, weight}, 0);
//    auto bias = otter::Tensor();
//
//    auto out = otter::convolution(cube, weight, bias, {1, 1}, {0, 0}, {1, 1}, false, {0, 0}, 1, false);
//    cout << out << endl;
    
//    auto input = otter::full({1, 3, 320, 320}, 1, otter::ScalarType::Float);
//
//    otter::Net net;
//    net.addLayer(otter::LayerOption{{"type", "Input"}, {"name", "data"}, {"output", "data"}, {"channel", "3"}, {"height", "320"}, {"width", "320"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_1"}, {"out_channels", "16"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "2"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_2"}, {"out_channels", "16"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_3"}, {"out_channels", "16"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "16"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_4"}, {"out_channels", "8"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_5"}, {"out_channels", "16"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_6"}, {"out_channels", "16"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "16"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_7"}, {"out_channels", "8"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_8"}, {"out_channels", "48"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_9"}, {"out_channels", "48"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "2"}, {"groups", "48"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_10"}, {"out_channels", "16"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_11"}, {"out_channels", "64"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_12"}, {"out_channels", "64"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "64"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_13"}, {"out_channels", "16"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_14"}, {"out_channels", "64"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_15"}, {"out_channels", "64"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "64"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_16"}, {"out_channels", "16"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_17"}, {"out_channels", "64"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_18"}, {"out_channels", "64"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "2"}, {"groups", "64"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_19"}, {"out_channels", "16"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_20"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_21"}, {"out_channels", "96"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "96"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_22"}, {"out_channels", "16"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_23"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_24"}, {"out_channels", "96"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "96"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_25"}, {"out_channels", "16"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_26"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_27"}, {"out_channels", "96"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "96"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_28"}, {"out_channels", "32"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_29"}, {"out_channels", "192"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_30"}, {"out_channels", "192"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "192"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_31"}, {"out_channels", "32"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_32"}, {"out_channels", "192"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_33"}, {"out_channels", "192"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "192"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_34"}, {"out_channels", "32"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_35"}, {"out_channels", "192"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_36"}, {"out_channels", "192"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "192"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_37"}, {"out_channels", "32"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_38"}, {"out_channels", "192"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_39"}, {"out_channels", "192"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "192"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_40"}, {"out_channels", "32"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_41"}, {"out_channels", "192"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_42"}, {"out_channels", "192"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "2"}, {"groups", "192"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_43"}, {"out_channels", "48"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_44"}, {"out_channels", "272"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_45"}, {"out_channels", "272"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "272"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_46"}, {"out_channels", "48"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_47"}, {"out_channels", "272"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_48"}, {"out_channels", "272"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "272"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_49"}, {"out_channels", "48"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_50"}, {"out_channels", "272"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_51"}, {"out_channels", "272"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "272"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_52"}, {"out_channels", "48"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_53"}, {"out_channels", "272"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_54"}, {"out_channels", "272"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "272"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_55"}, {"out_channels", "48"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_56"}, {"out_channels", "272"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_57"}, {"out_channels", "272"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "2"}, {"groups", "272"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_58"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_59"}, {"out_channels", "448"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_60"}, {"out_channels", "448"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "448"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_61"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_62"}, {"out_channels", "448"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_63"}, {"out_channels", "448"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "448"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_64"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_65"}, {"out_channels", "448"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_66"}, {"out_channels", "448"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "448"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_67"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_68"}, {"out_channels", "448"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_69"}, {"out_channels", "448"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "448"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_70"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_71"}, {"out_channels", "448"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_72"}, {"out_channels", "448"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "448"}, {"batchnorm", "true"}, {"activation", "LRelu"}});
//    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_73"}, {"out_channels", "96"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
//    net.compile();
//    net.summary();
//
//    otter::Clock c;
//    for (int i = 0; i < 10; ++i) {
//        auto extractor = net.create_extractor();
////        extractor.set_lightmode(false);
//        extractor.input("data", input);
//        otter::Tensor result;
//        extractor.extract("bn_conv_73", result, 0);
//    }
//    c.stop_and_show();
    
//    auto input = otter::full({1, 1, 6, 6}, 1, otter::ScalarType::Float);
//
//    otter::Net net;
//    net.addLayer(otter::LayerOption{{"type", "Input"}, {"name", "data"}, {"output", "data"}, {"channel", "3"}, {"height", "6"}, {"width", "6"}});
//    net.addLayer(otter::LayerOption{{"type", "LRelu"}, {"name", "lr_1"}});
//    net.addLayer(otter::LayerOption{{"type", "Split"}, {"name", "sp"}, {"output", "sp_0, sp_1"}});
//    net.addLayer(otter::LayerOption{{"type", "LRelu"}, {"name", "lr_2"}});
//    net.addLayer(otter::LayerOption{{"type", "LRelu"}, {"name", "lr_3"}});
//    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_1"}, {"input", "lr_3, sp_1"}});
//    net.addLayer(otter::LayerOption{{"type", "MaxPool"}, {"name", "p_1"}, {"stride", "2"}});
//    net.compile();
//    net.summary();
//
//    auto ex = net.create_extractor();
//    ex.input("data", input);
//    otter::Tensor output;
//    ex.extract("p_1", output, 0);
//
//    cout << output << endl;
    
//    otter::Net net;
//    net.addLayer(otter::LayerOption{{"type", "Input"}, {"name", "data"}, {"output", "data"}, {"channel", "1"}, {"height", "5"}, {"width", "5"}});
//    net.addLayer(otter::LayerOption{{"type", "MaxPool"}, {"name", "p_1"}, {"stride", "1"}, {"kernel", "4"}, {"darknet_mode", "true"}});
//    net.compile();
//    net.summary();
//
//    auto in = otter::range(1, 25, 1, otter::ScalarType::Float).view({1, 1, 5, 5});
//    auto ex = net.create_extractor();
//    ex.input("data", in);
//    otter::Tensor out;
//    ex.extract("p_1", out, 0);
//    cout << out << endl;
    
    otter::Net net;
    net.addLayer(otter::LayerOption{{"type", "Input"}, {"output", "data"}, {"channel", "1"}, {"height", "5"}, {"width", "5"}});
    net.addLayer(otter::LayerOption{{"type", "LRelu"}, {"name", "lr_1"}});
    net.addLayer(otter::LayerOption{{"type", "LRelu"}, {"name", "lr_2"}, {"input", "data"}});
    net.addLayer(otter::LayerOption{{"type", "Concat"}, {"name", "c_1"}, {"input", "lr_2, lr_1"}});
    net.compile();
    net.summary();

    auto in = otter::range(1, 25, 1, otter::ScalarType::Float).view({1, 1, 5, 5});
    auto ex = net.create_extractor();
    ex.input("data", in);
    otter::Tensor out;
    ex.extract("c_1", out, 0);
    cout << out << endl;
    
    auto pull_test = otter::tensor({1, 2, 3}, otter::ScalarType::Float);

    cout << pull_test << endl;

    return 0;
}
