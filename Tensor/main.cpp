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
    
    auto plane1 = otter::full({1, 3, 16, 16}, 2, otter::ScalarType::Float);
    auto plane2 = otter::full({1, 3, 16, 16}, 4, otter::ScalarType::Float);
    
//    auto out = plane1 + plane2;
//    out.print();
    
    otter::Net net;
    net.addLayer(otter::LayerOption{{"type", "Input"}, {"name", "data"}, {"channel", "3"}, {"height", "16"}, {"width", "16"}});
    net.addLayer(otter::LayerOption{{"type", "LRelu"}, {"name", "lr_1"}});
    net.addLayer(otter::LayerOption{{"type", "LRelu"}, {"name", "lr_2"}});
    net.addLayer(otter::LayerOption{{"type", "ShortCut"}, {"name", "sc_1"}, {"input", "lr_2, lr_1"}});
    net.compile();
    net.summary();
    
    otter::Tensor out;
    auto ex = net.create_extractor();
    ex.set_lightmode(false);
    ex.input("data", plane1);
    ex.extract("sc_1", out, 0);
    cout << out;
    
    
    
    return 0;
}
