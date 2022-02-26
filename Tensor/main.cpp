//
//  main.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "OTensor.hpp"
#include "Clock.hpp"
#include "Net.hpp"

using namespace std;

int main(int argc, const char * argv[]) {
    
    auto plane = otter::full({1, 3, 32, 32}, 1, otter::ScalarType::Float);
    auto kernel = otter::full({3, 1, 3, 3}, 1, otter::ScalarType::Float);
    otter::Tensor bias = otter::full({3}, 2, otter::ScalarType::Float);

    otter::Tensor out;
    otter::Clock a;
    out = otter::convolution(plane, kernel, bias, {1, 1}, {1, 1}, {1, 1}, false, {0, 0}, 3, false);
    a.stop_and_show();

    auto input = otter::full({1, 3, 5, 5}, 1, otter::ScalarType::Float);

    otter::Net net;
    net.addLayer(otter::LayerOption{{"type", "Input"}, {"name", "data"}, {"output", "data"}, {"channel", "3"}, {"height", "320"}, {"width", "320"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_1"}, {"out_channels", "16"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "2"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_2"}, {"out_channels", "16"}, {"kernel", "1"}, {"padding", "0"}, {"stride", "1"}, {"batchnorm", "true"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_3"}, {"out_channels", "16"}, {"kernel", "3"}, {"padding", "1"}, {"stride", "1"}, {"groups", "16"}, {"batchnorm", "true"}});
    net.compile();
//    net.summary();

    otter::Clock c;
    auto extractor = net.create_extractor();
    extractor.input("data", input);
    otter::Tensor result;
    extractor.extract("bn_conv_3", result, 0);
    c.stop_and_show();
    cout << result << endl;
    
    return 0;
}
