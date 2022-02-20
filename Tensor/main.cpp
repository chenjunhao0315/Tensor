//
//  main.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "OTensor.hpp"
#include "TensorShape.hpp"
#include "ConvolutionMM2D.hpp"
#include "Convolution.hpp"
#include "Clock.hpp"
#include "TensorPixel.hpp"
#include "Parallel.hpp"
#include "Net.hpp"
#include "ConvolutionLayer.hpp"

using namespace std;

int main(int argc, const char * argv[]) {
    
    auto plane = otter::full({1, 3, 32, 32}, 1, otter::ScalarType::Float);
    auto kernel = otter::full({3, 1, 3, 3}, 1, otter::ScalarType::Float);
    otter::Tensor bias;

    otter::Tensor out;
    otter::Clock a;
    out = otter::convolution(plane, kernel, bias, {1, 1}, {1, 1}, {1, 1}, false, {0, 0}, 3, false);
    a.stop_and_show();
    
    otter::Net net;
    net.addLayer(otter::LayerOption{{"type", "Input"}, {"name", "Input"}, {"output", "data"}, {"channel", "3"}, {"height", "32"}, {"width", "32"}});
    net.addLayer(otter::LayerOption{{"type", "Convolution"}, {"name", "conv_1"}, {"input", "data"}, {"output", "conv_1"}, {"stride", "1"}, {"groups", "3"}});
    net.compile();
    ((otter::ConvolutionLayer*)net.layers[1])->weight_data = kernel;
    auto extractor = net.create_extractor();
    extractor.input("data", plane);
    otter::Tensor result;
    extractor.extract("conv_1", result, 0);
    cout << result << endl;
    
    
    return 0;
}
