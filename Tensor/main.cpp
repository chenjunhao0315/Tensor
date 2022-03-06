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
    otter::Net net;
    net.addLayer(otter::LayerOption{{"type", "Input"}, {"output", "data"}, {"channel", "1"}, {"height", "5"}, {"width", "5"}});
    net.addLayer(otter::LayerOption{{"type", "LRelu"}, {"name", "lr_1"}});
    net.addLayer(otter::LayerOption{{"type", "LRelu"}, {"name", "lr_2"}, {"input", "data"}});
    net.addLayer(otter::LayerOption{{"type", "Concat"}, {"name", "c_1"}, {"input", "lr_2, lr_1"}});
    net.addLayer(otter::LayerOption{{"type", "Upsample"}, {"name", "up_1"}, {"darknet_mode", "true"}, {"stride", "2"}});
    net.compile();
    net.summary();

    auto in = otter::range(1, 25, 1, otter::ScalarType::Float).view({1, 1, 5, 5});
    auto ex = net.create_extractor();
    ex.input("data", in);
    otter::Tensor out;
    ex.extract("up_1", out, 0);
    cout << out << endl;
    return 0;
}
