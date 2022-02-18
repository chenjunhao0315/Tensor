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
#include "Vec.hpp"
#include "TensorFunction.hpp"
#include "Parallel.hpp"
#include "Exception.hpp"

using namespace std;

int main(int argc, const char * argv[]) {
    
    auto plane = otter::full({1, 3, 320, 320}, 1, otter::ScalarType::Float);
    auto kernel = otter::full({3, 1, 3, 3}, 1, otter::ScalarType::Float);
    otter::Tensor bias;

    otter::Tensor out;
    otter::Clock a;
    out = otter::convolution(plane, kernel, bias, {1, 1}, {1, 1}, {1, 1}, false, {0, 0}, 3, false);
    a.stop_and_show();
    
    
    
    return 0;
}
