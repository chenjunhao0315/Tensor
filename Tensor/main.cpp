//
//  main.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "OTensor.hpp"
#include "Parallel.hpp"

int main(int argc, const char * argv[]) {
    
    auto linspace = otter::linspace(-5, 5, 11, ScalarType::Float);
    
    std::cout << linspace << std::endl;
    
    auto sigmoid = 1 / (1 + exp(-linspace));
    
    std::cout << sigmoid << std::endl;
    
    return 0;
}
