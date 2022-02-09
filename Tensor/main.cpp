//
//  main.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include <iostream>

#include "OTensor.hpp"

int main(int argc, const char * argv[]) {
    
    otter::TensorPrinter printer(10);
    
    auto a = otter::ones({1, 3}, ScalarType::Float);

    auto b = otter::full({1, 3}, 9.0, ScalarType::Float);
    
    auto c = otter::full({1, 3}, M_PI_2, ScalarType::Float);
    
    auto zero = otter::zeros({1, 3}, ScalarType::Float);
    
    auto sig_test = 1.0 / (1 + exp(-zero));

    printer.print<float>(sig_test);

    auto time_three = sig_test * 7;

    printer.print<float>(time_three);

    auto mod_two = time_three % 2;
    printer.print<float>(mod_two);
    
    auto bitwise_eight = otter::full({1, 3}, 7, ScalarType::Int);
    auto bitwise_four = otter::full({1, 3}, 4, ScalarType::Int);

    auto bitwise_test = bitwise_eight & bitwise_four;

    printer.print<int>(bitwise_test);
    
    bitwise_test = abs(~bitwise_test + 1);
    
    printer.print<int>(bitwise_test);
        
    return 0;
}
