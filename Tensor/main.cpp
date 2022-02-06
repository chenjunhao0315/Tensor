//
//  main.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include <iostream>

#include "TensorFactory.hpp"
#include "TensorOperator.hpp"
#include "ScalarOps.hpp"
#include "Parallel.hpp"
#include "Utils.hpp"
#include "Dispatch.hpp"

int main(int argc, const char * argv[]) {
    
    otter::TensorPrinter printer(10);
    
    auto a = otter::ones({1, 3}, ScalarType::Float);

    auto b = otter::full({1, 3}, 9.0, ScalarType::Float);
    
    auto c = otter::full({1, 3}, 5, ScalarType::Int);
    
    auto out = a + b + c.to(ScalarType::Float);
    out.print();
    printer.print<float>(out);
    
    auto test_t = (b + 17.0) / 2.0;
    test_t.print();
    printer.print<float>(test_t);
//
////    test_t.zero_();
//
//    printer.print<float>(test_t);
//
//    auto test_c = test_t.to(ScalarType::Int);
//
//    printer.print<int>(test_c);
    
//    out -= b;
//    printer.print<float>(out);
//
//    auto mul_out = b.mul(c);
//    printer.print<float>(mul_out);
    
//    auto test = otter::structured_add
    
    return 0;
}
