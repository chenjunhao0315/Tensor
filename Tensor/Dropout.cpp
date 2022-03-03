//
//  Dropout.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/3.
//

#include "Dropout.hpp"
#include "Tensor.hpp"
#include "TensorFactory.hpp"

namespace otter {

std::tuple<Tensor, Tensor> dropout(const Tensor& input, double p, bool train) {
    if (input.numel() == 0) {
        return std::make_tuple(input, otter::empty_like(input, input.options()));
    }
    
    Tensor mask;
    Tensor output;
    if (train) {
        double p1m = 1. - p;
        double scale = p1m == 0 ? 0. : 1. / p1m;
        
        mask = otter::empty_like(input, input.options());
//        mask.bernoulli_(p1m);
        
        output = input.mul(mask).mul_(scale);
    } else {
        mask = otter::ones_like(input, input.options());
        output = input.clone();
    }
    
    return std::make_tuple(output, mask);
}

}
