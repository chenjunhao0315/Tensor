//
//  Dropout.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/3.
//

#ifndef Dropout_hpp
#define Dropout_hpp

#include <tuple>

namespace otter {

class Tensor;

std::tuple<Tensor, Tensor> dropout(const Tensor& input, double p, bool train);

}

#endif /* Dropout_hpp */
