//
//  Quantize.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/2.
//

#ifndef Quantize_hpp
#define Quantize_hpp

#include "Tensor.hpp"

namespace otter {

Tensor quantize_to_int8(const Tensor& src, const Tensor& scale_data);

}   // end namespace otter

#endif /* Quantize_hpp */
