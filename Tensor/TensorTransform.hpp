//
//  TensorTransform.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/4.
//

#ifndef TensorTransform_hpp
#define TensorTransform_hpp

#include "Tensor.hpp"

namespace otter {

// border: [left, right, top, bottom, channel_front, channel_rear, batch_front, batch_rear]
Tensor crop(const Tensor& input, IntArrayRef border);

Tensor& crop_(const Tensor& input, IntArrayRef border, Tensor& output);

}

#endif /* TensorTransform_hpp */
