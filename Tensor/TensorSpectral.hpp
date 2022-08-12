//
//  TensorSpectral.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/8/12.
//

#ifndef TensorSpectral_hpp
#define TensorSpectral_hpp

#include <tuple>

#include "Tensor.hpp"

namespace otter {

std::tuple<Tensor, Tensor> fft(const Tensor& real_part, const Tensor& imag_part = Tensor());

}

#endif /* TensorSpectral_hpp */
