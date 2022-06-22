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

Tensor quantize_to_int8(const Tensor& src, const Tensor& scale_data, bool pack = false);

Tensor dequantize_from_int32(const Tensor& src, const Tensor& scale_data, const Tensor& bias_data, bool pack = false);

Tensor requantize_from_int32_to_int8(const Tensor& src, const Tensor& scale_in_data, const Tensor& scale_out_data, const Tensor& bias_data, int activation_type, const Tensor& activation_params);

}   // end namespace otter

#endif /* Quantize_hpp */
