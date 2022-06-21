//
//  Convolution.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/15.
//

#ifndef Convolution_hpp
#define Convolution_hpp

#include "ConvolutionUtils.hpp"
#include "ConvolutionMM2D.hpp"

namespace otter {

std::ostream& operator<<(std::ostream & out, const ConvParams& params);

Tensor convolution(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& weight_o,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    bool packed_ = true,
    const Tensor& input_int8_scales = Tensor(),
    const Tensor& weight_int8_scales = Tensor());

Tensor convolution_nogroup_backend(const Tensor& self, const Tensor& weight, const Tensor& weight_o, const Tensor& bias, ConvBackend backend, ConvParams& params, const Tensor& input_int8_scales = Tensor(), const Tensor& weight_int8_scales = Tensor());

Tensor convolution_packed(
    const Tensor& input_r,
    const Tensor& weight_r,
    const Tensor& weight_o,
    const Tensor& bias_r,
    IntArrayRef stride_,
    IntArrayRef padding_,
    IntArrayRef dilation_,
    bool transposed_,
    IntArrayRef output_padding_,
    int64_t groups_,
    const Tensor& input_int8_scales = Tensor(),
    const Tensor& weight_int8_scales = Tensor());

Tensor convolution_packed_nogroup_backend(const Tensor& self, const Tensor& weight, const Tensor& weight_o, const Tensor& bias, ConvBackend backend, ConvParams& params, const Tensor& input_int8_scales = Tensor(), const Tensor& weight_int8_scales = Tensor());

}

#endif /* Convolution_hpp */
