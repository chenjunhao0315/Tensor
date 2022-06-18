//
//  ConvolutionMM2DNeonPack.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/17.
//

#ifndef ConvolutionMM2DNeonPack_hpp
#define ConvolutionMM2DNeonPack_hpp

#include "Tensor.hpp"

namespace otter {

#if __ARM_NEON__

void convolution_im2col_sgemm_transform_kernel_pack4_neon(const Tensor& kernel, Tensor& kernel_tf, int inch, int outch, int kernel_w, int kernel_h);

void convolution_im2col_sgemm_transform_kernel_pack4to1_neon(const Tensor& kernel, Tensor& kernel_tf, int inch, int outch, int kernel_w, int kernel_h);

void convolution_im2col_sgemm_transform_kernel_pack1to4_neon(const Tensor& kernel, Tensor& kernel_tf, int inch, int outch, int kernel_w, int kernel_h);

Tensor& sgemm_conv2d_pack4_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);
    
Tensor sgemm_conv2d_pack4_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor& sgemm_conv2d_pack4to1_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);
    
Tensor sgemm_conv2d_pack4to1_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor& sgemm_conv2d_pack1to4_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);
    
Tensor sgemm_conv2d_pack1to4_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor conv2d_1x1s1_sgemm_pack4_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_1x1s1_sgemm_pack4_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

Tensor conv2d_1x1s1_sgemm_pack4to1_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_1x1s1_sgemm_pack4to1_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

Tensor conv2d_1x1s1_sgemm_pack1to4_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_1x1s1_sgemm_pack1to4_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

#endif  // __ARM_NEON__

}   // end namespace otter

#endif /* ConvolutionMM2DNeonPack_hpp */
