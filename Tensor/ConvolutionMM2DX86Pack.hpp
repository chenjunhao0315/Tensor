//
//  ConvolutionMM2DX86Pack.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/17.
//

#ifndef ConvolutionMM2DX86Pack_hpp
#define ConvolutionMM2DX86Pack_hpp

#include "Tensor.hpp"

namespace otter {

#if __SSE2__

void convolution_im2col_sgemm_transform_kernel_pack4_sse(const Tensor& kernel, Tensor& kernel_tf, int inch, int outch, int kernel_w, int kernel_h);

void convolution_im2col_sgemm_transform_kernel_pack4to1_sse(const Tensor& kernel, Tensor& kernel_tf, int inch, int outch, int kernel_w, int kernel_h);

void convolution_im2col_sgemm_transform_kernel_pack1to4_sse(const Tensor& kernel, Tensor& kernel_tf, int inch, int outch, int kernel_w, int kernel_h);

Tensor& sgemm_conv2d_pack4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);
    
Tensor sgemm_conv2d_pack4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor& sgemm_conv2d_pack4to1_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);
    
Tensor sgemm_conv2d_pack4to1_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor& sgemm_conv2d_pack1to4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);
    
Tensor sgemm_conv2d_pack1to4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor conv2d_1x1s1_sgemm_pack4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_1x1s1_sgemm_pack4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

Tensor conv2d_1x1s2_sgemm_pack4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_1x1s2_sgemm_pack4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

Tensor conv2d_1x1s1_sgemm_pack1to4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_1x1s1_sgemm_pack1to4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

Tensor conv2d_1x1s1_sgemm_pack4to1_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_1x1s1_sgemm_pack4to1_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

void conv3x3s1_winograd63_transform_input_pack4_sse(const Tensor& bottom_blob, Tensor& bottom_blob_tm);

void conv3x3s1_winograd63_transform_output_pack4_sse(const Tensor& top_blob_tm, Tensor& top_blob, const Tensor& bias);

void conv3x3s1_winograd43_transform_input_pack4_sse(const Tensor& bottom_blob, Tensor& bottom_blob_tm);

void conv3x3s1_winograd43_transform_output_pack4_sse(const Tensor& top_blob_tm, Tensor& top_blob, const Tensor& bias);

void conv3x3s1_winograd23_transform_input_pack4_sse(const Tensor& bottom_blob, Tensor& bottom_blob_tm);

void conv3x3s1_winograd23_transform_output_pack4_sse(const Tensor& top_blob_tm, Tensor& top_blob, const Tensor& bias);

void convolution_winograd_dot_pack4_sse(Tensor& bottom_blob_tm, int outch, const Tensor& kernel_tm, Tensor& top_blob_tm);

Tensor conv2d_3x3s1_winograd63_pack4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_3x3s1_winograd63_pack4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

Tensor conv2d_3x3s1_winograd43_pack4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_3x3s1_winograd43_pack4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

Tensor conv2d_3x3s1_winograd23_pack4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_3x3s1_winograd23_pack4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

#if __AVX__

void convolution_im2col_sgemm_transform_kernel_pack1to8_avx(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);

void convolution_im2col_sgemm_transform_kernel_pack4to8_avx(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);

void convolution_im2col_sgemm_transform_kernel_pack8_avx(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);

void convolution_im2col_sgemm_transform_kernel_pack8to1_avx(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);

void convolution_im2col_sgemm_transform_kernel_pack8to4_avx(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h);

Tensor& sgemm_conv2d_pack1to8_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);
    
Tensor sgemm_conv2d_pack1to8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor& sgemm_conv2d_pack4to8_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);
    
Tensor sgemm_conv2d_pack4to8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor& sgemm_conv2d_pack8_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);
    
Tensor sgemm_conv2d_pack8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor& sgemm_conv2d_pack8to1_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);
    
Tensor sgemm_conv2d_pack8to1_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor& sgemm_conv2d_pack8to4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output);
    
Tensor sgemm_conv2d_pack8to4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

Tensor conv2d_1x1s1_sgemm_pack1to8_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_1x1s1_sgemm_pack1to8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

Tensor conv2d_1x1s1_sgemm_pack4to8_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_1x1s1_sgemm_pack4to8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

Tensor conv2d_1x1s1_sgemm_pack8_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_1x1s1_sgemm_pack8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

Tensor conv2d_1x1s2_sgemm_pack8_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_1x1s2_sgemm_pack8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

Tensor conv2d_1x1s1_sgemm_pack8to1_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_1x1s1_sgemm_pack8to1_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

Tensor conv2d_1x1s1_sgemm_pack8to4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_1x1s1_sgemm_pack8to4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

void conv3x3s1_winograd63_transform_kernel_pack8_avx(const Tensor& kernel, Tensor& kernel_tm_pack8, int inch, int outch);

void conv3x3s1_winograd43_transform_kernel_pack8_avx(const Tensor& kernel, Tensor& kernel_tm_pack8, int inch, int outch);

void conv3x3s1_winograd23_transform_kernel_pack8_avx(const Tensor& kernel, Tensor& kernel_tm_pack8, int inch, int outch);

Tensor conv2d_3x3s1_winograd63_pack8_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_3x3s1_winograd63_pack8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

Tensor conv2d_3x3s1_winograd43_pack8_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_3x3s1_winograd43_pack8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

Tensor conv2d_3x3s1_winograd23_pack8_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output);

Tensor conv2d_3x3s1_winograd23_pack8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding);

#endif  // __AVX__
#endif  // __SSE2__

}   // end namespace otter

#endif /* ConvolutionMM2DX86Pack_hpp */
