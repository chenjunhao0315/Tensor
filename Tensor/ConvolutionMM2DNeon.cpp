//
//  ConvolutionMM2DNeon.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/26.
//

#include "ConvolutionMM2DNeon.hpp"
#include "Tensor.hpp"
#include "TensorFactory.hpp"
#include "im2col.hpp"
#include "Parallel.hpp"
#include "Padding.hpp"
#include "TensorTransform.hpp"

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

namespace otter {

#ifdef __ARM_NEON__
void im2col_sgemm_conv2d_impl_neon(
    const Tensor& im2col_,
    const Tensor& kernel_pack4x4_,
    const Tensor& bias_,
    int64_t input_channels,
    int64_t output_channels,
    Tensor& output) {
    
    Tensor im2col = im2col_.view({input_channels, -1, im2col_.size(2)});
    const float* bias = (bias_.defined()) ? bias_.data_ptr<float>() : nullptr;
    
    const int64_t outSize = im2col.size(2); // im2col width = output_height * output_width
    const int64_t kernelSize = im2col.size(1); // kernel_height * kernel_width
    
    Tensor tmp;
    int64_t packChannel = outSize;
    int64_t packHeight  = input_channels;
    int64_t packWidth   = kernelSize;
    if (outSize >= 8) {        // pack 8x8
        packChannel = outSize / 8 + (outSize % 8) / 4 + outSize % 4;
        packHeight  = input_channels;
        packWidth   = 8 * kernelSize;
    } else if (outSize >= 4) { // pack 4x4
        packChannel = outSize / 4 + outSize % 4;
        packHeight  = input_channels;
        packWidth   = 4 * kernelSize;
    }
    tmp = otter::empty({packChannel, packHeight, packWidth}, ScalarType::Float);
    
    auto tmp_a    = tmp.accessor<float, 3>();
    auto im2col_a = im2col.accessor<float, 3>();
    
    // Pack 8x8
    int64_t colCount = outSize >> 3;
    int64_t remainColCount = 0;
    otter::parallel_for(0, colCount, 1, [&](int64_t start, int64_t end) {
        for (auto i : otter::irange(start, end)) {
            int64_t index = remainColCount + i * 8;
            float* tmpptr = tmp_a[index / 8].data();
            
            for (const auto q : otter::irange(input_channels)) {
                const float* img0 = (const float*)im2col_a[q].data() + index;
                
                for (const auto k : otter::irange(kernelSize)) {
                    (void)k;    // eliminate warning
                    vst1q_f32(tmpptr, vld1q_f32(img0));
                    vst1q_f32(tmpptr + 4, vld1q_f32(img0 + 4));
                    img0 += outSize;
                    tmpptr += 8;
                }
            }
        }
    });
    
    // Pack 4x4
    remainColCount += colCount << 3;
    colCount = (outSize - remainColCount) >> 2;
    otter::parallel_for(0, colCount, 1, [&](int64_t start, int64_t end) {
        for (auto i : otter::irange(start, end)) {
            int64_t index = remainColCount + i * 4;
            float* tmpptr = tmp_a[index / 8 + (index % 8) / 4].data();
            
            for (const auto q : otter::irange(input_channels)) {
                const float* img0 = (const float*)im2col_a[q].data() + index;
                
                for (const auto k : otter::irange(kernelSize)) {
                    (void)k;
                    vst1q_f32(tmpptr, vld1q_f32(img0));
                    img0 += outSize;
                    tmpptr += 4;
                }
            }
        }
    });
    
    // Remain
    remainColCount += colCount << 2;
    otter::parallel_for(remainColCount, outSize, 1, [&](int64_t start, int64_t end) {
        for (auto i : otter::irange(start, end)) {
            float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();
            
            for(const auto q : otter::irange(input_channels)) {
                const float* img0 = (const float*)im2col_a[q].data() + i;
                
                for (const auto k : otter::irange(kernelSize)) {
                    (void)k;
                    tmpptr[0] = img0[0];
                    img0 += outSize;
                    tmpptr += 1;
                }
            }
        }
    });
    
    int64_t ccOutChannel = 0;
    int64_t ccRemainOutChannel = 0;
    
    auto output_a = output.accessor<float, 4>()[0];
    auto kernel_a = kernel_pack4x4_.accessor<float, 3>();
    
#if __aarch64__
    ccOutChannel = output_channels >> 3;
    ccRemainOutChannel = ccOutChannel << 3;
    
    otter::parallel_for(0, ccOutChannel, 1, [&](int64_t start, int64_t end) {
        for (auto pp : otter::irange(start, end)) {
            int64_t p = pp << 3;
            
            float* outptr0 = output_a[p + 0].data();
            float* outptr1 = output_a[p + 1].data();
            float* outptr2 = output_a[p + 2].data();
            float* outptr3 = output_a[p + 3].data();
            float* outptr4 = output_a[p + 4].data();
            float* outptr5 = output_a[p + 5].data();
            float* outptr6 = output_a[p + 6].data();
            float* outptr7 = output_a[p + 7].data();
            
            const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + p : zeros;
            
            int i = 0;
            for ( ; i + 7 < outSize; i += 8) {
                const float* tmpptr = tmp_a[i / 8].data();
                const float* kptr = kernel_a[p / 8].data();
                
                int64_t nn = input_channels * kernelSize;
                
                float32x4_t _b0 = vld1q_f32(biasptr);
                float32x4_t _b1 = vld1q_f32(biasptr + 4);
                
                float32x4_t _sum0 = vdupq_n_f32(_b0[0]);
                float32x4_t _sum1 = vdupq_n_f32(_b0[0]);
                float32x4_t _sum2 = vdupq_n_f32(_b0[1]);
                float32x4_t _sum3 = vdupq_n_f32(_b0[1]);
                float32x4_t _sum4 = vdupq_n_f32(_b0[2]);
                float32x4_t _sum5 = vdupq_n_f32(_b0[2]);
                float32x4_t _sum6 = vdupq_n_f32(_b0[3]);
                float32x4_t _sum7 = vdupq_n_f32(_b0[3]);
                
                float32x4_t _sum8 = vdupq_n_f32(_b1[0]);
                float32x4_t _sum9 = vdupq_n_f32(_b1[0]);
                float32x4_t _sum10 = vdupq_n_f32(_b1[1]);
                float32x4_t _sum11 = vdupq_n_f32(_b1[1]);
                float32x4_t _sum12 = vdupq_n_f32(_b1[2]);
                float32x4_t _sum13 = vdupq_n_f32(_b1[2]);
                float32x4_t _sum14 = vdupq_n_f32(_b1[3]);
                float32x4_t _sum15 = vdupq_n_f32(_b1[3]);
                
                for (int64_t w = nn >> 2; w != 0; --w) {
                    float32x4_t _val0 = vld1q_f32(tmpptr);
                    float32x4_t _val1 = vld1q_f32(tmpptr + 4);
                    float32x4_t _val2 = vld1q_f32(tmpptr + 8);
                    float32x4_t _val3 = vld1q_f32(tmpptr + 12);
                    tmpptr += 16;
                    
                    float32x4_t _w0 = vld1q_f32(kptr);
                    float32x4_t _w1 = vld1q_f32(kptr + 4);
                    float32x4_t _w2 = vld1q_f32(kptr + 8);
                    float32x4_t _w3 = vld1q_f32(kptr + 12);
                    kptr += 16;
                    
                    _sum0 = vfmaq_f32(_sum0, _val0, vdupq_n_f32(_w0[0]));
                    _sum2 = vfmaq_f32(_sum2, _val0, vdupq_n_f32(_w0[1]));
                    _sum4 = vfmaq_f32(_sum4, _val0, vdupq_n_f32(_w0[2]));
                    _sum6 = vfmaq_f32(_sum6, _val0, vdupq_n_f32(_w0[3]));
                    _sum1 = vfmaq_f32(_sum1, _val1, vdupq_n_f32(_w0[0]));
                    _sum3 = vfmaq_f32(_sum3, _val1, vdupq_n_f32(_w0[1]));
                    _sum5 = vfmaq_f32(_sum5, _val1, vdupq_n_f32(_w0[2]));
                    _sum7 = vfmaq_f32(_sum7, _val1, vdupq_n_f32(_w0[3]));
                    
                    _sum8  = vfmaq_f32(_sum8,  _val0, vdupq_n_f32(_w1[0]));
                    _sum10 = vfmaq_f32(_sum10, _val0, vdupq_n_f32(_w1[1]));
                    _sum12 = vfmaq_f32(_sum12, _val0, vdupq_n_f32(_w1[2]));
                    _sum14 = vfmaq_f32(_sum14, _val0, vdupq_n_f32(_w1[3]));
                    _sum9  = vfmaq_f32(_sum9,  _val1, vdupq_n_f32(_w1[0]));
                    _sum11 = vfmaq_f32(_sum11, _val1, vdupq_n_f32(_w1[1]));
                    _sum13 = vfmaq_f32(_sum13, _val1, vdupq_n_f32(_w1[2]));
                    _sum15 = vfmaq_f32(_sum15, _val1, vdupq_n_f32(_w1[3]));
                    
                    float32x4_t _val4 = vld1q_f32(tmpptr);
                    float32x4_t _val5 = vld1q_f32(tmpptr + 4);
                    float32x4_t _val6 = vld1q_f32(tmpptr + 8);
                    float32x4_t _val7 = vld1q_f32(tmpptr + 12);
                    tmpptr += 16;
                    
                    _sum0 = vfmaq_f32(_sum0, _val2, vdupq_n_f32(_w2[0]));
                    _sum2 = vfmaq_f32(_sum2, _val2, vdupq_n_f32(_w2[1]));
                    _sum4 = vfmaq_f32(_sum4, _val2, vdupq_n_f32(_w2[2]));
                    _sum6 = vfmaq_f32(_sum6, _val2, vdupq_n_f32(_w2[3]));
                    _sum1 = vfmaq_f32(_sum1, _val3, vdupq_n_f32(_w2[0]));
                    _sum3 = vfmaq_f32(_sum3, _val3, vdupq_n_f32(_w2[1]));
                    _sum5 = vfmaq_f32(_sum5, _val3, vdupq_n_f32(_w2[2]));
                    _sum7 = vfmaq_f32(_sum7, _val3, vdupq_n_f32(_w2[3]));
                    
                    _sum8  = vfmaq_f32(_sum8,  _val2, vdupq_n_f32(_w3[0]));
                    _sum10 = vfmaq_f32(_sum10, _val2, vdupq_n_f32(_w3[1]));
                    _sum12 = vfmaq_f32(_sum12, _val2, vdupq_n_f32(_w3[2]));
                    _sum14 = vfmaq_f32(_sum14, _val2, vdupq_n_f32(_w3[3]));
                    _sum9  = vfmaq_f32(_sum9,  _val3, vdupq_n_f32(_w3[0]));
                    _sum11 = vfmaq_f32(_sum11, _val3, vdupq_n_f32(_w3[1]));
                    _sum13 = vfmaq_f32(_sum13, _val3, vdupq_n_f32(_w3[2]));
                    _sum15 = vfmaq_f32(_sum15, _val3, vdupq_n_f32(_w3[3]));
                    
                    float32x4_t _w4 = vld1q_f32(kptr);
                    float32x4_t _w5 = vld1q_f32(kptr + 4);
                    float32x4_t _w6 = vld1q_f32(kptr + 8);
                    float32x4_t _w7 = vld1q_f32(kptr + 12);
                    kptr += 16;
                    
                    _sum0 = vfmaq_f32(_sum0, _val4, vdupq_n_f32(_w4[0]));
                    _sum2 = vfmaq_f32(_sum2, _val4, vdupq_n_f32(_w4[1]));
                    _sum4 = vfmaq_f32(_sum4, _val4, vdupq_n_f32(_w4[2]));
                    _sum6 = vfmaq_f32(_sum6, _val4, vdupq_n_f32(_w4[3]));
                    _sum1 = vfmaq_f32(_sum1, _val5, vdupq_n_f32(_w4[0]));
                    _sum3 = vfmaq_f32(_sum3, _val5, vdupq_n_f32(_w4[1]));
                    _sum5 = vfmaq_f32(_sum5, _val5, vdupq_n_f32(_w4[2]));
                    _sum7 = vfmaq_f32(_sum7, _val5, vdupq_n_f32(_w4[3]));
                    
                    _sum8  = vfmaq_f32(_sum8,  _val4, vdupq_n_f32(_w5[0]));
                    _sum10 = vfmaq_f32(_sum10, _val4, vdupq_n_f32(_w5[1]));
                    _sum12 = vfmaq_f32(_sum12, _val4, vdupq_n_f32(_w5[2]));
                    _sum14 = vfmaq_f32(_sum14, _val4, vdupq_n_f32(_w5[3]));
                    _sum9  = vfmaq_f32(_sum9,  _val5, vdupq_n_f32(_w5[0]));
                    _sum11 = vfmaq_f32(_sum11, _val5, vdupq_n_f32(_w5[1]));
                    _sum13 = vfmaq_f32(_sum13, _val5, vdupq_n_f32(_w5[2]));
                    _sum15 = vfmaq_f32(_sum15, _val5, vdupq_n_f32(_w5[3]));
                    
                    _sum0 = vfmaq_f32(_sum0, _val6, vdupq_n_f32(_w6[0]));
                    _sum2 = vfmaq_f32(_sum2, _val6, vdupq_n_f32(_w6[1]));
                    _sum4 = vfmaq_f32(_sum4, _val6, vdupq_n_f32(_w6[2]));
                    _sum6 = vfmaq_f32(_sum6, _val6, vdupq_n_f32(_w6[3]));
                    _sum1 = vfmaq_f32(_sum1, _val7, vdupq_n_f32(_w6[0]));
                    _sum3 = vfmaq_f32(_sum3, _val7, vdupq_n_f32(_w6[1]));
                    _sum5 = vfmaq_f32(_sum5, _val7, vdupq_n_f32(_w6[2]));
                    _sum7 = vfmaq_f32(_sum7, _val7, vdupq_n_f32(_w6[3]));
                    
                    _sum8  = vfmaq_f32(_sum8,  _val6, vdupq_n_f32(_w7[0]));
                    _sum10 = vfmaq_f32(_sum10, _val6, vdupq_n_f32(_w7[1]));
                    _sum12 = vfmaq_f32(_sum12, _val6, vdupq_n_f32(_w7[2]));
                    _sum14 = vfmaq_f32(_sum14, _val6, vdupq_n_f32(_w7[3]));
                    _sum9  = vfmaq_f32(_sum9,  _val7, vdupq_n_f32(_w7[0]));
                    _sum11 = vfmaq_f32(_sum11, _val7, vdupq_n_f32(_w7[1]));
                    _sum13 = vfmaq_f32(_sum13, _val7, vdupq_n_f32(_w7[2]));
                    _sum15 = vfmaq_f32(_sum15, _val7, vdupq_n_f32(_w7[3]));
                }
                
                for (int64_t w = nn & 3; w != 0; --w) {
                    float32x4_t _val0 = vld1q_f32(tmpptr);
                    float32x4_t _val1 = vld1q_f32(tmpptr + 4);
                    tmpptr += 8;
                    
                    float32x4_t _w0 = vld1q_f32(kptr);
                    float32x4_t _w1 = vld1q_f32(kptr + 4);
                    kptr += 8;
                    
                    _sum0 = vfmaq_f32(_sum0, _val0, vdupq_n_f32(_w0[0]));
                    _sum2 = vfmaq_f32(_sum2, _val0, vdupq_n_f32(_w0[1]));
                    _sum4 = vfmaq_f32(_sum4, _val0, vdupq_n_f32(_w0[2]));
                    _sum6 = vfmaq_f32(_sum6, _val0, vdupq_n_f32(_w0[3]));
                    _sum1 = vfmaq_f32(_sum1, _val1, vdupq_n_f32(_w0[0]));
                    _sum3 = vfmaq_f32(_sum3, _val1, vdupq_n_f32(_w0[1]));
                    _sum5 = vfmaq_f32(_sum5, _val1, vdupq_n_f32(_w0[2]));
                    _sum7 = vfmaq_f32(_sum7, _val1, vdupq_n_f32(_w0[3]));
                    
                    _sum8  = vfmaq_f32(_sum8,  _val0, vdupq_n_f32(_w1[0]));
                    _sum10 = vfmaq_f32(_sum10, _val0, vdupq_n_f32(_w1[1]));
                    _sum12 = vfmaq_f32(_sum12, _val0, vdupq_n_f32(_w1[2]));
                    _sum14 = vfmaq_f32(_sum14, _val0, vdupq_n_f32(_w1[3]));
                    _sum9  = vfmaq_f32(_sum9,  _val1, vdupq_n_f32(_w1[0]));
                    _sum11 = vfmaq_f32(_sum11, _val1, vdupq_n_f32(_w1[1]));
                    _sum13 = vfmaq_f32(_sum13, _val1, vdupq_n_f32(_w1[2]));
                    _sum15 = vfmaq_f32(_sum15, _val1, vdupq_n_f32(_w1[3]));
                }
                
                vst1q_f32(outptr0, _sum0);
                vst1q_f32(outptr0 + 4, _sum1);
                outptr0 += 8;
                vst1q_f32(outptr1, _sum2);
                vst1q_f32(outptr1 + 4, _sum3);
                outptr1 += 8;
                vst1q_f32(outptr2, _sum4);
                vst1q_f32(outptr2 + 4, _sum5);
                outptr2 += 8;
                vst1q_f32(outptr3, _sum6);
                vst1q_f32(outptr3 + 4, _sum7);
                outptr3 += 8;
                vst1q_f32(outptr4, _sum8);
                vst1q_f32(outptr4 + 4, _sum9);
                outptr4 += 8;
                vst1q_f32(outptr5, _sum10);
                vst1q_f32(outptr5 + 4, _sum11);
                outptr5 += 8;
                vst1q_f32(outptr6, _sum12);
                vst1q_f32(outptr6 + 4, _sum13);
                outptr6 += 8;
                vst1q_f32(outptr7, _sum14);
                vst1q_f32(outptr7 + 4, _sum15);
                outptr7 += 8;
            }
            
            for ( ; i + 3 < outSize; i += 4) {
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();
                const float* kptr = kernel_a[p / 8].data();
                
                int64_t nn = input_channels * kernelSize;
                
                float32x4_t _b0 = vld1q_f32(biasptr);
                float32x4_t _b1 = vld1q_f32(biasptr + 4);
                
                float32x4_t _sum0 = vdupq_n_f32(_b0[0]);
                float32x4_t _sum1 = vdupq_n_f32(_b0[1]);
                float32x4_t _sum2 = vdupq_n_f32(_b0[2]);
                float32x4_t _sum3 = vdupq_n_f32(_b0[3]);
                float32x4_t _sum4 = vdupq_n_f32(_b1[0]);
                float32x4_t _sum5 = vdupq_n_f32(_b1[1]);
                float32x4_t _sum6 = vdupq_n_f32(_b1[2]);
                float32x4_t _sum7 = vdupq_n_f32(_b1[3]);
                
                for (int64_t w = nn >> 2; w != 0; --w) {
                    float32x4_t _val0 = vld1q_f32(tmpptr);
                    float32x4_t _val1 = vld1q_f32(tmpptr + 4);
                    float32x4_t _val2 = vld1q_f32(tmpptr + 8);
                    float32x4_t _val3 = vld1q_f32(tmpptr + 12);
                    tmpptr += 16;
                    
                    float32x4_t _w0 = vld1q_f32(kptr);
                    float32x4_t _w1 = vld1q_f32(kptr + 4);
                    float32x4_t _w2 = vld1q_f32(kptr + 8);
                    float32x4_t _w3 = vld1q_f32(kptr + 12);
                    kptr += 16;
                    
                    _sum0 = vfmaq_f32(_sum0, _val0, vdupq_n_f32(_w0[0]));
                    _sum1 = vfmaq_f32(_sum1, _val0, vdupq_n_f32(_w0[1]));
                    _sum2 = vfmaq_f32(_sum2, _val0, vdupq_n_f32(_w0[2]));
                    _sum3 = vfmaq_f32(_sum3, _val0, vdupq_n_f32(_w0[3]));
                    _sum4 = vfmaq_f32(_sum4, _val0, vdupq_n_f32(_w1[0]));
                    _sum5 = vfmaq_f32(_sum5, _val0, vdupq_n_f32(_w1[1]));
                    _sum6 = vfmaq_f32(_sum6, _val0, vdupq_n_f32(_w1[2]));
                    _sum7 = vfmaq_f32(_sum7, _val0, vdupq_n_f32(_w1[3]));
                    
                    float32x4_t _w4 = vld1q_f32(kptr);
                    float32x4_t _w5 = vld1q_f32(kptr + 4);
                    float32x4_t _w6 = vld1q_f32(kptr + 8);
                    float32x4_t _w7 = vld1q_f32(kptr + 12);
                    kptr += 16;
                    
                    _sum0 = vfmaq_f32(_sum0, _val1, vdupq_n_f32(_w2[0]));
                    _sum1 = vfmaq_f32(_sum1, _val1, vdupq_n_f32(_w2[1]));
                    _sum2 = vfmaq_f32(_sum2, _val1, vdupq_n_f32(_w2[2]));
                    _sum3 = vfmaq_f32(_sum3, _val1, vdupq_n_f32(_w2[3]));
                    _sum4 = vfmaq_f32(_sum4, _val1, vdupq_n_f32(_w3[0]));
                    _sum5 = vfmaq_f32(_sum5, _val1, vdupq_n_f32(_w3[1]));
                    _sum6 = vfmaq_f32(_sum6, _val1, vdupq_n_f32(_w3[2]));
                    _sum7 = vfmaq_f32(_sum7, _val1, vdupq_n_f32(_w3[3]));
                    
                    _sum0 = vfmaq_f32(_sum0, _val2, vdupq_n_f32(_w4[0]));
                    _sum1 = vfmaq_f32(_sum1, _val2, vdupq_n_f32(_w4[1]));
                    _sum2 = vfmaq_f32(_sum2, _val2, vdupq_n_f32(_w4[2]));
                    _sum3 = vfmaq_f32(_sum3, _val2, vdupq_n_f32(_w4[3]));
                    _sum4 = vfmaq_f32(_sum4, _val2, vdupq_n_f32(_w5[0]));
                    _sum5 = vfmaq_f32(_sum5, _val2, vdupq_n_f32(_w5[1]));
                    _sum6 = vfmaq_f32(_sum6, _val2, vdupq_n_f32(_w5[2]));
                    _sum7 = vfmaq_f32(_sum7, _val2, vdupq_n_f32(_w5[3]));
                    
                    _sum0 = vfmaq_f32(_sum0, _val3, vdupq_n_f32(_w6[0]));
                    _sum1 = vfmaq_f32(_sum1, _val3, vdupq_n_f32(_w6[1]));
                    _sum2 = vfmaq_f32(_sum2, _val3, vdupq_n_f32(_w6[2]));
                    _sum3 = vfmaq_f32(_sum3, _val3, vdupq_n_f32(_w6[3]));
                    _sum4 = vfmaq_f32(_sum4, _val3, vdupq_n_f32(_w7[0]));
                    _sum5 = vfmaq_f32(_sum5, _val3, vdupq_n_f32(_w7[1]));
                    _sum6 = vfmaq_f32(_sum6, _val3, vdupq_n_f32(_w7[2]));
                    _sum7 = vfmaq_f32(_sum7, _val3, vdupq_n_f32(_w7[3]));
                }
                
                for (int w = nn & 3; w != 0; --w) {
                    float32x4_t _val0 = vld1q_f32(tmpptr);
                    tmpptr += 4;
                    
                    float32x4_t _w0 = vld1q_f32(kptr);
                    float32x4_t _w1 = vld1q_f32(kptr + 4);
                    kptr += 8;
                    
                    _sum0 = vfmaq_f32(_sum0, _val0, vdupq_n_f32(_w0[0]));
                    _sum1 = vfmaq_f32(_sum1, _val0, vdupq_n_f32(_w0[1]));
                    _sum2 = vfmaq_f32(_sum2, _val0, vdupq_n_f32(_w0[2]));
                    _sum3 = vfmaq_f32(_sum3, _val0, vdupq_n_f32(_w0[3]));
                    _sum4 = vfmaq_f32(_sum4, _val0, vdupq_n_f32(_w1[0]));
                    _sum5 = vfmaq_f32(_sum5, _val0, vdupq_n_f32(_w1[1]));
                    _sum6 = vfmaq_f32(_sum6, _val0, vdupq_n_f32(_w1[2]));
                    _sum7 = vfmaq_f32(_sum7, _val0, vdupq_n_f32(_w1[3]));
                }
                
                vst1q_f32(outptr0, _sum0);
                outptr0 += 4;
                vst1q_f32(outptr1, _sum1);
                outptr1 += 4;
                vst1q_f32(outptr2, _sum2);
                outptr2 += 4;
                vst1q_f32(outptr3, _sum3);
                outptr3 += 4;
                vst1q_f32(outptr4, _sum4);
                outptr4 += 4;
                vst1q_f32(outptr5, _sum5);
                outptr5 += 4;
                vst1q_f32(outptr6, _sum6);
                outptr6 += 4;
                vst1q_f32(outptr7, _sum7);
                outptr7 += 4;
            }
            
            for ( ; i < outSize; ++i) {
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();
                const float* kptr = kernel_a[p / 8].data();
                
                int64_t nn = input_channels * kernelSize;
                
                float32x4_t _sum0 = vld1q_f32(biasptr);
                float32x4_t _sum1 = vld1q_f32(biasptr + 4);
                
                float32x4_t _t0 = vdupq_n_f32(0);
                float32x4_t _t1 = vdupq_n_f32(0);
                float32x4_t _t2 = vdupq_n_f32(0);
                float32x4_t _t3 = vdupq_n_f32(0);
                float32x4_t _t4 = vdupq_n_f32(0);
                float32x4_t _t5 = vdupq_n_f32(0);
                float32x4_t _t6 = vdupq_n_f32(0);
                float32x4_t _t7 = vdupq_n_f32(0);
                
                for (int64_t w = nn >> 2; w != 0; --w) {
                    float32x4_t _val0 = vld1q_f32(tmpptr);
                    tmpptr += 4;
                    
                    float32x4_t _w0 = vld1q_f32(kptr);
                    float32x4_t _w1 = vld1q_f32(kptr + 4);
                    float32x4_t _w2 = vld1q_f32(kptr + 8);
                    float32x4_t _w3 = vld1q_f32(kptr + 12);
                    kptr += 16;
                    
                    _t0 = vfmaq_f32(_t0, _w0, vdupq_n_f32(_val0[0]));
                    _t1 = vfmaq_f32(_t1, _w1, vdupq_n_f32(_val0[0]));
                    _t2 = vfmaq_f32(_t2, _w2, vdupq_n_f32(_val0[1]));
                    _t3 = vfmaq_f32(_t3, _w3, vdupq_n_f32(_val0[1]));
                    
                    float32x4_t _w4 = vld1q_f32(kptr);
                    float32x4_t _w5 = vld1q_f32(kptr + 4);
                    float32x4_t _w6 = vld1q_f32(kptr + 8);
                    float32x4_t _w7 = vld1q_f32(kptr + 12);
                    kptr += 16;
                    
                    _t4 = vfmaq_f32(_t4, _w4, vdupq_n_f32(_val0[2]));
                    _t5 = vfmaq_f32(_t5, _w5, vdupq_n_f32(_val0[2]));
                    _t6 = vfmaq_f32(_t6, _w6, vdupq_n_f32(_val0[3]));
                    _t7 = vfmaq_f32(_t7, _w7, vdupq_n_f32(_val0[3]));
                }
                
                _t0 = vaddq_f32(_t0, _t2);
                _t1 = vaddq_f32(_t1, _t3);
                _t4 = vaddq_f32(_t4, _t6);
                _t5 = vaddq_f32(_t5, _t7);
                _t0 = vaddq_f32(_t0, _t4);
                _t1 = vaddq_f32(_t1, _t5);
                _sum0 = vaddq_f32(_sum0, _t0);
                _sum1 = vaddq_f32(_sum1, _t1);
                
                for (int64_t w = nn & 3; w != 0; --w) {
                    float32x4_t _val0 = vld1q_dup_f32(tmpptr);
                    tmpptr += 1;
                    
                    float32x4_t _w0 = vld1q_f32(kptr);
                    float32x4_t _w1 = vld1q_f32(kptr + 4);
                    kptr += 8;
                    
                    _sum0 = vfmaq_f32(_sum0, _val0, _w0);
                    _sum1 = vfmaq_f32(_sum1, _val0, _w1);
                }
                
                outptr0[0] = _sum0[0];
                outptr1[0] = _sum0[1];
                outptr2[0] = _sum0[2];
                outptr3[0] = _sum0[3];
                outptr4[0] = _sum1[0];
                outptr5[0] = _sum1[1];
                outptr6[0] = _sum1[2];
                outptr7[0] = _sum1[3];
                
                ++outptr0;
                ++outptr1;
                ++outptr2;
                ++outptr3;
                ++outptr4;
                ++outptr5;
                ++outptr6;
                ++outptr7;
            }
        }
    });
#endif
    
    ccOutChannel = (output_channels - ccRemainOutChannel) >> 2;
    otter::parallel_for(0, ccOutChannel, 1, [&](int64_t start, int64_t end) {
        for (auto pp : otter::irange(start, end)) {
            int64_t p = ccRemainOutChannel + pp * 4;
            
            float* outptr0 = output_a[p + 0].data();
            float* outptr1 = output_a[p + 1].data();
            float* outptr2 = output_a[p + 2].data();
            float* outptr3 = output_a[p + 3].data();

            const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + p : zeros;

            int i = 0;
            for (; i + 7 < outSize; i += 8) {
                const float* tmpptr = tmp_a[i / 8].data();
    #if __aarch64__
                const float* kptr = kernel_a[p / 8 + (p % 8) / 4].data();
    #else
                const float* kptr = kernel_a[p / 4].data();
    #endif
                int64_t nn = input_channels * kernelSize;
                
                
    #if __aarch64__
                
                float32x4_t _b0 = vld1q_f32(biasptr);
                float32x4_t _sum0 = vdupq_n_f32(_b0[0]);
                float32x4_t _sum1 = vdupq_n_f32(_b0[0]);
                float32x4_t _sum2 = vdupq_n_f32(_b0[1]);
                float32x4_t _sum3 = vdupq_n_f32(_b0[1]);
                float32x4_t _sum4 = vdupq_n_f32(_b0[2]);
                float32x4_t _sum5 = vdupq_n_f32(_b0[2]);
                float32x4_t _sum6 = vdupq_n_f32(_b0[3]);
                float32x4_t _sum7 = vdupq_n_f32(_b0[3]);

                for (int64_t w = nn >> 2; w != 0; --w) {
                    float32x4_t _val0 = vld1q_f32(tmpptr + 0);
                    float32x4_t _val1 = vld1q_f32(tmpptr + 4);
                    float32x4_t _val2 = vld1q_f32(tmpptr + 8);
                    float32x4_t _val3 = vld1q_f32(tmpptr + 12);
                    tmpptr += 16;

                    float32x4_t _w0 = vld1q_f32(kptr + 0);
                    float32x4_t _w1 = vld1q_f32(kptr + 4);
                    float32x4_t _w2 = vld1q_f32(kptr + 8);
                    float32x4_t _w3 = vld1q_f32(kptr + 12);
                    kptr += 16;

                    _sum0 = vfmaq_f32(_sum0, _val0, vdupq_n_f32(_w0[0]));
                    _sum2 = vfmaq_f32(_sum2, _val0, vdupq_n_f32(_w0[1]));
                    _sum4 = vfmaq_f32(_sum4, _val0, vdupq_n_f32(_w0[2]));
                    _sum6 = vfmaq_f32(_sum6, _val0, vdupq_n_f32(_w0[3]));

                    _sum1 = vfmaq_f32(_sum1, _val1, vdupq_n_f32(_w0[0]));
                    _sum3 = vfmaq_f32(_sum3, _val1, vdupq_n_f32(_w0[1]));
                    _sum5 = vfmaq_f32(_sum5, _val1, vdupq_n_f32(_w0[2]));
                    _sum7 = vfmaq_f32(_sum7, _val1, vdupq_n_f32(_w0[3]));

                    float32x4_t _val4 = vld1q_f32(tmpptr + 0);
                    float32x4_t _val5 = vld1q_f32(tmpptr + 4);
                    float32x4_t _val6 = vld1q_f32(tmpptr + 8);
                    float32x4_t _val7 = vld1q_f32(tmpptr + 12);
                    tmpptr += 16;

                    _sum0 = vfmaq_f32(_sum0, _val2, vdupq_n_f32(_w1[0]));
                    _sum2 = vfmaq_f32(_sum2, _val2, vdupq_n_f32(_w1[1]));
                    _sum4 = vfmaq_f32(_sum4, _val2, vdupq_n_f32(_w1[2]));
                    _sum6 = vfmaq_f32(_sum6, _val2, vdupq_n_f32(_w1[3]));

                    _sum1 = vfmaq_f32(_sum1, _val3, vdupq_n_f32(_w1[0]));
                    _sum3 = vfmaq_f32(_sum3, _val3, vdupq_n_f32(_w1[1]));
                    _sum5 = vfmaq_f32(_sum5, _val3, vdupq_n_f32(_w1[2]));
                    _sum7 = vfmaq_f32(_sum7, _val3, vdupq_n_f32(_w1[3]));

                    _sum0 = vfmaq_f32(_sum0, _val4, vdupq_n_f32(_w2[0]));
                    _sum2 = vfmaq_f32(_sum2, _val4, vdupq_n_f32(_w2[1]));
                    _sum4 = vfmaq_f32(_sum4, _val4, vdupq_n_f32(_w2[2]));
                    _sum6 = vfmaq_f32(_sum6, _val4, vdupq_n_f32(_w2[3]));

                    _sum1 = vfmaq_f32(_sum1, _val5, vdupq_n_f32(_w2[0]));
                    _sum3 = vfmaq_f32(_sum3, _val5, vdupq_n_f32(_w2[1]));
                    _sum5 = vfmaq_f32(_sum5, _val5, vdupq_n_f32(_w2[2]));
                    _sum7 = vfmaq_f32(_sum7, _val5, vdupq_n_f32(_w2[3]));

                    _sum0 = vfmaq_f32(_sum0, _val6, vdupq_n_f32(_w3[0]));
                    _sum2 = vfmaq_f32(_sum2, _val6, vdupq_n_f32(_w3[1]));
                    _sum4 = vfmaq_f32(_sum4, _val6, vdupq_n_f32(_w3[2]));
                    _sum6 = vfmaq_f32(_sum6, _val6, vdupq_n_f32(_w3[3]));

                    _sum1 = vfmaq_f32(_sum1, _val7, vdupq_n_f32(_w3[0]));
                    _sum3 = vfmaq_f32(_sum3, _val7, vdupq_n_f32(_w3[1]));
                    _sum5 = vfmaq_f32(_sum5, _val7, vdupq_n_f32(_w3[2]));
                    _sum7 = vfmaq_f32(_sum7, _val7, vdupq_n_f32(_w3[3]));
                }

                for (int64_t w = nn & 3; w != 0; --w) {
                    float32x4_t _val0 = vld1q_f32(tmpptr);
                    float32x4_t _val1 = vld1q_f32(tmpptr + 4);
                    tmpptr += 8;

                    float32x4_t _w0 = vld1q_f32(kptr);
                    kptr += 4;

                    _sum0 = vfmaq_f32(_sum0, _val0, vdupq_n_f32(_w0[0]));
                    _sum2 = vfmaq_f32(_sum2, _val0, vdupq_n_f32(_w0[1]));
                    _sum4 = vfmaq_f32(_sum4, _val0, vdupq_n_f32(_w0[2]));
                    _sum6 = vfmaq_f32(_sum6, _val0, vdupq_n_f32(_w0[3]));

                    _sum1 = vfmaq_f32(_sum1, _val1, vdupq_n_f32(_w0[0]));
                    _sum3 = vfmaq_f32(_sum3, _val1, vdupq_n_f32(_w0[1]));
                    _sum5 = vfmaq_f32(_sum5, _val1, vdupq_n_f32(_w0[2]));
                    _sum7 = vfmaq_f32(_sum7, _val1, vdupq_n_f32(_w0[3]));
                }

                vst1q_f32(outptr0, _sum0);
                vst1q_f32(outptr0 + 4, _sum1);
                outptr0 += 8;
                vst1q_f32(outptr1, _sum2);
                vst1q_f32(outptr1 + 4, _sum3);
                outptr1 += 8;
                vst1q_f32(outptr2, _sum4);
                vst1q_f32(outptr2 + 4, _sum5);
                outptr2 += 8;
                vst1q_f32(outptr3, _sum6);
                vst1q_f32(outptr3 + 4, _sum7);
                outptr3 += 8;
    #else
                
    #endif
            }
            
            for (; i + 3 < outSize; i += 4) {
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();
    #if __aarch64__
                const float* kptr = kernel_a[p / 8 + (p % 8) / 4].data();
    #else
                const float* kptr = kernel_a[p / 4].data();
    #endif
                int64_t nn = input_channels * kernelSize;
                
    #if __aarch64__
                float32x4_t _b0 = vld1q_f32(biasptr);
                float32x4_t _sum0 = vdupq_n_f32(_b0[0]);
                float32x4_t _sum1 = vdupq_n_f32(_b0[1]);
                float32x4_t _sum2 = vdupq_n_f32(_b0[2]);
                float32x4_t _sum3 = vdupq_n_f32(_b0[3]);

                for (int64_t w = nn >> 2; w != 0; --w) {
                    float32x4_t _val0 = vld1q_f32(tmpptr + 0);
                    float32x4_t _val1 = vld1q_f32(tmpptr + 4);
                    float32x4_t _val2 = vld1q_f32(tmpptr + 8);
                    float32x4_t _val3 = vld1q_f32(tmpptr + 12);
                    tmpptr += 16;

                    float32x4_t _w0 = vld1q_f32(kptr + 0);
                    float32x4_t _w1 = vld1q_f32(kptr + 4);
                    float32x4_t _w2 = vld1q_f32(kptr + 8);
                    float32x4_t _w3 = vld1q_f32(kptr + 12);
                    kptr += 16;

                    _sum0 = vfmaq_f32(_sum0, _val0, vdupq_n_f32(_w0[0]));
                    _sum1 = vfmaq_f32(_sum1, _val0, vdupq_n_f32(_w0[1]));
                    _sum2 = vfmaq_f32(_sum2, _val0, vdupq_n_f32(_w0[2]));
                    _sum3 = vfmaq_f32(_sum3, _val0, vdupq_n_f32(_w0[3]));

                    _sum0 = vfmaq_f32(_sum0, _val1, vdupq_n_f32(_w1[0]));
                    _sum1 = vfmaq_f32(_sum1, _val1, vdupq_n_f32(_w1[1]));
                    _sum2 = vfmaq_f32(_sum2, _val1, vdupq_n_f32(_w1[2]));
                    _sum3 = vfmaq_f32(_sum3, _val1, vdupq_n_f32(_w1[3]));

                    _sum0 = vfmaq_f32(_sum0, _val2, vdupq_n_f32(_w2[0]));
                    _sum1 = vfmaq_f32(_sum1, _val2, vdupq_n_f32(_w2[1]));
                    _sum2 = vfmaq_f32(_sum2, _val2, vdupq_n_f32(_w2[2]));
                    _sum3 = vfmaq_f32(_sum3, _val2, vdupq_n_f32(_w2[3]));

                    _sum0 = vfmaq_f32(_sum0, _val3, vdupq_n_f32(_w3[0]));
                    _sum1 = vfmaq_f32(_sum1, _val3, vdupq_n_f32(_w3[1]));
                    _sum2 = vfmaq_f32(_sum2, _val3, vdupq_n_f32(_w3[2]));
                    _sum3 = vfmaq_f32(_sum3, _val3, vdupq_n_f32(_w3[3]));
                }

                for (int64_t w = nn & 3; w != 0; --w) {
                    float32x4_t _val0 = vld1q_f32(tmpptr);
                    tmpptr += 4;

                    float32x4_t _w0 = vld1q_f32(kptr);
                    kptr += 4;

                    _sum0 = vfmaq_f32(_sum0, _val0, vdupq_n_f32(_w0[0]));
                    _sum1 = vfmaq_f32(_sum1, _val0, vdupq_n_f32(_w0[1]));
                    _sum2 = vfmaq_f32(_sum2, _val0, vdupq_n_f32(_w0[2]));
                    _sum3 = vfmaq_f32(_sum3, _val0, vdupq_n_f32(_w0[3]));
                }

                vst1q_f32(outptr0, _sum0);
                outptr0 += 4;
                vst1q_f32(outptr1, _sum1);
                outptr1 += 4;
                vst1q_f32(outptr2, _sum2);
                outptr2 += 4;
                vst1q_f32(outptr3, _sum3);
                outptr3 += 4;
    #else
                
    #endif
            }
            for ( ; i < outSize; ++i) {
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();
    #if __aarch64__
                const float* kptr = kernel_a[p / 8 + (p % 8) / 4].data();
    #else
                const float* kptr = kernel_a[p / 4].data();
    #endif
                int64_t nn = input_channels * kernelSize;
                
    #if __aarch64__
                float32x4_t _sum0 = vld1q_f32(biasptr);

                float32x4_t _t0 = vdupq_n_f32(0);
                float32x4_t _t1 = vdupq_n_f32(0);
                float32x4_t _t2 = vdupq_n_f32(0);
                float32x4_t _t3 = vdupq_n_f32(0);

                for (int64_t w = nn >> 2; w != 0; --w) {

                    float32x4_t _val0 = vld1q_f32(tmpptr);
                    tmpptr += 4;

                    float32x4_t _w0 = vld1q_f32(kptr);
                    float32x4_t _w1 = vld1q_f32(kptr + 4);
                    float32x4_t _w2 = vld1q_f32(kptr + 8);
                    float32x4_t _w3 = vld1q_f32(kptr + 12);
                    kptr += 16;

                    _t0 = vfmaq_f32(_t0, _w0, vdupq_n_f32(_val0[0]));
                    _t1 = vfmaq_f32(_t1, _w1, vdupq_n_f32(_val0[0]));
                    _t2 = vfmaq_f32(_t2, _w2, vdupq_n_f32(_val0[0]));
                    _t3 = vfmaq_f32(_t3, _w3, vdupq_n_f32(_val0[0]));
                }

                _t0 = vaddq_f32(_t0, _t1);
                _t2 = vaddq_f32(_t2, _t3);
                _t0 = vaddq_f32(_t0, _t2);
                _sum0 = vaddq_f32(_sum0, _t0);

                for (int64_t w = nn & 3; w != 0; --w) {
                    float32x4_t _val0 = vld1q_dup_f32(tmpptr);
                    tmpptr += 1;

                    float32x4_t _w0 = vld1q_f32(kptr);
                    kptr += 4;

                    _sum0 = vfmaq_f32(_sum0, _val0, _w0);
                }

                outptr0[0] = _sum0[0];
                ++outptr0;

                outptr1[0] = _sum0[1];
                ++outptr1;

                outptr2[0] = _sum0[2];
                ++outptr2;

                outptr3[0] = _sum0[3];
                ++outptr3;
    #else
                
    #endif
            }
        }
    });
    
    ccRemainOutChannel += ccOutChannel << 2;
    
    otter::parallel_for(ccRemainOutChannel, output_channels, 1, [&](int64_t start, int64_t end) {
        for (auto c : otter::irange(start, end)) {
            float* outptr0 = output_a[c].data();
            const float bias0 = bias ? bias[c] : 0.f;
            
            int i = 0;
            for ( ; i + 7 < outSize; i += 8) {
                const float* tmpptr = tmp_a[i / 8].data();
    #if __aarch64__
                const float* kptr = kernel_a[c / 8 + (c % 8) / 4 + c % 4].data();
    #else
                const float* kptr = kernel_a[c / 4 + c % 4)].data();
    #endif
                int64_t nn = input_channels * kernelSize;
    #if __aarch64__
                
                float32x4_t _sum0 = vdupq_n_f32(bias0);
                float32x4_t _sum1 = vdupq_n_f32(bias0);
                
                for (int64_t w = nn >> 2; w != 0; --w) {
                    float32x4_t _val0 = vld1q_f32(tmpptr);
                    float32x4_t _val1 = vld1q_f32(tmpptr + 4);
                    float32x4_t _val2 = vld1q_f32(tmpptr + 8);
                    float32x4_t _val3 = vld1q_f32(tmpptr + 12);
                    tmpptr += 16;
                    
                    float32x4_t _w0 = vld1q_f32(kptr);
                    kptr += 4;
                    
                    _sum0 = vfmaq_f32(_sum0, _val0, vdupq_n_f32(_w0[0]));
                    _sum1 = vfmaq_f32(_sum1, _val1, vdupq_n_f32(_w0[0]));
                    
                    float32x4_t _val4 = vld1q_f32(tmpptr);
                    float32x4_t _val5 = vld1q_f32(tmpptr + 4);
                    float32x4_t _val6 = vld1q_f32(tmpptr + 8);
                    float32x4_t _val7 = vld1q_f32(tmpptr + 12);
                    tmpptr += 16;
                    
                    _sum0 = vfmaq_f32(_sum0, _val2, vdupq_n_f32(_w0[1]));
                    _sum1 = vfmaq_f32(_sum1, _val3, vdupq_n_f32(_w0[1]));
                    
                    _sum0 = vfmaq_f32(_sum0, _val4, vdupq_n_f32(_w0[2]));
                    _sum1 = vfmaq_f32(_sum1, _val5, vdupq_n_f32(_w0[2]));
                    
                    _sum0 = vfmaq_f32(_sum0, _val6, vdupq_n_f32(_w0[3]));
                    _sum1 = vfmaq_f32(_sum1, _val7, vdupq_n_f32(_w0[3]));
                }
                
                for (int64_t w = nn & 3; w != 0; --w) {
                    float32x4_t _val0 = vld1q_f32(tmpptr);
                    float32x4_t _val1 = vld1q_f32(tmpptr + 4);
                    tmpptr += 8;
                    
                    float32x4_t _w0 = vld1q_dup_f32(kptr);
                    kptr += 1;
                    
                    _sum0 = vfmaq_f32(_sum0, _val0, _w0);
                    _sum1 = vfmaq_f32(_sum1, _val1, _w0);
                }
                
                vst1q_f32(outptr0, _sum0);
                vst1q_f32(outptr0 + 4, _sum1);
                outptr0 += 8;
    #else

    #endif
            }
            
            for ( ; i + 3 < outSize; i += 4) {
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();
    #if __aarch64__
                const float* kptr = kernel_a[c / 8 + (c % 8) / 4 + c % 4].data();
    #else
                const float* kptr = kernel_a[c / 4 + c % 4].data();
    #endif
                
                int64_t nn = input_channels * kernelSize;
                
    #if __aarch64__
                float32x4_t _sum0 = vdupq_n_f32(bias0);
                
                int j = 0;
                for (; j + 3 < nn; j += 4) {
                    float32x4_t _val0 = vld1q_f32(tmpptr);
                    float32x4_t _w0   = vdupq_n_f32(*kptr);
                    _sum0 = vfmaq_f32(_sum0, _val0, _w0);
                    
                    float32x4_t _val1 = vld1q_f32(tmpptr + 4);
                    float32x4_t _w1   = vdupq_n_f32(*(kptr + 1));
                    _sum0 = vfmaq_f32(_sum0, _val1, _w1);
                    
                    float32x4_t _val2 = vld1q_f32(tmpptr + 8);
                    float32x4_t _w2   = vdupq_n_f32(*(kptr + 2));
                    _sum0 = vfmaq_f32(_sum0, _val2, _w2);
                    
                    float32x4_t _val3 = vld1q_f32(tmpptr + 12);
                    float32x4_t _w3   = vdupq_n_f32(*(kptr + 3));
                    _sum0 = vfmaq_f32(_sum0, _val3, _w3);

                    tmpptr += 16;
                    kptr += 4;
                }
                for (; j < nn; j++) {
                    float32x4_t _val0 = vld1q_f32(tmpptr);
                    float32x4_t _w0   = vdupq_n_f32(*kptr);
                    _sum0 = vfmaq_f32(_sum0, _val0, _w0);
                    
                    tmpptr += 4;
                    kptr++;
                }
                
                vst1q_f32(outptr0, _sum0);
                outptr0 += 4;
    #else
                
    #endif
            }
            
            for ( ; i < outSize; ++i) {
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();
    #if __aarch64__
                const float* kptr = kernel_a[c / 8 + (c % 8) / 4 + c % 4].data();
    #else
                const float* kptr = kernel_a[c / 4 + c % 4].data();
    #endif
                int64_t nn = input_channels * kernelSize;
                
                float32x4_t _sum0 = vdupq_n_f32(0.f);
                
                int q = 0;
                for (; q + 3 < nn; q += 4) {
                    float32x4_t _p0 = vld1q_f32(tmpptr);
                    tmpptr += 4;

                    float32x4_t _k0 = vld1q_f32(kptr);
                    kptr += 4;

    #if __aarch64__
                    _sum0 = vfmaq_f32(_sum0, _p0, _k0);
    #else
                    _sum0 = vmlaq_f32(_sum0, _p0, _k0);
    #endif
                }
    #if __aarch64__
                float sum0 = bias0 + vaddvq_f32(_sum0);
    #else
                float32x2_t _ss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                float sum0 = bias0 + vget_lane_f32(vpadd_f32(_ss, _ss), 0);
    #endif
                for (; q < nn; q++) {
                    sum0 += tmpptr[0] * kptr[0];
                    tmpptr++;
                    kptr++;
                }

                outptr0[0] = sum0;

                outptr0++;
            }
        }
    });
    
//    std::cout << output << std::endl;
}

#else
void im2col_sgemm_conv2d_impl_neon(const Tensor& /*im2col_*/, const Tensor& /*kernel_pack4x4_*/, const Tensor& /*bias_*/, int64_t /*input_channels*/, int64_t /*output_channels*/, Tensor& /*output*/) {}
#endif

#ifdef __ARM_NEON__
void convolution_im2col_sgemm_transform_kernel_neon(const Tensor& kernel_, Tensor& kernel_tf, int64_t input_channels, int64_t output_channels, int64_t kernel_width, int64_t kernel_height) {
    const int64_t kernelSize = kernel_width * kernel_height;
    
    auto kernel = kernel_.view({output_channels, input_channels, kernelSize});

#if __aarch64__
    kernel_tf = otter::empty({output_channels / 8 + (output_channels % 8) / 4 + output_channels % 4, input_channels / 4 + input_channels % 4, 32 * kernelSize}, ScalarType::Float);
#else
    kernel_tf = otter::empty({outch / 4 + outch % 4, inch / 4 + inch % 4, 16 * maxk}, ScalarType::Float);
#endif
    int q = 0;
    
    auto kernel_a = kernel.accessor<float, 3>();
    auto kernel_tf_a = kernel_tf.accessor<float, 3>();
    
#if __aarch64__
    for (; q + 7 < output_channels; q += 8) {
        const auto k0 = kernel_a[q];
        const auto k1 = kernel_a[q + 1];
        const auto k2 = kernel_a[q + 2];
        const auto k3 = kernel_a[q + 3];
        const auto k4 = kernel_a[q + 4];
        const auto k5 = kernel_a[q + 5];
        const auto k6 = kernel_a[q + 6];
        const auto k7 = kernel_a[q + 7];

        float* g00 = kernel_tf_a[q / 8].data();

        for (int p = 0; p < input_channels; p++) {
            const float* k00 = k0[p].data();
            const float* k10 = k1[p].data();
            const float* k20 = k2[p].data();
            const float* k30 = k3[p].data();
            const float* k40 = k4[p].data();
            const float* k50 = k5[p].data();
            const float* k60 = k6[p].data();
            const float* k70 = k7[p].data();

            for (int k = 0; k < kernelSize; k++) {
                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];
                g00[4] = k40[k];
                g00[5] = k50[k];
                g00[6] = k60[k];
                g00[7] = k70[k];

                g00 += 8;
            }
        }
    }
#endif
    
    for (; q + 3 < output_channels; q += 4) {
        const auto k0 = kernel_a[q];
        const auto k1 = kernel_a[q + 1];
        const auto k2 = kernel_a[q + 2];
        const auto k3 = kernel_a[q + 3];

#if __aarch64__
        float* g00 = kernel_tf_a[q / 8 + (q % 8) / 4].data();
#else
        float* g00 = kernel_tf_a[q / 4].data();
#endif

        for (int p = 0; p < input_channels; p++) {
            const float* k00 = k0[p].data();
            const float* k10 = k1[p].data();
            const float* k20 = k2[p].data();
            const float* k30 = k3[p].data();

            for (int k = 0; k < kernelSize; k++) {
                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];

                g00 += 4;
            }
        }
    }
    for (; q < output_channels; q++) {
        const auto k0 = kernel_a[q];

#if __aarch64__
        float* g00 = kernel_tf_a[q / 8 + (q % 8) / 4 + q % 4].data();
#else
        float* g00 = kernel_tf_a[q / 4 + q % 4].data();
#endif

        for (int p = 0; p < input_channels; p++) {
            const float* k00 = k0[p].data();

            for (int k = 0; k < kernelSize; k++) {
                g00[0] = k00[k];

                g00 += 1;
            }
        }
    }
}
#else
void convolution_im2col_sgemm_transform_kernel_neon(const Tensor& /*_kernel*/, Tensor& /*kernel_tf*/, int64_t /*input_channels*/, int64_t /*out_chnnels*/, int64_t /*kernel_width*/, int64_t /*kernel_height*/) {
}
#endif

Tensor& sgemm_conv2d_1x1s1_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef /*kernel_size*/,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_(output_size);
    
    const int64_t input_channels  = self.size(1);
    const int64_t output_channels = weight.size(0);
    
    Tensor im2col = self.view({self.size(0), self.size(1), -1});
    Tensor kernel_pack4x4;
    if (weight_o.defined())
        kernel_pack4x4 = weight_o;
    else
        otter::convolution_im2col_sgemm_transform_kernel_neon(weight, kernel_pack4x4, input_channels, output_channels, 1, 1);
    otter::im2col_sgemm_conv2d_impl_neon(im2col, kernel_pack4x4, bias, input_channels, output_channels, output);
    
    return output;
}

Tensor sgemm_conv2d_1x1s1_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, self.options());
    
    return sgemm_conv2d_1x1s1_neon_out(self, weight, weight_o, bias, kernel_size, stride, padding, output);
}

Tensor& sgemm_conv2d_1x1s2_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_(output_size);
    
    int64_t batch_size = self.size(0);
    int64_t input_channels = self.size(1);
    int64_t input_width = self.size(3);
    
    int64_t output_height = output_size[2];
    int64_t output_width = output_size[3];
    
    const int64_t tailstep = input_width - 2 * output_width + input_width;
    
    auto self_shrink = otter::empty({batch_size, input_channels, output_height, output_width}, self.options());
    
    auto input_a = self.accessor<float, 4>()[0];
    auto input_shrink_a = self_shrink.accessor<float, 4>()[0];
    
    otter::parallel_for(0, input_channels, 0, [&](int64_t start, int64_t end) {
        for (const auto p : otter::irange(start, end)) {
            const float *input_ptr = input_a[p].data();
            float *output_ptr = input_shrink_a[p].data();
            
            for (const auto i : otter::irange(0, output_height)) {
                for (const auto j : otter::irange(0, output_width)) {
                    (void)i;
                    (void)j;
                    output_ptr[0] = input_ptr[0];
                    
                    input_ptr += 2;
                    output_ptr += 1;
                }
                input_ptr += tailstep;
            }
        }
    });
    
    return otter::sgemm_conv2d_1x1s1_neon_out(self_shrink, weight, weight_o, bias, kernel_size, stride, padding, output);
}

Tensor sgemm_conv2d_1x1s2_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {

    auto output = otter::empty({}, self.options());
    
    return sgemm_conv2d_1x1s2_neon_out(self, weight, weight_o, bias, kernel_size, stride, padding, output);
}

Tensor& sgemm_conv2d_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_(output_size);
    
    const int64_t kernel_height = kernel_size[0];
    const int64_t kernel_width  = kernel_size[1];
    
    const Tensor input = self.contiguous();
    const int64_t input_channels  = input.size(1);
    const int64_t output_channels = weight.size(0);
    
    Tensor im2col = otter::im2col_cpu(input, kernel_size, stride, padding, {1, 1});
    Tensor kernel_pack4x4;
    if (weight_o.defined())
        kernel_pack4x4 = weight_o;
    else
        otter::convolution_im2col_sgemm_transform_kernel_neon(weight, kernel_pack4x4, input_channels, output_channels, kernel_width, kernel_height);
    
    im2col_sgemm_conv2d_impl_neon(
        im2col,
        kernel_pack4x4,
        bias,
        input_channels,
        output_channels,
        output);
    
    return output;
}

Tensor sgemm_conv2d_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {

    auto output = otter::empty({}, self.options());
    
    return sgemm_conv2d_neon_out(self, weight, weight_o, bias, kernel_size, stride, padding, output);
}

void conv2d_1x1s1_neon_impl(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias_,
    IntArrayRef /*kernel_size*/,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0);
    auto output_shape = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_(output_shape);
    
    int inch  = (int)self.size(1);

    int outw  = (int)output_shape[3];
    int outh  = (int)output_shape[2];
    int outch = (int)output_shape[1];

    const float* kernel = weight.data_ptr<float>();
    const float* bias = (bias_.defined()) ? bias_.data_ptr<float>() : nullptr;
    
    auto input_a = input.accessor<float, 4>()[0];
    auto output_t = output[0];

    int nn_outch = 0;
    int remain_outch_start = 0;

    #if __ARM_NEON && __aarch64__

    nn_outch = outch >> 3;
    remain_outch_start = nn_outch << 3;

    otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
        for (const int pp : otter::irange(begin, end))
        {
            int p = pp * 8;

            auto out0 = output_t[p + 0];
            auto out1 = output_t[p + 1];
            auto out2 = output_t[p + 2];
            auto out3 = output_t[p + 3];
            auto out4 = output_t[p + 4];
            auto out5 = output_t[p + 5];
            auto out6 = output_t[p + 6];
            auto out7 = output_t[p + 7];

            const float bias0 = bias ? bias[p] : 0.f;
            const float bias1 = bias ? bias[p + 1] : 0.f;
            const float bias2 = bias ? bias[p + 2] : 0.f;
            const float bias3 = bias ? bias[p + 3] : 0.f;
            const float bias4 = bias ? bias[p + 4] : 0.f;
            const float bias5 = bias ? bias[p + 5] : 0.f;
            const float bias6 = bias ? bias[p + 6] : 0.f;
            const float bias7 = bias ? bias[p + 7] : 0.f;

            out0.fill_(bias0);
            out1.fill_(bias1);
            out2.fill_(bias2);
            out3.fill_(bias3);
            out4.fill_(bias4);
            out5.fill_(bias5);
            out6.fill_(bias6);
            out7.fill_(bias7);

            int q = 0;

            for (; q + 7 < inch; q += 8)
            {
                float* outptr0 = out0.data_ptr<float>();
                float* outptr1 = out1.data_ptr<float>();
                float* outptr2 = out2.data_ptr<float>();
                float* outptr3 = out3.data_ptr<float>();
                float* outptr4 = out4.data_ptr<float>();
                float* outptr5 = out5.data_ptr<float>();
                float* outptr6 = out6.data_ptr<float>();
                float* outptr7 = out7.data_ptr<float>();

                const float* img0 = input_a[q + 0].data();
                const float* img1 = input_a[q + 1].data();
                const float* img2 = input_a[q + 2].data();
                const float* img3 = input_a[q + 3].data();
                const float* img4 = input_a[q + 4].data();
                const float* img5 = input_a[q + 5].data();
                const float* img6 = input_a[q + 6].data();
                const float* img7 = input_a[q + 7].data();

                const float* kernel0 = kernel + p * inch + q;
                const float* kernel1 = kernel + (p + 1) * inch + q;
                const float* kernel2 = kernel + (p + 2) * inch + q;
                const float* kernel3 = kernel + (p + 3) * inch + q;
                const float* kernel4 = kernel + (p + 4) * inch + q;
                const float* kernel5 = kernel + (p + 5) * inch + q;
                const float* kernel6 = kernel + (p + 6) * inch + q;
                const float* kernel7 = kernel + (p + 7) * inch + q;

                const float* r0 = img0;
                const float* r1 = img1;
                const float* r2 = img2;
                const float* r3 = img3;
                const float* r4 = img4;
                const float* r5 = img5;
                const float* r6 = img6;
                const float* r7 = img7;

                int size = outw * outh;

                int nn = size >> 2;
                int remain = size & 3;

                float32x4_t _k0 = vld1q_f32(kernel0);
                float32x4_t _k1 = vld1q_f32(kernel1);
                float32x4_t _k2 = vld1q_f32(kernel2);
                float32x4_t _k3 = vld1q_f32(kernel3);
                float32x4_t _k4 = vld1q_f32(kernel4);
                float32x4_t _k5 = vld1q_f32(kernel5);
                float32x4_t _k6 = vld1q_f32(kernel6);
                float32x4_t _k7 = vld1q_f32(kernel7);

                float32x4_t _k0n = vld1q_f32(kernel0 + 4);
                float32x4_t _k1n = vld1q_f32(kernel1 + 4);
                float32x4_t _k2n = vld1q_f32(kernel2 + 4);
                float32x4_t _k3n = vld1q_f32(kernel3 + 4);
                float32x4_t _k4n = vld1q_f32(kernel4 + 4);
                float32x4_t _k5n = vld1q_f32(kernel5 + 4);
                float32x4_t _k6n = vld1q_f32(kernel6 + 4);
                float32x4_t _k7n = vld1q_f32(kernel7 + 4);

    #ifdef __clang__
                // gcc reject over 30 oprands :(
                if (nn > 0)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%9, #128]       \n"
                        "ld1    {v17.4s}, [%9], #16         \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v18.4s}, [%1]              \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v19.4s}, [%2]              \n"

                        "0:                                 \n"

                        "fmla   v18.4s, v17.4s, %34.s[0]    \n"

                        "prfm   pldl1keep, [%3, #128]       \n"
                        "ld1    {v20.4s}, [%3]              \n"

                        "fmla   v19.4s, v17.4s, %35.s[0]    \n"

                        "prfm   pldl1keep, [%4, #128]       \n"
                        "ld1    {v21.4s}, [%4]              \n"

                        "fmla   v20.4s, v17.4s, %36.s[0]    \n"

                        "prfm   pldl1keep, [%5, #128]       \n"
                        "ld1    {v22.4s}, [%5]              \n"

                        "fmla   v21.4s, v17.4s, %37.s[0]    \n"

                        "prfm   pldl1keep, [%6, #128]       \n"
                        "ld1    {v23.4s}, [%6]              \n"

                        "fmla   v22.4s, v17.4s, %38.s[0]    \n"

                        "prfm   pldl1keep, [%10, #128]      \n"
                        "ld1    {v16.4s}, [%10], #16        \n"

                        "fmla   v23.4s, v17.4s, %39.s[0]    \n"

                        "prfm   pldl1keep, [%7, #128]       \n"
                        "ld1    {v24.4s}, [%7]              \n"

                        "fmla   v18.4s, v16.4s, %34.s[1]    \n"
                        "fmla   v19.4s, v16.4s, %35.s[1]    \n"

                        "prfm   pldl1keep, [%8, #128]       \n"
                        "ld1    {v25.4s}, [%8]              \n"

                        "fmla   v24.4s, v17.4s, %40.s[0]    \n"
                        "fmla   v25.4s, v17.4s, %41.s[0]    \n"

                        "fmla   v20.4s, v16.4s, %36.s[1]    \n"
                        "fmla   v21.4s, v16.4s, %37.s[1]    \n"

                        "prfm   pldl1keep, [%11, #128]      \n"
                        "ld1    {v17.4s}, [%11], #16        \n"

                        "fmla   v22.4s, v16.4s, %38.s[1]    \n"
                        "fmla   v23.4s, v16.4s, %39.s[1]    \n"

                        "fmla   v18.4s, v17.4s, %34.s[2]    \n"
                        "fmla   v19.4s, v17.4s, %35.s[2]    \n"

                        "fmla   v24.4s, v16.4s, %40.s[1]    \n"
                        "fmla   v25.4s, v16.4s, %41.s[1]    \n"

                        "fmla   v20.4s, v17.4s, %36.s[2]    \n"
                        "fmla   v21.4s, v17.4s, %37.s[2]    \n"

                        "prfm   pldl1keep, [%12, #128]      \n"
                        "ld1    {v16.4s}, [%12], #16        \n"

                        "fmla   v22.4s, v17.4s, %38.s[2]    \n"
                        "fmla   v23.4s, v17.4s, %39.s[2]    \n"

                        "fmla   v18.4s, v16.4s, %34.s[3]    \n"
                        "fmla   v19.4s, v16.4s, %35.s[3]    \n"

                        "fmla   v24.4s, v17.4s, %40.s[2]    \n"
                        "fmla   v25.4s, v17.4s, %41.s[2]    \n"

                        "fmla   v20.4s, v16.4s, %36.s[3]    \n"
                        "fmla   v21.4s, v16.4s, %37.s[3]    \n"

                        "prfm   pldl1keep, [%13, #128]      \n"
                        "ld1    {v17.4s}, [%13], #16        \n"

                        "fmla   v22.4s, v16.4s, %38.s[3]    \n"
                        "fmla   v23.4s, v16.4s, %39.s[3]    \n"

                        "fmla   v18.4s, v17.4s, %42.s[0]    \n"
                        "fmla   v19.4s, v17.4s, %43.s[0]    \n"

                        "fmla   v24.4s, v16.4s, %40.s[3]    \n"
                        "fmla   v25.4s, v16.4s, %41.s[3]    \n"

                        "fmla   v20.4s, v17.4s, %44.s[0]    \n"
                        "fmla   v21.4s, v17.4s, %45.s[0]    \n"

                        "prfm   pldl1keep, [%14, #128]      \n"
                        "ld1    {v16.4s}, [%14], #16        \n"

                        "fmla   v22.4s, v17.4s, %46.s[0]    \n"
                        "fmla   v23.4s, v17.4s, %47.s[0]    \n"

                        "fmla   v18.4s, v16.4s, %42.s[1]    \n"
                        "fmla   v19.4s, v16.4s, %43.s[1]    \n"

                        "fmla   v24.4s, v17.4s, %48.s[0]    \n"
                        "fmla   v25.4s, v17.4s, %49.s[0]    \n"

                        "fmla   v20.4s, v16.4s, %44.s[1]    \n"
                        "fmla   v21.4s, v16.4s, %45.s[1]    \n"

                        "prfm   pldl1keep, [%15, #128]      \n"
                        "ld1    {v17.4s}, [%15], #16        \n"

                        "fmla   v22.4s, v16.4s, %46.s[1]    \n"
                        "fmla   v23.4s, v16.4s, %47.s[1]    \n"

                        "fmla   v18.4s, v17.4s, %42.s[2]    \n"
                        "fmla   v19.4s, v17.4s, %43.s[2]    \n"

                        "fmla   v24.4s, v16.4s, %48.s[1]    \n"
                        "fmla   v25.4s, v16.4s, %49.s[1]    \n"

                        "fmla   v20.4s, v17.4s, %44.s[2]    \n"
                        "fmla   v21.4s, v17.4s, %45.s[2]    \n"

                        "prfm   pldl1keep, [%16, #128]      \n"
                        "ld1    {v16.4s}, [%16], #16        \n"

                        "fmla   v22.4s, v17.4s, %46.s[2]    \n"
                        "fmla   v23.4s, v17.4s, %47.s[2]    \n"

                        "fmla   v18.4s, v16.4s, %42.s[3]    \n"
                        "fmla   v19.4s, v16.4s, %43.s[3]    \n"

                        "fmla   v24.4s, v17.4s, %48.s[2]    \n"
                        "fmla   v25.4s, v17.4s, %49.s[2]    \n"

                        "fmla   v20.4s, v16.4s, %44.s[3]    \n"
                        "fmla   v21.4s, v16.4s, %45.s[3]    \n"

                        "st1    {v18.4s}, [%1], #16         \n"

                        "fmla   v22.4s, v16.4s, %46.s[3]    \n"

                        "st1    {v19.4s}, [%2], #16         \n"

                        "fmla   v23.4s, v16.4s, %47.s[3]    \n"

                        "st1    {v20.4s}, [%3], #16         \n"

                        "prfm   pldl1keep, [%9, #128]       \n"
                        "ld1    {v17.4s}, [%9], #16         \n"

                        "fmla   v24.4s, v16.4s, %48.s[3]    \n"

                        "st1    {v21.4s}, [%4], #16         \n"

                        "fmla   v25.4s, v16.4s, %49.s[3]    \n"

                        "st1    {v22.4s}, [%5], #16         \n"

                        "prfm   pldl1keep, [%1, #128]       \n"
                        "ld1    {v18.4s}, [%1]              \n"

                        "st1    {v23.4s}, [%6], #16         \n"

                        "prfm   pldl1keep, [%2, #128]       \n"
                        "ld1    {v19.4s}, [%2]              \n"

                        "st1    {v24.4s}, [%7], #16         \n"

                        "subs   %w0, %w0, #1                \n"

                        "st1    {v25.4s}, [%8], #16         \n"

                        "bne    0b                          \n"
                        "sub    %9, %9, #16                 \n"
                        : "=r"(nn),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(outptr2), // %3
                        "=r"(outptr3), // %4
                        "=r"(outptr4), // %5
                        "=r"(outptr5), // %6
                        "=r"(outptr6), // %7
                        "=r"(outptr7), // %8
                        "=r"(r0),      // %9
                        "=r"(r1),      // %10
                        "=r"(r2),      // %11
                        "=r"(r3),      // %12
                        "=r"(r4),      // %13
                        "=r"(r5),      // %14
                        "=r"(r6),      // %15
                        "=r"(r7)       // %16
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(outptr4),
                        "6"(outptr5),
                        "7"(outptr6),
                        "8"(outptr7),
                        "9"(r0),
                        "10"(r1),
                        "11"(r2),
                        "12"(r3),
                        "13"(r4),
                        "14"(r5),
                        "15"(r6),
                        "16"(r7),
                        "w"(_k0),                                                                            // %34
                        "w"(_k1),                                                                            // %35
                        "w"(_k2),                                                                            // %36
                        "w"(_k3),                                                                            // %37
                        "w"(_k4),                                                                            // %38
                        "w"(_k5),                                                                            // %39
                        "w"(_k6),                                                                            // %40
                        "w"(_k7),                                                                            // %41
                        "w"(_k0n),                                                                           // %42
                        "w"(_k1n),                                                                           // %43
                        "w"(_k2n),                                                                           // %44
                        "w"(_k3n),                                                                           // %45
                        "w"(_k4n),                                                                           // %46
                        "w"(_k5n),                                                                           // %47
                        "w"(_k6n),                                                                           // %48
                        "w"(_k7n)                                                                            // %49
                        : "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25" //, "v26", "v27", "v28", "v29", "v30", "v31"
                    );
                }
    #else
                for (; nn > 0; nn--)
                {
                    float32x4_t _p = vld1q_f32(r0);

                    float32x4_t _out0p = vld1q_f32(outptr0);
                    float32x4_t _out1p = vld1q_f32(outptr1);
                    float32x4_t _out2p = vld1q_f32(outptr2);
                    float32x4_t _out3p = vld1q_f32(outptr3);
                    float32x4_t _out4p = vld1q_f32(outptr4);
                    float32x4_t _out5p = vld1q_f32(outptr5);
                    float32x4_t _out6p = vld1q_f32(outptr6);
                    float32x4_t _out7p = vld1q_f32(outptr7);

                    _out0p = vfmaq_laneq_f32(_out0p, _p, _k0, 0);
                    _out1p = vfmaq_laneq_f32(_out1p, _p, _k1, 0);
                    _out2p = vfmaq_laneq_f32(_out2p, _p, _k2, 0);
                    _out3p = vfmaq_laneq_f32(_out3p, _p, _k3, 0);
                    _out4p = vfmaq_laneq_f32(_out4p, _p, _k4, 0);
                    _out5p = vfmaq_laneq_f32(_out5p, _p, _k5, 0);
                    _out6p = vfmaq_laneq_f32(_out6p, _p, _k6, 0);
                    _out7p = vfmaq_laneq_f32(_out7p, _p, _k7, 0);

                    float32x4_t _p1 = vld1q_f32(r1);

                    _out0p = vfmaq_laneq_f32(_out0p, _p1, _k0, 1);
                    _out1p = vfmaq_laneq_f32(_out1p, _p1, _k1, 1);
                    _out2p = vfmaq_laneq_f32(_out2p, _p1, _k2, 1);
                    _out3p = vfmaq_laneq_f32(_out3p, _p1, _k3, 1);
                    _out4p = vfmaq_laneq_f32(_out4p, _p1, _k4, 1);
                    _out5p = vfmaq_laneq_f32(_out5p, _p1, _k5, 1);
                    _out6p = vfmaq_laneq_f32(_out6p, _p1, _k6, 1);
                    _out7p = vfmaq_laneq_f32(_out7p, _p1, _k7, 1);

                    float32x4_t _p2 = vld1q_f32(r2);

                    _out0p = vfmaq_laneq_f32(_out0p, _p2, _k0, 2);
                    _out1p = vfmaq_laneq_f32(_out1p, _p2, _k1, 2);
                    _out2p = vfmaq_laneq_f32(_out2p, _p2, _k2, 2);
                    _out3p = vfmaq_laneq_f32(_out3p, _p2, _k3, 2);
                    _out4p = vfmaq_laneq_f32(_out4p, _p2, _k4, 2);
                    _out5p = vfmaq_laneq_f32(_out5p, _p2, _k5, 2);
                    _out6p = vfmaq_laneq_f32(_out6p, _p2, _k6, 2);
                    _out7p = vfmaq_laneq_f32(_out7p, _p2, _k7, 2);

                    float32x4_t _p3 = vld1q_f32(r3);

                    _out0p = vfmaq_laneq_f32(_out0p, _p3, _k0, 3);
                    _out1p = vfmaq_laneq_f32(_out1p, _p3, _k1, 3);
                    _out2p = vfmaq_laneq_f32(_out2p, _p3, _k2, 3);
                    _out3p = vfmaq_laneq_f32(_out3p, _p3, _k3, 3);
                    _out4p = vfmaq_laneq_f32(_out4p, _p3, _k4, 3);
                    _out5p = vfmaq_laneq_f32(_out5p, _p3, _k5, 3);
                    _out6p = vfmaq_laneq_f32(_out6p, _p3, _k6, 3);
                    _out7p = vfmaq_laneq_f32(_out7p, _p3, _k7, 3);

                    float32x4_t _p4 = vld1q_f32(r4);

                    _out0p = vfmaq_laneq_f32(_out0p, _p4, _k0n, 0);
                    _out1p = vfmaq_laneq_f32(_out1p, _p4, _k1n, 0);
                    _out2p = vfmaq_laneq_f32(_out2p, _p4, _k2n, 0);
                    _out3p = vfmaq_laneq_f32(_out3p, _p4, _k3n, 0);
                    _out4p = vfmaq_laneq_f32(_out4p, _p4, _k4n, 0);
                    _out5p = vfmaq_laneq_f32(_out5p, _p4, _k5n, 0);
                    _out6p = vfmaq_laneq_f32(_out6p, _p4, _k6n, 0);
                    _out7p = vfmaq_laneq_f32(_out7p, _p4, _k7n, 0);

                    float32x4_t _p5 = vld1q_f32(r5);

                    _out0p = vfmaq_laneq_f32(_out0p, _p5, _k0n, 1);
                    _out1p = vfmaq_laneq_f32(_out1p, _p5, _k1n, 1);
                    _out2p = vfmaq_laneq_f32(_out2p, _p5, _k2n, 1);
                    _out3p = vfmaq_laneq_f32(_out3p, _p5, _k3n, 1);
                    _out4p = vfmaq_laneq_f32(_out4p, _p5, _k4n, 1);
                    _out5p = vfmaq_laneq_f32(_out5p, _p5, _k5n, 1);
                    _out6p = vfmaq_laneq_f32(_out6p, _p5, _k6n, 1);
                    _out7p = vfmaq_laneq_f32(_out7p, _p5, _k7n, 1);

                    float32x4_t _p6 = vld1q_f32(r6);

                    _out0p = vfmaq_laneq_f32(_out0p, _p6, _k0n, 2);
                    _out1p = vfmaq_laneq_f32(_out1p, _p6, _k1n, 2);
                    _out2p = vfmaq_laneq_f32(_out2p, _p6, _k2n, 2);
                    _out3p = vfmaq_laneq_f32(_out3p, _p6, _k3n, 2);
                    _out4p = vfmaq_laneq_f32(_out4p, _p6, _k4n, 2);
                    _out5p = vfmaq_laneq_f32(_out5p, _p6, _k5n, 2);
                    _out6p = vfmaq_laneq_f32(_out6p, _p6, _k6n, 2);
                    _out7p = vfmaq_laneq_f32(_out7p, _p6, _k7n, 2);

                    float32x4_t _p7 = vld1q_f32(r7);

                    _out0p = vfmaq_laneq_f32(_out0p, _p7, _k0n, 3);
                    _out1p = vfmaq_laneq_f32(_out1p, _p7, _k1n, 3);
                    _out2p = vfmaq_laneq_f32(_out2p, _p7, _k2n, 3);
                    _out3p = vfmaq_laneq_f32(_out3p, _p7, _k3n, 3);
                    _out4p = vfmaq_laneq_f32(_out4p, _p7, _k4n, 3);
                    _out5p = vfmaq_laneq_f32(_out5p, _p7, _k5n, 3);
                    _out6p = vfmaq_laneq_f32(_out6p, _p7, _k6n, 3);
                    _out7p = vfmaq_laneq_f32(_out7p, _p7, _k7n, 3);

                    vst1q_f32(outptr0, _out0p);
                    vst1q_f32(outptr1, _out1p);
                    vst1q_f32(outptr2, _out2p);
                    vst1q_f32(outptr3, _out3p);
                    vst1q_f32(outptr4, _out4p);
                    vst1q_f32(outptr5, _out5p);
                    vst1q_f32(outptr6, _out6p);
                    vst1q_f32(outptr7, _out7p);

                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    r4 += 4;
                    r5 += 4;
                    r6 += 4;
                    r7 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                    outptr4 += 4;
                    outptr5 += 4;
                    outptr6 += 4;
                    outptr7 += 4;
                }
    #endif
                for (; remain > 0; remain--)
                {
                    // TODO neon optimize
                    float sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3] + *r4 * kernel0[4] + *r5 * kernel0[5] + *r6 * kernel0[6] + *r7 * kernel0[7];
                    float sum1 = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3] + *r4 * kernel1[4] + *r5 * kernel1[5] + *r6 * kernel1[6] + *r7 * kernel1[7];
                    float sum2 = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3] + *r4 * kernel2[4] + *r5 * kernel2[5] + *r6 * kernel2[6] + *r7 * kernel2[7];
                    float sum3 = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3] + *r4 * kernel3[4] + *r5 * kernel3[5] + *r6 * kernel3[6] + *r7 * kernel3[7];
                    float sum4 = *r0 * kernel4[0] + *r1 * kernel4[1] + *r2 * kernel4[2] + *r3 * kernel4[3] + *r4 * kernel4[4] + *r5 * kernel4[5] + *r6 * kernel4[6] + *r7 * kernel4[7];
                    float sum5 = *r0 * kernel5[0] + *r1 * kernel5[1] + *r2 * kernel5[2] + *r3 * kernel5[3] + *r4 * kernel5[4] + *r5 * kernel5[5] + *r6 * kernel5[6] + *r7 * kernel5[7];
                    float sum6 = *r0 * kernel6[0] + *r1 * kernel6[1] + *r2 * kernel6[2] + *r3 * kernel6[3] + *r4 * kernel6[4] + *r5 * kernel6[5] + *r6 * kernel6[6] + *r7 * kernel6[7];
                    float sum7 = *r0 * kernel7[0] + *r1 * kernel7[1] + *r2 * kernel7[2] + *r3 * kernel7[3] + *r4 * kernel7[4] + *r5 * kernel7[5] + *r6 * kernel7[6] + *r7 * kernel7[7];

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;
                    *outptr4 += sum4;
                    *outptr5 += sum5;
                    *outptr6 += sum6;
                    *outptr7 += sum7;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    r4++;
                    r5++;
                    r6++;
                    r7++;
                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                    outptr4++;
                    outptr5++;
                    outptr6++;
                    outptr7++;
                }
            }

            for (; q < inch; q++)
            {
                float* outptr0 = out0.data_ptr<float>();
                float* outptr1 = out1.data_ptr<float>();
                float* outptr2 = out2.data_ptr<float>();
                float* outptr3 = out3.data_ptr<float>();
                float* outptr4 = out4.data_ptr<float>();
                float* outptr5 = out5.data_ptr<float>();
                float* outptr6 = out6.data_ptr<float>();
                float* outptr7 = out7.data_ptr<float>();

                const float* img0 = input_a[q].data();

                const float* kernel0 = kernel + p * inch + q;
                const float* kernel1 = kernel + (p + 1) * inch + q;
                const float* kernel2 = kernel + (p + 2) * inch + q;
                const float* kernel3 = kernel + (p + 3) * inch + q;
                const float* kernel4 = kernel + (p + 4) * inch + q;
                const float* kernel5 = kernel + (p + 5) * inch + q;
                const float* kernel6 = kernel + (p + 6) * inch + q;
                const float* kernel7 = kernel + (p + 7) * inch + q;

                const float k0 = kernel0[0];
                const float k1 = kernel1[0];
                const float k2 = kernel2[0];
                const float k3 = kernel3[0];
                const float k4 = kernel4[0];
                const float k5 = kernel5[0];
                const float k6 = kernel6[0];
                const float k7 = kernel7[0];

                const float* r0 = img0;

                int size = outw * outh;

                int nn = size >> 2;
                int remain = size & 3;

                float32x4_t _k0 = vdupq_n_f32(k0);
                float32x4_t _k1 = vdupq_n_f32(k1);
                float32x4_t _k2 = vdupq_n_f32(k2);
                float32x4_t _k3 = vdupq_n_f32(k3);
                float32x4_t _k4 = vdupq_n_f32(k4);
                float32x4_t _k5 = vdupq_n_f32(k5);
                float32x4_t _k6 = vdupq_n_f32(k6);
                float32x4_t _k7 = vdupq_n_f32(k7);

                for (; nn > 0; nn--)
                {
                    float32x4_t _p = vld1q_f32(r0);

                    float32x4_t _out0p = vld1q_f32(outptr0);
                    float32x4_t _out1p = vld1q_f32(outptr1);
                    float32x4_t _out2p = vld1q_f32(outptr2);
                    float32x4_t _out3p = vld1q_f32(outptr3);
                    float32x4_t _out4p = vld1q_f32(outptr4);
                    float32x4_t _out5p = vld1q_f32(outptr5);
                    float32x4_t _out6p = vld1q_f32(outptr6);
                    float32x4_t _out7p = vld1q_f32(outptr7);

                    _out0p = vfmaq_f32(_out0p, _p, _k0);
                    _out1p = vfmaq_f32(_out1p, _p, _k1);
                    _out2p = vfmaq_f32(_out2p, _p, _k2);
                    _out3p = vfmaq_f32(_out3p, _p, _k3);
                    _out4p = vfmaq_f32(_out4p, _p, _k4);
                    _out5p = vfmaq_f32(_out5p, _p, _k5);
                    _out6p = vfmaq_f32(_out6p, _p, _k6);
                    _out7p = vfmaq_f32(_out7p, _p, _k7);

                    vst1q_f32(outptr0, _out0p);
                    vst1q_f32(outptr1, _out1p);
                    vst1q_f32(outptr2, _out2p);
                    vst1q_f32(outptr3, _out3p);
                    vst1q_f32(outptr4, _out4p);
                    vst1q_f32(outptr5, _out5p);
                    vst1q_f32(outptr6, _out6p);
                    vst1q_f32(outptr7, _out7p);

                    r0 += 4;
                    outptr0 += 4;
                    outptr1 += 4;
                    outptr2 += 4;
                    outptr3 += 4;
                    outptr4 += 4;
                    outptr5 += 4;
                    outptr6 += 4;
                    outptr7 += 4;
                }
                for (; remain > 0; remain--)
                {
                    // TODO neon optimize
                    float sum0 = *r0 * k0;
                    float sum1 = *r0 * k1;
                    float sum2 = *r0 * k2;
                    float sum3 = *r0 * k3;
                    float sum4 = *r0 * k4;
                    float sum5 = *r0 * k5;
                    float sum6 = *r0 * k6;
                    float sum7 = *r0 * k7;

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;
                    *outptr4 += sum4;
                    *outptr5 += sum5;
                    *outptr6 += sum6;
                    *outptr7 += sum7;

                    r0++;
                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                    outptr4++;
                    outptr5++;
                    outptr6++;
                    outptr7++;
                }
            }
        }
    });

    #else

    nn_outch = outch / 6;
    remain_outch_start = nn_outch * 6;

    otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto pp : otter::irange(begin, end))
        {
            int p = (int)pp * 6;

            auto out0 = output_t[p + 0];
            auto out1 = output_t[p + 1];
            auto out2 = output_t[p + 2];
            auto out3 = output_t[p + 3];
            auto out4 = output_t[p + 4];
            auto out5 = output_t[p + 5];

            const float bias0 = bias ? bias[p] : 0.f;
            const float bias1 = bias ? bias[p + 1] : 0.f;
            const float bias2 = bias ? bias[p + 2] : 0.f;
            const float bias3 = bias ? bias[p + 3] : 0.f;
            const float bias4 = bias ? bias[p + 4] : 0.f;
            const float bias5 = bias ? bias[p + 5] : 0.f;

            out0.fill_(bias0);
            out1.fill_(bias1);
            out2.fill_(bias2);
            out3.fill_(bias3);
            out4.fill_(bias4);
            out5.fill_(bias5);

            int q = 0;

            for (; q + 3 < inch; q += 4)
            {
                float* outptr0 = out0.data_ptr<float>();
                float* outptr1 = out1.data_ptr<float>();
                float* outptr2 = out2.data_ptr<float>();
                float* outptr3 = out3.data_ptr<float>();
                float* outptr4 = out4.data_ptr<float>();
                float* outptr5 = out5.data_ptr<float>();

                const float* img0 = input_a[q + 0].data();
                const float* img1 = input_a[q + 1].data();
                const float* img2 = input_a[q + 2].data();
                const float* img3 = input_a[q + 3].data();

                const float* kernel0 = kernel + p * inch + q;
                const float* kernel1 = kernel + (p + 1) * inch + q;
                const float* kernel2 = kernel + (p + 2) * inch + q;
                const float* kernel3 = kernel + (p + 3) * inch + q;
                const float* kernel4 = kernel + (p + 4) * inch + q;
                const float* kernel5 = kernel + (p + 5) * inch + q;

                const float* r0 = img0;
                const float* r1 = img1;
                const float* r2 = img2;
                const float* r3 = img3;

                int size = outw * outh;

    #if __ARM_NEON
                int nn = size >> 2;
                int remain = size & 3;
    #else
                int remain = size;
    #endif // __ARM_NEON

    #if __ARM_NEON
                float32x4_t _k0 = vld1q_f32(kernel0);
                float32x4_t _k1 = vld1q_f32(kernel1);
                float32x4_t _k2 = vld1q_f32(kernel2);
                float32x4_t _k3 = vld1q_f32(kernel3);
                float32x4_t _k4 = vld1q_f32(kernel4);
                float32x4_t _k5 = vld1q_f32(kernel5);

                for (; nn > 0; nn--) {
                    asm volatile(
                        "pld        [%6, #128]              \n"
                        "vld1.f32   {d24-d25}, [%6 :128]!   \n" // q12 = r0

                        "pld        [%0, #128]              \n"
                        "vld1.f32   {d12-d13}, [%0 :128]    \n" // q6 = outptr0

                        "pld        [%1, #128]              \n"
                        "vld1.f32   {d14-d15}, [%1 :128]    \n" // q7 = outptr1

                        "vmla.f32   q6, q12, %e20[0]        \n"

                        "pld        [%2, #128]              \n"
                        "vld1.f32   {d16-d17}, [%2 :128]    \n" // q8 = outptr2

                        "vmla.f32   q7, q12, %e21[0]        \n"

                        "pld        [%3, #128]              \n"
                        "vld1.f32   {d18-d19}, [%3 :128]    \n" // q9 = outptr3

                        "vmla.f32   q8, q12, %e22[0]        \n"

                        "pld        [%7, #128]              \n"
                        "vld1.f32   {d26-d27}, [%7 :128]!   \n" // q13 = r1

                        "vmla.f32   q9, q12, %e23[0]        \n"

                        "pld        [%4, #128]              \n"
                        "vld1.f32   {d20-d21}, [%4 :128]    \n" // q10 = outptr4

                        "vmla.f32   q6, q13, %e20[1]        \n"
                        "vmla.f32   q7, q13, %e21[1]        \n"

                        "pld        [%5, #128]              \n"
                        "vld1.f32   {d22-d23}, [%5 :128]    \n" // q11 = outptr5

                        "vmla.f32   q10, q12, %e24[0]       \n"
                        "vmla.f32   q11, q12, %e25[0]       \n"

                        "vmla.f32   q8, q13, %e22[1]        \n"
                        "vmla.f32   q9, q13, %e23[1]        \n"

                        "pld        [%8, #128]              \n"
                        "vld1.f32   {d28-d29}, [%8 :128]!   \n" // q14 = r2

                        "vmla.f32   q10, q13, %e24[1]       \n"
                        "vmla.f32   q11, q13, %e25[1]       \n"

                        "vmla.f32   q6, q14, %f20[0]        \n"
                        "vmla.f32   q7, q14, %f21[0]        \n"
                        "vmla.f32   q8, q14, %f22[0]        \n"
                        "vmla.f32   q9, q14, %f23[0]        \n"

                        "pld        [%9, #128]             \n"
                        "vld1.f32   {d30-d31}, [%9 :128]!  \n" // q15 = r3

                        "vmla.f32   q10, q14, %f24[0]       \n"
                        "vmla.f32   q11, q14, %f25[0]       \n"

                        "vmla.f32   q6, q15, %f20[1]        \n"
                        "vmla.f32   q7, q15, %f21[1]        \n"
                        "vmla.f32   q8, q15, %f22[1]        \n"
                        "vmla.f32   q9, q15, %f23[1]        \n"

                        "vmla.f32   q10, q15, %f24[1]       \n"
                        "vmla.f32   q11, q15, %f25[1]       \n"

                        "vst1.f32   {d12-d13}, [%0 :128]!   \n"
                        "vst1.f32   {d14-d15}, [%1 :128]!   \n"
                        "vst1.f32   {d16-d17}, [%2 :128]!   \n"
                        "vst1.f32   {d18-d19}, [%3 :128]!   \n"
                        "vst1.f32   {d20-d21}, [%4 :128]!   \n"
                        "vst1.f32   {d22-d23}, [%5 :128]!   \n"

                        : "=r"(outptr0), // %0
                        "=r"(outptr1), // %1
                        "=r"(outptr2), // %2
                        "=r"(outptr3), // %3
                        "=r"(outptr4), // %4
                        "=r"(outptr5), // %5
                        "=r"(r0),      // %6
                        "=r"(r1),      // %7
                        "=r"(r2),      // %8
                        "=r"(r3)       // %9
                        : "0"(outptr0),
                        "1"(outptr1),
                        "2"(outptr2),
                        "3"(outptr3),
                        "4"(outptr4),
                        "5"(outptr5),
                        "6"(r0),
                        "7"(r1),
                        "8"(r2),
                        "9"(r3),
                        "w"(_k0), // %20
                        "w"(_k1), // %21
                        "w"(_k2), // %22
                        "w"(_k3), // %23
                        "w"(_k4), // %24
                        "w"(_k5)  // %25
                        : "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
    #endif // __ARM_NEON

                for (; remain > 0; remain--)
                {
                    // TODO neon optimize
                    float sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3];
                    float sum1 = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3];
                    float sum2 = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3];
                    float sum3 = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3];
                    float sum4 = *r0 * kernel4[0] + *r1 * kernel4[1] + *r2 * kernel4[2] + *r3 * kernel4[3];
                    float sum5 = *r0 * kernel5[0] + *r1 * kernel5[1] + *r2 * kernel5[2] + *r3 * kernel5[3];

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;
                    *outptr4 += sum4;
                    *outptr5 += sum5;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                    outptr4++;
                    outptr5++;
                }
            }

            for (; q < inch; q++)
            {
                float* outptr0 = out0.data_ptr<float>();
                float* outptr1 = out1.data_ptr<float>();
                float* outptr2 = out2.data_ptr<float>();
                float* outptr3 = out3.data_ptr<float>();
                float* outptr4 = out4.data_ptr<float>();
                float* outptr5 = out5.data_ptr<float>();

                const float* img0 = input_a[q].data();

                const float* kernel0 = kernel + p * inch + q;
                const float* kernel1 = kernel + (p + 1) * inch + q;
                const float* kernel2 = kernel + (p + 2) * inch + q;
                const float* kernel3 = kernel + (p + 3) * inch + q;
                const float* kernel4 = kernel + (p + 4) * inch + q;
                const float* kernel5 = kernel + (p + 5) * inch + q;

                const float k0 = kernel0[0];
                const float k1 = kernel1[0];
                const float k2 = kernel2[0];
                const float k3 = kernel3[0];
                const float k4 = kernel4[0];
                const float k5 = kernel5[0];

                const float* r0 = img0;

                int size = outw * outh;

    #if __ARM_NEON
                int nn = size >> 2;
                int remain = size & 3;
    #else
                int remain = size;
    #endif // __ARM_NEON

    #if __ARM_NEON
                float32x4_t _k0 = vdupq_n_f32(k0);
                float32x4_t _k1 = vdupq_n_f32(k1);
                float32x4_t _k2 = vdupq_n_f32(k2);
                float32x4_t _k3 = vdupq_n_f32(k3);
                float32x4_t _k4 = vdupq_n_f32(k4);
                float32x4_t _k5 = vdupq_n_f32(k5);

                if (nn > 0)
                {
                    asm volatile(
                        "pld        [%7, #128]              \n"
                        "vld1.f32   {d24-d25}, [%7 :128]!   \n" // q12 = r0

                        "pld        [%1, #128]              \n"
                        "vld1.f32   {d12-d13}, [%1 :128]    \n" // q6 = outptr0

                        "0:                                 \n"

                        "pld        [%2, #128]              \n"
                        "vld1.f32   {d14-d15}, [%2 :128]    \n" // q7 = outptr1

                        "vmla.f32   q6, q12, %q16           \n"

                        "pld        [%3, #128]              \n"
                        "vld1.f32   {d16-d17}, [%3 :128]    \n" // q8 = outptr2

                        "vmla.f32   q7, q12, %q17           \n"

                        "pld        [%4, #128]              \n"
                        "vld1.f32   {d18-d19}, [%4 :128]    \n" // q9 = outptr3

                        "vmla.f32   q8, q12, %q18           \n"

                        "pld        [%5, #128]              \n"
                        "vld1.f32   {d20-d21}, [%5 :128]    \n" // q10 = outptr4

                        "vmla.f32   q9, q12, %q19           \n"

                        "pld        [%6, #128]              \n"
                        "vld1.f32   {d22-d23}, [%6 :128]    \n" // q11 = outptr5

                        "vmla.f32   q10, q12, %q20          \n"
                        "vmla.f32   q11, q12, %q21          \n"

                        "pld        [%7, #128]              \n"
                        "vld1.f32   {d24-d25}, [%7 :128]!   \n" // q12 = r0

                        "vst1.f32   {d12-d13}, [%1 :128]!   \n"
                        "vst1.f32   {d14-d15}, [%2 :128]!   \n"

                        "pld        [%1, #128]              \n"
                        "vld1.f32   {d12-d13}, [%1 :128]    \n" // q6 = outptr0

                        "vst1.f32   {d16-d17}, [%3 :128]!   \n"
                        "vst1.f32   {d18-d19}, [%4 :128]!   \n"

                        "subs       %0, #1                  \n"

                        "vst1.f32   {d20-d21}, [%5 :128]!   \n"
                        "vst1.f32   {d22-d23}, [%6 :128]!   \n"

                        "bne        0b                      \n"

                        "sub        %7, #16                 \n"

                        : "=r"(nn),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(outptr2), // %3
                        "=r"(outptr3), // %4
                        "=r"(outptr4), // %5
                        "=r"(outptr5), // %6
                        "=r"(r0)       // %7
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(outptr4),
                        "6"(outptr5),
                        "7"(r0),
                        "w"(_k0), // %16
                        "w"(_k1), // %17
                        "w"(_k2), // %18
                        "w"(_k3), // %19
                        "w"(_k4), // %20
                        "w"(_k5)  // %21
                        : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12");
                }
    #endif // __ARM_NEON

                for (; remain > 0; remain--)
                {
                    // TODO neon optimize
                    float sum0 = *r0 * k0;
                    float sum1 = *r0 * k1;
                    float sum2 = *r0 * k2;
                    float sum3 = *r0 * k3;
                    float sum4 = *r0 * k4;
                    float sum5 = *r0 * k5;

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;
                    *outptr4 += sum4;
                    *outptr5 += sum5;

                    r0++;
                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                    outptr4++;
                    outptr5++;
                }
            }
        }
    });
    #endif // __ARM_NEON && __aarch64__

    nn_outch = (outch - remain_outch_start) >> 2;

    otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto pp : otter::irange(begin, end))
        {
            int p = remain_outch_start + (int)pp * 4;

            auto out0 = output_t[p + 0];
            auto out1 = output_t[p + 1];
            auto out2 = output_t[p + 2];
            auto out3 = output_t[p + 3];

            const float bias0 = bias ? bias[p] : 0.f;
            const float bias1 = bias ? bias[p + 1] : 0.f;
            const float bias2 = bias ? bias[p + 2] : 0.f;
            const float bias3 = bias ? bias[p + 3] : 0.f;

            out0.fill_(bias0);
            out1.fill_(bias1);
            out2.fill_(bias2);
            out3.fill_(bias3);

            int q = 0;

            for (; q + 3 < inch; q += 4)
            {
                float* outptr0 = out0.data_ptr<float>();
                float* outptr1 = out1.data_ptr<float>();
                float* outptr2 = out2.data_ptr<float>();
                float* outptr3 = out3.data_ptr<float>();

                const float* img0 = input_a[q + 0].data();
                const float* img1 = input_a[q + 1].data();
                const float* img2 = input_a[q + 2].data();
                const float* img3 = input_a[q + 3].data();

                const float* kernel0 = kernel + p * inch + q;
                const float* kernel1 = kernel + (p + 1) * inch + q;
                const float* kernel2 = kernel + (p + 2) * inch + q;
                const float* kernel3 = kernel + (p + 3) * inch + q;

                const float* r0 = img0;
                const float* r1 = img1;
                const float* r2 = img2;
                const float* r3 = img3;

                int size = outw * outh;

    #if __ARM_NEON
                int nn = size >> 3;
                int remain = size & 7;
    #else
                int remain = size;
    #endif // __ARM_NEON

    #if __ARM_NEON
                float32x4_t _k0 = vld1q_f32(kernel0);
                float32x4_t _k1 = vld1q_f32(kernel1);
                float32x4_t _k2 = vld1q_f32(kernel2);
                float32x4_t _k3 = vld1q_f32(kernel3);

    #if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v6.4s, v7.4s}, [%5], #32   \n"

                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%1]        \n"

                        "0:                                 \n"

                        "fmla   v8.4s, v6.4s, %18.s[0]      \n"

                        "prfm   pldl1keep, [%2, #256]       \n"
                        "ld1    {v10.4s, v11.4s}, [%2]      \n"

                        "fmla   v9.4s, v7.4s, %18.s[0]      \n"

                        "fmla   v10.4s, v6.4s, %19.s[0]     \n"

                        "prfm   pldl1keep, [%3, #256]       \n"
                        "ld1    {v12.4s, v13.4s}, [%3]      \n"

                        "fmla   v11.4s, v7.4s, %19.s[0]     \n"

                        "fmla   v12.4s, v6.4s, %20.s[0]     \n"

                        "prfm   pldl1keep, [%4, #256]       \n"
                        "ld1    {v14.4s, v15.4s}, [%4]      \n"

                        "fmla   v13.4s, v7.4s, %20.s[0]     \n"

                        "prfm   pldl1keep, [%6, #256]       \n"
                        "ld1    {v4.4s, v5.4s}, [%6], #32   \n"

                        "fmla   v14.4s, v6.4s, %21.s[0]     \n"
                        "fmla   v15.4s, v7.4s, %21.s[0]     \n"

                        "fmla   v8.4s, v4.4s, %18.s[1]      \n"
                        "fmla   v9.4s, v5.4s, %18.s[1]      \n"

                        "fmla   v10.4s, v4.4s, %19.s[1]     \n"
                        "fmla   v11.4s, v5.4s, %19.s[1]     \n"

                        "fmla   v12.4s, v4.4s, %20.s[1]     \n"
                        "fmla   v13.4s, v5.4s, %20.s[1]     \n"

                        "prfm   pldl1keep, [%7, #256]       \n"
                        "ld1    {v6.4s, v7.4s}, [%7], #32   \n"

                        "fmla   v14.4s, v4.4s, %21.s[1]     \n"
                        "fmla   v15.4s, v5.4s, %21.s[1]     \n"

                        "fmla   v8.4s, v6.4s, %18.s[2]      \n"
                        "fmla   v9.4s, v7.4s, %18.s[2]      \n"

                        "fmla   v10.4s, v6.4s, %19.s[2]     \n"
                        "fmla   v11.4s, v7.4s, %19.s[2]     \n"

                        "fmla   v12.4s, v6.4s, %20.s[2]     \n"
                        "fmla   v13.4s, v7.4s, %20.s[2]     \n"

                        "prfm   pldl1keep, [%8, #256]       \n"
                        "ld1    {v4.4s, v5.4s}, [%8], #32   \n"

                        "fmla   v14.4s, v6.4s, %21.s[2]     \n"
                        "fmla   v15.4s, v7.4s, %21.s[2]     \n"

                        "fmla   v8.4s, v4.4s, %18.s[3]      \n"
                        "fmla   v9.4s, v5.4s, %18.s[3]      \n"

                        "fmla   v10.4s, v4.4s, %19.s[3]     \n"
                        "fmla   v11.4s, v5.4s, %19.s[3]     \n"

                        "st1    {v8.4s, v9.4s}, [%1], #32   \n"

                        "fmla   v12.4s, v4.4s, %20.s[3]     \n"
                        "fmla   v13.4s, v5.4s, %20.s[3]     \n"

                        "st1    {v10.4s, v11.4s}, [%2], #32 \n"

                        "prfm   pldl1keep, [%5, #256]       \n"
                        "ld1    {v6.4s, v7.4s}, [%5], #32   \n"

                        "fmla   v14.4s, v4.4s, %21.s[3]     \n"
                        "fmla   v15.4s, v5.4s, %21.s[3]     \n"

                        "st1    {v12.4s, v13.4s}, [%3], #32 \n"

                        "prfm   pldl1keep, [%1, #256]       \n"
                        "ld1    {v8.4s, v9.4s}, [%1]        \n"

                        "subs   %w0, %w0, #1                \n"

                        "st1    {v14.4s, v15.4s}, [%4], #32 \n"

                        "bne    0b                          \n"
                        "sub    %5, %5, #32                 \n"
                        : "=r"(nn),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(outptr2), // %3
                        "=r"(outptr3), // %4
                        "=r"(r0),      // %5
                        "=r"(r1),      // %6
                        "=r"(r2),      // %7
                        "=r"(r3)       // %8
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(r0),
                        "6"(r1),
                        "7"(r2),
                        "8"(r3),
                        "w"(_k0), // %18
                        "w"(_k1), // %19
                        "w"(_k2), // %20
                        "w"(_k3)  // %21
                        : "cc", "memory", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }
    #else
                if (nn > 0)
                {
                    asm volatile(
                        "pld        [%5, #256]              \n"
                        "vld1.f32   {d12-d15}, [%5 :128]!   \n"
                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"
                        "0:                                 \n"

                        "vmla.f32   q8, q6, %e18[0]         \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d20-d23}, [%2 :128]    \n"
                        "vmla.f32   q9, q7, %e18[0]         \n"

                        "vmla.f32   q10, q6, %e19[0]        \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d24-d27}, [%3 :128]    \n"
                        "vmla.f32   q11, q7, %e19[0]        \n"

                        "vmla.f32   q12, q6, %e20[0]        \n"

                        "pld        [%4, #256]              \n"
                        "vld1.f32   {d28-d31}, [%4 :128]    \n"
                        "vmla.f32   q13, q7, %e20[0]        \n"

                        "pld        [%6, #256]              \n"
                        "vld1.f32   {d8-d11}, [%6 :128]!    \n"

                        "vmla.f32   q14, q6, %e21[0]        \n"
                        "vmla.f32   q15, q7, %e21[0]        \n"

                        "vmla.f32   q8, q4, %e18[1]         \n"
                        "vmla.f32   q9, q5, %e18[1]         \n"

                        "vmla.f32   q10, q4, %e19[1]        \n"
                        "vmla.f32   q11, q5, %e19[1]        \n"

                        "vmla.f32   q12, q4, %e20[1]        \n"
                        "vmla.f32   q13, q5, %e20[1]        \n"

                        "pld        [%7, #256]              \n"
                        "vld1.f32   {d12-d15}, [%7 :128]!   \n"

                        "vmla.f32   q14, q4, %e21[1]        \n"
                        "vmla.f32   q15, q5, %e21[1]        \n"

                        "vmla.f32   q8, q6, %f18[0]         \n"
                        "vmla.f32   q9, q7, %f18[0]         \n"

                        "vmla.f32   q10, q6, %f19[0]        \n"
                        "vmla.f32   q11, q7, %f19[0]        \n"

                        "vmla.f32   q12, q6, %f20[0]        \n"
                        "vmla.f32   q13, q7, %f20[0]        \n"

                        "pld        [%8, #256]              \n"
                        "vld1.f32   {d8-d11}, [%8 :128]!    \n"

                        "vmla.f32   q14, q6, %f21[0]        \n"
                        "vmla.f32   q15, q7, %f21[0]        \n"

                        "vmla.f32   q8, q4, %f18[1]         \n"
                        "vmla.f32   q9, q5, %f18[1]         \n"

                        "vmla.f32   q10, q4, %f19[1]        \n"
                        "vmla.f32   q11, q5, %f19[1]        \n"

                        "vmla.f32   q12, q4, %f20[1]        \n"
                        "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                        "vmla.f32   q13, q5, %f20[1]        \n"

                        "vst1.f32   {d20-d23}, [%2 :128]!   \n"

                        "vmla.f32   q14, q4, %f21[1]        \n"
                        "pld        [%5, #256]              \n"
                        "vld1.f32   {d12-d15}, [%5 :128]!   \n"

                        "vmla.f32   q15, q5, %f21[1]        \n"

                        "vst1.f32   {d24-d27}, [%3 :128]!   \n"

                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"

                        "subs       %0, #1                  \n"
                        "vst1.f32   {d28-d31}, [%4 :128]!   \n"

                        "bne        0b                      \n"
                        "sub        %5, #32                 \n"
                        : "=r"(nn),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(outptr2), // %3
                        "=r"(outptr3), // %4
                        "=r"(r0),      // %5
                        "=r"(r1),      // %6
                        "=r"(r2),      // %7
                        "=r"(r3)       // %8
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(r0),
                        "6"(r1),
                        "7"(r2),
                        "8"(r3),
                        "w"(_k0), // %18
                        "w"(_k1), // %19
                        "w"(_k2), // %20
                        "w"(_k3)  // %21
                        : "cc", "memory", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
    #endif // __aarch64__
    #endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    // TODO neon optimize
                    float sum0 = *r0 * kernel0[0] + *r1 * kernel0[1] + *r2 * kernel0[2] + *r3 * kernel0[3];
                    float sum1 = *r0 * kernel1[0] + *r1 * kernel1[1] + *r2 * kernel1[2] + *r3 * kernel1[3];
                    float sum2 = *r0 * kernel2[0] + *r1 * kernel2[1] + *r2 * kernel2[2] + *r3 * kernel2[3];
                    float sum3 = *r0 * kernel3[0] + *r1 * kernel3[1] + *r2 * kernel3[2] + *r3 * kernel3[3];

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                }
            }

            for (; q < inch; q++)
            {
                float* outptr0 = out0.data_ptr<float>();
                float* outptr1 = out1.data_ptr<float>();
                float* outptr2 = out2.data_ptr<float>();
                float* outptr3 = out3.data_ptr<float>();

                const float* img0 = input_a[q].data();

                const float* kernel0 = kernel + p * inch + q;
                const float* kernel1 = kernel + (p + 1) * inch + q;
                const float* kernel2 = kernel + (p + 2) * inch + q;
                const float* kernel3 = kernel + (p + 3) * inch + q;

                const float k0 = kernel0[0];
                const float k1 = kernel1[0];
                const float k2 = kernel2[0];
                const float k3 = kernel3[0];

                const float* r0 = img0;

                int size = outw * outh;

    #if __ARM_NEON
                int nn = size >> 3;
                int remain = size & 7;
    #else
                int remain = size;
    #endif // __ARM_NEON

    #if __ARM_NEON
                float32x4_t _k0 = vdupq_n_f32(k0);
                float32x4_t _k1 = vdupq_n_f32(k1);
                float32x4_t _k2 = vdupq_n_f32(k2);
                float32x4_t _k3 = vdupq_n_f32(k3);
    #if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "prfm       pldl1keep, [%5, #256]          \n"
                        "ld1        {v6.4s, v7.4s}, [%5], #32      \n"
                        "0:                                        \n"
                        "prfm       pldl1keep, [%1, #256]          \n"
                        "ld1        {v8.4s, v9.4s}, [%1]           \n"
                        "fmla       v8.4s, v6.4s, %12.4s           \n"
                        "fmla       v9.4s, v7.4s, %12.4s           \n"

                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld1        {v10.4s, v11.4s}, [%2]         \n"
                        "fmla       v10.4s, v6.4s, %13.4s          \n"
                        "fmla       v11.4s, v7.4s, %13.4s          \n"

                        "st1        {v8.4s, v9.4s}, [%1], #32      \n"

                        "prfm       pldl1keep, [%3, #256]          \n"
                        "ld1        {v12.4s, v13.4s}, [%3]         \n"
                        "fmla       v12.4s, v6.4s, %14.4s          \n"
                        "fmla       v13.4s, v7.4s, %14.4s          \n"

                        "st1        {v10.4s, v11.4s}, [%2], #32    \n"

                        "prfm       pldl1keep, [%4, #256]          \n"
                        "ld1        {v14.4s, v15.4s}, [%4]         \n"
                        "fmla       v14.4s, v6.4s, %15.4s          \n"
                        "fmla       v15.4s, v7.4s, %15.4s          \n"

                        "st1        {v12.4s, v13.4s}, [%3], #32    \n"

                        "prfm       pldl1keep, [%5, #256]          \n"
                        "ld1        {v6.4s, v7.4s}, [%5], #32      \n"
                        "subs       %w0, %w0, #1                   \n"
                        "st1        {v14.4s, v15.4s}, [%4], #32    \n"
                        "bne        0b                             \n"
                        "sub        %5, %5, #32                    \n"
                        : "=r"(nn),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(outptr2), // %3
                        "=r"(outptr3), // %4
                        "=r"(r0)       // %5
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(r0),
                        "w"(_k0), // %12
                        "w"(_k1), // %13
                        "w"(_k2), // %14
                        "w"(_k3)  // %15
                        : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }
    #else
                if (nn > 0)
                {
                    asm volatile(
                        "pld        [%5, #256]              \n"
                        "vld1.f32   {d12-d15}, [%5 :128]!   \n"
                        "0:                                 \n"
                        "pld        [%1, #256]              \n"
                        "vld1.f32   {d16-d19}, [%1 :128]    \n"
                        "vmla.f32   q8, q6, %q12            \n"
                        "vmla.f32   q9, q7, %q12            \n"

                        "pld        [%2, #256]              \n"
                        "vld1.f32   {d20-d23}, [%2 :128]    \n"
                        "vmla.f32   q10, q6, %q13           \n"
                        "vmla.f32   q11, q7, %q13           \n"

                        "vst1.f32   {d16-d19}, [%1 :128]!   \n"

                        "pld        [%3, #256]              \n"
                        "vld1.f32   {d24-d27}, [%3 :128]    \n"
                        "vmla.f32   q12, q6, %q14           \n"
                        "vmla.f32   q13, q7, %q14           \n"

                        "vst1.f32   {d20-d23}, [%2 :128]!   \n"

                        "pld        [%4, #256]              \n"
                        "vld1.f32   {d28-d31}, [%4 :128]    \n"
                        "vmla.f32   q14, q6, %q15           \n"
                        "vmla.f32   q15, q7, %q15           \n"

                        "vst1.f32   {d24-d27}, [%3 :128]!   \n"

                        "pld        [%5, #256]              \n"
                        "vld1.f32   {d12-d15}, [%5 :128]!   \n"
                        "subs       %0, #1                  \n"
                        "vst1.f32   {d28-d31}, [%4 :128]!   \n"
                        "bne        0b                      \n"
                        "sub        %5, #32                 \n"
                        : "=r"(nn),      // %0
                        "=r"(outptr0), // %1
                        "=r"(outptr1), // %2
                        "=r"(outptr2), // %3
                        "=r"(outptr3), // %4
                        "=r"(r0)       // %5
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(r0),
                        "w"(_k0), // %12
                        "w"(_k1), // %13
                        "w"(_k2), // %14
                        "w"(_k3)  // %15
                        : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
    #endif // __aarch64__
    #endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    // TODO neon optimize
                    float sum0 = *r0 * k0;
                    float sum1 = *r0 * k1;
                    float sum2 = *r0 * k2;
                    float sum3 = *r0 * k3;

                    *outptr0 += sum0;
                    *outptr1 += sum1;
                    *outptr2 += sum2;
                    *outptr3 += sum3;

                    r0++;
                    outptr0++;
                    outptr1++;
                    outptr2++;
                    outptr3++;
                }
            }
        }
    });

    remain_outch_start += nn_outch << 2;

    otter::parallel_for(remain_outch_start, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end))
        {
            auto out = output_t[p];

            const float bias0 = bias ? bias[p] : 0.f;

            out.fill_(bias0);

            int q = 0;

            for (; q + 3 < inch; q += 4)
            {
                float* outptr = out.data_ptr<float>();

                const float* img0 = input_a[q + 0].data();
                const float* img1 = input_a[q + 1].data();
                const float* img2 = input_a[q + 2].data();
                const float* img3 = input_a[q + 3].data();

                const float* kernel0 = kernel + p * inch + q;
                const float k0 = kernel0[0];
                const float k1 = kernel0[1];
                const float k2 = kernel0[2];
                const float k3 = kernel0[3];

                const float* r0 = img0;
                const float* r1 = img1;
                const float* r2 = img2;
                const float* r3 = img3;

                int size = outw * outh;

    #if __ARM_NEON
                int nn = size >> 3;
                int remain = size & 7;
    #else
                int remain = size;
    #endif // __ARM_NEON

    #if __ARM_NEON
                float32x4_t _k0 = vdupq_n_f32(k0);
                float32x4_t _k1 = vdupq_n_f32(k1);
                float32x4_t _k2 = vdupq_n_f32(k2);
                float32x4_t _k3 = vdupq_n_f32(k3);
    #if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld1        {v2.4s, v3.4s}, [%2], #32      \n"
                        "0:                                        \n"
                        "prfm       pldl1keep, [%1, #256]          \n"
                        "ld1        {v0.4s, v1.4s}, [%1]           \n"
                        "fmla       v0.4s, v2.4s, %12.4s           \n"
                        "fmla       v1.4s, v3.4s, %12.4s           \n"

                        "prfm       pldl1keep, [%3, #256]          \n"
                        "ld1        {v2.4s, v3.4s}, [%3], #32      \n"
                        "fmla       v0.4s, v2.4s, %13.4s           \n"
                        "fmla       v1.4s, v3.4s, %13.4s           \n"

                        "prfm       pldl1keep, [%4, #256]          \n"
                        "ld1        {v2.4s, v3.4s}, [%4], #32      \n"
                        "fmla       v0.4s, v2.4s, %14.4s           \n"
                        "fmla       v1.4s, v3.4s, %14.4s           \n"

                        "prfm       pldl1keep, [%5, #256]          \n"
                        "ld1        {v2.4s, v3.4s}, [%5], #32      \n"
                        "fmla       v0.4s, v2.4s, %15.4s           \n"
                        "fmla       v1.4s, v3.4s, %15.4s           \n"

                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld1        {v2.4s, v3.4s}, [%2], #32      \n"
                        "subs       %w0, %w0, #1                   \n"
                        "st1        {v0.4s, v1.4s}, [%1], #32      \n"
                        "bne        0b                             \n"
                        "sub        %2, %2, #32                    \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2),     // %4
                        "=r"(r3)      // %5
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "w"(_k0), // %12
                        "w"(_k1), // %13
                        "w"(_k2), // %14
                        "w"(_k3)  // %15
                        : "cc", "memory", "v0", "v1", "v2", "v3");
                }
    #else
                if (nn > 0)
                {
                    asm volatile(
                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d4-d7}, [%2 :128]! \n"
                        "0:                             \n"
                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]  \n"
                        "vmla.f32   q0, q2, %q12        \n"
                        "vmla.f32   q1, q3, %q12        \n"
                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d4-d7}, [%3 :128]! \n"
                        "vmla.f32   q0, q2, %q13        \n"
                        "vmla.f32   q1, q3, %q13        \n"
                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d4-d7}, [%4 :128]! \n"
                        "vmla.f32   q0, q2, %q14        \n"
                        "vmla.f32   q1, q3, %q14        \n"
                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d4-d7}, [%5 :128]! \n"
                        "vmla.f32   q0, q2, %q15        \n"
                        "vmla.f32   q1, q3, %q15        \n"
                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d4-d7}, [%2 :128]! \n"
                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d3}, [%1 :128]! \n"
                        "bne        0b                  \n"
                        "sub        %2, #32             \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2),     // %4
                        "=r"(r3)      // %5
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "w"(_k0), // %12
                        "w"(_k1), // %13
                        "w"(_k2), // %14
                        "w"(_k3)  // %15
                        : "cc", "memory", "q0", "q1", "q2", "q3");
                }
    #endif // __aarch64__
    #endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    float sum = *r0 * k0;
                    float sum1 = *r1 * k1;
                    float sum2 = *r2 * k2;
                    float sum3 = *r3 * k3;

                    *outptr += sum + sum1 + sum2 + sum3;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    outptr++;
                }
            }

            for (; q < inch; q++)
            {
                float* outptr = out.data_ptr<float>();

                const float* img0 = input_a[q].data();

                const float* kernel0 = kernel + p * inch + q;
                const float k0 = kernel0[0];

                const float* r0 = img0;

                int size = outw * outh;

    #if __ARM_NEON
                int nn = size >> 3;
                int remain = size & 7;
    #else
                int remain = size;
    #endif // __ARM_NEON

    #if __ARM_NEON
                float32x4_t _k0 = vdupq_n_f32(k0);
    #if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld1        {v2.4s, v3.4s}, [%2], #32      \n"
                        "0:                                        \n"
                        "prfm       pldl1keep, [%1, #256]          \n"
                        "ld1        {v0.4s, v1.4s}, [%1]           \n"
                        "fmla       v0.4s, v2.4s, %6.4s            \n"
                        "fmla       v1.4s, v3.4s, %6.4s            \n"
                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld1        {v2.4s, v3.4s}, [%2], #32      \n"
                        "subs       %w0, %w0, #1                   \n"
                        "st1        {v0.4s, v1.4s}, [%1], #32      \n"
                        "bne        0b                             \n"
                        "sub        %2, %2, #32                    \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0)      // %2
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "w"(_k0) // %6
                        : "cc", "memory", "v0", "v1", "v2", "v3");
                }
    #else
                if (nn > 0)
                {
                    asm volatile(
                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d4-d7}, [%2 :128]! \n"
                        "0:                             \n"
                        "pld        [%1, #256]          \n"
                        "vld1.f32   {d0-d3}, [%1 :128]  \n"
                        "vmla.f32   q0, q2, %q6         \n"
                        "vmla.f32   q1, q3, %q6         \n"
                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d4-d7}, [%2 :128]! \n"
                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d3}, [%1 :128]! \n"
                        "bne        0b                  \n"
                        "sub        %2, #32             \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0)      // %2
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "w"(_k0) // %6
                        : "cc", "memory", "q0", "q1", "q2", "q3");
                }
    #endif // __aarch64__
    #endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    float sum = *r0 * k0;

                    *outptr += sum;

                    r0++;
                    outptr++;
                }
            }
        }
    });
}

Tensor& conv2d_1x1s1_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    conv2d_1x1s1_neon_impl(self, weight, bias, kernel_size, stride, padding, output);
    
    return output;
}

Tensor conv2d_1x1s1_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, self.options());
    
    return conv2d_1x1s1_neon_out(self, weight, bias, kernel_size, stride, padding, output);
}

Tensor& conv2d_3x3s1_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0);
    auto output_shape = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_(output_shape);
    
    int inch  = (int)self.size(1);
    int w = (int)input.size(3);

    int outw  = (int)output_shape[3];
    int outh  = (int)output_shape[2];
    int outch = (int)output_shape[1];
    
    const float* kernel = weight.data_ptr<float>();
    const float* bias = (bias_.defined()) ? bias_.data_ptr<float>() : nullptr;

    int nn_outch = outch >> 1;
    int remain_outch_start = nn_outch << 1;
    
    auto input_a = input.accessor<float, 4>()[0];
    auto output_t = output[0];

    otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto pp : otter::irange(begin, end)) {
            int p = pp * 2;

            auto out0 = output_t[p];
            auto out1 = output_t[p + 1];

            const float bias0 = bias ? bias[p] : 0.f;
            const float bias1 = bias ? bias[p + 1] : 0.f;

            out0.fill_(bias0);
            out1.fill_(bias1);

            const float* k0 = kernel + p * inch * 9;
            const float* k1 = kernel + (p + 1) * inch * 9;

            for (int q = 0; q < inch; q++)
            {
                float* outptr0 = out0.data_ptr<float>();
                float* outptr1 = out1.data_ptr<float>();
                float* outptr0n = outptr0 + outw;
                float* outptr1n = outptr1 + outw;

                const float* img0 = input_a[q].data();

                const float* r0 = img0;
                const float* r1 = img0 + w;
                const float* r2 = img0 + w * 2;
                const float* r3 = img0 + w * 3;

    #if __ARM_NEON
                float32x4_t _k00 = vld1q_f32(k0);
                float32x4_t _k03 = vld1q_f32(k0 + 3);
                float32x4_t _k06 = vld1q_f32(k0 + 6);

                float32x4_t _k10 = vld1q_f32(k1);
                float32x4_t _k13 = vld1q_f32(k1 + 3);
                float32x4_t _k16 = vld1q_f32(k1 + 6);
    #endif // __ARM_NEON

                int i = 0;

                for (; i + 1 < outh; i += 2)
                {
    #if __ARM_NEON
                    int nn = outw >> 2;
                    int remain = outw & 3;
    #else
                    int remain = outw;
    #endif // __ARM_NEON

    #if __ARM_NEON
    #if __aarch64__
                    if (nn > 0)
                    {
                        asm volatile(
                            "prfm   pldl1keep, [%5, #256]       \n"
                            "ld1    {v8.4s, v9.4s}, [%5]        \n" // r0
                            "add    %5, %5, #16                 \n"

                            "prfm   pldl1keep, [%8, #256]       \n"
                            "ld1    {v14.4s, v15.4s}, [%8]      \n" // r3
                            "add    %8, %8, #16                 \n"

                            "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                            "ext    v11.16b, v14.16b, v15.16b, #8 \n"

                            "0:                                 \n"

                            "prfm   pldl1keep, [%1, #128]       \n"
                            "ld1    {v6.4s}, [%1]               \n" // _sum0

                            "prfm   pldl1keep, [%2, #128]       \n"
                            "ld1    {v7.4s}, [%2]               \n" // _sum1

                            "fmla   v6.4s, v8.4s, %18.s[0]      \n"
                            "fmla   v7.4s, v8.4s, %21.s[0]      \n"

                            "prfm   pldl1keep, [%3, #128]       \n"
                            "ld1    {v12.4s}, [%3]              \n" // _sum0n

                            "prfm   pldl1keep, [%4, #128]       \n"
                            "ld1    {v13.4s}, [%4]              \n" // _sum1n

                            "fmla   v12.4s, v14.4s, %20.s[0]    \n"
                            "fmla   v13.4s, v14.4s, %23.s[0]    \n"

                            "ext    v8.16b, v8.16b, v9.16b, #8  \n"
                            "ext    v9.16b, v14.16b, v15.16b, #4 \n"

                            "fmla   v6.4s, v10.4s, %18.s[1]     \n"
                            "fmla   v7.4s, v10.4s, %21.s[1]     \n"
                            "fmla   v12.4s, v11.4s, %20.s[2]    \n"
                            "fmla   v13.4s, v11.4s, %23.s[2]    \n"

                            "prfm   pldl1keep, [%6, #256]       \n"
                            "ld1    {v14.4s, v15.4s}, [%6]      \n" // r1
                            "add    %6, %6, #16                 \n"

                            "fmla   v6.4s, v8.4s, %18.s[2]      \n"
                            "fmla   v7.4s, v8.4s, %21.s[2]      \n"
                            "fmla   v12.4s, v9.4s, %20.s[1]     \n"
                            "fmla   v13.4s, v9.4s, %23.s[1]     \n"

                            "ext    v10.16b, v14.16b, v15.16b, #4 \n"

                            "fmla   v6.4s, v14.4s, %19.s[0]     \n"
                            "fmla   v7.4s, v14.4s, %22.s[0]     \n"
                            "fmla   v12.4s, v14.4s, %18.s[0]    \n"
                            "fmla   v13.4s, v14.4s, %21.s[0]    \n"

                            "ext    v11.16b, v14.16b, v15.16b, #8 \n"

                            "fmla   v6.4s, v10.4s, %19.s[1]     \n"
                            "fmla   v7.4s, v10.4s, %22.s[1]     \n"
                            "fmla   v12.4s, v10.4s, %18.s[1]    \n"
                            "fmla   v13.4s, v10.4s, %21.s[1]    \n"

                            "prfm   pldl1keep, [%7, #256]       \n"
                            "ld1    {v8.4s, v9.4s}, [%7]        \n" // r2
                            "add    %7, %7, #16                 \n"

                            "fmla   v6.4s, v11.4s, %19.s[2]     \n"
                            "fmla   v7.4s, v11.4s, %22.s[2]     \n"
                            "fmla   v12.4s, v11.4s, %18.s[2]    \n"
                            "fmla   v13.4s, v11.4s, %21.s[2]    \n"

                            "ext    v10.16b, v8.16b, v9.16b, #4 \n"

                            "fmla   v6.4s, v8.4s, %20.s[0]      \n"
                            "fmla   v7.4s, v8.4s, %23.s[0]      \n"
                            "fmla   v12.4s, v8.4s, %19.s[0]     \n"
                            "fmla   v13.4s, v8.4s, %22.s[0]     \n"

                            "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                            "fmla   v6.4s, v10.4s, %20.s[1]     \n"
                            "fmla   v7.4s, v10.4s, %23.s[1]     \n"
                            "fmla   v12.4s, v10.4s, %19.s[1]    \n"
                            "fmla   v13.4s, v10.4s, %22.s[1]    \n"

                            "prfm   pldl1keep, [%5, #256]       \n"
                            "ld1    {v8.4s, v9.4s}, [%5]        \n" // r0
                            "add    %5, %5, #16                 \n"

                            "fmla   v6.4s, v11.4s, %20.s[2]     \n"
                            "fmla   v7.4s, v11.4s, %23.s[2]     \n"
                            "fmla   v12.4s, v11.4s, %19.s[2]    \n"
                            "fmla   v13.4s, v11.4s, %22.s[2]    \n"

                            "prfm   pldl1keep, [%8, #256]       \n"
                            "ld1    {v14.4s, v15.4s}, [%8]      \n" // r3
                            "add    %8, %8, #16                 \n"

                            "ext    v10.16b, v8.16b, v9.16b, #4 \n"

                            "st1    {v6.4s}, [%1], #16          \n"
                            "st1    {v7.4s}, [%2], #16          \n"

                            "ext    v11.16b, v14.16b, v15.16b, #8 \n"

                            "st1    {v12.4s}, [%3], #16         \n"
                            "st1    {v13.4s}, [%4], #16         \n"

                            "subs   %w0, %w0, #1                \n"
                            "bne    0b                          \n"

                            "sub    %5, %5, #16                 \n"
                            "sub    %8, %8, #16                 \n"
                            : "=r"(nn),       // %0
                            "=r"(outptr0),  // %1
                            "=r"(outptr1),  // %2
                            "=r"(outptr0n), // %3
                            "=r"(outptr1n), // %4
                            "=r"(r0),       // %5
                            "=r"(r1),       // %6
                            "=r"(r2),       // %7
                            "=r"(r3)        // %8
                            : "0"(nn),
                            "1"(outptr0),
                            "2"(outptr1),
                            "3"(outptr0n),
                            "4"(outptr1n),
                            "5"(r0),
                            "6"(r1),
                            "7"(r2),
                            "8"(r3),
                            "w"(_k00), // %18
                            "w"(_k03), // %19
                            "w"(_k06), // %20
                            "w"(_k10), // %21
                            "w"(_k13), // %22
                            "w"(_k16)  // %23
                            : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                    }
    #else
                    if (nn > 0)
                    {
                        asm volatile(

                            "pld        [%5, #192]          \n"
                            "vld1.f32   {d16-d18}, [%5 :64] \n" // r0
                            "add        %5, #16             \n"

                            "pld        [%8, #192]          \n"
                            "vld1.f32   {d28-d30}, [%8]     \n" // r3
                            "add        %8, #16             \n"

                            "vext.32    q10, q8, q9, #1     \n"
                            "vext.32    q11, q14, q15, #2   \n"

                            "0:                             \n"

                            "pld        [%1, #128]          \n"
                            "vld1.f32   {d12-d13}, [%1 :64] \n" // _sum0

                            "pld        [%2, #128]          \n"
                            "vld1.f32   {d14-d15}, [%2 :64] \n" // _sum1

                            "vmla.f32   q6, q8, %e18[0]     \n"
                            "vmla.f32   q7, q8, %e21[0]     \n"

                            "pld        [%3, #128]          \n"
                            "vld1.f32   {d24-d25}, [%3]     \n" // _sum0n

                            "pld        [%4, #128]          \n"
                            "vld1.f32   {d26-d27}, [%4]     \n" // _sum1n

                            "vmla.f32   q12, q14, %e20[0]   \n"
                            "vmla.f32   q13, q14, %e23[0]   \n"

                            "vext.32    q8, q8, q9, #2      \n"
                            "vext.32    q9, q14, q15, #1    \n"

                            "vmla.f32   q6, q10, %e18[1]    \n"
                            "vmla.f32   q7, q10, %e21[1]    \n"
                            "vmla.f32   q12, q11, %f20[0]   \n"
                            "vmla.f32   q13, q11, %f23[0]   \n"

                            "pld        [%6, #192]          \n"
                            "vld1.f32   {d28-d30}, [%6]     \n" // r1
                            "add        %6, #16             \n"

                            "vmla.f32   q6, q8, %f18[0]     \n"
                            "vmla.f32   q7, q8, %f21[0]     \n"
                            "vmla.f32   q12, q9, %e20[1]    \n"
                            "vmla.f32   q13, q9, %e23[1]    \n"

                            "vext.32    q10, q14, q15, #1   \n"

                            "vmla.f32   q6, q14, %e19[0]    \n"
                            "vmla.f32   q7, q14, %e22[0]    \n"
                            "vmla.f32   q12, q14, %e18[0]   \n"
                            "vmla.f32   q13, q14, %e21[0]   \n"

                            "vext.32    q11, q14, q15, #2   \n"

                            "vmla.f32   q6, q10, %e19[1]    \n"
                            "vmla.f32   q7, q10, %e22[1]    \n"
                            "vmla.f32   q12, q10, %e18[1]   \n"
                            "vmla.f32   q13, q10, %e21[1]   \n"

                            "pld        [%7, #192]          \n"
                            "vld1.f32   {d16-d18}, [%7 :64] \n" // r2
                            "add        %7, #16             \n"

                            "vmla.f32   q6, q11, %f19[0]    \n"
                            "vmla.f32   q7, q11, %f22[0]    \n"
                            "vmla.f32   q12, q11, %f18[0]   \n"
                            "vmla.f32   q13, q11, %f21[0]   \n"

                            "vext.32    q10, q8, q9, #1     \n"

                            "vmla.f32   q6, q8, %e20[0]     \n"
                            "vmla.f32   q7, q8, %e23[0]     \n"
                            "vmla.f32   q12, q8, %e19[0]    \n"
                            "vmla.f32   q13, q8, %e22[0]    \n"

                            "vext.32    q11, q8, q9, #2     \n"

                            "vmla.f32   q6, q10, %e20[1]    \n"
                            "vmla.f32   q7, q10, %e23[1]    \n"
                            "vmla.f32   q12, q10, %e19[1]   \n"
                            "vmla.f32   q13, q10, %e22[1]   \n"

                            "pld        [%5, #192]          \n"
                            "vld1.f32   {d16-d18}, [%5 :64] \n" // r0
                            "add        %5, #16             \n"

                            "vmla.f32   q6, q11, %f20[0]    \n"
                            "vmla.f32   q7, q11, %f23[0]    \n"
                            "vmla.f32   q12, q11, %f19[0]   \n"
                            "vmla.f32   q13, q11, %f22[0]   \n"

                            "pld        [%8, #192]          \n"
                            "vld1.f32   {d28-d30}, [%8]     \n" // r3
                            "add        %8, #16             \n"

                            "vext.32    q10, q8, q9, #1     \n"

                            "vst1.f32   {d12-d13}, [%1 : 64]!\n"
                            "vst1.f32   {d14-d15}, [%2 : 64]!\n"

                            "vext.32    q11, q14, q15, #2   \n"

                            "vst1.f32   {d24-d25}, [%3]!    \n"
                            "vst1.f32   {d26-d27}, [%4]!    \n"

                            "subs       %0, #1              \n"
                            "bne        0b                  \n"

                            "sub        %5, #16             \n"
                            "sub        %8, #16             \n"
                            : "=r"(nn),       // %0
                            "=r"(outptr0),  // %1
                            "=r"(outptr1),  // %2
                            "=r"(outptr0n), // %3
                            "=r"(outptr1n), // %4
                            "=r"(r0),       // %5
                            "=r"(r1),       // %6
                            "=r"(r2),       // %7
                            "=r"(r3)        // %8
                            : "0"(nn),
                            "1"(outptr0),
                            "2"(outptr1),
                            "3"(outptr0n),
                            "4"(outptr1n),
                            "5"(r0),
                            "6"(r1),
                            "7"(r2),
                            "8"(r3),
                            "w"(_k00), // %18
                            "w"(_k03), // %19
                            "w"(_k06), // %20
                            "w"(_k10), // %21
                            "w"(_k13), // %22
                            "w"(_k16)  // %23
                            : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                    }
    #endif // __aarch64__
    #endif // __ARM_NEON
                    for (; remain > 0; remain--)
                    {
    #if __ARM_NEON
                        float32x4_t _r00 = vld1q_f32(r0);
                        float32x4_t _r10 = vld1q_f32(r1);
                        float32x4_t _r20 = vld1q_f32(r2);
                        float32x4_t _r30 = vld1q_f32(r3);

                        float32x4_t _sum0 = vmulq_f32(_r00, _k00);
                        float32x4_t _sum1 = vmulq_f32(_r00, _k10);
                        _sum0 = vmlaq_f32(_sum0, _r10, _k03);
                        _sum1 = vmlaq_f32(_sum1, _r10, _k13);
                        _sum0 = vmlaq_f32(_sum0, _r20, _k06);
                        _sum1 = vmlaq_f32(_sum1, _r20, _k16);

                        float32x4_t _sum0n = vmulq_f32(_r10, _k00);
                        float32x4_t _sum1n = vmulq_f32(_r10, _k10);
                        _sum0n = vmlaq_f32(_sum0n, _r20, _k03);
                        _sum1n = vmlaq_f32(_sum1n, _r20, _k13);
                        _sum0n = vmlaq_f32(_sum0n, _r30, _k06);
                        _sum1n = vmlaq_f32(_sum1n, _r30, _k16);

                        _sum0 = vsetq_lane_f32(*outptr0, _sum0, 3);
                        _sum1 = vsetq_lane_f32(*outptr1, _sum1, 3);
                        _sum0n = vsetq_lane_f32(*outptr0n, _sum0n, 3);
                        _sum1n = vsetq_lane_f32(*outptr1n, _sum1n, 3);
    #if __aarch64__
                        *outptr0 = vaddvq_f32(_sum0);
                        *outptr1 = vaddvq_f32(_sum1);
                        *outptr0n = vaddvq_f32(_sum0n);
                        *outptr1n = vaddvq_f32(_sum1n);
    #else
                        float32x2_t _ss0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                        float32x2_t _ss1 = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
                        float32x2_t _ss0n = vadd_f32(vget_low_f32(_sum0n), vget_high_f32(_sum0n));
                        float32x2_t _ss1n = vadd_f32(vget_low_f32(_sum1n), vget_high_f32(_sum1n));

                        float32x2_t _ss01 = vpadd_f32(_ss0, _ss1);
                        float32x2_t _ss01n = vpadd_f32(_ss0n, _ss1n);

                        *outptr0 = vget_lane_f32(_ss01, 0);
                        *outptr1 = vget_lane_f32(_ss01, 1);
                        *outptr0n = vget_lane_f32(_ss01n, 0);
                        *outptr1n = vget_lane_f32(_ss01n, 1);
    #endif // __aarch64__
    #else
                        float sum0 = 0.f;
                        float sum0n = 0.f;
                        float sum1 = 0.f;
                        float sum1n = 0.f;

                        sum0 += r0[0] * k0[0];
                        sum0 += r0[1] * k0[1];
                        sum0 += r0[2] * k0[2];
                        sum0 += r1[0] * k0[3];
                        sum0 += r1[1] * k0[4];
                        sum0 += r1[2] * k0[5];
                        sum0 += r2[0] * k0[6];
                        sum0 += r2[1] * k0[7];
                        sum0 += r2[2] * k0[8];

                        sum1 += r0[0] * k1[0];
                        sum1 += r0[1] * k1[1];
                        sum1 += r0[2] * k1[2];
                        sum1 += r1[0] * k1[3];
                        sum1 += r1[1] * k1[4];
                        sum1 += r1[2] * k1[5];
                        sum1 += r2[0] * k1[6];
                        sum1 += r2[1] * k1[7];
                        sum1 += r2[2] * k1[8];

                        sum0n += r1[0] * k0[0];
                        sum0n += r1[1] * k0[1];
                        sum0n += r1[2] * k0[2];
                        sum0n += r2[0] * k0[3];
                        sum0n += r2[1] * k0[4];
                        sum0n += r2[2] * k0[5];
                        sum0n += r3[0] * k0[6];
                        sum0n += r3[1] * k0[7];
                        sum0n += r3[2] * k0[8];

                        sum1n += r1[0] * k1[0];
                        sum1n += r1[1] * k1[1];
                        sum1n += r1[2] * k1[2];
                        sum1n += r2[0] * k1[3];
                        sum1n += r2[1] * k1[4];
                        sum1n += r2[2] * k1[5];
                        sum1n += r3[0] * k1[6];
                        sum1n += r3[1] * k1[7];
                        sum1n += r3[2] * k1[8];

                        *outptr0 += sum0;
                        *outptr1 += sum1;
                        *outptr0n += sum0n;
                        *outptr1n += sum1n;
    #endif // __ARM_NEON
                        r0++;
                        r1++;
                        r2++;
                        r3++;
                        outptr0++;
                        outptr1++;
                        outptr0n++;
                        outptr1n++;
                    }

                    r0 += 2 + w;
                    r1 += 2 + w;
                    r2 += 2 + w;
                    r3 += 2 + w;

                    outptr0 += outw;
                    outptr1 += outw;
                    outptr0n += outw;
                    outptr1n += outw;
                }

                for (; i < outh; i++)
                {
    #if __ARM_NEON
                    int nn = outw >> 2;
                    int remain = outw & 3;
    #else
                    int remain = outw;
    #endif // __ARM_NEON

    #if __ARM_NEON
    #if __aarch64__
                    if (nn > 0)
                    {
                        asm volatile(
                            "0:                                 \n"

                            "prfm   pldl1keep, [%3, #256]       \n"
                            "ld1    {v8.4s, v9.4s}, [%3]        \n" // r0
                            "add    %3, %3, #16                 \n"

                            "prfm   pldl1keep, [%1, #128]       \n"
                            "ld1    {v6.4s}, [%1]               \n" // _sum0

                            "prfm   pldl1keep, [%2, #128]       \n"
                            "ld1    {v7.4s}, [%2]               \n" // _sum1

                            "fmul   v14.4s, v8.4s, %12.s[0]     \n"
                            "fmul   v15.4s, v8.4s, %15.s[0]     \n"

                            "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                            "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                            "fmla   v6.4s, v10.4s, %12.s[1]     \n"
                            "fmla   v7.4s, v10.4s, %15.s[1]     \n"

                            "prfm   pldl1keep, [%4, #256]       \n"
                            "ld1    {v8.4s, v9.4s}, [%4]        \n" // r1
                            "add    %4, %4, #16                 \n"

                            "fmla   v14.4s, v11.4s, %12.s[2]    \n"
                            "fmla   v15.4s, v11.4s, %15.s[2]    \n"

                            "fmla   v6.4s, v8.4s, %13.s[0]      \n"
                            "fmla   v7.4s, v8.4s, %16.s[0]      \n"

                            "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                            "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                            "fmla   v14.4s, v10.4s, %13.s[1]    \n"
                            "fmla   v15.4s, v10.4s, %16.s[1]    \n"

                            "prfm   pldl1keep, [%5, #256]       \n"
                            "ld1    {v8.4s, v9.4s}, [%5]        \n" // r2
                            "add    %5, %5, #16                 \n"

                            "fmla   v6.4s, v11.4s, %13.s[2]     \n"
                            "fmla   v7.4s, v11.4s, %16.s[2]     \n"

                            "fmla   v14.4s, v8.4s, %14.s[0]     \n"
                            "fmla   v15.4s, v8.4s, %17.s[0]     \n"

                            "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                            "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                            "fmla   v6.4s, v10.4s, %14.s[1]     \n"
                            "fmla   v7.4s, v10.4s, %17.s[1]     \n"

                            "fmla   v14.4s, v11.4s, %14.s[2]    \n"
                            "fmla   v15.4s, v11.4s, %17.s[2]    \n"

                            "fadd   v6.4s, v6.4s, v14.4s        \n"
                            "fadd   v7.4s, v7.4s, v15.4s        \n"

                            "st1    {v6.4s}, [%1], #16          \n"
                            "st1    {v7.4s}, [%2], #16          \n"

                            "subs   %w0, %w0, #1                \n"
                            "bne    0b                          \n"

                            : "=r"(nn),      // %0
                            "=r"(outptr0), // %1
                            "=r"(outptr1), // %2
                            "=r"(r0),      // %3
                            "=r"(r1),      // %4
                            "=r"(r2)       // %5
                            : "0"(nn),
                            "1"(outptr0),
                            "2"(outptr1),
                            "3"(r0),
                            "4"(r1),
                            "5"(r2),
                            "w"(_k00), // %12
                            "w"(_k03), // %13
                            "w"(_k06), // %14
                            "w"(_k10), // %15
                            "w"(_k13), // %16
                            "w"(_k16)  // %17
                            : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                    }
    #else
                    if (nn > 0)
                    {
                        asm volatile(
                            "0:                             \n"

                            "pld        [%3, #192]          \n"
                            "vld1.f32   {d16-d18}, [%3]     \n" // r0
                            "add        %3, #16             \n"

                            "pld        [%1, #128]          \n"
                            "vld1.f32   {d12-d13}, [%1]     \n" // _sum0

                            "pld        [%2, #128]          \n"
                            "vld1.f32   {d14-d15}, [%2]     \n" // _sum1

                            "vmul.f32   q14, q8, %e12[0]    \n"
                            "vmul.f32   q15, q8, %e15[0]    \n"

                            "vext.32    q10, q8, q9, #1     \n"
                            "vext.32    q11, q8, q9, #2     \n"

                            "vmla.f32   q6, q10, %e12[1]    \n"
                            "vmla.f32   q7, q10, %e15[1]    \n"

                            "pld        [%4, #192]          \n"
                            "vld1.f32   {d16-d18}, [%4]     \n" // r1
                            "add        %4, #16             \n"

                            "vmla.f32   q14, q11, %f12[0]   \n"
                            "vmla.f32   q15, q11, %f15[0]   \n"

                            "vmla.f32   q6, q8, %e13[0]     \n"
                            "vmla.f32   q7, q8, %e16[0]     \n"

                            "vext.32    q10, q8, q9, #1     \n"
                            "vext.32    q11, q8, q9, #2     \n"

                            "vmla.f32   q14, q10, %e13[1]   \n"
                            "vmla.f32   q15, q10, %e16[1]   \n"

                            "pld        [%5, #192]          \n"
                            "vld1.f32   {d16-d18}, [%5]     \n" // r2
                            "add        %5, #16             \n"

                            "vmla.f32   q6, q11, %f13[0]    \n"
                            "vmla.f32   q7, q11, %f16[0]    \n"

                            "vmla.f32   q14, q8, %e14[0]    \n"
                            "vmla.f32   q15, q8, %e17[0]    \n"

                            "vext.32    q10, q8, q9, #1     \n"
                            "vext.32    q11, q8, q9, #2     \n"

                            "vmla.f32   q6, q10, %e14[1]    \n"
                            "vmla.f32   q7, q10, %e17[1]    \n"

                            "vmla.f32   q14, q11, %f14[0]   \n"
                            "vmla.f32   q15, q11, %f17[0]   \n"

                            "vadd.f32   q6, q6, q14         \n"
                            "vadd.f32   q7, q7, q15         \n"

                            "vst1.f32   {d12-d13}, [%1]!    \n"

                            "vst1.f32   {d14-d15}, [%2]!    \n"

                            "subs       %0, #1              \n"
                            "bne        0b                  \n"

                            : "=r"(nn),      // %0
                            "=r"(outptr0), // %1
                            "=r"(outptr1), // %2
                            "=r"(r0),      // %3
                            "=r"(r1),      // %4
                            "=r"(r2)       // %5
                            : "0"(nn),
                            "1"(outptr0),
                            "2"(outptr1),
                            "3"(r0),
                            "4"(r1),
                            "5"(r2),
                            "w"(_k00), // %12
                            "w"(_k03), // %13
                            "w"(_k06), // %14
                            "w"(_k10), // %15
                            "w"(_k13), // %16
                            "w"(_k16)  // %17
                            : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                    }
    #endif // __aarch64__
    #endif // __ARM_NEON
                    for (; remain > 0; remain--)
                    {
    #if __ARM_NEON
                        float32x4_t _r00 = vld1q_f32(r0);
                        float32x4_t _r10 = vld1q_f32(r1);
                        float32x4_t _r20 = vld1q_f32(r2);

                        float32x4_t _sum0 = vmulq_f32(_r00, _k00);
                        float32x4_t _sum1 = vmulq_f32(_r00, _k10);
                        _sum0 = vmlaq_f32(_sum0, _r10, _k03);
                        _sum1 = vmlaq_f32(_sum1, _r10, _k13);
                        _sum0 = vmlaq_f32(_sum0, _r20, _k06);
                        _sum1 = vmlaq_f32(_sum1, _r20, _k16);

                        _sum0 = vsetq_lane_f32(*outptr0, _sum0, 3);
                        _sum1 = vsetq_lane_f32(*outptr1, _sum1, 3);
    #if __aarch64__
                        *outptr0 = vaddvq_f32(_sum0);
                        *outptr1 = vaddvq_f32(_sum1);
    #else
                        float32x2_t _ss0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                        float32x2_t _ss1 = vadd_f32(vget_low_f32(_sum1), vget_high_f32(_sum1));
                        float32x2_t _ss01 = vpadd_f32(_ss0, _ss1);

                        *outptr0 = vget_lane_f32(_ss01, 0);
                        *outptr1 = vget_lane_f32(_ss01, 1);
    #endif // __aarch64__
    #else
                        float sum0 = 0.f;
                        float sum1 = 0.f;

                        sum0 += r0[0] * k0[0];
                        sum0 += r0[1] * k0[1];
                        sum0 += r0[2] * k0[2];
                        sum0 += r1[0] * k0[3];
                        sum0 += r1[1] * k0[4];
                        sum0 += r1[2] * k0[5];
                        sum0 += r2[0] * k0[6];
                        sum0 += r2[1] * k0[7];
                        sum0 += r2[2] * k0[8];

                        sum1 += r0[0] * k1[0];
                        sum1 += r0[1] * k1[1];
                        sum1 += r0[2] * k1[2];
                        sum1 += r1[0] * k1[3];
                        sum1 += r1[1] * k1[4];
                        sum1 += r1[2] * k1[5];
                        sum1 += r2[0] * k1[6];
                        sum1 += r2[1] * k1[7];
                        sum1 += r2[2] * k1[8];

                        *outptr0 += sum0;
                        *outptr1 += sum1;
    #endif // __ARM_NEON
                        r0++;
                        r1++;
                        r2++;
                        outptr0++;
                        outptr1++;
                    }

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                }

                k0 += 9;
                k1 += 9;
            }
        }
    });

    otter::parallel_for(remain_outch_start, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            auto out = output_t[p];

            const float bias0 = bias ? bias[p] : 0.f;

            out.fill_(bias0);

            const float* kernel0 = kernel + p * inch * 9;

            for (int q = 0; q < inch; q++)
            {
                float* outptr = out.data_ptr<float>();
                float* outptr2 = outptr + outw;

                const float* img0 = input_a[q].data();

                const float* r0 = img0;
                const float* r1 = img0 + w;
                const float* r2 = img0 + w * 2;
                const float* r3 = img0 + w * 3;

    #if __ARM_NEON
                float32x4_t _k0123 = vld1q_f32(kernel0);
                float32x4_t _k3456 = vld1q_f32(kernel0 + 3);
                float32x4_t _k6789 = vld1q_f32(kernel0 + 6);
    #else
                const float* k0 = kernel0;
                const float* k1 = kernel0 + 3;
                const float* k2 = kernel0 + 6;
    #endif // __ARM_NEON

                int i = 0;

                for (; i + 1 < outh; i += 2)
                {
    #if __ARM_NEON
                    int nn = outw >> 2;
                    int remain = outw & 3;
    #else
                    int remain = outw;
    #endif // __ARM_NEON

    #if __ARM_NEON
    #if __aarch64__
                    if (nn > 0)
                    {
                        asm volatile(
                            "prfm   pldl1keep, [%3, #256]       \n"
                            "ld1    {v9.4s, v10.4s}, [%3]       \n" // r0
                            "add    %3, %3, #16                 \n"

                            "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                            "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                            "0:                                 \n"

                            "prfm   pldl1keep, [%1, #128]       \n"
                            "ld1    {v7.4s}, [%1]               \n" // _sum

                            "fmla   v7.4s, v9.4s, %14.s[0]      \n"
                            "fmul   v6.4s, v11.4s, %14.s[1]     \n"
                            "fmul   v13.4s, v12.4s, %14.s[2]    \n"

                            "prfm   pldl1keep, [%4, #256]       \n"
                            "ld1    {v9.4s, v10.4s}, [%4]       \n" // r1
                            "add    %4, %4, #16                 \n"

                            "fmla   v7.4s, v9.4s, %15.s[0]      \n"

                            "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                            "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                            "fmla   v6.4s, v11.4s, %15.s[1]     \n"
                            "fmla   v13.4s, v12.4s, %15.s[2]    \n"

                            "prfm   pldl1keep, [%2, #128]       \n"
                            "ld1    {v8.4s}, [%2]               \n" // _sum2

                            "fmla   v8.4s, v9.4s, %14.s[0]      \n"
                            "fmul   v14.4s, v11.4s, %14.s[1]    \n"
                            "fmul   v15.4s, v12.4s, %14.s[2]    \n"

                            "prfm   pldl1keep, [%5, #256]       \n"
                            "ld1    {v9.4s, v10.4s}, [%5]       \n" // r2
                            "add    %5, %5, #16                 \n"

                            "fmla   v7.4s, v9.4s, %16.s[0]      \n"

                            "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                            "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                            "fmla   v6.4s, v11.4s, %16.s[1]     \n"
                            "fmla   v13.4s, v12.4s, %16.s[2]    \n"

                            "fmla   v8.4s, v9.4s, %15.s[0]      \n"
                            "fmla   v14.4s, v11.4s, %15.s[1]    \n"
                            "fmla   v15.4s, v12.4s, %15.s[2]    \n"

                            "prfm   pldl1keep, [%6, #256]       \n"
                            "ld1    {v9.4s, v10.4s}, [%6]       \n" // r3
                            "add    %6, %6, #16                 \n"

                            "fmla   v8.4s, v9.4s, %16.s[0]      \n"

                            "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                            "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                            "fmla   v14.4s, v11.4s, %16.s[1]    \n"
                            "fmla   v15.4s, v12.4s, %16.s[2]    \n"

                            "fadd   v7.4s, v7.4s, v6.4s         \n"

                            "prfm   pldl1keep, [%3, #256]       \n"
                            "ld1    {v9.4s, v10.4s}, [%3]       \n" // r0

                            "fadd   v8.4s, v8.4s, v14.4s        \n"
                            "fadd   v7.4s, v7.4s, v13.4s        \n"
                            "fadd   v8.4s, v8.4s, v15.4s        \n"

                            "ext    v11.16b, v9.16b, v10.16b, #4 \n"
                            "ext    v12.16b, v9.16b, v10.16b, #8 \n"

                            "add    %3, %3, #16                 \n"

                            "st1    {v7.4s}, [%1], #16          \n"
                            "st1    {v8.4s}, [%2], #16          \n"

                            "subs   %w0, %w0, #1                \n"
                            "bne    0b                          \n"

                            "sub    %3, %3, #16                 \n"
                            : "=r"(nn),      // %0
                            "=r"(outptr),  // %1
                            "=r"(outptr2), // %2
                            "=r"(r0),      // %3
                            "=r"(r1),      // %4
                            "=r"(r2),      // %5
                            "=r"(r3)       // %6
                            : "0"(nn),
                            "1"(outptr),
                            "2"(outptr2),
                            "3"(r0),
                            "4"(r1),
                            "5"(r2),
                            "6"(r3),
                            "w"(_k0123), // %14
                            "w"(_k3456), // %15
                            "w"(_k6789)  // %16
                            : "cc", "memory", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                    }
    #else
                    if (nn > 0)
                    {
                        asm volatile(
                            "pld        [%3, #192]          \n"
                            "vld1.f32   {d18-d20}, [%3 :64] \n" // r0
                            "add        %3, #16             \n"

                            "vext.32    q11, q9, q10, #1    \n"
                            "vext.32    q12, q9, q10, #2    \n"

                            "0:                             \n"

                            "pld        [%1, #128]          \n"
                            "vld1.f32   {d14-d15}, [%1 :64] \n" // _sum

                            "vmla.f32   q7, q9, %e14[0]     \n"
                            "vmul.f32   q6, q11, %e14[1]    \n"
                            "vmul.f32   q13, q12, %f14[0]   \n"

                            "pld        [%4, #192]          \n"
                            "vld1.f32   {d18-d20}, [%4]     \n" // r1
                            "add        %4, #16             \n"

                            "vmla.f32   q7, q9, %e15[0]     \n"

                            "vext.32    q11, q9, q10, #1    \n"
                            "vext.32    q12, q9, q10, #2    \n"

                            "vmla.f32   q6, q11, %e15[1]    \n"
                            "vmla.f32   q13, q12, %f15[0]   \n"

                            "pld        [%2, #128]          \n"
                            "vld1.f32   {d16-d17}, [%2]     \n" // _sum2

                            "vmla.f32   q8, q9, %e14[0]     \n"
                            "vmul.f32   q14, q11, %e14[1]   \n"
                            "vmul.f32   q15, q12, %f14[0]   \n"

                            "pld        [%5, #192]          \n"
                            "vld1.f32   {d18-d20}, [%5 :64] \n" // r2
                            "add        %5, #16             \n"

                            "vmla.f32   q7, q9, %e16[0]     \n"

                            "vext.32    q11, q9, q10, #1    \n"
                            "vext.32    q12, q9, q10, #2    \n"

                            "vmla.f32   q6, q11, %e16[1]    \n"
                            "vmla.f32   q13, q12, %f16[0]   \n"

                            "vmla.f32   q8, q9, %e15[0]     \n"
                            "vmla.f32   q14, q11, %e15[1]   \n"
                            "vmla.f32   q15, q12, %f15[0]   \n"

                            "pld        [%6, #192]          \n"
                            "vld1.f32   {d18-d20}, [%6]     \n" // r3
                            "add        %6, #16             \n"

                            "vmla.f32   q8, q9, %e16[0]     \n"

                            "vext.32    q11, q9, q10, #1    \n"
                            "vext.32    q12, q9, q10, #2    \n"

                            "vmla.f32   q14, q11, %e16[1]   \n"
                            "vmla.f32   q15, q12, %f16[0]   \n"

                            "vadd.f32   q7, q7, q6          \n"

                            "pld        [%3, #192]          \n"
                            "vld1.f32   {d18-d20}, [%3 :64] \n" // r0

                            "vadd.f32   q8, q8, q14         \n"
                            "vadd.f32   q7, q7, q13         \n"
                            "vadd.f32   q8, q8, q15         \n"

                            "vext.32    q11, q9, q10, #1    \n"
                            "vext.32    q12, q9, q10, #2    \n"

                            "add        %3, #16             \n"

                            "vst1.f32   {d14-d15}, [%1]!    \n"
                            "vst1.f32   {d16-d17}, [%2]!    \n"

                            "subs       %0, #1              \n"
                            "bne        0b                  \n"

                            "sub        %3, #16             \n"
                            : "=r"(nn),      // %0
                            "=r"(outptr),  // %1
                            "=r"(outptr2), // %2
                            "=r"(r0),      // %3
                            "=r"(r1),      // %4
                            "=r"(r2),      // %5
                            "=r"(r3)       // %6
                            : "0"(nn),
                            "1"(outptr),
                            "2"(outptr2),
                            "3"(r0),
                            "4"(r1),
                            "5"(r2),
                            "6"(r3),
                            "w"(_k0123), // %14
                            "w"(_k3456), // %15
                            "w"(_k6789)  // %16
                            : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                    }
    #endif // __aarch64__
    #endif // __ARM_NEON
                    for (; remain > 0; remain--)
                    {
    #if __ARM_NEON
                        float32x4_t _r00 = vld1q_f32(r0);
                        float32x4_t _r10 = vld1q_f32(r1);
                        float32x4_t _r20 = vld1q_f32(r2);
                        float32x4_t _r30 = vld1q_f32(r3);

                        float32x4_t _sum = vmulq_f32(_r00, _k0123);
                        _sum = vmlaq_f32(_sum, _r10, _k3456);
                        _sum = vmlaq_f32(_sum, _r20, _k6789);

                        float32x4_t _sum2 = vmulq_f32(_r10, _k0123);
                        _sum2 = vmlaq_f32(_sum2, _r20, _k3456);
                        _sum2 = vmlaq_f32(_sum2, _r30, _k6789);

                        _sum = vsetq_lane_f32(*outptr, _sum, 3);
                        _sum2 = vsetq_lane_f32(*outptr2, _sum2, 3);

    #if __aarch64__
                        *outptr = vaddvq_f32(_sum);
                        *outptr2 = vaddvq_f32(_sum2);
    #else
                        float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                        float32x2_t _ss2 = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));

                        float32x2_t _sss2 = vpadd_f32(_ss, _ss2);

                        *outptr = vget_lane_f32(_sss2, 0);
                        *outptr2 = vget_lane_f32(_sss2, 1);
    #endif // __aarch64__
    #else
                        float sum = 0;
                        float sum2 = 0;

                        sum += r0[0] * k0[0];
                        sum += r0[1] * k0[1];
                        sum += r0[2] * k0[2];
                        sum += r1[0] * k1[0];
                        sum += r1[1] * k1[1];
                        sum += r1[2] * k1[2];
                        sum += r2[0] * k2[0];
                        sum += r2[1] * k2[1];
                        sum += r2[2] * k2[2];

                        sum2 += r1[0] * k0[0];
                        sum2 += r1[1] * k0[1];
                        sum2 += r1[2] * k0[2];
                        sum2 += r2[0] * k1[0];
                        sum2 += r2[1] * k1[1];
                        sum2 += r2[2] * k1[2];
                        sum2 += r3[0] * k2[0];
                        sum2 += r3[1] * k2[1];
                        sum2 += r3[2] * k2[2];

                        *outptr += sum;
                        *outptr2 += sum2;
    #endif
                        r0++;
                        r1++;
                        r2++;
                        r3++;
                        outptr++;
                        outptr2++;
                    }

                    r0 += 2 + w;
                    r1 += 2 + w;
                    r2 += 2 + w;
                    r3 += 2 + w;

                    outptr += outw;
                    outptr2 += outw;
                }

                for (; i < outh; i++)
                {
    #if __ARM_NEON
                    int nn = outw >> 2;
                    int remain = outw & 3;
    #else
                    int remain = outw;
    #endif // __ARM_NEON

    #if __ARM_NEON
    #if __aarch64__
                    if (nn > 0)
                    {
                        asm volatile(
                            "prfm   pldl1keep, [%2, #256]       \n"
                            "ld1    {v8.4s, v9.4s}, [%2]        \n" // r0
                            "add    %2, %2, #16                 \n"

                            "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                            "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                            "0:                                 \n"

                            "prfm   pldl1keep, [%1, #128]       \n"
                            "ld1    {v7.4s}, [%1]               \n" // _sum

                            "fmla   v7.4s, v8.4s, %10.s[0]      \n"
                            "fmul   v13.4s, v10.4s, %10.s[1]    \n"
                            "fmul   v14.4s, v11.4s, %10.s[2]    \n"

                            "prfm   pldl1keep, [%3, #256]       \n"
                            "ld1    {v8.4s, v9.4s}, [%3]        \n" // r1
                            "add    %3, %3, #16                 \n"

                            "fmla   v7.4s, v8.4s, %11.s[0]      \n"

                            "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                            "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                            "fmla   v13.4s, v10.4s, %11.s[1]    \n"
                            "fmla   v14.4s, v11.4s, %11.s[2]    \n"

                            "prfm   pldl1keep, [%4, #256]       \n"
                            "ld1    {v8.4s, v9.4s}, [%4]        \n" // r2
                            "add    %4, %4, #16                 \n"

                            "fmla   v7.4s, v8.4s, %12.s[0]      \n"

                            "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                            "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                            "fmla   v13.4s, v10.4s, %12.s[1]    \n"
                            "fmla   v14.4s, v11.4s, %12.s[2]    \n"

                            "prfm   pldl1keep, [%2, #256]       \n"
                            "ld1    {v8.4s, v9.4s}, [%2]        \n" // r0
                            "add    %2, %2, #16                 \n"

                            "fadd   v7.4s, v7.4s, v13.4s        \n"
                            "fadd   v7.4s, v7.4s, v14.4s        \n"

                            "ext    v10.16b, v8.16b, v9.16b, #4 \n"
                            "ext    v11.16b, v8.16b, v9.16b, #8 \n"

                            "st1    {v7.4s}, [%1], #16          \n"

                            "subs   %w0, %w0, #1                \n"
                            "bne    0b                          \n"

                            "sub    %2, %2, #16                 \n"
                            : "=r"(nn),     // %0
                            "=r"(outptr), // %1
                            "=r"(r0),     // %2
                            "=r"(r1),     // %3
                            "=r"(r2)      // %4
                            : "0"(nn),
                            "1"(outptr),
                            "2"(r0),
                            "3"(r1),
                            "4"(r2),
                            "w"(_k0123), // %10
                            "w"(_k3456), // %11
                            "w"(_k6789)  // %12
                            : "cc", "memory", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                    }
    #else
                    if (nn > 0)
                    {
                        asm volatile(
                            "pld        [%2, #192]          \n"
                            "vld1.f32   {d16-d18}, [%2]     \n" // r0
                            "add        %2, #16             \n"

                            "vext.32    q10, q8, q9, #1     \n"
                            "vext.32    q11, q8, q9, #2     \n"

                            "0:                             \n"

                            "pld        [%1, #128]          \n"
                            "vld1.f32   {d14-d15}, [%1]     \n" // _sum

                            "vmla.f32   q7, q8, %e10[0]     \n"
                            "vmul.f32   q13, q10, %e10[1]   \n"
                            "vmul.f32   q14, q11, %f10[0]   \n"

                            "pld        [%3, #192]          \n"
                            "vld1.f32   {d16-d18}, [%3]     \n" // r1
                            "add        %3, #16             \n"

                            "vmla.f32   q7, q8, %e11[0]     \n"

                            "vext.32    q10, q8, q9, #1     \n"
                            "vext.32    q11, q8, q9, #2     \n"

                            "vmla.f32   q13, q10, %e11[1]   \n"
                            "vmla.f32   q14, q11, %f11[0]   \n"

                            "pld        [%4, #192]          \n"
                            "vld1.f32   {d16-d18}, [%4]     \n" // r2
                            "add        %4, #16             \n"

                            "vmla.f32   q7, q8, %e12[0]     \n"

                            "vext.32    q10, q8, q9, #1     \n"
                            "vext.32    q11, q8, q9, #2     \n"

                            "vmla.f32   q13, q10, %e12[1]   \n"
                            "vmla.f32   q14, q11, %f12[0]   \n"

                            "pld        [%2, #192]          \n"
                            "vld1.f32   {d16-d18}, [%2]     \n" // r0
                            "add        %2, #16             \n"

                            "vadd.f32   q7, q7, q13         \n"
                            "vadd.f32   q7, q7, q14         \n"

                            "vext.32    q10, q8, q9, #1     \n"
                            "vext.32    q11, q8, q9, #2     \n"

                            "vst1.f32   {d14-d15}, [%1]!    \n"

                            "subs       %0, #1              \n"
                            "bne        0b                  \n"

                            "sub        %2, #16             \n"
                            : "=r"(nn),     // %0
                            "=r"(outptr), // %1
                            "=r"(r0),     // %2
                            "=r"(r1),     // %3
                            "=r"(r2)      // %4
                            : "0"(nn),
                            "1"(outptr),
                            "2"(r0),
                            "3"(r1),
                            "4"(r2),
                            "w"(_k0123), // %10
                            "w"(_k3456), // %11
                            "w"(_k6789)  // %12
                            : "cc", "memory", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                    }
    #endif // __aarch64__
    #endif // __ARM_NEON
                    for (; remain > 0; remain--)
                    {
    #if __ARM_NEON
                        float32x4_t _r00 = vld1q_f32(r0);
                        float32x4_t _r10 = vld1q_f32(r1);
                        float32x4_t _r20 = vld1q_f32(r2);

                        float32x4_t _sum = vmulq_f32(_r00, _k0123);
                        _sum = vmlaq_f32(_sum, _r10, _k3456);
                        _sum = vmlaq_f32(_sum, _r20, _k6789);

                        _sum = vsetq_lane_f32(*outptr, _sum, 3);

    #if __aarch64__
                        *outptr = vaddvq_f32(_sum);
    #else
                        float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                        _ss = vpadd_f32(_ss, _ss);

                        *outptr = vget_lane_f32(_ss, 0);
    #endif // __aarch64__
    #else
                        float sum = 0;

                        sum += r0[0] * k0[0];
                        sum += r0[1] * k0[1];
                        sum += r0[2] * k0[2];
                        sum += r1[0] * k1[0];
                        sum += r1[1] * k1[1];
                        sum += r1[2] * k1[2];
                        sum += r2[0] * k2[0];
                        sum += r2[1] * k2[1];
                        sum += r2[2] * k2[2];

                        *outptr += sum;
    #endif
                        r0++;
                        r1++;
                        r2++;
                        outptr++;
                    }

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                }

                kernel0 += 9;
            }
        }
    });
    
    return output;
}

Tensor conv2d_3x3s1_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, otter::ScalarType::Float);
    
    return conv2d_3x3s1_neon_out(self, weight, bias, kernel_size, stride, padding, output);
}

#ifdef __ARM_NEON__
void conv3x3s1_winograd64_transform_kernel_neon5(const Tensor& kernel_, Tensor& kernel_tf, int64_t input_channels, int64_t output_channels) {
    const int64_t kernelSize = 3 * 3;
    
    float* kernel = kernel_.view({output_channels, input_channels, kernelSize}).data_ptr<float>();
    
    kernel_tf = otter::empty({output_channels, input_channels, 8 * 8}, otter::ScalarType::Float);
    
    const float ktm[8][3] = {
        {1.0f, 0.0f, 0.0f},
        {-2.0f / 9, -2.0f / 9, -2.0f / 9},
        {-2.0f / 9, 2.0f / 9, -2.0f / 9},
        {1.0f / 90, 1.0f / 45, 2.0f / 45},
        {1.0f / 90, -1.0f / 45, 2.0f / 45},
        {1.0f / 45, 1.0f / 90, 1.0f / 180},
        {1.0f / 45, -1.0f / 90, 1.0f / 180},
        {0.0f, 0.0f, 1.0f}
    };
    
    auto kernel_tf_a = kernel_tf.accessor<float, 3>();

    otter::parallel_for(0, output_channels, 0, [&](int64_t begin, int64_t end) {
        for (int p = 0; p < output_channels; p++)
        {
            for (int q = 0; q < input_channels; q++)
            {
                const float* kernel0 = (const float*)kernel + p * input_channels * 9 + q * 9;
                float* kernel_tm0 = kernel_tf_a[p][q].data();

                // transform kernel, transposed
                const float* k0 = kernel0;
                const float* k1 = kernel0 + 3;
                const float* k2 = kernel0 + 6;

                // h
                float tmp[8][3];
                for (int i = 0; i < 8; i++)
                {
                    tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                    tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                    tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
                }

                // v
                for (int j = 0; j < 8; j++)
                {
                    float* tmpp = &tmp[j][0];

                    for (int i = 0; i < 8; i++)
                    {
                        kernel_tm0[j * 8 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                    }
                }
            }
        }
    });

    // optimized layout for winograd5
    // interleave weights
//     Mat kernel_tm2(8*8, inch, outch);
//     Mat kernel_tm2(inch, 64, outch);
#if __ARM_NEON && __aarch64__
    Tensor kernel_tf2 = otter::empty({output_channels / 8 + (output_channels % 8) / 4 + output_channels % 4, 64, 8 * 4 * (input_channels / 4) + 8 * (input_channels % 4)}, otter::ScalarType::Float);
#else
    Tensor kernel_tf2 = otter::empty({output_channels / 4 + output_channels % 4, 64, 4 * 4 * (input_channels / 4) + 4 * (input_channels % 4)}, otter::ScalarType::Float);
#endif
    
    auto kernel_tf2_a = kernel_tf2.accessor<float, 3>();

    int p = 0;
#if __aarch64__
    for (; p + 7 < output_channels; p += 8)
    {
        const auto kernel0_tm = kernel_tf_a[p];
        const auto kernel1_tm = kernel_tf_a[p + 1];
        const auto kernel2_tm = kernel_tf_a[p + 2];
        const auto kernel3_tm = kernel_tf_a[p + 3];
        const auto kernel4_tm = kernel_tf_a[p + 4];
        const auto kernel5_tm = kernel_tf_a[p + 5];
        const auto kernel6_tm = kernel_tf_a[p + 6];
        const auto kernel7_tm = kernel_tf_a[p + 7];

        auto ktm2 = kernel_tf2_a[p / 8];

        for (int r = 0; r < 64; r++)
        {
            float* ktm2p = ktm2[r].data();

            for (int q = 0; q < input_channels; q++)
            {
                const float* ktm0_0 = kernel0_tm[q].data();
                const float* ktm1_0 = kernel1_tm[q].data();
                const float* ktm2_0 = kernel2_tm[q].data();
                const float* ktm3_0 = kernel3_tm[q].data();
                const float* ktm4_0 = kernel4_tm[q].data();
                const float* ktm5_0 = kernel5_tm[q].data();
                const float* ktm6_0 = kernel6_tm[q].data();
                const float* ktm7_0 = kernel7_tm[q].data();

                ktm2p[0] = ktm0_0[r];
                ktm2p[1] = ktm1_0[r];
                ktm2p[2] = ktm2_0[r];
                ktm2p[3] = ktm3_0[r];
                ktm2p[4] = ktm4_0[r];
                ktm2p[5] = ktm5_0[r];
                ktm2p[6] = ktm6_0[r];
                ktm2p[7] = ktm7_0[r];

                ktm2p += 8;
            }
        }
    }
#endif // __aarch64__
    for (; p + 3 < output_channels; p += 4)
    {
        const auto kernel0_tm = kernel_tf_a[p];
        const auto kernel1_tm = kernel_tf_a[p + 1];
        const auto kernel2_tm = kernel_tf_a[p + 2];
        const auto kernel3_tm = kernel_tf_a[p + 3];

#if __ARM_NEON && __aarch64__
        auto ktm2 = kernel_tf2_a[p / 8 + (p % 8) / 4];
#else
        auto ktm2 = kernel_tf2_a[p / 4];
#endif

        for (int r = 0; r < 64; r++)
        {
            float* ktm2p = ktm2[r].data();

            for (int q = 0; q < input_channels; q++)
            {
                const float* ktm0_0 = kernel0_tm[q].data();
                const float* ktm1_0 = kernel1_tm[q].data();
                const float* ktm2_0 = kernel2_tm[q].data();
                const float* ktm3_0 = kernel3_tm[q].data();

                ktm2p[0] = ktm0_0[r];
                ktm2p[1] = ktm1_0[r];
                ktm2p[2] = ktm2_0[r];
                ktm2p[3] = ktm3_0[r];

                ktm2p += 4;
            }
        }
    }
    for (; p < output_channels; p++)
    {
        const auto kernel0_tm = kernel_tf_a[p];

#if __ARM_NEON && __aarch64__
        auto ktm2 = kernel_tf2_a[p / 8 + (p % 8) / 4 + p % 4];
#else
        auto ktm2 = kernel_tf2_a[p / 4 + p % 4];
#endif

        for (int r = 0; r < 64; r++)
        {
            float* ktm2p = ktm2[r].data();

            for (int q = 0; q < input_channels; q++)
            {
                const float* ktm0_0 = kernel0_tm[q].data();

                ktm2p[0] = ktm0_0[r];

                ktm2p += 1;
            }
        }
    }

    kernel_tf = kernel_tf2;
}
#else
void conv3x3s1_winograd64_transform_kernel_neon5(const Tensor& /*kernel_*/, Tensor& /*kernel_tf*/, int64_t /*input_channels*/, int64_t /*output_channels*/) {}
#endif

Tensor& conv2d_3x3s1_winograd64_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    // padding: [height, width]
    // constnat_pad: [left, right, top, bottom]
    auto output_shape = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_(output_shape);
    
    int origin_w = (int)self.size(3) + 2 * (int)padding[1];
    int origin_h = (int)self.size(2) + 2 * (int)padding[0];
    
    int w = origin_w;
    int h = origin_h;
    int inch  = (int)self.size(1);
    
    int outw  = (int)output_shape[3];
    int outh  = (int)output_shape[2];
    int outch = (int)output_shape[1];
    
    outw = (outw + 5) / 6 * 6;
    outh = (outh + 5) / 6 * 6;
    
    w = outw + 2;
    h = outh + 2;
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1] + w - origin_w, padding[0], padding[0] + h - origin_h}, 0);
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        otter::conv3x3s1_winograd64_transform_kernel_neon5(weight, kernel_tf, inch, outch);
    auto kernel_tf_a = kernel_tf.accessor<float, 3>();
    
    const float* bias = (bias_.defined()) ? bias_.data_ptr<float>() : nullptr;
    
    auto input_a = input.accessor<float, 4>()[0];
    
    Tensor input_tf;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        const int tiles = w_tm / 8 * h_tm / 8;
        input_tf = otter::empty({inch, 64 * tiles, 1}, otter::ScalarType::Float);
        auto input_tf_a = input_tf.accessor<float, 3>();

        //         const float itm[8][8] = {
        //             {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
        //
        //             {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
        //             {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
        //
        //             {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
        //             {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
        //
        //             {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
        //             {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
        //
        //             {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
        //         };

        // 0 = r00 - r06 + (r04 - r02) * 5.25
        // 7 = r07 - r01 + (r03 - r05) * 5.25

        // 1 = (r02 + r06 - r04 * 4.25) + (r01 - r03 * 4.25 + r05)
        // 2 = (r02 + r06 - r04 * 4.25) - (r01 - r03 * 4.25 + r05)

        // 3 = (r06 + r02 * 0.25 - r04 * 1.25) + (r01 * 0.5 - r03 * 2.5 + r05 * 2)
        // 4 = (r06 + r02 * 0.25 - r04 * 1.25) - (r01 * 0.5 - r03 * 2.5 + r05 * 2)

        // reuse r04 * 1.25
        // reuse r03 * 2.5
        // 5 = (r06 + (r02 - r04 * 1.25) * 4) + (r01 * 2 - r03 * 2.5 + r05 * 0.5)
        // 6 = (r06 + (r02 - r04 * 1.25) * 4) - (r01 * 2 - r03 * 2.5 + r05 * 0.5)

#if __ARM_NEON
        const float coeff[8] = {
            0.25f, 0.5f, -1.25f, 2.f,
            -2.5f, 4.f, 4.25f, 5.25f
        };
        float32x4_t _coeff0 = vld1q_f32(coeff);
        float32x4_t _coeff1 = vld1q_f32(coeff + 4);
#endif // __ARM_NEON

        otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
            for (const auto q : otter::irange(begin, end)) {
                const auto img0 = input_a[q];
                auto img0_tm = input_tf_a[q];
                int img0_tm_w = input_tf.size(2);

                float tmp[8][8];

                // tile
                for (int i = 0; i < h_tm / 8; i++)
                {
                    for (int j = 0; j < w_tm / 8; j++)
                    {
    #if __ARM_NEON
                        const float* r0 = img0[i * 6].data() + j * 6;
                        const float* r1 = r0 + w;
                        const float* r2 = r0 + w * 2;
                        const float* r3 = r0 + w * 3;

    #if __aarch64__
                        for (int m = 0; m + 3 < 8; m += 4)
                        {
                            float32x4_t _r0_0123 = vld1q_f32(r0);
                            float32x4_t _r0_4567 = vld1q_f32(r0 + 4);
                            float32x4_t _r1_0123 = vld1q_f32(r1);
                            float32x4_t _r1_4567 = vld1q_f32(r1 + 4);
                            float32x4_t _r2_0123 = vld1q_f32(r2);
                            float32x4_t _r2_4567 = vld1q_f32(r2 + 4);
                            float32x4_t _r3_0123 = vld1q_f32(r3);
                            float32x4_t _r3_4567 = vld1q_f32(r3 + 4);

                            float32x4x2_t _r01_00221133 = vtrnq_f32(_r0_0123, _r1_0123);
                            float32x4x2_t _r01_44665577 = vtrnq_f32(_r0_4567, _r1_4567);
                            float32x4x2_t _r23_00221133 = vtrnq_f32(_r2_0123, _r3_0123);
                            float32x4x2_t _r23_44665577 = vtrnq_f32(_r2_4567, _r3_4567);

                            // no vswp intrinsic  :(
                            float32x4_t _r_00 = vcombine_f32(vget_low_f32(_r01_00221133.val[0]), vget_low_f32(_r23_00221133.val[0]));
                            float32x4_t _r_11 = vcombine_f32(vget_low_f32(_r01_00221133.val[1]), vget_low_f32(_r23_00221133.val[1]));
                            float32x4_t _r_22 = vcombine_f32(vget_high_f32(_r01_00221133.val[0]), vget_high_f32(_r23_00221133.val[0]));
                            float32x4_t _r_33 = vcombine_f32(vget_high_f32(_r01_00221133.val[1]), vget_high_f32(_r23_00221133.val[1]));
                            float32x4_t _r_44 = vcombine_f32(vget_low_f32(_r01_44665577.val[0]), vget_low_f32(_r23_44665577.val[0]));
                            float32x4_t _r_55 = vcombine_f32(vget_low_f32(_r01_44665577.val[1]), vget_low_f32(_r23_44665577.val[1]));
                            float32x4_t _r_66 = vcombine_f32(vget_high_f32(_r01_44665577.val[0]), vget_high_f32(_r23_44665577.val[0]));
                            float32x4_t _r_77 = vcombine_f32(vget_high_f32(_r01_44665577.val[1]), vget_high_f32(_r23_44665577.val[1]));

                            float32x4_t _r_0_m_6 = vsubq_f32(_r_00, _r_66);
                            float32x4_t _r_7_m_1 = vsubq_f32(_r_77, _r_11);

                            float32x4_t _r_4_m_2 = vsubq_f32(_r_44, _r_22);
                            float32x4_t _r_3_m_5 = vsubq_f32(_r_33, _r_55);

                            float32x4_t _tmp0 = vmlaq_lane_f32(_r_0_m_6, _r_4_m_2, vget_high_f32(_coeff1), 1);
                            float32x4_t _tmp7 = vmlaq_lane_f32(_r_7_m_1, _r_3_m_5, vget_high_f32(_coeff1), 1);

                            vst1q_f32(&tmp[0][m], _tmp0);
                            vst1q_f32(&tmp[7][m], _tmp7);

                            float32x4_t _r_2_a_6 = vaddq_f32(_r_22, _r_66);
                            float32x4_t _r_1_a_5 = vaddq_f32(_r_11, _r_55);

                            float32x4_t _tmp12a = vmlsq_lane_f32(_r_2_a_6, _r_44, vget_high_f32(_coeff1), 0);
                            float32x4_t _tmp12b = vmlsq_lane_f32(_r_1_a_5, _r_33, vget_high_f32(_coeff1), 0);

                            float32x4_t _tmp1 = vaddq_f32(_tmp12a, _tmp12b);
                            float32x4_t _tmp2 = vsubq_f32(_tmp12a, _tmp12b);

                            vst1q_f32(&tmp[1][m], _tmp1);
                            vst1q_f32(&tmp[2][m], _tmp2);

                            float32x4_t _r_4_x_c = vmulq_lane_f32(_r_44, vget_high_f32(_coeff0), 0);
                            float32x4_t _r_3_x_c = vmulq_lane_f32(_r_33, vget_low_f32(_coeff1), 0);

                            float32x4_t _tmp34a = vaddq_f32(_r_66, _r_4_x_c);
                            _tmp34a = vmlaq_lane_f32(_tmp34a, _r_22, vget_low_f32(_coeff0), 0);

                            float32x4_t _tmp34b = vmlaq_lane_f32(_r_3_x_c, _r_11, vget_low_f32(_coeff0), 1);
                            _tmp34b = vmlaq_lane_f32(_tmp34b, _r_55, vget_high_f32(_coeff0), 1);

                            float32x4_t _tmp3 = vaddq_f32(_tmp34a, _tmp34b);
                            float32x4_t _tmp4 = vsubq_f32(_tmp34a, _tmp34b);

                            vst1q_f32(&tmp[3][m], _tmp3);
                            vst1q_f32(&tmp[4][m], _tmp4);

                            // reuse r04 * 1.25
                            // reuse r03 * 2.5
                            float32x4_t _r_2_a_4c = vaddq_f32(_r_22, _r_4_x_c);
                            float32x4_t _tmp56a = vmlaq_lane_f32(_r_66, _r_2_a_4c, vget_low_f32(_coeff1), 1);
                            float32x4_t _tmp56b = vmlaq_lane_f32(_r_3_x_c, _r_11, vget_high_f32(_coeff0), 1);
                            _tmp56b = vmlaq_lane_f32(_tmp56b, _r_55, vget_low_f32(_coeff0), 1);

                            float32x4_t _tmp5 = vaddq_f32(_tmp56a, _tmp56b);
                            float32x4_t _tmp6 = vsubq_f32(_tmp56a, _tmp56b);

                            vst1q_f32(&tmp[5][m], _tmp5);
                            vst1q_f32(&tmp[6][m], _tmp6);

                            r0 += w * 4;
                            r1 += w * 4;
                            r2 += w * 4;
                            r3 += w * 4;
                        }

                        const float* t0 = tmp[0];
                        const float* t1 = tmp[1];
                        const float* t2 = tmp[2];
                        const float* t3 = tmp[3];

                        float* r0_tm0 = img0_tm[i * w_tm / 8 + j].data();
                        float* r0_tm1 = img0_tm[i * w_tm / 8 + j + tiles * 8].data();
                        float* r0_tm2 = img0_tm[i * w_tm / 8 + j + tiles * 16].data();
                        float* r0_tm3 = img0_tm[i * w_tm / 8 + j + tiles * 24].data();

                        for (int m = 0; m + 3 < 8; m += 4)
                        {
                            float32x4_t _t0_0123 = vld1q_f32(t0);
                            float32x4_t _t0_4567 = vld1q_f32(t0 + 4);
                            float32x4_t _t1_0123 = vld1q_f32(t1);
                            float32x4_t _t1_4567 = vld1q_f32(t1 + 4);
                            float32x4_t _t2_0123 = vld1q_f32(t2);
                            float32x4_t _t2_4567 = vld1q_f32(t2 + 4);
                            float32x4_t _t3_0123 = vld1q_f32(t3);
                            float32x4_t _t3_4567 = vld1q_f32(t3 + 4);

                            float32x4x2_t _t01_00221133 = vtrnq_f32(_t0_0123, _t1_0123);
                            float32x4x2_t _t01_44665577 = vtrnq_f32(_t0_4567, _t1_4567);
                            float32x4x2_t _t23_00221133 = vtrnq_f32(_t2_0123, _t3_0123);
                            float32x4x2_t _t23_44665577 = vtrnq_f32(_t2_4567, _t3_4567);

                            // no vswp intrinsic  :(
                            float32x4_t _t_00 = vcombine_f32(vget_low_f32(_t01_00221133.val[0]), vget_low_f32(_t23_00221133.val[0]));
                            float32x4_t _t_11 = vcombine_f32(vget_low_f32(_t01_00221133.val[1]), vget_low_f32(_t23_00221133.val[1]));
                            float32x4_t _t_22 = vcombine_f32(vget_high_f32(_t01_00221133.val[0]), vget_high_f32(_t23_00221133.val[0]));
                            float32x4_t _t_33 = vcombine_f32(vget_high_f32(_t01_00221133.val[1]), vget_high_f32(_t23_00221133.val[1]));
                            float32x4_t _t_44 = vcombine_f32(vget_low_f32(_t01_44665577.val[0]), vget_low_f32(_t23_44665577.val[0]));
                            float32x4_t _t_55 = vcombine_f32(vget_low_f32(_t01_44665577.val[1]), vget_low_f32(_t23_44665577.val[1]));
                            float32x4_t _t_66 = vcombine_f32(vget_high_f32(_t01_44665577.val[0]), vget_high_f32(_t23_44665577.val[0]));
                            float32x4_t _t_77 = vcombine_f32(vget_high_f32(_t01_44665577.val[1]), vget_high_f32(_t23_44665577.val[1]));

                            float32x4_t _t_0_m_6 = vsubq_f32(_t_00, _t_66);
                            float32x4_t _t_7_m_1 = vsubq_f32(_t_77, _t_11);

                            float32x4_t _t_4_m_2 = vsubq_f32(_t_44, _t_22);
                            float32x4_t _t_3_m_5 = vsubq_f32(_t_33, _t_55);

                            float32x4_t _r0_tm_0_0 = vmlaq_lane_f32(_t_0_m_6, _t_4_m_2, vget_high_f32(_coeff1), 1);
                            float32x4_t _r0_tm_4_3 = vmlaq_lane_f32(_t_7_m_1, _t_3_m_5, vget_high_f32(_coeff1), 1);

                            r0_tm0[0] = vgetq_lane_f32(_r0_tm_0_0, 0);
                            r0_tm1[0] = vgetq_lane_f32(_r0_tm_0_0, 1);
                            r0_tm2[0] = vgetq_lane_f32(_r0_tm_0_0, 2);
                            r0_tm3[0] = vgetq_lane_f32(_r0_tm_0_0, 3);

                            r0_tm0 += img0_tm_w * tiles;
                            r0_tm1 += img0_tm_w * tiles;
                            r0_tm2 += img0_tm_w * tiles;
                            r0_tm3 += img0_tm_w * tiles;

                            float32x4_t _t_2_m_6 = vaddq_f32(_t_22, _t_66);
                            float32x4_t _t_1_m_5 = vaddq_f32(_t_11, _t_55);

                            float32x4_t _tmp12a = vmlsq_lane_f32(_t_2_m_6, _t_44, vget_high_f32(_coeff1), 0);
                            float32x4_t _tmp12b = vmlsq_lane_f32(_t_1_m_5, _t_33, vget_high_f32(_coeff1), 0);

                            float32x4_t _r0_tm_0_1 = vaddq_f32(_tmp12a, _tmp12b);
                            float32x4_t _r0_tm_0_2 = vsubq_f32(_tmp12a, _tmp12b);

                            r0_tm0[0] = vgetq_lane_f32(_r0_tm_0_1, 0);
                            r0_tm1[0] = vgetq_lane_f32(_r0_tm_0_1, 1);
                            r0_tm2[0] = vgetq_lane_f32(_r0_tm_0_1, 2);
                            r0_tm3[0] = vgetq_lane_f32(_r0_tm_0_1, 3);

                            r0_tm0 += img0_tm_w * tiles;
                            r0_tm1 += img0_tm_w * tiles;
                            r0_tm2 += img0_tm_w * tiles;
                            r0_tm3 += img0_tm_w * tiles;

                            r0_tm0[0] = vgetq_lane_f32(_r0_tm_0_2, 0);
                            r0_tm1[0] = vgetq_lane_f32(_r0_tm_0_2, 1);
                            r0_tm2[0] = vgetq_lane_f32(_r0_tm_0_2, 2);
                            r0_tm3[0] = vgetq_lane_f32(_r0_tm_0_2, 3);

                            r0_tm0 += img0_tm_w * tiles;
                            r0_tm1 += img0_tm_w * tiles;
                            r0_tm2 += img0_tm_w * tiles;
                            r0_tm3 += img0_tm_w * tiles;

                            float32x4_t _t_4_x_c = vmulq_lane_f32(_t_44, vget_high_f32(_coeff0), 0);
                            float32x4_t _t_3_x_c = vmulq_lane_f32(_t_33, vget_low_f32(_coeff1), 0);

                            float32x4_t _tmp34a = vaddq_f32(_t_66, _t_4_x_c);
                            _tmp34a = vmlaq_lane_f32(_tmp34a, _t_22, vget_low_f32(_coeff0), 0);

                            float32x4_t _tmp34b = vmlaq_lane_f32(_t_3_x_c, _t_11, vget_low_f32(_coeff0), 1);
                            _tmp34b = vmlaq_lane_f32(_tmp34b, _t_55, vget_high_f32(_coeff0), 1);

                            float32x4_t _r0_tm_0_3 = vaddq_f32(_tmp34a, _tmp34b);
                            float32x4_t _r0_tm_4_0 = vsubq_f32(_tmp34a, _tmp34b);

                            r0_tm0[0] = vgetq_lane_f32(_r0_tm_0_3, 0);
                            r0_tm1[0] = vgetq_lane_f32(_r0_tm_0_3, 1);
                            r0_tm2[0] = vgetq_lane_f32(_r0_tm_0_3, 2);
                            r0_tm3[0] = vgetq_lane_f32(_r0_tm_0_3, 3);

                            r0_tm0 += img0_tm_w * tiles;
                            r0_tm1 += img0_tm_w * tiles;
                            r0_tm2 += img0_tm_w * tiles;
                            r0_tm3 += img0_tm_w * tiles;

                            r0_tm0[0] = vgetq_lane_f32(_r0_tm_4_0, 0);
                            r0_tm1[0] = vgetq_lane_f32(_r0_tm_4_0, 1);
                            r0_tm2[0] = vgetq_lane_f32(_r0_tm_4_0, 2);
                            r0_tm3[0] = vgetq_lane_f32(_r0_tm_4_0, 3);

                            r0_tm0 += img0_tm_w * tiles;
                            r0_tm1 += img0_tm_w * tiles;
                            r0_tm2 += img0_tm_w * tiles;
                            r0_tm3 += img0_tm_w * tiles;

                            float32x4_t _t_2_a_4c = vaddq_f32(_t_22, _t_4_x_c);
                            float32x4_t _tmp56a = vmlaq_lane_f32(_t_66, _t_2_a_4c, vget_low_f32(_coeff1), 1);
                            float32x4_t _tmp56b = vmlaq_lane_f32(_t_3_x_c, _t_11, vget_high_f32(_coeff0), 1);
                            _tmp56b = vmlaq_lane_f32(_tmp56b, _t_55, vget_low_f32(_coeff0), 1);

                            float32x4_t _r0_tm_4_1 = vaddq_f32(_tmp56a, _tmp56b);
                            float32x4_t _r0_tm_4_2 = vsubq_f32(_tmp56a, _tmp56b);

                            r0_tm0[0] = vgetq_lane_f32(_r0_tm_4_1, 0);
                            r0_tm1[0] = vgetq_lane_f32(_r0_tm_4_1, 1);
                            r0_tm2[0] = vgetq_lane_f32(_r0_tm_4_1, 2);
                            r0_tm3[0] = vgetq_lane_f32(_r0_tm_4_1, 3);

                            r0_tm0 += img0_tm_w * tiles;
                            r0_tm1 += img0_tm_w * tiles;
                            r0_tm2 += img0_tm_w * tiles;
                            r0_tm3 += img0_tm_w * tiles;

                            r0_tm0[0] = vgetq_lane_f32(_r0_tm_4_2, 0);
                            r0_tm1[0] = vgetq_lane_f32(_r0_tm_4_2, 1);
                            r0_tm2[0] = vgetq_lane_f32(_r0_tm_4_2, 2);
                            r0_tm3[0] = vgetq_lane_f32(_r0_tm_4_2, 3);

                            r0_tm0 += img0_tm_w * tiles;
                            r0_tm1 += img0_tm_w * tiles;
                            r0_tm2 += img0_tm_w * tiles;
                            r0_tm3 += img0_tm_w * tiles;

                            r0_tm0[0] = vgetq_lane_f32(_r0_tm_4_3, 0);
                            r0_tm1[0] = vgetq_lane_f32(_r0_tm_4_3, 1);
                            r0_tm2[0] = vgetq_lane_f32(_r0_tm_4_3, 2);
                            r0_tm3[0] = vgetq_lane_f32(_r0_tm_4_3, 3);

                            t0 += 8 * 4;
                            t1 += 8 * 4;
                            t2 += 8 * 4;
                            t3 += 8 * 4;

                            r0_tm0 += img0_tm_w * tiles * 25;
                            r0_tm1 += img0_tm_w * tiles * 25;
                            r0_tm2 += img0_tm_w * tiles * 25;
                            r0_tm3 += img0_tm_w * tiles * 25;
                        }
    #else  // __aarch64__
                        float* t0 = tmp[0];
                        float* t1 = tmp[1];
                        float* t2 = tmp[2];
                        float* t3 = tmp[3];

                        int stepw = w * 4 * 4;

                        asm volatile(

                            // loop0
                            "vld1.f32   {d16-d19}, [%4], %18    \n"
                            "vld1.f32   {d20-d23}, [%5], %18    \n"
                            "vld1.f32   {d24-d27}, [%6], %18    \n"

                            "vtrn.32    q8, q10             \n"

                            "vld1.f32   {d28-d31}, [%7], %18    \n"

                            "vtrn.32    q9, q11             \n"
                            "vtrn.32    q12, q14            \n"
                            "vtrn.32    q13, q15            \n"

                            "vswp       d17, d24            \n"
                            "vswp       d19, d26            \n"
                            "vswp       d21, d28            \n" //  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                            "vswp       d23, d30            \n" // q12 = 22  q13 = 66  q14 = 33  q15 = 77

                            "vsub.f32   q2, q8, q13         \n"
                            "vsub.f32   q3, q9, q12         \n"

                            "vadd.f32   q4, q12, q13        \n"
                            "vadd.f32   q5, q10, q11        \n"

                            "vmla.f32   q2, q3, %f17[1]     \n"

                            "vmul.f32   q7, q14, %e17[0]    \n" // q7 = _r_3_x_c
                            "vmul.f32   q6, q9, %f16[0]     \n" // q6 = _r_4_x_c

                            "vmls.f32   q4, q9, %f17[0]     \n"
                            "vmls.f32   q5, q14, %f17[0]    \n"

                            "vst1.f32   {d4-d5}, [%0]       \n" // tmp[0][m]
                            "add        %0, %0, #128        \n"

                            "vmov       q3, q7              \n" // use q7

                            "vadd.f32   q2, q13, q6         \n" // use q6
                            "vmla.f32   q3, q10, %e16[1]    \n"

                            "vadd.f32   q8, q4, q5          \n"
                            "vsub.f32   q9, q4, q5          \n"

                            "vmov       q5, q7              \n" // use q7

                            "vadd.f32   q6, q12, q6         \n" // use q6
                            "vmla.f32   q5, q10, %f16[1]    \n"

                            "vmov       q4, q13             \n"

                            "vmla.f32   q2, q12, %e16[0]    \n"
                            "vmla.f32   q3, q11, %f16[1]    \n"

                            "vst1.f32   {d16-d17}, [%1]     \n" // tmp[1][m]
                            "add        %1, %1, #128        \n"

                            "vmla.f32   q4, q6, %e17[1]     \n"
                            "vmla.f32   q5, q11, %e16[1]    \n"

                            "vst1.f32   {d18-d19}, [%2]     \n" // tmp[2][m]
                            "add        %2, %2, #128        \n"

                            "vadd.f32   q8, q2, q3          \n"
                            "vsub.f32   q9, q2, q3          \n"

                            "vsub.f32   q6, q15, q10        \n"
                            "vsub.f32   q7, q14, q11        \n"

                            "vadd.f32   q2, q4, q5          \n"
                            "vsub.f32   q3, q4, q5          \n"

                            "vst1.f32   {d16-d17}, [%3]     \n" // tmp[3][m]
                            "add        %3, %3, #128        \n"
                            "vst1.f32   {d18-d19}, [%0]     \n" // tmp[4][m]
                            "sub        %0, %0, #112        \n"

                            "vmla.f32   q6, q7, %f17[1]     \n"

                            "vst1.f32   {d4-d5}, [%1]       \n" // tmp[5][m]
                            "sub        %1, %1, #112        \n"
                            "vst1.f32   {d6-d7}, [%2]       \n" // tmp[6][m]
                            "sub        %2, %2, #112        \n"

                            "vst1.f32   {d12-d13}, [%3]     \n" // tmp[7][m]
                            "sub        %3, %3, #112        \n"

                            // loop1
                            "vld1.f32   {d16-d19}, [%4]     \n"
                            "vld1.f32   {d20-d23}, [%5]     \n"
                            "vld1.f32   {d24-d27}, [%6]     \n"

                            "vtrn.32    q8, q10             \n"

                            "vld1.f32   {d28-d31}, [%7]     \n"

                            "vtrn.32    q9, q11             \n"
                            "vtrn.32    q12, q14            \n"
                            "vtrn.32    q13, q15            \n"

                            "vswp       d17, d24            \n"
                            "vswp       d19, d26            \n"
                            "vswp       d21, d28            \n" //  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                            "vswp       d23, d30            \n" // q12 = 22  q13 = 66  q14 = 33  q15 = 77

                            "vsub.f32   q2, q8, q13         \n"
                            "vsub.f32   q3, q9, q12         \n"

                            "vadd.f32   q4, q12, q13        \n"
                            "vadd.f32   q5, q10, q11        \n"

                            "vmla.f32   q2, q3, %f17[1]     \n"

                            "vmul.f32   q7, q14, %e17[0]    \n" // q7 = _r_3_x_c
                            "vmul.f32   q6, q9, %f16[0]     \n" // q6 = _r_4_x_c

                            "vmls.f32   q4, q9, %f17[0]     \n"
                            "vmls.f32   q5, q14, %f17[0]    \n"

                            "vst1.f32   {d4-d5}, [%0]       \n" // tmp[0][m]
                            "add        %0, %0, #128        \n"

                            "vmov       q3, q7              \n" // use q7

                            "vadd.f32   q2, q13, q6         \n" // use q6
                            "vmla.f32   q3, q10, %e16[1]    \n"

                            "vadd.f32   q8, q4, q5          \n"
                            "vsub.f32   q9, q4, q5          \n"

                            "vmov       q5, q7              \n" // use q7

                            "vadd.f32   q6, q12, q6         \n" // use q6
                            "vmla.f32   q5, q10, %f16[1]    \n"

                            "vmov       q4, q13             \n"

                            "vmla.f32   q2, q12, %e16[0]    \n"
                            "vmla.f32   q3, q11, %f16[1]    \n"

                            "vst1.f32   {d16-d17}, [%1]     \n" // tmp[1][m]
                            "add        %1, %1, #128        \n"

                            "vmla.f32   q4, q6, %e17[1]     \n"
                            "vmla.f32   q5, q11, %e16[1]    \n"

                            "vst1.f32   {d18-d19}, [%2]     \n" // tmp[2][m]
                            "add        %2, %2, #128        \n"

                            "vadd.f32   q8, q2, q3          \n"
                            "vsub.f32   q9, q2, q3          \n"

                            "vsub.f32   q6, q15, q10        \n"
                            "vsub.f32   q7, q14, q11        \n"

                            "vadd.f32   q2, q4, q5          \n"
                            "vsub.f32   q3, q4, q5          \n"

                            "vst1.f32   {d16-d17}, [%3]     \n" // tmp[3][m]
                            "add        %3, %3, #128        \n"
                            "vst1.f32   {d18-d19}, [%0]     \n" // tmp[4][m]

                            "vmla.f32   q6, q7, %f17[1]     \n"

                            "vst1.f32   {d4-d5}, [%1]       \n" // tmp[5][m]
                            "vst1.f32   {d6-d7}, [%2]       \n" // tmp[6][m]

                            "vst1.f32   {d12-d13}, [%3]     \n" // tmp[7][m]

                            : "=r"(t0), // %0
                            "=r"(t1), // %1
                            "=r"(t2), // %2
                            "=r"(t3), // %3
                            "=r"(r0), // %4
                            "=r"(r1), // %5
                            "=r"(r2), // %6
                            "=r"(r3)  // %7
                            : "0"(t0),
                            "1"(t1),
                            "2"(t2),
                            "3"(t3),
                            "4"(r0),
                            "5"(r1),
                            "6"(r2),
                            "7"(r3),
                            "w"(_coeff0), // %16
                            "w"(_coeff1), // %17
                            "r"(stepw)    // %18
                            : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");

                        t0 = tmp[0];
                        t1 = tmp[1];
                        t2 = tmp[2];
                        t3 = tmp[3];

                        float* r0_tm0_0 = img0_tm[i * w_tm / 8 + j].data();
                        float* r0_tm1_0 = img0_tm[i * w_tm / 8 + j + tiles * 8].data();
                        float* r0_tm2_0 = img0_tm[i * w_tm / 8 + j + tiles * 16].data();
                        float* r0_tm3_0 = img0_tm[i * w_tm / 8 + j + tiles * 24].data();

                        int step = img0_tm_w * tiles * 4;
                        int step2 = img0_tm_w * tiles * 25 * 4;

                        asm volatile(

                            // loop0
                            "vld1.f32   {d16-d19}, [%4]     \n"
                            "add        %4, %4, #128        \n"
                            "vld1.f32   {d20-d23}, [%5]     \n"
                            "add        %5, %5, #128        \n"
                            "vld1.f32   {d24-d27}, [%6]     \n"
                            "add        %6, %6, #128        \n"

                            "vtrn.32    q8, q10             \n"

                            "vld1.f32   {d28-d31}, [%7]     \n"
                            "add        %7, %7, #128        \n"

                            "vtrn.32    q9, q11             \n"
                            "vtrn.32    q12, q14            \n"
                            "vtrn.32    q13, q15            \n"

                            "vswp       d17, d24            \n"
                            "vswp       d19, d26            \n"
                            "vswp       d21, d28            \n" //  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                            "vswp       d23, d30            \n" // q12 = 22  q13 = 66  q14 = 33  q15 = 77

                            "vsub.f32   q2, q8, q13         \n"
                            "vsub.f32   q3, q9, q12         \n"

                            "vadd.f32   q4, q12, q13        \n"
                            "vadd.f32   q5, q10, q11        \n"

                            "vmla.f32   q2, q3, %f17[1]     \n"

                            "vmul.f32   q7, q14, %e17[0]    \n" // q7 = _r_3_x_c
                            "vmul.f32   q6, q9, %f16[0]     \n" // q6 = _r_4_x_c

                            "vmls.f32   q4, q9, %f17[0]     \n"
                            "vmls.f32   q5, q14, %f17[0]    \n"

                            "vst1.f32   {d4[0]}, [%0], %18  \n"
                            "vst1.f32   {d4[1]}, [%1], %18  \n"

                            "vmov       q3, q7              \n" // use q7

                            "vst1.f32   {d5[0]}, [%2], %18  \n"
                            "vst1.f32   {d5[1]}, [%3], %18  \n"

                            "vadd.f32   q2, q13, q6         \n" // use q6
                            "vmla.f32   q3, q10, %e16[1]    \n"

                            "vadd.f32   q8, q4, q5          \n"
                            "vsub.f32   q9, q4, q5          \n"

                            "vmov       q5, q7              \n" // use q7

                            "vadd.f32   q6, q12, q6         \n" // use q6
                            "vmla.f32   q5, q10, %f16[1]    \n"

                            "vmov       q4, q13             \n"

                            "vmla.f32   q2, q12, %e16[0]    \n"
                            "vmla.f32   q3, q11, %f16[1]    \n"

                            "vst1.f32   {d16[0]}, [%0], %18 \n"
                            "vst1.f32   {d16[1]}, [%1], %18 \n"

                            "vmla.f32   q4, q6, %e17[1]     \n"

                            "vst1.f32   {d17[0]}, [%2], %18 \n"
                            "vst1.f32   {d17[1]}, [%3], %18 \n"

                            "vmla.f32   q5, q11, %e16[1]    \n"

                            "vst1.f32   {d18[0]}, [%0], %18 \n"
                            "vst1.f32   {d18[1]}, [%1], %18 \n"

                            "vadd.f32   q8, q2, q3          \n"

                            "vst1.f32   {d19[0]}, [%2], %18 \n"
                            "vst1.f32   {d19[1]}, [%3], %18 \n"

                            "vsub.f32   q9, q2, q3          \n"

                            "vsub.f32   q6, q15, q10        \n"
                            "vsub.f32   q7, q14, q11        \n"

                            "vst1.f32   {d16[0]}, [%0], %18 \n"
                            "vst1.f32   {d16[1]}, [%1], %18 \n"
                            "vst1.f32   {d17[0]}, [%2], %18 \n"
                            "vst1.f32   {d17[1]}, [%3], %18 \n"

                            "vadd.f32   q2, q4, q5          \n"

                            "vst1.f32   {d18[0]}, [%0], %18 \n"
                            "vst1.f32   {d18[1]}, [%1], %18 \n"
                            "vst1.f32   {d19[0]}, [%2], %18 \n"
                            "vst1.f32   {d19[1]}, [%3], %18 \n"

                            "vsub.f32   q3, q4, q5          \n"

                            "vst1.f32   {d4[0]}, [%0], %18  \n"
                            "vst1.f32   {d4[1]}, [%1], %18  \n"
                            "vst1.f32   {d5[0]}, [%2], %18  \n"
                            "vst1.f32   {d5[1]}, [%3], %18  \n"

                            "vmla.f32   q6, q7, %f17[1]     \n"

                            "vst1.f32   {d6[0]}, [%0], %18  \n"
                            "vst1.f32   {d6[1]}, [%1], %18  \n"
                            "vst1.f32   {d7[0]}, [%2], %18  \n"
                            "vst1.f32   {d7[1]}, [%3], %18  \n"

                            "vst1.f32   {d12[0]}, [%0], %19 \n"
                            "vst1.f32   {d12[1]}, [%1], %19 \n"
                            "vst1.f32   {d13[0]}, [%2], %19 \n"
                            "vst1.f32   {d13[1]}, [%3], %19 \n"

                            // loop1
                            "vld1.f32   {d16-d19}, [%4]     \n"
                            "vld1.f32   {d20-d23}, [%5]     \n"
                            "vld1.f32   {d24-d27}, [%6]     \n"

                            "vtrn.32    q8, q10             \n"

                            "vld1.f32   {d28-d31}, [%7]     \n"

                            "vtrn.32    q9, q11             \n"
                            "vtrn.32    q12, q14            \n"
                            "vtrn.32    q13, q15            \n"

                            "vswp       d17, d24            \n"
                            "vswp       d19, d26            \n"
                            "vswp       d21, d28            \n" //  q8 = 00   q9 = 44  q10 = 11  q11 = 55
                            "vswp       d23, d30            \n" // q12 = 22  q13 = 66  q14 = 33  q15 = 77

                            "vsub.f32   q2, q8, q13         \n"
                            "vsub.f32   q3, q9, q12         \n"

                            "vadd.f32   q4, q12, q13        \n"
                            "vadd.f32   q5, q10, q11        \n"

                            "vmla.f32   q2, q3, %f17[1]     \n"

                            "vmul.f32   q7, q14, %e17[0]    \n" // q7 = _r_3_x_c
                            "vmul.f32   q6, q9, %f16[0]     \n" // q6 = _r_4_x_c

                            "vmls.f32   q4, q9, %f17[0]     \n"
                            "vmls.f32   q5, q14, %f17[0]    \n"

                            "vst1.f32   {d4[0]}, [%0], %18  \n"
                            "vst1.f32   {d4[1]}, [%1], %18  \n"

                            "vmov       q3, q7              \n" // use q7

                            "vst1.f32   {d5[0]}, [%2], %18  \n"
                            "vst1.f32   {d5[1]}, [%3], %18  \n"

                            "vadd.f32   q2, q13, q6         \n" // use q6
                            "vmla.f32   q3, q10, %e16[1]    \n"

                            "vadd.f32   q8, q4, q5          \n"
                            "vsub.f32   q9, q4, q5          \n"

                            "vmov       q5, q7              \n" // use q7

                            "vadd.f32   q6, q12, q6         \n" // use q6
                            "vmla.f32   q5, q10, %f16[1]    \n"

                            "vmov       q4, q13             \n"

                            "vmla.f32   q2, q12, %e16[0]    \n"
                            "vmla.f32   q3, q11, %f16[1]    \n"

                            "vst1.f32   {d16[0]}, [%0], %18 \n"
                            "vst1.f32   {d16[1]}, [%1], %18 \n"

                            "vmla.f32   q4, q6, %e17[1]     \n"

                            "vst1.f32   {d17[0]}, [%2], %18 \n"
                            "vst1.f32   {d17[1]}, [%3], %18 \n"

                            "vmla.f32   q5, q11, %e16[1]    \n"

                            "vst1.f32   {d18[0]}, [%0], %18 \n"
                            "vst1.f32   {d18[1]}, [%1], %18 \n"

                            "vadd.f32   q8, q2, q3          \n"

                            "vst1.f32   {d19[0]}, [%2], %18 \n"
                            "vst1.f32   {d19[1]}, [%3], %18 \n"

                            "vsub.f32   q9, q2, q3          \n"

                            "vsub.f32   q6, q15, q10        \n"
                            "vsub.f32   q7, q14, q11        \n"

                            "vst1.f32   {d16[0]}, [%0], %18 \n"
                            "vst1.f32   {d16[1]}, [%1], %18 \n"
                            "vst1.f32   {d17[0]}, [%2], %18 \n"
                            "vst1.f32   {d17[1]}, [%3], %18 \n"

                            "vadd.f32   q2, q4, q5          \n"

                            "vst1.f32   {d18[0]}, [%0], %18 \n"
                            "vst1.f32   {d18[1]}, [%1], %18 \n"
                            "vst1.f32   {d19[0]}, [%2], %18 \n"
                            "vst1.f32   {d19[1]}, [%3], %18 \n"

                            "vsub.f32   q3, q4, q5          \n"

                            "vst1.f32   {d4[0]}, [%0], %18  \n"
                            "vst1.f32   {d4[1]}, [%1], %18  \n"
                            "vst1.f32   {d5[0]}, [%2], %18  \n"
                            "vst1.f32   {d5[1]}, [%3], %18  \n"

                            "vmla.f32   q6, q7, %f17[1]     \n"

                            "vst1.f32   {d6[0]}, [%0], %18  \n"
                            "vst1.f32   {d6[1]}, [%1], %18  \n"
                            "vst1.f32   {d7[0]}, [%2], %18  \n"
                            "vst1.f32   {d7[1]}, [%3], %18  \n"

                            "vst1.f32   {d12[0]}, [%0]      \n"
                            "vst1.f32   {d12[1]}, [%1]      \n"
                            "vst1.f32   {d13[0]}, [%2]      \n"
                            "vst1.f32   {d13[1]}, [%3]      \n"

                            : "=r"(r0_tm0_0), // %0
                            "=r"(r0_tm1_0), // %1
                            "=r"(r0_tm2_0), // %2
                            "=r"(r0_tm3_0), // %3
                            "=r"(t0),       // %4
                            "=r"(t1),       // %5
                            "=r"(t2),       // %6
                            "=r"(t3)        // %7
                            : "0"(r0_tm0_0),
                            "1"(r0_tm1_0),
                            "2"(r0_tm2_0),
                            "3"(r0_tm3_0),
                            "4"(t0),
                            "5"(t1),
                            "6"(t2),
                            "7"(t3),
                            "w"(_coeff0), // %16
                            "w"(_coeff1), // %17
                            "r"(step),    // %18
                            "r"(step2)    // %19
                            : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
    #endif // __aarch64__
    #else
                        const float* r0 = img0[i * 6].data() + j * 6;

                        for (int m = 0; m < 8; m++)
                        {
                            tmp[0][m] = r0[0] - r0[6] + (r0[4] - r0[2]) * 5.25f;
                            tmp[7][m] = r0[7] - r0[1] + (r0[3] - r0[5]) * 5.25f;

                            float tmp12a = (r0[2] + r0[6] - r0[4] * 4.25f);
                            float tmp12b = (r0[1] + r0[5] - r0[3] * 4.25f);

                            tmp[1][m] = tmp12a + tmp12b;
                            tmp[2][m] = tmp12a - tmp12b;

                            float tmp34a = (r0[6] + r0[2] * 0.25f - r0[4] * 1.25f);
                            float tmp34b = (r0[1] * 0.5f - r0[3] * 2.5f + r0[5] * 2.f);

                            tmp[3][m] = tmp34a + tmp34b;
                            tmp[4][m] = tmp34a - tmp34b;

                            float tmp56a = (r0[6] + (r0[2] - r0[4] * 1.25f) * 4.f);
                            float tmp56b = (r0[1] * 2.f - r0[3] * 2.5f + r0[5] * 0.5f);

                            tmp[5][m] = tmp56a + tmp56b;
                            tmp[6][m] = tmp56a - tmp56b;

                            r0 += w;
                        }

                        float* r0_tm_0 = img0_tm[i * w_tm / 8 + j].data();
                        float* r0_tm_1 = img0_tm[i * w_tm / 8 + j + tiles].data();
                        float* r0_tm_2 = img0_tm[i * w_tm / 8 + j + tiles * 2].data();
                        float* r0_tm_3 = img0_tm[i * w_tm / 8 + j + tiles * 3].data();
                        float* r0_tm_4 = img0_tm[i * w_tm / 8 + j + tiles * 4].data();
                        float* r0_tm_5 = img0_tm[i * w_tm / 8 + j + tiles * 5].data();
                        float* r0_tm_6 = img0_tm[i * w_tm / 8 + j + tiles * 6].data();
                        float* r0_tm_7 = img0_tm[i * w_tm / 8 + j + tiles * 7].data();

                        for (int m = 0; m < 8; m++)
                        {
                            const float* tmp0 = tmp[m];

                            r0_tm_0[0] = tmp0[0] - tmp0[6] + (tmp0[4] - tmp0[2]) * 5.25f;
                            r0_tm_7[0] = tmp0[7] - tmp0[1] + (tmp0[3] - tmp0[5]) * 5.25f;

                            float tmp12a = (tmp0[2] + tmp0[6] - tmp0[4] * 4.25f);
                            float tmp12b = (tmp0[1] - tmp0[3] * 4.25f + tmp0[5]);

                            r0_tm_1[0] = tmp12a + tmp12b;
                            r0_tm_2[0] = tmp12a - tmp12b;

                            float tmp34a = (tmp0[6] + tmp0[2] * 0.25f - tmp0[4] * 1.25f);
                            float tmp34b = (tmp0[1] * 0.5f - tmp0[3] * 2.5f + tmp0[5] * 2.f);

                            r0_tm_3[0] = tmp34a + tmp34b;
                            r0_tm_4[0] = tmp34a - tmp34b;

                            float tmp56a = (tmp0[6] + (tmp0[2] - tmp0[4] * 1.25f) * 4.f);
                            float tmp56b = (tmp0[1] * 2.f - tmp0[3] * 2.5f + tmp0[5] * 0.5f);

                            r0_tm_5[0] = tmp56a + tmp56b;
                            r0_tm_6[0] = tmp56a - tmp56b;

                            r0_tm_0 += img0_tm_w * tiles * 8;
                            r0_tm_1 += img0_tm_w * tiles * 8;
                            r0_tm_2 += img0_tm_w * tiles * 8;
                            r0_tm_3 += img0_tm_w * tiles * 8;
                            r0_tm_4 += img0_tm_w * tiles * 8;
                            r0_tm_5 += img0_tm_w * tiles * 8;
                            r0_tm_6 += img0_tm_w * tiles * 8;
                            r0_tm_7 += img0_tm_w * tiles * 8;
                        }
    #endif // __ARM_NEON
                    }
                }
            }
        });
    }
    input.reset();
    
    Tensor output_tf;
    {
        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        const int tiles = w_tm / 8 * h_tm / 8;

        // permute
        // bottom_blob_tm.create(1, 64 * tiles, inch);
        //         Mat bottom_blob_tm2(inch, tiles, 64);
        Tensor input_tf2 = otter::empty({64, tiles / 8 + (tiles % 8) / 4 + tiles % 4, 8 * inch}, otter::ScalarType::Float);
        auto input_tf2_a = input_tf2.accessor<float, 3>();
        auto input_tf_a = input_tf.accessor<float, 3>();

        otter::parallel_for(0, 64, 0, [&](int64_t begin, int64_t end) {
            for (int r = 0; r < 64; r++)
            {
                auto tm2 = input_tf2_a[r];

                // tile
                int i = 0;
                for (; i + 7 < tiles; i += 8)
                {
                    float* tm2p = tm2[i / 8].data();

                    const float* r0 = input_tf_a.data();

                    r0 += r * tiles + i;

                    for (int q = 0; q < inch; q++)
                    {
    #if __ARM_NEON
                        float32x4_t _r0 = vld1q_f32(r0);
                        float32x4_t _r0n = vld1q_f32(r0 + 4);
                        vst1q_f32(tm2p, _r0);
                        vst1q_f32(tm2p + 4, _r0n);
    #else
                        tm2p[0] = r0[0];
                        tm2p[1] = r0[1];
                        tm2p[2] = r0[2];
                        tm2p[3] = r0[3];
                        tm2p[4] = r0[4];
                        tm2p[5] = r0[5];
                        tm2p[6] = r0[6];
                        tm2p[7] = r0[7];
    #endif // __ARM_NEON

//                        r0 += bottom_blob_tm.cstep;
                        tm2p += 8;
                    }
                }
                for (; i + 3 < tiles; i += 4)
                {
                    float* tm2p = tm2[i / 8 + (i % 8) / 4].data();

                    const float* r0 = input_tf_a.data();

                    r0 += r * tiles + i;

                    for (int q = 0; q < inch; q++)
                    {
    #if __ARM_NEON
                        float32x4_t _r0 = vld1q_f32(r0);
                        vst1q_f32(tm2p, _r0);
    #else
                        tm2p[0] = r0[0];
                        tm2p[1] = r0[1];
                        tm2p[2] = r0[2];
                        tm2p[3] = r0[3];
    #endif // __ARM_NEON

//                        r0 += bottom_blob_tm.cstep;
                        tm2p += 4;
                    }
                }
                for (; i < tiles; i++)
                {
                    float* tm2p = tm2[i / 8 + (i % 8) / 4 + i % 4].data();

                    const float* r0 = input_tf_a.data();

                    r0 += r * tiles + i;

                    for (int q = 0; q < inch; q++)
                    {
                        tm2p[0] = r0[0];

//                        r0 += bottom_blob_tm.cstep;
                        tm2p += 1;
                    }
                }
            }
        });
        input_tf.reset();
        // permute end

        output_tf = otter::empty({outch, 64 * tiles, 1}, otter::ScalarType::Float);
        auto output_tf_a = output_tf.accessor<float, 3>();

        int nn_outch = 0;
        int remain_outch_start = 0;

#if __ARM_NEON && __aarch64__
        nn_outch = outch >> 3;
        remain_outch_start = nn_outch << 3;

        otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
            for (int pp = 0; pp < nn_outch; pp++)
            {
                int p = pp * 8;

                const auto kernel_tm0 = kernel_tf_a[p / 8];

                auto out0_tm = output_tf_a[p];
                auto out1_tm = output_tf_a[p + 1];
                auto out2_tm = output_tf_a[p + 2];
                auto out3_tm = output_tf_a[p + 3];
                auto out4_tm = output_tf_a[p + 4];
                auto out5_tm = output_tf_a[p + 5];
                auto out6_tm = output_tf_a[p + 6];
                auto out7_tm = output_tf_a[p + 7];

                float* output0_tm = out0_tm.data();
                float* output1_tm = out1_tm.data();
                float* output2_tm = out2_tm.data();
                float* output3_tm = out3_tm.data();
                float* output4_tm = out4_tm.data();
                float* output5_tm = out5_tm.data();
                float* output6_tm = out6_tm.data();
                float* output7_tm = out7_tm.data();

                for (int r = 0; r < 64; r++)
                {
                    const auto bb2 = input_tf2_a[r];

                    // tile
                    int i = 0;
                    for (; i + 7 < tiles; i += 8)
                    {
                        const float* bb2p0 = bb2[i / 8].data();

                        const float* ktm0 = kernel_tm0[r].data();

                        asm volatile(
                            "eor    v16.16b, v16.16b, v16.16b  \n"
                            "eor    v17.16b, v17.16b, v17.16b  \n"
                            "eor    v18.16b, v18.16b, v18.16b  \n"
                            "eor    v19.16b, v19.16b, v19.16b  \n"
                            "eor    v20.16b, v20.16b, v20.16b  \n"
                            "eor    v21.16b, v21.16b, v21.16b  \n"
                            "eor    v22.16b, v22.16b, v22.16b  \n"
                            "eor    v23.16b, v23.16b, v23.16b  \n"
                            "eor    v24.16b, v24.16b, v24.16b  \n"
                            "eor    v25.16b, v25.16b, v25.16b  \n"
                            "eor    v26.16b, v26.16b, v26.16b  \n"
                            "eor    v27.16b, v27.16b, v27.16b  \n"
                            "eor    v28.16b, v28.16b, v28.16b  \n"
                            "eor    v29.16b, v29.16b, v29.16b  \n"
                            "eor    v30.16b, v30.16b, v30.16b  \n"
                            "eor    v31.16b, v31.16b, v31.16b  \n"

                            // inch loop
                            "lsr    w4, %w20, #2            \n" // w4 = nn = inch >> 2
                            "cmp    w4, #0                  \n"
                            "beq    1f                      \n"

                            "0:                             \n"

                            "prfm   pldl1keep, [%8, #512]   \n"
                            "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%8], #64   \n"

                            "prfm   pldl1keep, [%9, #512]   \n"
                            "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64   \n"

                            "fmla   v16.4s, v8.4s, v0.s[0]  \n"
                            "fmla   v17.4s, v9.4s, v0.s[0]  \n"
                            "fmla   v18.4s, v8.4s, v0.s[1]  \n"
                            "fmla   v19.4s, v9.4s, v0.s[1]  \n"
                            "fmla   v20.4s, v8.4s, v0.s[2]  \n"
                            "fmla   v21.4s, v9.4s, v0.s[2]  \n"
                            "fmla   v22.4s, v8.4s, v0.s[3]  \n"
                            "fmla   v23.4s, v9.4s, v0.s[3]  \n"

                            "prfm   pldl1keep, [%9, #512]   \n"
                            "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%9], #64   \n"

                            "fmla   v24.4s, v8.4s, v1.s[0]  \n"
                            "fmla   v25.4s, v9.4s, v1.s[0]  \n"
                            "fmla   v26.4s, v8.4s, v1.s[1]  \n"
                            "fmla   v27.4s, v9.4s, v1.s[1]  \n"
                            "fmla   v28.4s, v8.4s, v1.s[2]  \n"
                            "fmla   v29.4s, v9.4s, v1.s[2]  \n"
                            "fmla   v30.4s, v8.4s, v1.s[3]  \n"
                            "fmla   v31.4s, v9.4s, v1.s[3]  \n"

                            "fmla   v16.4s, v10.4s, v2.s[0] \n"
                            "fmla   v17.4s, v11.4s, v2.s[0] \n"
                            "fmla   v18.4s, v10.4s, v2.s[1] \n"
                            "fmla   v19.4s, v11.4s, v2.s[1] \n"
                            "fmla   v20.4s, v10.4s, v2.s[2] \n"
                            "fmla   v21.4s, v11.4s, v2.s[2] \n"
                            "fmla   v22.4s, v10.4s, v2.s[3] \n"
                            "fmla   v23.4s, v11.4s, v2.s[3] \n"

                            "prfm   pldl1keep, [%8, #512]   \n"
                            "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%8], #64 \n"

                            "fmla   v24.4s, v10.4s, v3.s[0] \n"
                            "fmla   v25.4s, v11.4s, v3.s[0] \n"
                            "fmla   v26.4s, v10.4s, v3.s[1] \n"
                            "fmla   v27.4s, v11.4s, v3.s[1] \n"
                            "fmla   v28.4s, v10.4s, v3.s[2] \n"
                            "fmla   v29.4s, v11.4s, v3.s[2] \n"
                            "fmla   v30.4s, v10.4s, v3.s[3] \n"
                            "fmla   v31.4s, v11.4s, v3.s[3] \n"

                            "fmla   v16.4s, v12.4s, v4.s[0] \n"
                            "fmla   v17.4s, v13.4s, v4.s[0] \n"
                            "fmla   v18.4s, v12.4s, v4.s[1] \n"
                            "fmla   v19.4s, v13.4s, v4.s[1] \n"
                            "fmla   v20.4s, v12.4s, v4.s[2] \n"
                            "fmla   v21.4s, v13.4s, v4.s[2] \n"
                            "fmla   v22.4s, v12.4s, v4.s[3] \n"
                            "fmla   v23.4s, v13.4s, v4.s[3] \n"

                            "fmla   v24.4s, v12.4s, v5.s[0] \n"
                            "fmla   v25.4s, v13.4s, v5.s[0] \n"
                            "fmla   v26.4s, v12.4s, v5.s[1] \n"
                            "fmla   v27.4s, v13.4s, v5.s[1] \n"
                            "fmla   v28.4s, v12.4s, v5.s[2] \n"
                            "fmla   v29.4s, v13.4s, v5.s[2] \n"
                            "fmla   v30.4s, v12.4s, v5.s[3] \n"
                            "fmla   v31.4s, v13.4s, v5.s[3] \n"

                            "fmla   v16.4s, v14.4s, v6.s[0] \n"
                            "fmla   v17.4s, v15.4s, v6.s[0] \n"
                            "fmla   v18.4s, v14.4s, v6.s[1] \n"
                            "fmla   v19.4s, v15.4s, v6.s[1] \n"
                            "fmla   v20.4s, v14.4s, v6.s[2] \n"
                            "fmla   v21.4s, v15.4s, v6.s[2] \n"
                            "fmla   v22.4s, v14.4s, v6.s[3] \n"
                            "fmla   v23.4s, v15.4s, v6.s[3] \n"

                            "subs   w4, w4, #1              \n"

                            "fmla   v24.4s, v14.4s, v7.s[0] \n"
                            "fmla   v25.4s, v15.4s, v7.s[0] \n"
                            "fmla   v26.4s, v14.4s, v7.s[1] \n"
                            "fmla   v27.4s, v15.4s, v7.s[1] \n"
                            "fmla   v28.4s, v14.4s, v7.s[2] \n"
                            "fmla   v29.4s, v15.4s, v7.s[2] \n"
                            "fmla   v30.4s, v14.4s, v7.s[3] \n"
                            "fmla   v31.4s, v15.4s, v7.s[3] \n"

                            "bne    0b                      \n"

                            "1:                             \n"

                            // remain loop
                            "and    w4, %w20, #3            \n" // w4 = remain = tiles & 3
                            "cmp    w4, #0                  \n"
                            "beq    3f                      \n"

                            "2:                             \n"

                            "prfm   pldl1keep, [%8, #256]   \n"
                            "ld1    {v8.4s, v9.4s}, [%8], #32   \n"

                            "prfm   pldl1keep, [%9, #256]   \n"
                            "ld1    {v0.4s, v1.4s}, [%9], #32   \n"

                            "fmla   v16.4s, v8.4s, v0.s[0]  \n"
                            "fmla   v17.4s, v9.4s, v0.s[0]  \n"
                            "fmla   v18.4s, v8.4s, v0.s[1]  \n"
                            "fmla   v19.4s, v9.4s, v0.s[1]  \n"
                            "fmla   v20.4s, v8.4s, v0.s[2]  \n"
                            "fmla   v21.4s, v9.4s, v0.s[2]  \n"
                            "fmla   v22.4s, v8.4s, v0.s[3]  \n"
                            "fmla   v23.4s, v9.4s, v0.s[3]  \n"

                            "subs   w4, w4, #1              \n"

                            "fmla   v24.4s, v8.4s, v1.s[0]  \n"
                            "fmla   v25.4s, v9.4s, v1.s[0]  \n"
                            "fmla   v26.4s, v8.4s, v1.s[1]  \n"
                            "fmla   v27.4s, v9.4s, v1.s[1]  \n"
                            "fmla   v28.4s, v8.4s, v1.s[2]  \n"
                            "fmla   v29.4s, v9.4s, v1.s[2]  \n"
                            "fmla   v30.4s, v8.4s, v1.s[3]  \n"
                            "fmla   v31.4s, v9.4s, v1.s[3]  \n"

                            "bne    2b                      \n"

                            "3:                             \n"

                            "st1    {v16.4s, v17.4s}, [%0], #32 \n"
                            "st1    {v18.4s, v19.4s}, [%1], #32 \n"
                            "st1    {v20.4s, v21.4s}, [%2], #32 \n"
                            "st1    {v22.4s, v23.4s}, [%3], #32 \n"
                            "st1    {v24.4s, v25.4s}, [%4], #32 \n"
                            "st1    {v26.4s, v27.4s}, [%5], #32 \n"
                            "st1    {v28.4s, v29.4s}, [%6], #32 \n"
                            "st1    {v30.4s, v31.4s}, [%7], #32 \n"

                            : "=r"(output0_tm), // %0
                            "=r"(output1_tm), // %1
                            "=r"(output2_tm), // %2
                            "=r"(output3_tm), // %3
                            "=r"(output4_tm), // %4
                            "=r"(output5_tm), // %5
                            "=r"(output6_tm), // %6
                            "=r"(output7_tm), // %7
                            "=r"(bb2p0),      // %8
                            "=r"(ktm0)        // %9
                            : "0"(output0_tm),
                            "1"(output1_tm),
                            "2"(output2_tm),
                            "3"(output3_tm),
                            "4"(output4_tm),
                            "5"(output5_tm),
                            "6"(output6_tm),
                            "7"(output7_tm),
                            "8"(bb2p0),
                            "9"(ktm0),
                            "r"(inch) // %20
                            : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
                    }
                    for (; i + 3 < tiles; i += 4)
                    {
                        const float* bb2p0 = bb2[i / 8 + (i % 8) / 4].data();

                        const float* ktm0 = kernel_tm0[r].data();

                        asm volatile(
                            "eor    v16.16b, v16.16b, v16.16b  \n"
                            "eor    v17.16b, v17.16b, v17.16b  \n"
                            "eor    v18.16b, v18.16b, v18.16b  \n"
                            "eor    v19.16b, v19.16b, v19.16b  \n"
                            "eor    v20.16b, v20.16b, v20.16b  \n"
                            "eor    v21.16b, v21.16b, v21.16b  \n"
                            "eor    v22.16b, v22.16b, v22.16b  \n"
                            "eor    v23.16b, v23.16b, v23.16b  \n"

                            // inch loop
                            "lsr    w4, %w20, #2            \n" // w4 = nn = inch >> 2
                            "cmp    w4, #0                  \n"
                            "beq    1f                      \n"

                            "0:                             \n"

                            "prfm   pldl1keep, [%8, #512]   \n"
                            "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%8], #64 \n"

                            "prfm   pldl1keep, [%9, #512]   \n"
                            "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64   \n"

                            "fmla   v16.4s, v8.4s, v0.s[0]  \n"
                            "fmla   v17.4s, v8.4s, v0.s[1]  \n"
                            "fmla   v18.4s, v8.4s, v0.s[2]  \n"
                            "fmla   v19.4s, v8.4s, v0.s[3]  \n"
                            "fmla   v20.4s, v8.4s, v1.s[0]  \n"
                            "fmla   v21.4s, v8.4s, v1.s[1]  \n"
                            "fmla   v22.4s, v8.4s, v1.s[2]  \n"
                            "fmla   v23.4s, v8.4s, v1.s[3]  \n"

                            "prfm   pldl1keep, [%9, #512]   \n"
                            "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%9], #64   \n"

                            "fmla   v16.4s, v9.4s, v2.s[0]  \n"
                            "fmla   v17.4s, v9.4s, v2.s[1]  \n"
                            "fmla   v18.4s, v9.4s, v2.s[2]  \n"
                            "fmla   v19.4s, v9.4s, v2.s[3]  \n"
                            "fmla   v20.4s, v9.4s, v3.s[0]  \n"
                            "fmla   v21.4s, v9.4s, v3.s[1]  \n"
                            "fmla   v22.4s, v9.4s, v3.s[2]  \n"
                            "fmla   v23.4s, v9.4s, v3.s[3]  \n"

                            "fmla   v16.4s, v10.4s, v4.s[0] \n"
                            "fmla   v17.4s, v10.4s, v4.s[1] \n"
                            "fmla   v18.4s, v10.4s, v4.s[2] \n"
                            "fmla   v19.4s, v10.4s, v4.s[3] \n"
                            "fmla   v20.4s, v10.4s, v5.s[0] \n"
                            "fmla   v21.4s, v10.4s, v5.s[1] \n"
                            "fmla   v22.4s, v10.4s, v5.s[2] \n"
                            "fmla   v23.4s, v10.4s, v5.s[3] \n"

                            "subs   w4, w4, #1              \n"

                            "fmla   v16.4s, v11.4s, v6.s[0] \n"
                            "fmla   v17.4s, v11.4s, v6.s[1] \n"
                            "fmla   v18.4s, v11.4s, v6.s[2] \n"
                            "fmla   v19.4s, v11.4s, v6.s[3] \n"
                            "fmla   v20.4s, v11.4s, v7.s[0] \n"
                            "fmla   v21.4s, v11.4s, v7.s[1] \n"
                            "fmla   v22.4s, v11.4s, v7.s[2] \n"
                            "fmla   v23.4s, v11.4s, v7.s[3] \n"

                            "bne    0b                      \n"

                            "1:                             \n"

                            // remain loop
                            "and    w4, %w20, #3            \n" // w4 = remain = tiles & 3
                            "cmp    w4, #0                  \n"
                            "beq    3f                      \n"

                            "2:                             \n"

                            "prfm   pldl1keep, [%8, #128]   \n"
                            "ld1    {v8.4s}, [%8], #16      \n"

                            "prfm   pldl1keep, [%9, #256]   \n"
                            "ld1    {v0.4s, v1.4s}, [%9], #32   \n"

                            "fmla   v16.4s, v8.4s, v0.s[0]  \n"
                            "fmla   v17.4s, v8.4s, v0.s[1]  \n"
                            "fmla   v18.4s, v8.4s, v0.s[2]  \n"
                            "fmla   v19.4s, v8.4s, v0.s[3]  \n"

                            "subs   w4, w4, #1              \n"

                            "fmla   v20.4s, v8.4s, v1.s[0]  \n"
                            "fmla   v21.4s, v8.4s, v1.s[1]  \n"
                            "fmla   v22.4s, v8.4s, v1.s[2]  \n"
                            "fmla   v23.4s, v8.4s, v1.s[3]  \n"

                            "bne    2b                      \n"

                            "3:                             \n"

                            "st1    {v16.4s}, [%0], #16     \n"
                            "st1    {v17.4s}, [%1], #16     \n"
                            "st1    {v18.4s}, [%2], #16     \n"
                            "st1    {v19.4s}, [%3], #16     \n"
                            "st1    {v20.4s}, [%4], #16     \n"
                            "st1    {v21.4s}, [%5], #16     \n"
                            "st1    {v22.4s}, [%6], #16     \n"
                            "st1    {v23.4s}, [%7], #16     \n"

                            : "=r"(output0_tm), // %0
                            "=r"(output1_tm), // %1
                            "=r"(output2_tm), // %2
                            "=r"(output3_tm), // %3
                            "=r"(output4_tm), // %4
                            "=r"(output5_tm), // %5
                            "=r"(output6_tm), // %6
                            "=r"(output7_tm), // %7
                            "=r"(bb2p0),      // %8
                            "=r"(ktm0)        // %9
                            : "0"(output0_tm),
                            "1"(output1_tm),
                            "2"(output2_tm),
                            "3"(output3_tm),
                            "4"(output4_tm),
                            "5"(output5_tm),
                            "6"(output6_tm),
                            "7"(output7_tm),
                            "8"(bb2p0),
                            "9"(ktm0),
                            "r"(inch) // %20
                            : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
                    }
                    for (; i < tiles; i++)
                    {
                        const float* bb2p0 = bb2[i / 8 + (i % 8) / 4 + i % 4].data();

                        const float* ktm0 = kernel_tm0[r].data();

                        float32x4_t _sum0123 = vdupq_n_f32(0.f);
                        float32x4_t _sum4567 = vdupq_n_f32(0.f);

                        int q = 0;
                        for (; q + 3 < inch; q += 4)
                        {
                            //                         asm volatile("prfm pldl1keep, [%0, #128] \n" : :"r"(bb2p0) :);
                            float32x4_t _bb2p0 = vld1q_f32(bb2p0);
                            bb2p0 += 4;

                            //                         asm volatile("prfm pldl1keep, [%0, #512] \n" : :"r"(ktm0) :);
                            float32x4_t _ktm0 = vld1q_f32(ktm0 + 0);
                            float32x4_t _ktm1 = vld1q_f32(ktm0 + 4);
                            float32x4_t _ktm2 = vld1q_f32(ktm0 + 8);
                            float32x4_t _ktm3 = vld1q_f32(ktm0 + 12);
                            ktm0 += 16;

                            _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm0, _bb2p0, 0);
                            _sum4567 = vmlaq_laneq_f32(_sum4567, _ktm1, _bb2p0, 0);
                            _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm2, _bb2p0, 1);
                            _sum4567 = vmlaq_laneq_f32(_sum4567, _ktm3, _bb2p0, 1);

                            //                         asm volatile("prfm pldl1keep, [%0, #512] \n" : :"r"(ktm0) :);
                            float32x4_t _ktm4 = vld1q_f32(ktm0 + 0);
                            float32x4_t _ktm5 = vld1q_f32(ktm0 + 4);
                            float32x4_t _ktm6 = vld1q_f32(ktm0 + 8);
                            float32x4_t _ktm7 = vld1q_f32(ktm0 + 12);
                            ktm0 += 16;

                            _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm4, _bb2p0, 2);
                            _sum4567 = vmlaq_laneq_f32(_sum4567, _ktm5, _bb2p0, 2);
                            _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm6, _bb2p0, 3);
                            _sum4567 = vmlaq_laneq_f32(_sum4567, _ktm7, _bb2p0, 3);
                        }

                        for (; q < inch; q++)
                        {
                            float32x4_t _bb2p0 = vld1q_dup_f32(bb2p0);
                            float32x4_t _ktm0123 = vld1q_f32(ktm0 + 0);
                            float32x4_t _ktm4567 = vld1q_f32(ktm0 + 4);

                            _sum0123 = vmlaq_f32(_sum0123, _bb2p0, _ktm0123);
                            _sum4567 = vmlaq_f32(_sum4567, _bb2p0, _ktm4567);

                            bb2p0 += 1;
                            ktm0 += 8;
                        }

                        float sum0 = vgetq_lane_f32(_sum0123, 0);
                        float sum1 = vgetq_lane_f32(_sum0123, 1);
                        float sum2 = vgetq_lane_f32(_sum0123, 2);
                        float sum3 = vgetq_lane_f32(_sum0123, 3);
                        float sum4 = vgetq_lane_f32(_sum4567, 0);
                        float sum5 = vgetq_lane_f32(_sum4567, 1);
                        float sum6 = vgetq_lane_f32(_sum4567, 2);
                        float sum7 = vgetq_lane_f32(_sum4567, 3);

                        output0_tm[0] = sum0;
                        output1_tm[0] = sum1;
                        output2_tm[0] = sum2;
                        output3_tm[0] = sum3;
                        output4_tm[0] = sum4;
                        output5_tm[0] = sum5;
                        output6_tm[0] = sum6;
                        output7_tm[0] = sum7;

                        output0_tm += 1;
                        output1_tm += 1;
                        output2_tm += 1;
                        output3_tm += 1;
                        output4_tm += 1;
                        output5_tm += 1;
                        output6_tm += 1;
                        output7_tm += 1;
                    }
                }
            }
        });
#endif // __aarch64__

        nn_outch = (outch - remain_outch_start) >> 2;

        otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
            for (const auto pp : otter::irange(begin, end)) {
                int p = remain_outch_start + pp * 4;

    #if __ARM_NEON && __aarch64__
                const auto kernel_tm0 = kernel_tf_a[p / 8 + (p % 8) / 4];
    #else
                const auto kernel_tm0 = kernel_tf_a[p / 4];
    #endif

                auto out0_tm = output_tf_a[p];
                auto out1_tm = output_tf_a[p + 1];
                auto out2_tm = output_tf_a[p + 2];
                auto out3_tm = output_tf_a[p + 3];

                float* output0_tm = out0_tm.data();
                float* output1_tm = out1_tm.data();
                float* output2_tm = out2_tm.data();
                float* output3_tm = out3_tm.data();

                for (int r = 0; r < 64; r++)
                {
                    const auto bb2 = input_tf2_a[r];

                    // tile
                    int i = 0;
                    for (; i + 7 < tiles; i += 8)
                    {
                        const float* bb2p0 = bb2[i / 8].data();

                        const float* ktm0 = kernel_tm0[r].data();
    #if __ARM_NEON
    #if __aarch64__
                        asm volatile(
                            "eor    v8.16b, v8.16b, v8.16b     \n"
                            "eor    v9.16b, v9.16b, v9.16b     \n"
                            "eor    v10.16b, v10.16b, v10.16b  \n"
                            "eor    v11.16b, v11.16b, v11.16b  \n"
                            "eor    v12.16b, v12.16b, v12.16b  \n"
                            "eor    v13.16b, v13.16b, v13.16b  \n"
                            "eor    v14.16b, v14.16b, v14.16b  \n"
                            "eor    v15.16b, v15.16b, v15.16b  \n"

                            // inch loop
                            "lsr    w4, %w12, #2            \n" // w4 = nn = inch >> 2
                            "cmp    w4, #0                  \n"
                            "beq    1f                      \n"

                            "0:                             \n"

                            "prfm   pldl1keep, [%4, #512]   \n"
                            "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64     \n"

                            "prfm   pldl1keep, [%5, #512]   \n"
                            "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64     \n"

                            "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                            "fmla   v9.4s, v5.4s, v0.s[0]   \n"
                            "fmla   v10.4s, v4.4s, v0.s[1]  \n"
                            "fmla   v11.4s, v5.4s, v0.s[1]  \n"
                            "fmla   v12.4s, v4.4s, v0.s[2]  \n"
                            "fmla   v13.4s, v5.4s, v0.s[2]  \n"
                            "fmla   v14.4s, v4.4s, v0.s[3]  \n"
                            "fmla   v15.4s, v5.4s, v0.s[3]  \n"

                            "prfm   pldl1keep, [%4, #512]   \n"
                            "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%4], #64 \n"

                            "fmla   v8.4s, v6.4s, v1.s[0]   \n"
                            "fmla   v9.4s, v7.4s, v1.s[0]   \n"
                            "fmla   v10.4s, v6.4s, v1.s[1]  \n"
                            "fmla   v11.4s, v7.4s, v1.s[1]  \n"
                            "fmla   v12.4s, v6.4s, v1.s[2]  \n"
                            "fmla   v13.4s, v7.4s, v1.s[2]  \n"
                            "fmla   v14.4s, v6.4s, v1.s[3]  \n"
                            "fmla   v15.4s, v7.4s, v1.s[3]  \n"

                            "fmla   v8.4s, v16.4s, v2.s[0]  \n"
                            "fmla   v9.4s, v17.4s, v2.s[0]  \n"
                            "fmla   v10.4s, v16.4s, v2.s[1] \n"
                            "fmla   v11.4s, v17.4s, v2.s[1] \n"
                            "fmla   v12.4s, v16.4s, v2.s[2] \n"
                            "fmla   v13.4s, v17.4s, v2.s[2] \n"
                            "fmla   v14.4s, v16.4s, v2.s[3] \n"
                            "fmla   v15.4s, v17.4s, v2.s[3] \n"

                            "fmla   v8.4s, v18.4s, v3.s[0]  \n"
                            "fmla   v9.4s, v19.4s, v3.s[0]  \n"
                            "fmla   v10.4s, v18.4s, v3.s[1] \n"
                            "fmla   v11.4s, v19.4s, v3.s[1] \n"
                            "fmla   v12.4s, v18.4s, v3.s[2] \n"
                            "fmla   v13.4s, v19.4s, v3.s[2] \n"
                            "fmla   v14.4s, v18.4s, v3.s[3] \n"
                            "fmla   v15.4s, v19.4s, v3.s[3] \n"

                            "subs   w4, w4, #1              \n"
                            "bne    0b                      \n"

                            "1:                             \n"

                            // remain loop
                            "and    w4, %w12, #3            \n" // w4 = remain = tiles & 3
                            "cmp    w4, #0                  \n"
                            "beq    3f                      \n"

                            "2:                             \n"

                            "prfm   pldl1keep, [%4, #256]   \n"
                            "ld1    {v4.4s, v5.4s}, [%4], #32      \n"

                            "prfm   pldl1keep, [%5, #128]   \n"
                            "ld1    {v0.4s}, [%5], #16      \n"

                            "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                            "fmla   v9.4s, v5.4s, v0.s[0]   \n"
                            "fmla   v10.4s, v4.4s, v0.s[1]  \n"
                            "fmla   v11.4s, v5.4s, v0.s[1]  \n"
                            "fmla   v12.4s, v4.4s, v0.s[2]  \n"
                            "fmla   v13.4s, v5.4s, v0.s[2]  \n"
                            "fmla   v14.4s, v4.4s, v0.s[3]  \n"
                            "fmla   v15.4s, v5.4s, v0.s[3]  \n"

                            "subs   w4, w4, #1              \n"
                            "bne    2b                      \n"

                            "3:                             \n"

                            "st1    {v8.4s, v9.4s}, [%0], #32       \n"
                            "st1    {v10.4s, v11.4s}, [%1], #32     \n"
                            "st1    {v12.4s, v13.4s}, [%2], #32     \n"
                            "st1    {v14.4s, v15.4s}, [%3], #32     \n"

                            : "=r"(output0_tm), // %0
                            "=r"(output1_tm), // %1
                            "=r"(output2_tm), // %2
                            "=r"(output3_tm), // %3
                            "=r"(bb2p0),      // %4
                            "=r"(ktm0)        // %5
                            : "0"(output0_tm),
                            "1"(output1_tm),
                            "2"(output2_tm),
                            "3"(output3_tm),
                            "4"(bb2p0),
                            "5"(ktm0),
                            "r"(inch) // %12
                            : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
    #else  // __aarch64__
                        asm volatile(
                            "veor       q8, q8, q8      \n"
                            "veor       q9, q9, q9      \n"
                            "veor       q10, q10, q10   \n"
                            "veor       q11, q11, q11   \n"
                            "veor       q12, q12, q12   \n"
                            "veor       q13, q13, q13   \n"
                            "veor       q14, q14, q14   \n"
                            "veor       q15, q15, q15   \n"

                            // inch loop
                            "lsr        r4, %12, #2     \n" // r4 = nn = inch >> 2
                            "cmp        r4, #0          \n"
                            "beq        1f              \n"

                            "0:                         \n"

                            "pld        [%4, #512]      \n"
                            "vldm       %4!, {d8-d15}   \n"
                            //                         "vld1.f32   {d8-d11}, [%4 :128]! \n"
                            //                         "vld1.f32   {d12-d15}, [%4 :128]! \n"

                            "pld        [%5, #512]      \n"
                            "vldm       %5!, {d0-d7}    \n"
                            //                         "vld1.f32   {d0-d3}, [%5 :128]!  \n"
                            //                         "vld1.f32   {d4-d7}, [%5 :128]!  \n"

                            "vmla.f32   q8, q4, d0[0]   \n"
                            "vmla.f32   q9, q5, d0[0]   \n"
                            "vmla.f32   q10, q4, d0[1]  \n"
                            "vmla.f32   q11, q5, d0[1]  \n"
                            "vmla.f32   q12, q4, d1[0]  \n"
                            "vmla.f32   q13, q5, d1[0]  \n"
                            "vmla.f32   q14, q4, d1[1]  \n"
                            "vmla.f32   q15, q5, d1[1]  \n"

                            "vmla.f32   q8, q6, d2[0]   \n"
                            "vmla.f32   q9, q7, d2[0]   \n"
                            "vmla.f32   q10, q6, d2[1]  \n"
                            "vmla.f32   q11, q7, d2[1]  \n"
                            "vmla.f32   q12, q6, d3[0]  \n"
                            "vmla.f32   q13, q7, d3[0]  \n"
                            "vmla.f32   q14, q6, d3[1]  \n"
                            "vmla.f32   q15, q7, d3[1]  \n"

                            "pld        [%4, #512]      \n"
                            "vldm       %4!, {d8-d15}   \n"
                            //                         "vld1.f32   {d8-d11}, [%4 :128]! \n"
                            //                         "vld1.f32   {d12-d15}, [%4 :128]! \n"

                            "vmla.f32   q8, q4, d4[0]   \n"
                            "vmla.f32   q9, q5, d4[0]   \n"
                            "vmla.f32   q10, q4, d4[1]  \n"
                            "vmla.f32   q11, q5, d4[1]  \n"
                            "vmla.f32   q12, q4, d5[0]  \n"
                            "vmla.f32   q13, q5, d5[0]  \n"
                            "vmla.f32   q14, q4, d5[1]  \n"
                            "vmla.f32   q15, q5, d5[1]  \n"

                            "subs       r4, r4, #1      \n"

                            "vmla.f32   q8, q6, d6[0]   \n"
                            "vmla.f32   q9, q7, d6[0]   \n"
                            "vmla.f32   q10, q6, d6[1]  \n"
                            "vmla.f32   q11, q7, d6[1]  \n"
                            "vmla.f32   q12, q6, d7[0]  \n"
                            "vmla.f32   q13, q7, d7[0]  \n"
                            "vmla.f32   q14, q6, d7[1]  \n"
                            "vmla.f32   q15, q7, d7[1]  \n"

                            "bne        0b              \n"

                            "1:                         \n"

                            // remain loop
                            "and        r4, %12, #3     \n" // r4 = remain = tiles & 3
                            "cmp        r4, #0          \n"
                            "beq        3f              \n"

                            "2:                         \n"

                            "pld        [%4, #256]      \n"
                            "vld1.f32   {d8-d11}, [%4 :128]! \n"

                            "pld        [%5, #128]      \n"
                            "vld1.f32   {d0-d1}, [%5 :128]!  \n"

                            "vmla.f32   q8, q4, d0[0]   \n"
                            "vmla.f32   q9, q5, d0[0]   \n"
                            "vmla.f32   q10, q4, d0[1]  \n"
                            "vmla.f32   q11, q5, d0[1]  \n"

                            "subs       r4, r4, #1      \n"

                            "vmla.f32   q12, q4, d1[0]  \n"
                            "vmla.f32   q13, q5, d1[0]  \n"
                            "vmla.f32   q14, q4, d1[1]  \n"
                            "vmla.f32   q15, q5, d1[1]  \n"

                            "bne        2b              \n"

                            "3:                         \n"

                            "vst1.f32   {d16-d19}, [%0]! \n"
                            "vst1.f32   {d20-d23}, [%1]! \n"
                            "vst1.f32   {d24-d27}, [%2]! \n"
                            "vst1.f32   {d28-d31}, [%3]! \n"

                            : "=r"(output0_tm), // %0
                            "=r"(output1_tm), // %1
                            "=r"(output2_tm), // %2
                            "=r"(output3_tm), // %3
                            "=r"(bb2p0),      // %4
                            "=r"(ktm0)        // %5
                            : "0"(output0_tm),
                            "1"(output1_tm),
                            "2"(output2_tm),
                            "3"(output3_tm),
                            "4"(bb2p0),
                            "5"(ktm0),
                            "r"(inch) // %12
                            : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
    #endif // __aarch64__
    #else
                        float sum0_0 = 0.f;
                        float sum0_1 = 0.f;
                        float sum0_2 = 0.f;
                        float sum0_3 = 0.f;
                        float sum0_4 = 0.f;
                        float sum0_5 = 0.f;
                        float sum0_6 = 0.f;
                        float sum0_7 = 0.f;

                        float sum1_0 = 0.f;
                        float sum1_1 = 0.f;
                        float sum1_2 = 0.f;
                        float sum1_3 = 0.f;
                        float sum1_4 = 0.f;
                        float sum1_5 = 0.f;
                        float sum1_6 = 0.f;
                        float sum1_7 = 0.f;

                        float sum2_0 = 0.f;
                        float sum2_1 = 0.f;
                        float sum2_2 = 0.f;
                        float sum2_3 = 0.f;
                        float sum2_4 = 0.f;
                        float sum2_5 = 0.f;
                        float sum2_6 = 0.f;
                        float sum2_7 = 0.f;

                        float sum3_0 = 0.f;
                        float sum3_1 = 0.f;
                        float sum3_2 = 0.f;
                        float sum3_3 = 0.f;
                        float sum3_4 = 0.f;
                        float sum3_5 = 0.f;
                        float sum3_6 = 0.f;
                        float sum3_7 = 0.f;

                        for (int q = 0; q < inch; q++)
                        {
                            sum0_0 += bb2p0[0] * ktm0[0];
                            sum0_1 += bb2p0[1] * ktm0[0];
                            sum0_2 += bb2p0[2] * ktm0[0];
                            sum0_3 += bb2p0[3] * ktm0[0];
                            sum0_4 += bb2p0[4] * ktm0[0];
                            sum0_5 += bb2p0[5] * ktm0[0];
                            sum0_6 += bb2p0[6] * ktm0[0];
                            sum0_7 += bb2p0[7] * ktm0[0];

                            sum1_0 += bb2p0[0] * ktm0[1];
                            sum1_1 += bb2p0[1] * ktm0[1];
                            sum1_2 += bb2p0[2] * ktm0[1];
                            sum1_3 += bb2p0[3] * ktm0[1];
                            sum1_4 += bb2p0[4] * ktm0[1];
                            sum1_5 += bb2p0[5] * ktm0[1];
                            sum1_6 += bb2p0[6] * ktm0[1];
                            sum1_7 += bb2p0[7] * ktm0[1];

                            sum2_0 += bb2p0[0] * ktm0[2];
                            sum2_1 += bb2p0[1] * ktm0[2];
                            sum2_2 += bb2p0[2] * ktm0[2];
                            sum2_3 += bb2p0[3] * ktm0[2];
                            sum2_4 += bb2p0[4] * ktm0[2];
                            sum2_5 += bb2p0[5] * ktm0[2];
                            sum2_6 += bb2p0[6] * ktm0[2];
                            sum2_7 += bb2p0[7] * ktm0[2];

                            sum3_0 += bb2p0[0] * ktm0[3];
                            sum3_1 += bb2p0[1] * ktm0[3];
                            sum3_2 += bb2p0[2] * ktm0[3];
                            sum3_3 += bb2p0[3] * ktm0[3];
                            sum3_4 += bb2p0[4] * ktm0[3];
                            sum3_5 += bb2p0[5] * ktm0[3];
                            sum3_6 += bb2p0[6] * ktm0[3];
                            sum3_7 += bb2p0[7] * ktm0[3];

                            bb2p0 += 8;
                            ktm0 += 4;
                        }

                        output0_tm[0] = sum0_0;
                        output0_tm[1] = sum0_1;
                        output0_tm[2] = sum0_2;
                        output0_tm[3] = sum0_3;
                        output0_tm[4] = sum0_4;
                        output0_tm[5] = sum0_5;
                        output0_tm[6] = sum0_6;
                        output0_tm[7] = sum0_7;

                        output1_tm[0] = sum1_0;
                        output1_tm[1] = sum1_1;
                        output1_tm[2] = sum1_2;
                        output1_tm[3] = sum1_3;
                        output1_tm[4] = sum1_4;
                        output1_tm[5] = sum1_5;
                        output1_tm[6] = sum1_6;
                        output1_tm[7] = sum1_7;

                        output2_tm[0] = sum2_0;
                        output2_tm[1] = sum2_1;
                        output2_tm[2] = sum2_2;
                        output2_tm[3] = sum2_3;
                        output2_tm[4] = sum2_4;
                        output2_tm[5] = sum2_5;
                        output2_tm[6] = sum2_6;
                        output2_tm[7] = sum2_7;

                        output3_tm[0] = sum3_0;
                        output3_tm[1] = sum3_1;
                        output3_tm[2] = sum3_2;
                        output3_tm[3] = sum3_3;
                        output3_tm[4] = sum3_4;
                        output3_tm[5] = sum3_5;
                        output3_tm[6] = sum3_6;
                        output3_tm[7] = sum3_7;

                        output0_tm += 8;
                        output1_tm += 8;
                        output2_tm += 8;
                        output3_tm += 8;
    #endif // __ARM_NEON
                    }
                    for (; i + 3 < tiles; i += 4)
                    {
                        const float* bb2p0 = bb2[i / 8 + (i % 8) / 4].data();

                        const float* ktm0 = kernel_tm0[r].data();
    #if __ARM_NEON
    #if __aarch64__
                        asm volatile(
                            "eor    v8.16b, v8.16b, v8.16b     \n"
                            "eor    v9.16b, v9.16b, v9.16b     \n"
                            "eor    v10.16b, v10.16b, v10.16b  \n"
                            "eor    v11.16b, v11.16b, v11.16b  \n"

                            // inch loop
                            "lsr    w4, %w12, #2            \n" // w4 = nn = inch >> 2
                            "cmp    w4, #0                  \n"
                            "beq    1f                      \n"

                            "0:                             \n"

                            "prfm   pldl1keep, [%4, #512]   \n"
                            "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64     \n"

                            "prfm   pldl1keep, [%5, #512]   \n"
                            "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64     \n"

                            "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                            "fmla   v9.4s, v4.4s, v0.s[1]   \n"
                            "fmla   v10.4s, v4.4s, v0.s[2]  \n"
                            "fmla   v11.4s, v4.4s, v0.s[3]  \n"

                            "fmla   v8.4s, v5.4s, v1.s[0]   \n"
                            "fmla   v9.4s, v5.4s, v1.s[1]   \n"
                            "fmla   v10.4s, v5.4s, v1.s[2]  \n"
                            "fmla   v11.4s, v5.4s, v1.s[3]  \n"

                            "fmla   v8.4s, v6.4s, v2.s[0]   \n"
                            "fmla   v9.4s, v6.4s, v2.s[1]   \n"
                            "fmla   v10.4s, v6.4s, v2.s[2]  \n"
                            "fmla   v11.4s, v6.4s, v2.s[3]  \n"

                            "fmla   v8.4s, v7.4s, v3.s[0]   \n"
                            "fmla   v9.4s, v7.4s, v3.s[1]   \n"
                            "fmla   v10.4s, v7.4s, v3.s[2]  \n"
                            "fmla   v11.4s, v7.4s, v3.s[3]  \n"

                            "subs   w4, w4, #1              \n"
                            "bne    0b                      \n"

                            "1:                             \n"

                            // remain loop
                            "and    w4, %w12, #3            \n" // w4 = remain = tiles & 3
                            "cmp    w4, #0                  \n"
                            "beq    3f                      \n"

                            "2:                             \n"

                            "prfm   pldl1keep, [%4, #128]   \n"
                            "ld1    {v4.4s}, [%4], #16      \n"

                            "prfm   pldl1keep, [%5, #128]   \n"
                            "ld1    {v0.4s}, [%5], #16      \n"

                            "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                            "fmla   v9.4s, v4.4s, v0.s[1]   \n"
                            "fmla   v10.4s, v4.4s, v0.s[2]  \n"
                            "fmla   v11.4s, v4.4s, v0.s[3]  \n"

                            "subs   w4, w4, #1              \n"
                            "bne    2b                      \n"

                            "3:                             \n"

                            "st1    {v8.4s}, [%0], #16      \n"
                            "st1    {v9.4s}, [%1], #16      \n"
                            "st1    {v10.4s}, [%2], #16     \n"
                            "st1    {v11.4s}, [%3], #16     \n"

                            : "=r"(output0_tm), // %0
                            "=r"(output1_tm), // %1
                            "=r"(output2_tm), // %2
                            "=r"(output3_tm), // %3
                            "=r"(bb2p0),      // %4
                            "=r"(ktm0)        // %5
                            : "0"(output0_tm),
                            "1"(output1_tm),
                            "2"(output2_tm),
                            "3"(output3_tm),
                            "4"(bb2p0),
                            "5"(ktm0),
                            "r"(inch) // %12
                            : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
    #else  // __aarch64__
                        asm volatile(
                            "veor       q8, q8, q8      \n"
                            "veor       q9, q9, q9      \n"
                            "veor       q10, q10, q10   \n"
                            "veor       q11, q11, q11   \n"

                            // inch loop
                            "lsr        r4, %12, #2     \n" // r4 = nn = inch >> 2
                            "cmp        r4, #0          \n"
                            "beq        1f              \n"

                            "0:                         \n"

                            "pld        [%4, #512]      \n"
                            "vldm       %4!, {d8-d15}   \n"
                            //                         "vld1.f32   {d8-d11}, [%4 :128]! \n"
                            //                         "vld1.f32   {d12-d15}, [%4 :128]! \n"

                            "pld        [%5, #512]      \n"
                            "vldm       %5!, {d0-d7}    \n"
                            //                         "vld1.f32   {d0-d3}, [%5 :128]!  \n"
                            //                         "vld1.f32   {d4-d7}, [%5 :128]!  \n"

                            "vmla.f32   q8, q4, d0[0]   \n"
                            "vmla.f32   q9, q4, d0[1]   \n"
                            "vmla.f32   q10, q4, d1[0]  \n"
                            "vmla.f32   q11, q4, d1[1]  \n"

                            "vmla.f32   q8, q5, d2[0]   \n"
                            "vmla.f32   q9, q5, d2[1]   \n"
                            "vmla.f32   q10, q5, d3[0]  \n"
                            "vmla.f32   q11, q5, d3[1]  \n"

                            "subs       r4, r4, #1      \n"

                            "vmla.f32   q8, q6, d4[0]   \n"
                            "vmla.f32   q9, q6, d4[1]   \n"
                            "vmla.f32   q10, q6, d5[0]  \n"
                            "vmla.f32   q11, q6, d5[1]  \n"

                            "vmla.f32   q8, q7, d6[0]   \n"
                            "vmla.f32   q9, q7, d6[1]   \n"
                            "vmla.f32   q10, q7, d7[0]  \n"
                            "vmla.f32   q11, q7, d7[1]  \n"

                            "bne        0b              \n"

                            "1:                         \n"

                            // remain loop
                            "and        r4, %12, #3     \n" // r4 = remain = tiles & 3
                            "cmp        r4, #0          \n"
                            "beq        3f              \n"

                            "2:                         \n"

                            "pld        [%4, #128]      \n"
                            "vld1.f32   {d8-d9}, [%4 :128]!  \n"

                            "pld        [%5, #128]      \n"
                            "vld1.f32   {d0-d1}, [%5 :128]!  \n"

                            "subs       r4, r4, #1      \n"

                            "vmla.f32   q8, q4, d0[0]   \n"
                            "vmla.f32   q9, q4, d0[1]   \n"
                            "vmla.f32   q10, q4, d1[0]  \n"
                            "vmla.f32   q11, q4, d1[1]  \n"

                            "bne        2b              \n"

                            "3:                         \n"

                            "vst1.f32   {d16-d17}, [%0]! \n"
                            "vst1.f32   {d18-d19}, [%1]! \n"
                            "vst1.f32   {d20-d21}, [%2]! \n"
                            "vst1.f32   {d22-d23}, [%3]! \n"

                            : "=r"(output0_tm), // %0
                            "=r"(output1_tm), // %1
                            "=r"(output2_tm), // %2
                            "=r"(output3_tm), // %3
                            "=r"(bb2p0),      // %4
                            "=r"(ktm0)        // %5
                            : "0"(output0_tm),
                            "1"(output1_tm),
                            "2"(output2_tm),
                            "3"(output3_tm),
                            "4"(bb2p0),
                            "5"(ktm0),
                            "r"(inch) // %12
                            : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
    #endif // __aarch64__
    #else
                        float sum0_0 = 0.f;
                        float sum0_1 = 0.f;
                        float sum0_2 = 0.f;
                        float sum0_3 = 0.f;

                        float sum1_0 = 0.f;
                        float sum1_1 = 0.f;
                        float sum1_2 = 0.f;
                        float sum1_3 = 0.f;

                        float sum2_0 = 0.f;
                        float sum2_1 = 0.f;
                        float sum2_2 = 0.f;
                        float sum2_3 = 0.f;

                        float sum3_0 = 0.f;
                        float sum3_1 = 0.f;
                        float sum3_2 = 0.f;
                        float sum3_3 = 0.f;

                        for (int q = 0; q < inch; q++)
                        {
                            sum0_0 += bb2p0[0] * ktm0[0];
                            sum0_1 += bb2p0[1] * ktm0[0];
                            sum0_2 += bb2p0[2] * ktm0[0];
                            sum0_3 += bb2p0[3] * ktm0[0];

                            sum1_0 += bb2p0[0] * ktm0[1];
                            sum1_1 += bb2p0[1] * ktm0[1];
                            sum1_2 += bb2p0[2] * ktm0[1];
                            sum1_3 += bb2p0[3] * ktm0[1];

                            sum2_0 += bb2p0[0] * ktm0[2];
                            sum2_1 += bb2p0[1] * ktm0[2];
                            sum2_2 += bb2p0[2] * ktm0[2];
                            sum2_3 += bb2p0[3] * ktm0[2];

                            sum3_0 += bb2p0[0] * ktm0[3];
                            sum3_1 += bb2p0[1] * ktm0[3];
                            sum3_2 += bb2p0[2] * ktm0[3];
                            sum3_3 += bb2p0[3] * ktm0[3];

                            bb2p0 += 4;
                            ktm0 += 4;
                        }

                        output0_tm[0] = sum0_0;
                        output0_tm[1] = sum0_1;
                        output0_tm[2] = sum0_2;
                        output0_tm[3] = sum0_3;

                        output1_tm[0] = sum1_0;
                        output1_tm[1] = sum1_1;
                        output1_tm[2] = sum1_2;
                        output1_tm[3] = sum1_3;

                        output2_tm[0] = sum2_0;
                        output2_tm[1] = sum2_1;
                        output2_tm[2] = sum2_2;
                        output2_tm[3] = sum2_3;

                        output3_tm[0] = sum3_0;
                        output3_tm[1] = sum3_1;
                        output3_tm[2] = sum3_2;
                        output3_tm[3] = sum3_3;

                        output0_tm += 4;
                        output1_tm += 4;
                        output2_tm += 4;
                        output3_tm += 4;
    #endif // __ARM_NEON
                    }
                    for (; i < tiles; i++)
                    {
                        const float* bb2p0 = bb2[i / 8 + (i % 8) / 4 + i % 4].data();

                        const float* ktm0 = kernel_tm0[r].data();

    #if __ARM_NEON
                        float32x4_t _sum0123 = vdupq_n_f32(0.f);

                        int q = 0;
                        for (; q + 3 < inch; q += 4)
                        {
                            //                         asm volatile("prfm pldl1keep, [%0, #128] \n" : :"r"(bb2p0) :);
                            float32x4_t _bb2p0 = vld1q_f32(bb2p0);
                            bb2p0 += 4;

                            //                         asm volatile("prfm pldl1keep, [%0, #512] \n" : :"r"(ktm0) :);
                            float32x4_t _ktm0 = vld1q_f32(ktm0 + 0);
                            float32x4_t _ktm1 = vld1q_f32(ktm0 + 4);
                            float32x4_t _ktm2 = vld1q_f32(ktm0 + 8);
                            float32x4_t _ktm3 = vld1q_f32(ktm0 + 12);
                            ktm0 += 16;

    #if __aarch64__
                            _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm0, _bb2p0, 0);
                            _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm1, _bb2p0, 1);
                            _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm2, _bb2p0, 2);
                            _sum0123 = vmlaq_laneq_f32(_sum0123, _ktm3, _bb2p0, 3);
    #else
                            _sum0123 = vmlaq_lane_f32(_sum0123, _ktm0, vget_low_f32(_bb2p0), 0);
                            _sum0123 = vmlaq_lane_f32(_sum0123, _ktm1, vget_low_f32(_bb2p0), 1);
                            _sum0123 = vmlaq_lane_f32(_sum0123, _ktm2, vget_high_f32(_bb2p0), 0);
                            _sum0123 = vmlaq_lane_f32(_sum0123, _ktm3, vget_high_f32(_bb2p0), 1);
    #endif // __aarch64__
                        }

                        for (; q < inch; q++)
                        {
                            float32x4_t _bb2p0 = vld1q_dup_f32(bb2p0);
                            float32x4_t _ktm0 = vld1q_f32(ktm0);

                            _sum0123 = vmlaq_f32(_sum0123, _bb2p0, _ktm0);

                            bb2p0 += 1;
                            ktm0 += 4;
                        }

                        float sum0 = vgetq_lane_f32(_sum0123, 0);
                        float sum1 = vgetq_lane_f32(_sum0123, 1);
                        float sum2 = vgetq_lane_f32(_sum0123, 2);
                        float sum3 = vgetq_lane_f32(_sum0123, 3);
    #else
                        float sum0 = 0.f;
                        float sum1 = 0.f;
                        float sum2 = 0.f;
                        float sum3 = 0.f;

                        for (int q = 0; q < inch; q++)
                        {
                            sum0 += bb2p0[0] * ktm0[0];
                            sum1 += bb2p0[0] * ktm0[1];
                            sum2 += bb2p0[0] * ktm0[2];
                            sum3 += bb2p0[0] * ktm0[3];

                            bb2p0 += 1;
                            ktm0 += 4;
                        }
    #endif // __ARM_NEON

                        output0_tm[0] = sum0;
                        output1_tm[0] = sum1;
                        output2_tm[0] = sum2;
                        output3_tm[0] = sum3;

                        output0_tm += 1;
                        output1_tm += 1;
                        output2_tm += 1;
                        output3_tm += 1;
                    }
                }
            }
        });

        remain_outch_start += nn_outch << 2;

        otter::parallel_for(remain_outch_start, outch, 0, [&](int64_t begin, int64_t end) {
            for (const auto p : otter::irange(begin, end)) {
    #if __ARM_NEON && __aarch64__
                const auto kernel_tm0 = kernel_tf_a[p / 8 + (p % 8) / 4 + p % 4];
    #else
                const auto kernel_tm0 = kernel_tf_a[p / 4 + p % 4];
    #endif

                auto out0_tm = output_tf_a[p];

                float* output0_tm = out0_tm.data();

                for (int r = 0; r < 64; r++)
                {
                    const auto bb2 = input_tf2_a[r];

                    // tile
                    int i = 0;
                    for (; i + 7 < tiles; i += 8)
                    {
                        const float* bb2p0 = bb2[i / 8].data();

                        const float* ktm0 = kernel_tm0[r].data();
    #if __ARM_NEON
    #if __aarch64__
                        asm volatile(
                            "eor    v8.16b, v8.16b, v8.16b     \n"
                            "eor    v9.16b, v9.16b, v9.16b     \n"

                            // inch loop
                            "lsr    w4, %w6, #2             \n" // w4 = nn = inch >> 2
                            "cmp    w4, #0                  \n"
                            "beq    1f                      \n"

                            "0:                             \n"

                            "prfm   pldl1keep, [%1, #512]   \n"
                            "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64     \n"

                            "prfm   pldl1keep, [%2, #128]   \n"
                            "ld1    {v0.4s}, [%2], #16      \n"

                            "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                            "fmla   v9.4s, v5.4s, v0.s[0]   \n"
                            "fmla   v8.4s, v6.4s, v0.s[1]   \n"
                            "fmla   v9.4s, v7.4s, v0.s[1]   \n"

                            "prfm   pldl1keep, [%1, #512]   \n"
                            "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%1], #64 \n"

                            "fmla   v8.4s, v12.4s, v0.s[2]  \n"
                            "fmla   v9.4s, v13.4s, v0.s[2]  \n"
                            "fmla   v8.4s, v14.4s, v0.s[3]  \n"
                            "fmla   v9.4s, v15.4s, v0.s[3]  \n"

                            "subs   w4, w4, #1              \n"
                            "bne    0b                      \n"

                            "1:                             \n"

                            // remain loop
                            "and    w4, %w6, #3             \n" // w4 = remain = tiles & 3
                            "cmp    w4, #0                  \n"
                            "beq    3f                      \n"

                            "2:                             \n"

                            "prfm   pldl1keep, [%1, #256]   \n"
                            "ld1    {v4.4s, v5.4s}, [%1], #32      \n"

                            "prfm   pldl1keep, [%2, #32]    \n"
                            "ld1r   {v0.4s}, [%2], #4       \n"

                            "fmla   v8.4s, v4.4s, v0.4s     \n"
                            "fmla   v9.4s, v5.4s, v0.4s     \n"

                            "subs   w4, w4, #1              \n"
                            "bne    2b                      \n"

                            "3:                             \n"

                            "st1    {v8.4s, v9.4s}, [%0], #32       \n"

                            : "=r"(output0_tm), // %0
                            "=r"(bb2p0),      // %1
                            "=r"(ktm0)        // %2
                            : "0"(output0_tm),
                            "1"(bb2p0),
                            "2"(ktm0),
                            "r"(inch) // %6
                            : "cc", "memory", "x4", "v0", "v4", "v5", "v6", "v7", "v8", "v9", "v12", "v13", "v14", "v15");
    #else  // __aarch64__
                        asm volatile(
                            "veor       q8, q8, q8          \n"
                            "veor       q9, q9, q9          \n"

                            // inch loop
                            "lsr        r4, %6, #2          \n" // r4 = nn = inch >> 2
                            "cmp        r4, #0              \n"
                            "beq        1f                  \n"

                            "0:                             \n"

                            "pld        [%1, #512]          \n"
                            "vldm       %1!, {d8-d15}       \n"
                            //                         "vld1.f32   {d8-d11}, [%1 :128]! \n"
                            //                         "vld1.f32   {d12-d15}, [%1 :128]! \n"

                            "pld        [%2, #128]          \n"
                            "vld1.f32   {d0-d1}, [%2 :128]! \n"

                            "vmla.f32   q8, q4, d0[0]       \n"
                            "vmla.f32   q9, q5, d0[0]       \n"
                            "vmla.f32   q8, q6, d0[1]       \n"
                            "vmla.f32   q9, q7, d0[1]       \n"

                            "pld        [%1, #512]          \n"
                            "vldm       %1!, {d24-d31}      \n"
                            //                         "vld1.f32   {d24-d27}, [%1 :128]! \n"
                            //                         "vld1.f32   {d28-d31}, [%1 :128]! \n"

                            "subs       r4, r4, #1          \n"

                            "vmla.f32   q8, q12, d1[0]      \n"
                            "vmla.f32   q9, q13, d1[0]      \n"
                            "vmla.f32   q8, q14, d1[1]      \n"
                            "vmla.f32   q9, q15, d1[1]      \n"

                            "bne        0b                  \n"

                            "1:                             \n"

                            // remain loop
                            "and        r4, %6, #3          \n" // r4 = remain = tiles & 3
                            "cmp        r4, #0              \n"
                            "beq        3f                  \n"

                            "2:                             \n"

                            "pld        [%1, #256]          \n"
                            "vld1.f32   {d8-d11}, [%1 :128]! \n"

                            "pld        [%2, #32]           \n"
                            "vld1.f32   {d0[],d1[]}, [%2]!  \n"

                            "subs       r4, r4, #1          \n"

                            "vmla.f32   q8, q4, q0          \n"
                            "vmla.f32   q9, q5, q0          \n"

                            "bne        2b                  \n"

                            "3:                             \n"

                            "vst1.f32   {d16-d19}, [%0]!    \n"

                            : "=r"(output0_tm), // %0
                            "=r"(bb2p0),      // %1
                            "=r"(ktm0)        // %2
                            : "0"(output0_tm),
                            "1"(bb2p0),
                            "2"(ktm0),
                            "r"(inch) // %6
                            : "cc", "memory", "r4", "q0", "q4", "q5", "q6", "q7", "q8", "q9", "q12", "q13", "q14", "q15");
    #endif // __aarch64__
    #else
                        float sum0 = 0.f;
                        float sum1 = 0.f;
                        float sum2 = 0.f;
                        float sum3 = 0.f;
                        float sum4 = 0.f;
                        float sum5 = 0.f;
                        float sum6 = 0.f;
                        float sum7 = 0.f;

                        for (int q = 0; q < inch; q++)
                        {
                            sum0 += bb2p0[0] * ktm0[0];
                            sum1 += bb2p0[1] * ktm0[0];
                            sum2 += bb2p0[2] * ktm0[0];
                            sum3 += bb2p0[3] * ktm0[0];
                            sum4 += bb2p0[4] * ktm0[0];
                            sum5 += bb2p0[5] * ktm0[0];
                            sum6 += bb2p0[6] * ktm0[0];
                            sum7 += bb2p0[7] * ktm0[0];

                            bb2p0 += 8;
                            ktm0 += 1;
                        }

                        output0_tm[0] = sum0;
                        output0_tm[1] = sum1;
                        output0_tm[2] = sum2;
                        output0_tm[3] = sum3;
                        output0_tm[4] = sum4;
                        output0_tm[5] = sum5;
                        output0_tm[6] = sum6;
                        output0_tm[7] = sum7;

                        output0_tm += 8;
    #endif // __ARM_NEON
                    }
                    for (; i + 3 < tiles; i += 4)
                    {
                        const float* bb2p0 = bb2[i / 8 + (i % 8) / 4].data();

                        const float* ktm0 = kernel_tm0[r].data();
    #if __ARM_NEON
    #if __aarch64__
                        asm volatile(
                            "eor    v8.16b, v8.16b, v8.16b     \n"

                            // inch loop
                            "lsr    w4, %w6, #2             \n" // w4 = nn = inch >> 2
                            "cmp    w4, #0                  \n"
                            "beq    1f                      \n"

                            "0:                             \n"

                            "prfm   pldl1keep, [%4, #512]   \n"
                            "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64     \n"

                            "prfm   pldl1keep, [%5, #128]   \n"
                            "ld1    {v0.4s}, [%5], #16      \n"

                            "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                            "fmla   v8.4s, v5.4s, v0.s[1]   \n"
                            "fmla   v8.4s, v6.4s, v0.s[2]   \n"
                            "fmla   v8.4s, v7.4s, v0.s[3]   \n"

                            "subs   w4, w4, #1              \n"
                            "bne    0b                      \n"

                            "1:                             \n"

                            // remain loop
                            "and    w4, %w6, #3             \n" // w4 = remain = tiles & 3
                            "cmp    w4, #0                  \n"
                            "beq    3f                      \n"

                            "2:                             \n"

                            "prfm   pldl1keep, [%4, #128]   \n"
                            "ld1    {v4.4s}, [%4], #16      \n"

                            "prfm   pldl1keep, [%5, #32]    \n"
                            "ld1r   {v0.4s}, [%5], #4       \n"

                            "fmla   v8.4s, v4.4s, v0.4s     \n"

                            "subs   w4, w4, #1              \n"
                            "bne    2b                      \n"

                            "3:                             \n"

                            "st1    {v8.4s}, [%0], #16      \n"

                            : "=r"(output0_tm), // %0
                            "=r"(bb2p0),      // %1
                            "=r"(ktm0)        // %2
                            : "0"(output0_tm),
                            "1"(bb2p0),
                            "2"(ktm0),
                            "r"(inch) // %6
                            : "cc", "memory", "x4", "v0", "v4", "v5", "v6", "v7", "v8");
    #else  // __aarch64__
                        asm volatile(
                            "veor       q8, q8, q8          \n"

                            // inch loop
                            "lsr        r4, %6, #2          \n" // r4 = nn = inch >> 2
                            "cmp        r4, #0              \n"
                            "beq        1f                  \n"

                            "0:                             \n"

                            "pld        [%4, #512]          \n"
                            "vldm       %4!, {d8-d15}       \n"
                            //                         "vld1.f32   {d8-d11}, [%4 :128]! \n"
                            //                         "vld1.f32   {d12-d15}, [%4 :128]! \n"

                            "pld        [%5, #128]          \n"
                            "vld1.f32   {d0-d1}, [%5 :128]! \n"

                            "subs       r4, r4, #1          \n"

                            "vmla.f32   q8, q4, d0[0]       \n"
                            "vmla.f32   q8, q5, d0[1]       \n"
                            "vmla.f32   q8, q6, d1[0]       \n"
                            "vmla.f32   q8, q7, d1[1]       \n"

                            "bne        0b                  \n"

                            "1:                             \n"

                            // remain loop
                            "and        r4, %6, #3          \n" // r4 = remain = tiles & 3
                            "cmp        r4, #0              \n"
                            "beq        3f                  \n"

                            "2:                             \n"

                            "pld        [%4, #128]          \n"
                            "vld1.f32   {d8-d9}, [%4]!      \n"

                            "pld        [%5, #32]           \n"
                            "vld1.f32   {d0[],d1[]}, [%5]!  \n"

                            "subs       r4, r4, #1          \n"

                            "vmla.f32   q8, q4, q0          \n"

                            "bne        2b                  \n"

                            "3:                             \n"

                            "vst1.f32   {d16-d17}, [%0]!    \n"

                            : "=r"(output0_tm), // %0
                            "=r"(bb2p0),      // %1
                            "=r"(ktm0)        // %2
                            : "0"(output0_tm),
                            "1"(bb2p0),
                            "2"(ktm0),
                            "r"(inch) // %6
                            : "cc", "memory", "r4", "q0", "q4", "q5", "q6", "q7", "q8");
    #endif // __aarch64__
    #else
                        float sum0 = 0.f;
                        float sum1 = 0.f;
                        float sum2 = 0.f;
                        float sum3 = 0.f;

                        for (int q = 0; q < inch; q++)
                        {
                            sum0 += bb2p0[0] * ktm0[0];
                            sum1 += bb2p0[1] * ktm0[0];
                            sum2 += bb2p0[2] * ktm0[0];
                            sum3 += bb2p0[3] * ktm0[0];

                            bb2p0 += 4;
                            ktm0 += 1;
                        }

                        output0_tm[0] = sum0;
                        output0_tm[1] = sum1;
                        output0_tm[2] = sum2;
                        output0_tm[3] = sum3;

                        output0_tm += 4;
    #endif // __ARM_NEON
                    }
                    for (; i < tiles; i++)
                    {
                        const float* bb2p0 = bb2[i / 8 + (i % 8) / 4 + i % 4].data();

                        const float* ktm0 = kernel_tm0[r].data();

                        int q = 0;
    #if __ARM_NEON
                        float32x4_t _sum0 = vdupq_n_f32(0.f);
                        for (; q + 3 < inch; q += 4)
                        {
                            //                         asm volatile("prfm pldl1keep, [%0, #128] \n" : :"r"(bb2p0) :);
                            float32x4_t _bb2p0 = vld1q_f32(bb2p0);
                            bb2p0 += 4;

                            float32x4_t _ktm0 = vld1q_f32(ktm0);
                            ktm0 += 4;

                            _sum0 = vmlaq_f32(_sum0, _bb2p0, _ktm0);
                        }

    #if __aarch64__
                        float sum0 = vaddvq_f32(_sum0);
    #else
                        float32x2_t _ss0 = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                        float sum0 = vget_lane_f32(vpadd_f32(_ss0, _ss0), 0);
    #endif // __aarch64__
    #else
                        float sum0 = 0.f;
    #endif
                        for (; q < inch; q++)
                        {
                            sum0 += bb2p0[0] * ktm0[0];

                            bb2p0 += 1;
                            ktm0 += 1;
                        }

                        output0_tm[0] = sum0;

                        output0_tm += 1;
                    }
                }
            }
        });
    }
    input_tf.reset();
    
    Tensor output_bordered;
    if (outw == output_shape[3] && outh == output_shape[2]) {
        output_bordered = output;
    } else {
        // assume batchsize = 1
        output_bordered = otter::empty({1, outch, outh, outw}, otter::ScalarType::Float);
    }
    
    auto output_tf_a = output_tf.accessor<float, 3>();
    auto output_bordered_a = output_bordered.accessor<float, 4>()[0];
    
    {
        //         const float otm[6][8] = {
        //             {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
        //             {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
        //             {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
        //             {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
        //             {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
        //             {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
        //         };

        // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
        // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
        // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
        // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
        // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
        // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)

#if __ARM_NEON
        const float coeff[4] = {4.f, 8.f, 16.f, 32.f};
        float32x4_t _coeff = vld1q_f32(coeff);
#endif // __ARM_NEON

        int w_tm = outw / 6 * 8;
        int h_tm = outh / 6 * 8;
        const int tiles = w_tm / 8 * h_tm / 8;

        otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
            for (const auto p : otter::irange(begin, end)) {
                const auto out0_tm = output_tf_a[p];
                auto out0 = output_bordered_a[p];
                
                int out0_tm_w = (int)output_tf.size(2);

                const float bias0 = bias ? bias[p] : 0.f;
    #if __ARM_NEON
                float32x2_t _bias0 = vdup_n_f32(bias0);
    #endif // __ARM_NEON

                float tmp[6][8];

                // tile
                for (int i = 0; i < outh / 6; i++)
                {
                    for (int j = 0; j < outw / 6; j++)
                    {
    #if __ARM_NEON
    #if __aarch64__
                        const float* output0_tm0 = out0_tm[i * w_tm / 8 + j].data();
                        const float* output0_tm1 = out0_tm[i * w_tm / 8 + j + tiles * 8].data();
                        const float* output0_tm2 = out0_tm[i * w_tm / 8 + j + tiles * 16].data();
                        const float* output0_tm3 = out0_tm[i * w_tm / 8 + j + tiles * 24].data();

                        for (int m = 0; m + 3 < 8; m += 4)
                        {
                            float32x4_t _output0_tm_00 = {};
                            float32x4_t _output0_tm_11 = {};
                            float32x4_t _output0_tm_22 = {};
                            float32x4_t _output0_tm_33 = {};
                            float32x4_t _output0_tm_44 = {};
                            float32x4_t _output0_tm_55 = {};
                            float32x4_t _output0_tm_66 = {};
                            float32x4_t _output0_tm_77 = {};

                            _output0_tm_00 = vsetq_lane_f32(output0_tm0[0], _output0_tm_00, 0);
                            output0_tm0 += out0_tm_w * tiles;
                            _output0_tm_00 = vsetq_lane_f32(output0_tm1[0], _output0_tm_00, 1);
                            output0_tm1 += out0_tm_w * tiles;
                            _output0_tm_00 = vsetq_lane_f32(output0_tm2[0], _output0_tm_00, 2);
                            output0_tm2 += out0_tm_w * tiles;
                            _output0_tm_00 = vsetq_lane_f32(output0_tm3[0], _output0_tm_00, 3);
                            output0_tm3 += out0_tm_w * tiles;

                            _output0_tm_11 = vsetq_lane_f32(output0_tm0[0], _output0_tm_11, 0);
                            output0_tm0 += out0_tm_w * tiles;
                            _output0_tm_11 = vsetq_lane_f32(output0_tm1[0], _output0_tm_11, 1);
                            output0_tm1 += out0_tm_w * tiles;
                            _output0_tm_11 = vsetq_lane_f32(output0_tm2[0], _output0_tm_11, 2);
                            output0_tm2 += out0_tm_w * tiles;
                            _output0_tm_11 = vsetq_lane_f32(output0_tm3[0], _output0_tm_11, 3);
                            output0_tm3 += out0_tm_w * tiles;

                            _output0_tm_22 = vsetq_lane_f32(output0_tm0[0], _output0_tm_22, 0);
                            output0_tm0 += out0_tm_w * tiles;
                            _output0_tm_22 = vsetq_lane_f32(output0_tm1[0], _output0_tm_22, 1);
                            output0_tm1 += out0_tm_w * tiles;
                            _output0_tm_22 = vsetq_lane_f32(output0_tm2[0], _output0_tm_22, 2);
                            output0_tm2 += out0_tm_w * tiles;
                            _output0_tm_22 = vsetq_lane_f32(output0_tm3[0], _output0_tm_22, 3);
                            output0_tm3 += out0_tm_w * tiles;

                            _output0_tm_33 = vsetq_lane_f32(output0_tm0[0], _output0_tm_33, 0);
                            output0_tm0 += out0_tm_w * tiles;
                            _output0_tm_33 = vsetq_lane_f32(output0_tm1[0], _output0_tm_33, 1);
                            output0_tm1 += out0_tm_w * tiles;
                            _output0_tm_33 = vsetq_lane_f32(output0_tm2[0], _output0_tm_33, 2);
                            output0_tm2 += out0_tm_w * tiles;
                            _output0_tm_33 = vsetq_lane_f32(output0_tm3[0], _output0_tm_33, 3);
                            output0_tm3 += out0_tm_w * tiles;

                            _output0_tm_44 = vsetq_lane_f32(output0_tm0[0], _output0_tm_44, 0);
                            output0_tm0 += out0_tm_w * tiles;
                            _output0_tm_44 = vsetq_lane_f32(output0_tm1[0], _output0_tm_44, 1);
                            output0_tm1 += out0_tm_w * tiles;
                            _output0_tm_44 = vsetq_lane_f32(output0_tm2[0], _output0_tm_44, 2);
                            output0_tm2 += out0_tm_w * tiles;
                            _output0_tm_44 = vsetq_lane_f32(output0_tm3[0], _output0_tm_44, 3);
                            output0_tm3 += out0_tm_w * tiles;

                            _output0_tm_55 = vsetq_lane_f32(output0_tm0[0], _output0_tm_55, 0);
                            output0_tm0 += out0_tm_w * tiles;
                            _output0_tm_55 = vsetq_lane_f32(output0_tm1[0], _output0_tm_55, 1);
                            output0_tm1 += out0_tm_w * tiles;
                            _output0_tm_55 = vsetq_lane_f32(output0_tm2[0], _output0_tm_55, 2);
                            output0_tm2 += out0_tm_w * tiles;
                            _output0_tm_55 = vsetq_lane_f32(output0_tm3[0], _output0_tm_55, 3);
                            output0_tm3 += out0_tm_w * tiles;

                            _output0_tm_66 = vsetq_lane_f32(output0_tm0[0], _output0_tm_66, 0);
                            output0_tm0 += out0_tm_w * tiles;
                            _output0_tm_66 = vsetq_lane_f32(output0_tm1[0], _output0_tm_66, 1);
                            output0_tm1 += out0_tm_w * tiles;
                            _output0_tm_66 = vsetq_lane_f32(output0_tm2[0], _output0_tm_66, 2);
                            output0_tm2 += out0_tm_w * tiles;
                            _output0_tm_66 = vsetq_lane_f32(output0_tm3[0], _output0_tm_66, 3);
                            output0_tm3 += out0_tm_w * tiles;

                            _output0_tm_77 = vsetq_lane_f32(output0_tm0[0], _output0_tm_77, 0);
                            _output0_tm_77 = vsetq_lane_f32(output0_tm1[0], _output0_tm_77, 1);
                            _output0_tm_77 = vsetq_lane_f32(output0_tm2[0], _output0_tm_77, 2);
                            _output0_tm_77 = vsetq_lane_f32(output0_tm3[0], _output0_tm_77, 3);

                            float32x4_t _tmp024a = vaddq_f32(_output0_tm_11, _output0_tm_22);
                            float32x4_t _tmp135a = vsubq_f32(_output0_tm_11, _output0_tm_22);

                            float32x4_t _tmp024b = vaddq_f32(_output0_tm_33, _output0_tm_44);
                            float32x4_t _tmp135b = vsubq_f32(_output0_tm_33, _output0_tm_44);

                            float32x4_t _tmp024c = vaddq_f32(_output0_tm_55, _output0_tm_66);
                            float32x4_t _tmp135c = vsubq_f32(_output0_tm_55, _output0_tm_66);

                            float32x4_t _tmp0 = vaddq_f32(_output0_tm_00, _tmp024a);
                            _tmp0 = vmlaq_lane_f32(_tmp0, _tmp024c, vget_high_f32(_coeff), 1);
                            _tmp0 = vaddq_f32(_tmp0, _tmp024b);

                            float32x4_t _tmp2 = vmlaq_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeff), 0);
                            _tmp2 = vmlaq_lane_f32(_tmp2, _tmp024c, vget_low_f32(_coeff), 1);

                            float32x4_t _tmp4 = vmlaq_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeff), 0);
                            _tmp4 = vaddq_f32(_tmp4, _tmp024c);
                            _tmp4 = vaddq_f32(_tmp4, _tmp024c);

                            vst1q_f32(&tmp[0][m], _tmp0);
                            vst1q_f32(&tmp[2][m], _tmp2);
                            vst1q_f32(&tmp[4][m], _tmp4);

                            float32x4_t _tmp1 = vmlaq_lane_f32(_tmp135a, _tmp135c, vget_high_f32(_coeff), 0);
                            _tmp1 = vaddq_f32(_tmp1, _tmp135b);
                            _tmp1 = vaddq_f32(_tmp1, _tmp135b);

                            float32x4_t _tmp3 = vmlaq_lane_f32(_tmp135a, _tmp135b, vget_low_f32(_coeff), 1);
                            _tmp3 = vmlaq_lane_f32(_tmp3, _tmp135c, vget_low_f32(_coeff), 0);

                            float32x4_t _tmp5 = vaddq_f32(_output0_tm_77, _tmp135a);
                            _tmp5 = vmlaq_lane_f32(_tmp5, _tmp135b, vget_high_f32(_coeff), 1);
                            _tmp5 = vaddq_f32(_tmp5, _tmp135c);

                            vst1q_f32(&tmp[1][m], _tmp1);
                            vst1q_f32(&tmp[3][m], _tmp3);
                            vst1q_f32(&tmp[5][m], _tmp5);

                            output0_tm0 += out0_tm_w * tiles * 25;
                            output0_tm1 += out0_tm_w * tiles * 25;
                            output0_tm2 += out0_tm_w * tiles * 25;
                            output0_tm3 += out0_tm_w * tiles * 25;
                        }

                        const float* t0 = tmp[0];
                        const float* t1 = tmp[1];

                        float* output0 = out0[i * 6].data() + j * 6;
                        float* output1 = output0 + outw;

                        for (int m = 0; m + 1 < 6; m += 2)
                        {
                            float32x4_t _t0_0123 = vld1q_f32(t0);
                            float32x4_t _t0_4567 = vld1q_f32(t0 + 4);
                            float32x4_t _t1_0123 = vld1q_f32(t1);
                            float32x4_t _t1_4567 = vld1q_f32(t1 + 4);

                            float32x4x2_t _t01_00221133 = vtrnq_f32(_t0_0123, _t1_0123);
                            float32x4x2_t _t01_44665577 = vtrnq_f32(_t0_4567, _t1_4567);

                            float32x2_t _t_00 = vget_low_f32(_t01_00221133.val[0]);
                            float32x2_t _t_11 = vget_low_f32(_t01_00221133.val[1]);
                            float32x2_t _t_22 = vget_high_f32(_t01_00221133.val[0]);
                            float32x2_t _t_33 = vget_high_f32(_t01_00221133.val[1]);
                            float32x2_t _t_44 = vget_low_f32(_t01_44665577.val[0]);
                            float32x2_t _t_55 = vget_low_f32(_t01_44665577.val[1]);
                            float32x2_t _t_66 = vget_high_f32(_t01_44665577.val[0]);
                            float32x2_t _t_77 = vget_high_f32(_t01_44665577.val[1]);

                            float32x2_t _tmp024a = vadd_f32(_t_11, _t_22);
                            float32x2_t _tmp135a = vsub_f32(_t_11, _t_22);

                            float32x2_t _tmp024b = vadd_f32(_t_33, _t_44);
                            float32x2_t _tmp135b = vsub_f32(_t_33, _t_44);

                            float32x2_t _tmp024c = vadd_f32(_t_55, _t_66);
                            float32x2_t _tmp135c = vsub_f32(_t_55, _t_66);

                            float32x2_t _output_0 = vadd_f32(_t_00, _tmp024a);
                            _output_0 = vmla_lane_f32(_output_0, _tmp024c, vget_high_f32(_coeff), 1);
                            _output_0 = vadd_f32(_output_0, _tmp024b);
                            _output_0 = vadd_f32(_output_0, _bias0);

                            float32x2_t _output_2 = vmla_lane_f32(_tmp024a, _tmp024b, vget_low_f32(_coeff), 0);
                            _output_2 = vmla_lane_f32(_output_2, _tmp024c, vget_low_f32(_coeff), 1);
                            _output_2 = vadd_f32(_output_2, _bias0);

                            float32x2_t _output_4 = vmla_lane_f32(_tmp024a, _tmp024b, vget_high_f32(_coeff), 0);
                            _output_4 = vadd_f32(_output_4, _tmp024c);
                            _output_4 = vadd_f32(_output_4, _tmp024c);
                            _output_4 = vadd_f32(_output_4, _bias0);

                            output0[0] = vget_lane_f32(_output_0, 0);
                            output1[0] = vget_lane_f32(_output_0, 1);
                            output0[2] = vget_lane_f32(_output_2, 0);
                            output1[2] = vget_lane_f32(_output_2, 1);
                            output0[4] = vget_lane_f32(_output_4, 0);
                            output1[4] = vget_lane_f32(_output_4, 1);

                            float32x2_t _output_1 = vmla_lane_f32(_tmp135a, _tmp135c, vget_high_f32(_coeff), 0);
                            _output_1 = vadd_f32(_output_1, _tmp135b);
                            _output_1 = vadd_f32(_output_1, _tmp135b);
                            _output_1 = vadd_f32(_output_1, _bias0);

                            float32x2_t _output_3 = vmla_lane_f32(_tmp135a, _tmp135b, vget_low_f32(_coeff), 1);
                            _output_3 = vmla_lane_f32(_output_3, _tmp135c, vget_low_f32(_coeff), 0);
                            _output_3 = vadd_f32(_output_3, _bias0);

                            float32x2_t _output_5 = vadd_f32(_t_77, _tmp135a);
                            _output_5 = vmla_lane_f32(_output_5, _tmp135b, vget_high_f32(_coeff), 1);
                            _output_5 = vadd_f32(_output_5, _tmp135c);
                            _output_5 = vadd_f32(_output_5, _bias0);

                            output0[1] = vget_lane_f32(_output_1, 0);
                            output1[1] = vget_lane_f32(_output_1, 1);
                            output0[3] = vget_lane_f32(_output_3, 0);
                            output1[3] = vget_lane_f32(_output_3, 1);
                            output0[5] = vget_lane_f32(_output_5, 0);
                            output1[5] = vget_lane_f32(_output_5, 1);

                            t0 += 8 * 2;
                            t1 += 8 * 2;
                            output0 += outw * 2;
                            output1 += outw * 2;
                        }
    #else  // __aarch64__
                        const float* output0_tm0_0 = out0_tm[i * w_tm / 8 + j].data();
                        const float* output0_tm1_0 = out0_tm[i * w_tm / 8 + j + tiles * 8].data();
                        const float* output0_tm2_0 = out0_tm[i * w_tm / 8 + j + tiles * 16].data();
                        const float* output0_tm3_0 = out0_tm[i * w_tm / 8 + j + tiles * 24].data();

                        float* t0 = tmp[0];
                        float* t1 = tmp[1];

                        int step = out0_tm_w * tiles * 4;
                        int step2 = out0_tm_w * tiles * 25 * 4;

                        asm volatile(

                            // loop0
                            "vld1.f32   {d16[0]}, [%2], %13 \n"
                            "vld1.f32   {d16[1]}, [%3], %13 \n"
                            "vld1.f32   {d17[0]}, [%4], %13 \n"
                            "vld1.f32   {d17[1]}, [%5], %13 \n"

                            "vld1.f32   {d20[0]}, [%2], %13 \n"
                            "vld1.f32   {d20[1]}, [%3], %13 \n"
                            "vld1.f32   {d21[0]}, [%4], %13 \n"
                            "vld1.f32   {d21[1]}, [%5], %13 \n"

                            "vld1.f32   {d24[0]}, [%2], %13 \n"
                            "vld1.f32   {d24[1]}, [%3], %13 \n"
                            "vld1.f32   {d25[0]}, [%4], %13 \n"
                            "vld1.f32   {d25[1]}, [%5], %13 \n"

                            "vadd.f32   q2, q10, q12        \n"
                            "vsub.f32   q3, q10, q12        \n"

                            "vld1.f32   {d28[0]}, [%2], %13 \n"
                            "vld1.f32   {d28[1]}, [%3], %13 \n"
                            "vld1.f32   {d29[0]}, [%4], %13 \n"
                            "vld1.f32   {d29[1]}, [%5], %13 \n"

                            "vld1.f32   {d18[0]}, [%2], %13 \n"
                            "vld1.f32   {d18[1]}, [%3], %13 \n"
                            "vld1.f32   {d19[0]}, [%4], %13 \n"
                            "vld1.f32   {d19[1]}, [%5], %13 \n"

                            "vadd.f32   q4, q14, q9         \n"
                            "vsub.f32   q5, q14, q9         \n"

                            "vld1.f32   {d22[0]}, [%2], %13 \n"
                            "vld1.f32   {d22[1]}, [%3], %13 \n"
                            "vld1.f32   {d23[0]}, [%4], %13 \n"
                            "vld1.f32   {d23[1]}, [%5], %13 \n"

                            "vld1.f32   {d26[0]}, [%2], %13 \n"
                            "vld1.f32   {d26[1]}, [%3], %13 \n"
                            "vld1.f32   {d27[0]}, [%4], %13 \n"
                            "vld1.f32   {d27[1]}, [%5], %13 \n"

                            "vadd.f32   q6, q11, q13        \n"
                            "vsub.f32   q7, q11, q13        \n" // spare q9 q10 q11 q12 q13 q14

                            "vld1.f32   {d30[0]}, [%2], %14 \n"
                            "vld1.f32   {d30[1]}, [%3], %14 \n"
                            "vld1.f32   {d31[0]}, [%4], %14 \n"
                            "vld1.f32   {d31[1]}, [%5], %14 \n"

                            "vmov       q9, q3              \n"
                            "vadd.f32   q8, q8, q2          \n"
                            "vmla.f32   q9, q7, %f12[0]     \n"
                            "vmov       q12, q2             \n"
                            "vmov       q10, q2             \n"
                            "vmov       q11, q3             \n"
                            "vmla.f32   q12, q4, %f12[0]    \n"
                            "vadd.f32   q15, q15, q3        \n"
                            "vmla.f32   q8, q6, %f12[1]     \n"
                            "vadd.f32   q9, q9, q5          \n"
                            "vmla.f32   q10, q4, %e12[0]    \n"
                            "vmla.f32   q11, q5, %e12[1]    \n"
                            "vadd.f32   q12, q12, q6        \n"
                            "vmla.f32   q15, q5, %f12[1]    \n"
                            "vadd.f32   q8, q8, q4          \n"
                            "vadd.f32   q9, q9, q5          \n"
                            "vmla.f32   q10, q6, %e12[1]    \n"
                            "vmla.f32   q11, q7, %e12[0]    \n"
                            "vadd.f32   q12, q12, q6        \n"
                            "vadd.f32   q15, q15, q7        \n"

                            "vst1.f32   {d16-d17}, [%0]     \n"
                            "add        %0, %0, #64         \n"

                            "vst1.f32   {d18-d19}, [%1]     \n"
                            "add        %1, %1, #64         \n"

                            "vst1.f32   {d20-d21}, [%0]     \n"
                            "add        %0, %0, #64         \n"

                            "vst1.f32   {d22-d23}, [%1]     \n"
                            "add        %1, %1, #64         \n"

                            "vst1.f32   {d24-d25}, [%0]     \n"
                            "sub        %0, %0, #112        \n"

                            "vst1.f32   {d30-d31}, [%1]     \n"
                            "sub        %1, %1, #112        \n"

                            // loop1
                            "vld1.f32   {d16[0]}, [%2], %13 \n"
                            "vld1.f32   {d16[1]}, [%3], %13 \n"
                            "vld1.f32   {d17[0]}, [%4], %13 \n"
                            "vld1.f32   {d17[1]}, [%5], %13 \n"

                            "vld1.f32   {d20[0]}, [%2], %13 \n"
                            "vld1.f32   {d20[1]}, [%3], %13 \n"
                            "vld1.f32   {d21[0]}, [%4], %13 \n"
                            "vld1.f32   {d21[1]}, [%5], %13 \n"

                            "vld1.f32   {d24[0]}, [%2], %13 \n"
                            "vld1.f32   {d24[1]}, [%3], %13 \n"
                            "vld1.f32   {d25[0]}, [%4], %13 \n"
                            "vld1.f32   {d25[1]}, [%5], %13 \n"

                            "vadd.f32   q2, q10, q12        \n"
                            "vsub.f32   q3, q10, q12        \n"

                            "vld1.f32   {d28[0]}, [%2], %13 \n"
                            "vld1.f32   {d28[1]}, [%3], %13 \n"
                            "vld1.f32   {d29[0]}, [%4], %13 \n"
                            "vld1.f32   {d29[1]}, [%5], %13 \n"

                            "vld1.f32   {d18[0]}, [%2], %13 \n"
                            "vld1.f32   {d18[1]}, [%3], %13 \n"
                            "vld1.f32   {d19[0]}, [%4], %13 \n"
                            "vld1.f32   {d19[1]}, [%5], %13 \n"

                            "vadd.f32   q4, q14, q9         \n"
                            "vsub.f32   q5, q14, q9         \n"

                            "vld1.f32   {d22[0]}, [%2], %13 \n"
                            "vld1.f32   {d22[1]}, [%3], %13 \n"
                            "vld1.f32   {d23[0]}, [%4], %13 \n"
                            "vld1.f32   {d23[1]}, [%5], %13 \n"

                            "vld1.f32   {d26[0]}, [%2], %13 \n"
                            "vld1.f32   {d26[1]}, [%3], %13 \n"
                            "vld1.f32   {d27[0]}, [%4], %13 \n"
                            "vld1.f32   {d27[1]}, [%5], %13 \n"

                            "vadd.f32   q6, q11, q13        \n"
                            "vsub.f32   q7, q11, q13        \n" // spare q9 q10 q11 q12 q13 q14

                            "vld1.f32   {d30[0]}, [%2]      \n"
                            "vld1.f32   {d30[1]}, [%3]      \n"
                            "vld1.f32   {d31[0]}, [%4]      \n"
                            "vld1.f32   {d31[1]}, [%5]      \n"

                            "vmov       q9, q3              \n"
                            "vadd.f32   q8, q8, q2          \n"
                            "vmla.f32   q9, q7, %f12[0]     \n"
                            "vmov       q12, q2             \n"
                            "vmov       q10, q2             \n"
                            "vmov       q11, q3             \n"
                            "vmla.f32   q12, q4, %f12[0]    \n"
                            "vadd.f32   q15, q15, q3        \n"
                            "vmla.f32   q8, q6, %f12[1]     \n"
                            "vadd.f32   q9, q9, q5          \n"
                            "vmla.f32   q10, q4, %e12[0]    \n"
                            "vmla.f32   q11, q5, %e12[1]    \n"
                            "vadd.f32   q12, q12, q6        \n"
                            "vmla.f32   q15, q5, %f12[1]    \n"
                            "vadd.f32   q8, q8, q4          \n"
                            "vadd.f32   q9, q9, q5          \n"
                            "vmla.f32   q10, q6, %e12[1]    \n"
                            "vmla.f32   q11, q7, %e12[0]    \n"
                            "vadd.f32   q12, q12, q6        \n"
                            "vadd.f32   q15, q15, q7        \n"

                            "vst1.f32   {d16-d17}, [%0]     \n"
                            "add        %0, %0, #64         \n"

                            "vst1.f32   {d18-d19}, [%1]     \n"
                            "add        %1, %1, #64         \n"

                            "vst1.f32   {d20-d21}, [%0]     \n"
                            "add        %0, %0, #64         \n"

                            "vst1.f32   {d22-d23}, [%1]     \n"
                            "add        %1, %1, #64         \n"

                            "vst1.f32   {d24-d25}, [%0]     \n"

                            "vst1.f32   {d30-d31}, [%1]     \n"

                            : "=r"(t0),            // %0
                            "=r"(t1),            // %1
                            "=r"(output0_tm0_0), // %2
                            "=r"(output0_tm1_0), // %3
                            "=r"(output0_tm2_0), // %4
                            "=r"(output0_tm3_0)  // %5
                            : "0"(t0),
                            "1"(t1),
                            "2"(output0_tm0_0),
                            "3"(output0_tm1_0),
                            "4"(output0_tm2_0),
                            "5"(output0_tm3_0),
                            "w"(_coeff), // %12
                            "r"(step),   // %13
                            "r"(step2)   // %14
                            : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");

                        t0 = tmp[0];
                        t1 = tmp[1];

                        float* output0 = out0[i * 6].data() + j * 6;
                        float* output1 = output0 + outw;

                        int stepw = outw * 2 * 4;

                        asm volatile(

                            // loop0
                            "vld1.f32   {d16-d19}, [%2]     \n"
                            "vld1.f32   {d20-d23}, [%3]     \n"

                            "add        %2, %2, #64         \n"
                            "add        %3, %3, #64         \n"

                            "vtrn.32    q8, q10             \n" // q8 = 0 2  q10 = 1 3
                            "vtrn.32    q9, q11             \n" // q9 = 4 6  q11 = 5 7

                            "vadd.f32   d4, d20, d17        \n"
                            "vsub.f32   d5, d20, d17        \n"

                            "vadd.f32   d6, d21, d18        \n"
                            "vsub.f32   d7, d21, d18        \n"

                            "vadd.f32   d8, d22, d19        \n"
                            "vsub.f32   d9, d22, d19        \n" // spare d17 ~ d22

                            "vmov       d20, d5             \n"
                            "vmov       d18, d4             \n"

                            "vadd.f32   d16, d16, d4        \n"
                            "vmla.f32   d20, d9, %f8[0]     \n"
                            "vmov       d17, d4             \n"
                            "vmov       d21, d5             \n"
                            "vmla.f32   d18, d6, %f8[0]     \n"
                            "vadd.f32   d22, d23, d5        \n"

                            "vmla.f32   d16, d8, %f8[1]     \n"
                            "vadd.f32   d20, d20, d7        \n"
                            "vmla.f32   d17, d6, %e8[0]     \n"
                            "vmla.f32   d21, d7, %e8[1]     \n"
                            "vadd.f32   d18, d18, d8        \n"
                            "vmla.f32   d22, d7, %f8[1]     \n"

                            "vadd.f32   d16, d16, d6        \n"
                            "vadd.f32   d20, d20, d7        \n"
                            "vmla.f32   d17, d8, %e8[1]     \n"
                            "vmla.f32   d21, d9, %e8[0]     \n"
                            "vadd.f32   d18, d18, d8        \n"
                            "vadd.f32   d22, d22, d9        \n"

                            "vadd.f32   d16, d16, %P9       \n" // _bias0
                            "vadd.f32   d20, d20, %P9       \n" // _bias0
                            "vadd.f32   d17, d17, %P9       \n" // _bias0
                            "vadd.f32   d21, d21, %P9       \n" // _bias0
                            "vadd.f32   d18, d18, %P9       \n" // _bias0
                            "vadd.f32   d22, d22, %P9       \n" // _bias0

                            "vtrn.f32   q8, q10             \n"
                            "vtrn.f32   d18, d22            \n"

                            "vst1.f32   {d16-d18}, [%0], %10 \n"
                            "vst1.f32   {d20-d22}, [%1], %10 \n"

                            // loop1
                            "vld1.f32   {d16-d19}, [%2]     \n"
                            "vld1.f32   {d20-d23}, [%3]     \n"

                            "add        %2, %2, #64         \n"
                            "add        %3, %3, #64         \n"

                            "vtrn.32    q8, q10             \n" // q8 = 0 2  q10 = 1 3
                            "vtrn.32    q9, q11             \n" // q9 = 4 6  q11 = 5 7

                            "vadd.f32   d4, d20, d17        \n"
                            "vsub.f32   d5, d20, d17        \n"

                            "vadd.f32   d6, d21, d18        \n"
                            "vsub.f32   d7, d21, d18        \n"

                            "vadd.f32   d8, d22, d19        \n"
                            "vsub.f32   d9, d22, d19        \n" // spare d17 ~ d22

                            "vmov       d20, d5             \n"
                            "vmov       d18, d4             \n"

                            "vadd.f32   d16, d16, d4        \n"
                            "vmla.f32   d20, d9, %f8[0]     \n"
                            "vmov       d17, d4             \n"
                            "vmov       d21, d5             \n"
                            "vmla.f32   d18, d6, %f8[0]     \n"
                            "vadd.f32   d22, d23, d5        \n"

                            "vmla.f32   d16, d8, %f8[1]     \n"
                            "vadd.f32   d20, d20, d7        \n"
                            "vmla.f32   d17, d6, %e8[0]     \n"
                            "vmla.f32   d21, d7, %e8[1]     \n"
                            "vadd.f32   d18, d18, d8        \n"
                            "vmla.f32   d22, d7, %f8[1]     \n"

                            "vadd.f32   d16, d16, d6        \n"
                            "vadd.f32   d20, d20, d7        \n"
                            "vmla.f32   d17, d8, %e8[1]     \n"
                            "vmla.f32   d21, d9, %e8[0]     \n"
                            "vadd.f32   d18, d18, d8        \n"
                            "vadd.f32   d22, d22, d9        \n"

                            "vadd.f32   d16, d16, %P9       \n" // _bias0
                            "vadd.f32   d20, d20, %P9       \n" // _bias0
                            "vadd.f32   d17, d17, %P9       \n" // _bias0
                            "vadd.f32   d21, d21, %P9       \n" // _bias0
                            "vadd.f32   d18, d18, %P9       \n" // _bias0
                            "vadd.f32   d22, d22, %P9       \n" // _bias0

                            "vtrn.f32   q8, q10             \n"
                            "vtrn.f32   d18, d22            \n"

                            "vst1.f32   {d16-d18}, [%0], %10 \n"
                            "vst1.f32   {d20-d22}, [%1], %10 \n"

                            // loop2
                            "vld1.f32   {d16-d19}, [%2]     \n"
                            "vld1.f32   {d20-d23}, [%3]     \n"

                            "add        %2, %2, #64         \n"
                            "add        %3, %3, #64         \n"

                            "vtrn.32    q8, q10             \n" // q8 = 0 2  q10 = 1 3
                            "vtrn.32    q9, q11             \n" // q9 = 4 6  q11 = 5 7

                            "vadd.f32   d4, d20, d17        \n"
                            "vsub.f32   d5, d20, d17        \n"

                            "vadd.f32   d6, d21, d18        \n"
                            "vsub.f32   d7, d21, d18        \n"

                            "vadd.f32   d8, d22, d19        \n"
                            "vsub.f32   d9, d22, d19        \n" // spare d17 ~ d22

                            "vmov       d20, d5             \n"
                            "vmov       d18, d4             \n"

                            "vadd.f32   d16, d16, d4        \n"
                            "vmla.f32   d20, d9, %f8[0]     \n"
                            "vmov       d17, d4             \n"
                            "vmov       d21, d5             \n"
                            "vmla.f32   d18, d6, %f8[0]     \n"
                            "vadd.f32   d22, d23, d5        \n"

                            "vmla.f32   d16, d8, %f8[1]     \n"
                            "vadd.f32   d20, d20, d7        \n"
                            "vmla.f32   d17, d6, %e8[0]     \n"
                            "vmla.f32   d21, d7, %e8[1]     \n"
                            "vadd.f32   d18, d18, d8        \n"
                            "vmla.f32   d22, d7, %f8[1]     \n"

                            "vadd.f32   d16, d16, d6        \n"
                            "vadd.f32   d20, d20, d7        \n"
                            "vmla.f32   d17, d8, %e8[1]     \n"
                            "vmla.f32   d21, d9, %e8[0]     \n"
                            "vadd.f32   d18, d18, d8        \n"
                            "vadd.f32   d22, d22, d9        \n"

                            "vadd.f32   d16, d16, %P9       \n" // _bias0
                            "vadd.f32   d20, d20, %P9       \n" // _bias0
                            "vadd.f32   d17, d17, %P9       \n" // _bias0
                            "vadd.f32   d21, d21, %P9       \n" // _bias0
                            "vadd.f32   d18, d18, %P9       \n" // _bias0
                            "vadd.f32   d22, d22, %P9       \n" // _bias0

                            "vtrn.f32   q8, q10             \n"
                            "vtrn.f32   d18, d22            \n"

                            "vst1.f32   {d16-d18}, [%0], %10 \n"
                            "vst1.f32   {d20-d22}, [%1], %10 \n"

                            : "=r"(output0), // %0
                            "=r"(output1), // %1
                            "=r"(t0),      // %2
                            "=r"(t1)       // %3
                            : "0"(output0),
                            "1"(output1),
                            "2"(t0),
                            "3"(t1),
                            "w"(_coeff), // %8
                            "w"(_bias0), // %9
                            "r"(stepw)   // %10
                            : "memory", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
    #endif // __aarch64__
    #else
                        const float* output0_tm_0 = out0_tm[i * w_tm / 8 + j].data();
                        const float* output0_tm_1 = out0_tm[i * w_tm / 8 + j + tiles].data();
                        const float* output0_tm_2 = out0_tm[i * w_tm / 8 + j + tiles * 2].data();
                        const float* output0_tm_3 = out0_tm[i * w_tm / 8 + j + tiles * 3].data();
                        const float* output0_tm_4 = out0_tm[i * w_tm / 8 + j + tiles * 4].data();
                        const float* output0_tm_5 = out0_tm[i * w_tm / 8 + j + tiles * 5].data();
                        const float* output0_tm_6 = out0_tm[i * w_tm / 8 + j + tiles * 6].data();
                        const float* output0_tm_7 = out0_tm[i * w_tm / 8 + j + tiles * 7].data();

                        for (int m = 0; m < 8; m++)
                        {
                            float tmp024a = output0_tm_1[0] + output0_tm_2[0];
                            float tmp135a = output0_tm_1[0] - output0_tm_2[0];

                            float tmp024b = output0_tm_3[0] + output0_tm_4[0];
                            float tmp135b = output0_tm_3[0] - output0_tm_4[0];

                            float tmp024c = output0_tm_5[0] + output0_tm_6[0];
                            float tmp135c = output0_tm_5[0] - output0_tm_6[0];

                            tmp[0][m] = output0_tm_0[0] + tmp024a + tmp024b + tmp024c * 32;
                            tmp[2][m] = tmp024a + tmp024b * 4 + tmp024c * 8;
                            tmp[4][m] = tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                            tmp[1][m] = tmp135a + tmp135b + tmp135b + tmp135c * 16;
                            tmp[3][m] = tmp135a + tmp135b * 8 + tmp135c * 4;
                            tmp[5][m] = output0_tm_7[0] + tmp135a + tmp135b * 32 + tmp135c;

                            output0_tm_0 += out0_tm_w * tiles * 8;
                            output0_tm_1 += out0_tm_w * tiles * 8;
                            output0_tm_2 += out0_tm_w * tiles * 8;
                            output0_tm_3 += out0_tm_w * tiles * 8;
                            output0_tm_4 += out0_tm_w * tiles * 8;
                            output0_tm_5 += out0_tm_w * tiles * 8;
                            output0_tm_6 += out0_tm_w * tiles * 8;
                            output0_tm_7 += out0_tm_w * tiles * 8;
                        }

                        float* output0 = out0[i * 6].data() + j * 6;

                        for (int m = 0; m < 6; m++)
                        {
                            const float* tmp0 = tmp[m];

                            float tmp024a = tmp0[1] + tmp0[2];
                            float tmp135a = tmp0[1] - tmp0[2];

                            float tmp024b = tmp0[3] + tmp0[4];
                            float tmp135b = tmp0[3] - tmp0[4];

                            float tmp024c = tmp0[5] + tmp0[6];
                            float tmp135c = tmp0[5] - tmp0[6];

                            output0[0] = bias0 + tmp0[0] + tmp024a + tmp024b + tmp024c * 32;
                            output0[2] = bias0 + tmp024a + tmp024b * 4 + tmp024c * 8;
                            output0[4] = bias0 + tmp024a + tmp024b * 16 + tmp024c + tmp024c;

                            output0[1] = bias0 + tmp135a + tmp135b + tmp135b + tmp135c * 16;
                            output0[3] = bias0 + tmp135a + tmp135b * 8 + tmp135c * 4;
                            output0[5] = bias0 + tmp0[7] + tmp135a + tmp135b * 32 + tmp135c;

                            output0 += outw;
                        }
    #endif // __ARM_NEON
                    }
                }
            }
        });
    }
    if (output_bordered.size(3) != output_shape[3] || output.size(2) != output_shape[2])
        otter::crop_(output_bordered, {0, output_bordered.size(3) - output_shape[3], 0, output_bordered.size(2) - output_shape[2]}, output);
    
    return output;
}

Tensor conv2d_3x3s1_winograd64_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, self.options());
    
    return conv2d_3x3s1_winograd64_neon_out(self, weight, weight_o, bias, kernel_size, stride, padding, output);
}

#ifdef __ARM_NEON__
void conv3x3s2_transform_kernel_neon(const Tensor& kernel_, Tensor& kernel_tf, int64_t input_channels, int64_t output_channels) {
    const int64_t kernelSize = 3 * 3;
    
    float* kernel = kernel_.view({output_channels, input_channels, kernelSize}).data_ptr<float>();
    
    kernel_tf = otter::empty({output_channels / 8 + output_channels % 8, input_channels, 8 * 9}, ScalarType::Float);
    
    auto kernel_tf_a = kernel_tf.accessor<float, 3>();
    
    int p = 0;
    for (; p + 7 < output_channels; p += 8)
    {
        const float* k0 = kernel + (p + 0) * input_channels * 9;
        const float* k1 = kernel + (p + 1) * input_channels * 9;
        const float* k2 = kernel + (p + 2) * input_channels * 9;
        const float* k3 = kernel + (p + 3) * input_channels * 9;
        const float* k4 = kernel + (p + 4) * input_channels * 9;
        const float* k5 = kernel + (p + 5) * input_channels * 9;
        const float* k6 = kernel + (p + 6) * input_channels * 9;
        const float* k7 = kernel + (p + 7) * input_channels * 9;

        float* ktmp = kernel_tf_a[p / 8].data();

        for (int q = 0; q < input_channels; q++)
        {
            for (int k = 0; k < 9; k++)
            {
                ktmp[0] = k0[k];
                ktmp[1] = k1[k];
                ktmp[2] = k2[k];
                ktmp[3] = k3[k];
                ktmp[4] = k4[k];
                ktmp[5] = k5[k];
                ktmp[6] = k6[k];
                ktmp[7] = k7[k];
                ktmp += 8;
            }

            k0 += 9;
            k1 += 9;
            k2 += 9;
            k3 += 9;
            k4 += 9;
            k5 += 9;
            k6 += 9;
            k7 += 9;
        }
    }
    for (; p < output_channels; p++)
    {
        const float* k0 = kernel + (p + 0) * input_channels * 9;

        float* ktmp = kernel_tf_a[p / 8 + p % 8].data();

        for (int q = 0; q < input_channels; q++)
        {
            for (int k = 0; k < 9; k++)
            {
                ktmp[k] = k0[k];
            }
            ktmp += 9;

            k0 += 9;
        }
    }
}
#else
void conv3x3s2_transform_kernel_neon(const Tensor& /*kernel_*/, Tensor& /*kernel_tf*/, int64_t /*input_channels*/, int64_t /*output_channels*/) {}
#endif

Tensor& conv2d_3x3s2_packed_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0);
    auto output_shape = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_(output_shape);
    
    int inch  = (int)self.size(1);
    int w = (int)input.size(3);

    int outw  = (int)output_shape[3];
    int outh  = (int)output_shape[2];
    int outch = (int)output_shape[1];

    const int tailstep = w - 2 * outw + w;

    const float* bias = (bias_.defined()) ? bias_.data_ptr<float>() : nullptr;
    
    Tensor kernel_packed;
    if (weight_o.defined())
        kernel_packed = weight_o;
    else
        otter::conv3x3s2_transform_kernel_neon(weight, kernel_packed, inch, outch);
    
    auto input_a = input.accessor<float, 4>()[0];
    auto output_t = output[0];
    auto kernel_a = kernel_packed.accessor<float, 3>();

    int nn_outch = outch >> 3;
    int remain_outch_start = nn_outch << 3;

    otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto pp : otter::irange(begin, end)) {
            int p = pp * 8;

            auto out0 = output_t[p + 0];
            auto out1 = output_t[p + 1];
            auto out2 = output_t[p + 2];
            auto out3 = output_t[p + 3];
            auto out4 = output_t[p + 4];
            auto out5 = output_t[p + 5];
            auto out6 = output_t[p + 6];
            auto out7 = output_t[p + 7];

            const float bias0 = bias ? bias[p + 0] : 0.f;
            const float bias1 = bias ? bias[p + 1] : 0.f;
            const float bias2 = bias ? bias[p + 2] : 0.f;
            const float bias3 = bias ? bias[p + 3] : 0.f;
            const float bias4 = bias ? bias[p + 4] : 0.f;
            const float bias5 = bias ? bias[p + 5] : 0.f;
            const float bias6 = bias ? bias[p + 6] : 0.f;
            const float bias7 = bias ? bias[p + 7] : 0.f;

            out0.fill_(bias0);
            out1.fill_(bias1);
            out2.fill_(bias2);
            out3.fill_(bias3);
            out4.fill_(bias4);
            out5.fill_(bias5);
            out6.fill_(bias6);
            out7.fill_(bias7);

            const float* ktmp = kernel_a[p / 8].data();

            for (int q = 0; q < inch; q++)
            {
                float* outptr0 = out0.data_ptr<float>();
                float* outptr1 = out1.data_ptr<float>();
                float* outptr2 = out2.data_ptr<float>();
                float* outptr3 = out3.data_ptr<float>();
                float* outptr4 = out4.data_ptr<float>();
                float* outptr5 = out5.data_ptr<float>();
                float* outptr6 = out6.data_ptr<float>();
                float* outptr7 = out7.data_ptr<float>();

                const float* img0 = input_a[q].data();

                const float* r0 = img0;
                const float* r1 = img0 + w;
                const float* r2 = img0 + w * 2;

                int i = 0;

                for (; i < outh; i++)
                {
    #if __ARM_NEON
                    int nn = outw >> 2;
                    int remain = outw & 3;
    #else
                    int remain = outw;
    #endif // __ARM_NEON

    #if __ARM_NEON
    #if __aarch64__
                    if (nn > 0)
                    {
                        asm volatile(
                            "0:                                 \n"

                            "prfm   pldl1keep, [%1, #128]       \n"
                            "ld1    {v8.4s}, [%1]               \n"
                            "prfm   pldl1keep, [%2, #128]       \n"
                            "ld1    {v9.4s}, [%2]               \n"

                            "prfm   pldl1keep, [%3, #128]       \n"
                            "ld1    {v10.4s}, [%3]              \n"
                            "prfm   pldl1keep, [%4, #128]       \n"
                            "ld1    {v11.4s}, [%4]              \n"

                            ///
                            "prfm   pldl1keep, [%9, #256]       \n"
                            "ld2    {v4.4s, v5.4s}, [%9], #32   \n" // v4=00 v5=01

                            "ld1    {v0.4s, v1.4s}, [%12], #32  \n"

                            "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                            "fmla   v9.4s, v4.4s, v0.s[1]       \n"

                            "prfm   pldl1keep, [%5, #128]       \n"
                            "ld1    {v12.4s}, [%5]              \n"
                            "prfm   pldl1keep, [%6, #128]       \n"
                            "ld1    {v13.4s}, [%6]              \n"

                            "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                            "fmla   v11.4s, v4.4s, v0.s[3]      \n"

                            "prfm   pldl1keep, [%7, #128]       \n"
                            "ld1    {v14.4s}, [%7]              \n"
                            "prfm   pldl1keep, [%8, #128]       \n"
                            "ld1    {v15.4s}, [%8]              \n"

                            "ld1    {v2.4s, v3.4s}, [%12], #32  \n"

                            "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                            "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                            "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                            "fmla   v15.4s, v4.4s, v1.s[3]      \n"

                            "prfm   pldl1keep, [%9, #256]       \n"
                            "ld2    {v6.4s, v7.4s}, [%9]        \n" // v6

                            "fmla   v8.4s, v5.4s, v2.s[0]       \n"
                            "fmla   v9.4s, v5.4s, v2.s[1]       \n"
                            "fmla   v10.4s, v5.4s, v2.s[2]      \n"
                            "fmla   v11.4s, v5.4s, v2.s[3]      \n"

                            "ext    v6.16b, v4.16b, v6.16b, #4  \n" // v6=02

                            "ld1    {v0.4s, v1.4s}, [%12], #32  \n"

                            "fmla   v12.4s, v5.4s, v3.s[0]      \n"
                            "fmla   v13.4s, v5.4s, v3.s[1]      \n"
                            "fmla   v14.4s, v5.4s, v3.s[2]      \n"
                            "fmla   v15.4s, v5.4s, v3.s[3]      \n"

                            ///
                            "prfm   pldl1keep, [%10, #256]      \n"
                            "ld2    {v4.4s, v5.4s}, [%10], #32  \n" // v4=10 v5=11

                            "fmla   v8.4s, v6.4s, v0.s[0]       \n"
                            "fmla   v9.4s, v6.4s, v0.s[1]       \n"
                            "fmla   v10.4s, v6.4s, v0.s[2]      \n"
                            "fmla   v11.4s, v6.4s, v0.s[3]      \n"

                            "ld1    {v2.4s, v3.4s}, [%12], #32  \n"

                            "fmla   v12.4s, v6.4s, v1.s[0]      \n"
                            "fmla   v13.4s, v6.4s, v1.s[1]      \n"
                            "fmla   v14.4s, v6.4s, v1.s[2]      \n"
                            "fmla   v15.4s, v6.4s, v1.s[3]      \n"

                            "fmla   v8.4s, v4.4s, v2.s[0]       \n"
                            "fmla   v9.4s, v4.4s, v2.s[1]       \n"
                            "fmla   v10.4s, v4.4s, v2.s[2]      \n"
                            "fmla   v11.4s, v4.4s, v2.s[3]      \n"

                            "ld1    {v0.4s, v1.4s}, [%12], #32  \n"

                            "fmla   v12.4s, v4.4s, v3.s[0]      \n"
                            "fmla   v13.4s, v4.4s, v3.s[1]      \n"
                            "fmla   v14.4s, v4.4s, v3.s[2]      \n"
                            "fmla   v15.4s, v4.4s, v3.s[3]      \n"

                            "prfm   pldl1keep, [%10, #256]      \n"
                            "ld2    {v6.4s, v7.4s}, [%10]       \n" // v6

                            "fmla   v8.4s, v5.4s, v0.s[0]       \n"
                            "fmla   v9.4s, v5.4s, v0.s[1]       \n"
                            "fmla   v10.4s, v5.4s, v0.s[2]      \n"
                            "fmla   v11.4s, v5.4s, v0.s[3]      \n"

                            "ld1    {v2.4s, v3.4s}, [%12], #32  \n"

                            "ext    v6.16b, v4.16b, v6.16b, #4  \n" // v6=12

                            "fmla   v12.4s, v5.4s, v1.s[0]      \n"
                            "fmla   v13.4s, v5.4s, v1.s[1]      \n"
                            "fmla   v14.4s, v5.4s, v1.s[2]      \n"
                            "fmla   v15.4s, v5.4s, v1.s[3]      \n"

                            ///
                            "prfm   pldl1keep, [%11, #256]      \n"
                            "ld2    {v4.4s, v5.4s}, [%11], #32  \n" // v4=20 v5=21

                            "fmla   v8.4s, v6.4s, v2.s[0]       \n"
                            "fmla   v9.4s, v6.4s, v2.s[1]       \n"
                            "fmla   v10.4s, v6.4s, v2.s[2]      \n"
                            "fmla   v11.4s, v6.4s, v2.s[3]      \n"

                            "ld1    {v0.4s, v1.4s}, [%12], #32  \n"

                            "fmla   v12.4s, v6.4s, v3.s[0]      \n"
                            "fmla   v13.4s, v6.4s, v3.s[1]      \n"
                            "fmla   v14.4s, v6.4s, v3.s[2]      \n"
                            "fmla   v15.4s, v6.4s, v3.s[3]      \n"

                            "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                            "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                            "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                            "fmla   v11.4s, v4.4s, v0.s[3]      \n"

                            "ld1    {v2.4s, v3.4s}, [%12], #32  \n"

                            "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                            "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                            "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                            "fmla   v15.4s, v4.4s, v1.s[3]      \n"

                            "prfm   pldl1keep, [%11, #256]      \n"
                            "ld2    {v6.4s, v7.4s}, [%11]       \n" // v6

                            "fmla   v8.4s, v5.4s, v2.s[0]       \n"
                            "fmla   v9.4s, v5.4s, v2.s[1]       \n"
                            "fmla   v10.4s, v5.4s, v2.s[2]      \n"
                            "fmla   v11.4s, v5.4s, v2.s[3]      \n"

                            "ext    v6.16b, v4.16b, v6.16b, #4  \n" // v6=22

                            "ld1    {v0.4s, v1.4s}, [%12], #32  \n"

                            "fmla   v12.4s, v5.4s, v3.s[0]      \n"
                            "fmla   v13.4s, v5.4s, v3.s[1]      \n"
                            "fmla   v14.4s, v5.4s, v3.s[2]      \n"
                            "fmla   v15.4s, v5.4s, v3.s[3]      \n"

                            "fmla   v8.4s, v6.4s, v0.s[0]       \n"
                            "fmla   v9.4s, v6.4s, v0.s[1]       \n"
                            "fmla   v10.4s, v6.4s, v0.s[2]      \n"
                            "fmla   v11.4s, v6.4s, v0.s[3]      \n"

                            "fmla   v12.4s, v6.4s, v1.s[0]      \n"
                            "fmla   v13.4s, v6.4s, v1.s[1]      \n"

                            "st1    {v8.4s}, [%1], #16          \n"
                            "st1    {v9.4s}, [%2], #16          \n"

                            "fmla   v14.4s, v6.4s, v1.s[2]      \n"
                            "fmla   v15.4s, v6.4s, v1.s[3]      \n"

                            "st1    {v10.4s}, [%3], #16         \n"
                            "st1    {v11.4s}, [%4], #16         \n"

                            "sub    %12, %12, #288              \n"

                            "st1    {v12.4s}, [%5], #16         \n"
                            "st1    {v13.4s}, [%6], #16         \n"

                            "subs   %w0, %w0, #1                \n"

                            "st1    {v14.4s}, [%7], #16         \n"
                            "st1    {v15.4s}, [%8], #16         \n"

                            "bne    0b                          \n"
                            : "=r"(nn),      // %0
                            "=r"(outptr0), // %1
                            "=r"(outptr1), // %2
                            "=r"(outptr2), // %3
                            "=r"(outptr3), // %4
                            "=r"(outptr4), // %5
                            "=r"(outptr5), // %6
                            "=r"(outptr6), // %7
                            "=r"(outptr7), // %8
                            "=r"(r0),      // %9
                            "=r"(r1),      // %10
                            "=r"(r2),      // %11
                            "=r"(ktmp)     // %12
                            : "0"(nn),
                            "1"(outptr0),
                            "2"(outptr1),
                            "3"(outptr2),
                            "4"(outptr3),
                            "5"(outptr4),
                            "6"(outptr5),
                            "7"(outptr6),
                            "8"(outptr7),
                            "9"(r0),
                            "10"(r1),
                            "11"(r2),
                            "12"(ktmp)
                            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                    }
    #else  // __aarch64__
                    for (; nn > 0; nn--)
                    {
                        asm volatile(
                            "pld        [%0, #128]          \n"
                            "vld1.f32   {d16-d17}, [%0]     \n"
                            "pld        [%1, #128]          \n"
                            "vld1.f32   {d18-d19}, [%1]     \n"

                            "pld        [%2, #128]          \n"
                            "vld1.f32   {d20-d21}, [%2]     \n"
                            "pld        [%3, #128]          \n"
                            "vld1.f32   {d22-d23}, [%3]     \n"

                            ///
                            "pld        [%8, #256]          \n"
                            "vld2.f32   {d8-d11}, [%8]!     \n" // q4=00 q5=01

                            "vld1.f32   {d0-d3}, [%11 :128]! \n"

                            "vmla.f32   q8, q4, d0[0]       \n"
                            "vmla.f32   q9, q4, d0[1]       \n"

                            "pld        [%4, #128]          \n"
                            "vld1.f32   {d24-d25}, [%4]     \n"
                            "pld        [%5, #128]          \n"
                            "vld1.f32   {d26-d27}, [%5]     \n"

                            "vmla.f32   q10, q4, d1[0]      \n"
                            "vmla.f32   q11, q4, d1[1]      \n"

                            "pld        [%6, #128]          \n"
                            "vld1.f32   {d28-d29}, [%6]     \n"
                            "pld        [%7, #128]          \n"
                            "vld1.f32   {d30-d31}, [%7]     \n"

                            "vld1.f32   {d4-d7}, [%11 :128]! \n"

                            "vmla.f32   q12, q4, d2[0]      \n"
                            "vmla.f32   q13, q4, d2[1]      \n"
                            "vmla.f32   q14, q4, d3[0]      \n"
                            "vmla.f32   q15, q4, d3[1]      \n"

                            "pld        [%8, #128]          \n"
                            "vld2.f32   {d12-d13}, [%8]     \n" // q6

                            "vmla.f32   q8, q5, d4[0]       \n"
                            "vmla.f32   q9, q5, d4[1]       \n"
                            "vmla.f32   q10, q5, d5[0]      \n"
                            "vmla.f32   q11, q5, d5[1]      \n"

                            "vext.f32   q6, q4, q6, #1      \n" // q6=02

                            "vld1.f32   {d0-d3}, [%11 :128]! \n"

                            "vmla.f32   q12, q5, d6[0]      \n"
                            "vmla.f32   q13, q5, d6[1]      \n"
                            "vmla.f32   q14, q5, d7[0]      \n"
                            "vmla.f32   q15, q5, d7[1]      \n"

                            ///
                            "pld        [%9, #256]          \n"
                            "vld2.f32   {d8-d11}, [%9]!     \n" // q4=10 q5=11

                            "vmla.f32   q8, q6, d0[0]       \n"
                            "vmla.f32   q9, q6, d0[1]       \n"
                            "vmla.f32   q10, q6, d1[0]      \n"
                            "vmla.f32   q11, q6, d1[1]      \n"

                            "vld1.f32   {d4-d7}, [%11 :128]! \n"

                            "vmla.f32   q12, q6, d2[0]      \n"
                            "vmla.f32   q13, q6, d2[1]      \n"
                            "vmla.f32   q14, q6, d3[0]      \n"
                            "vmla.f32   q15, q6, d3[1]      \n"

                            "vmla.f32   q8, q4, d4[0]       \n"
                            "vmla.f32   q9, q4, d4[1]       \n"
                            "vmla.f32   q10, q4, d5[0]      \n"
                            "vmla.f32   q11, q4, d5[1]      \n"

                            "vld1.f32   {d0-d3}, [%11 :128]! \n"

                            "vmla.f32   q12, q4, d6[0]      \n"
                            "vmla.f32   q13, q4, d6[1]      \n"
                            "vmla.f32   q14, q4, d7[0]      \n"
                            "vmla.f32   q15, q4, d7[1]      \n"

                            "pld        [%9, #128]          \n"
                            "vld2.f32   {d12-d13}, [%9]     \n" // q6

                            "vmla.f32   q8, q5, d0[0]       \n"
                            "vmla.f32   q9, q5, d0[1]       \n"
                            "vmla.f32   q10, q5, d1[0]      \n"
                            "vmla.f32   q11, q5, d1[1]      \n"

                            "vld1.f32   {d4-d7}, [%11 :128]! \n"

                            "vext.f32   q6, q4, q6, #1      \n" // q6=12

                            "vmla.f32   q12, q5, d2[0]      \n"
                            "vmla.f32   q13, q5, d2[1]      \n"
                            "vmla.f32   q14, q5, d3[0]      \n"
                            "vmla.f32   q15, q5, d3[1]      \n"

                            ///
                            "pld        [%10, #256]         \n"
                            "vld2.f32   {d8-d11}, [%10]!    \n" // q4=20 q5=21

                            "vmla.f32   q8, q6, d4[0]       \n"
                            "vmla.f32   q9, q6, d4[1]       \n"
                            "vmla.f32   q10, q6, d5[0]      \n"
                            "vmla.f32   q11, q6, d5[1]      \n"

                            "vld1.f32   {d0-d3}, [%11 :128]! \n"

                            "vmla.f32   q12, q6, d6[0]      \n"
                            "vmla.f32   q13, q6, d6[1]      \n"
                            "vmla.f32   q14, q6, d7[0]      \n"
                            "vmla.f32   q15, q6, d7[1]      \n"

                            "vmla.f32   q8, q4, d0[0]       \n"
                            "vmla.f32   q9, q4, d0[1]       \n"
                            "vmla.f32   q10, q4, d1[0]      \n"
                            "vmla.f32   q11, q4, d1[1]      \n"

                            "vld1.f32   {d4-d7}, [%11 :128]! \n"

                            "vmla.f32   q12, q4, d2[0]      \n"
                            "vmla.f32   q13, q4, d2[1]      \n"
                            "vmla.f32   q14, q4, d3[0]      \n"
                            "vmla.f32   q15, q4, d3[1]      \n"

                            "pld        [%10, #128]         \n"
                            "vld2.f32   {d12-d13}, [%10]    \n" // q6

                            "vmla.f32   q8, q5, d4[0]       \n"
                            "vmla.f32   q9, q5, d4[1]       \n"
                            "vmla.f32   q10, q5, d5[0]      \n"
                            "vmla.f32   q11, q5, d5[1]      \n"

                            "vext.f32   q6, q4, q6, #1      \n" // q6=22

                            "vld1.f32   {d0-d3}, [%11 :128]! \n"

                            "vmla.f32   q12, q5, d6[0]      \n"
                            "vmla.f32   q13, q5, d6[1]      \n"
                            "vmla.f32   q14, q5, d7[0]      \n"
                            "vmla.f32   q15, q5, d7[1]      \n"

                            "vmla.f32   q8, q6, d0[0]       \n"
                            "vmla.f32   q9, q6, d0[1]       \n"
                            "vmla.f32   q10, q6, d1[0]      \n"
                            "vmla.f32   q11, q6, d1[1]      \n"

                            "vmla.f32   q12, q6, d2[0]      \n"
                            "vmla.f32   q13, q6, d2[1]      \n"

                            "vst1.f32   {d16-d17}, [%0]!    \n"
                            "vst1.f32   {d18-d19}, [%1]!    \n"

                            "vmla.f32   q14, q6, d3[0]      \n"
                            "vmla.f32   q15, q6, d3[1]      \n"

                            "vst1.f32   {d20-d21}, [%2]!    \n"
                            "vst1.f32   {d22-d23}, [%3]!    \n"

                            "sub        %11, %11, #288      \n"

                            "vst1.f32   {d24-d25}, [%4]!    \n"
                            "vst1.f32   {d26-d27}, [%5]!    \n"
                            "vst1.f32   {d28-d29}, [%6]!    \n"
                            "vst1.f32   {d30-d31}, [%7]!    \n"
                            : "=r"(outptr0), // %0
                            "=r"(outptr1), // %1
                            "=r"(outptr2), // %2
                            "=r"(outptr3), // %3
                            "=r"(outptr4), // %4
                            "=r"(outptr5), // %5
                            "=r"(outptr6), // %6
                            "=r"(outptr7), // %7
                            "=r"(r0),      // %8
                            "=r"(r1),      // %9
                            "=r"(r2),      // %10
                            "=r"(ktmp)     // %11
                            : "0"(outptr0),
                            "1"(outptr1),
                            "2"(outptr2),
                            "3"(outptr3),
                            "4"(outptr4),
                            "5"(outptr5),
                            "6"(outptr6),
                            "7"(outptr7),
                            "8"(r0),
                            "9"(r1),
                            "10"(r2),
                            "11"(ktmp)
                            : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                    }
    #endif // __aarch64__
    #endif // __ARM_NEON
                    for (; remain > 0; remain--)
                    {
    #if __ARM_NEON
    #if __aarch64__
                        asm volatile(
                            "ld1    {v10.4s, v11.4s}, [%11], #32    \n"

                            "prfm   pldl1keep, [%8, #128]   \n"
                            "ld1    {v0.4s}, [%8]           \n"

                            "ld1    {v12.4s, v13.4s}, [%11], #32    \n"

                            "ld1    {v8.s}[0], [%0]         \n"
                            "ld1    {v8.s}[1], [%1]         \n"
                            "ld1    {v8.s}[2], [%2]         \n"
                            "ld1    {v8.s}[3], [%3]         \n"

                            "fmul   v14.4s, v10.4s, v0.s[0] \n"
                            "fmul   v15.4s, v11.4s, v0.s[0] \n"

                            "ld1    {v9.s}[0], [%4]         \n"
                            "ld1    {v9.s}[1], [%5]         \n"
                            "ld1    {v9.s}[2], [%6]         \n"
                            "ld1    {v9.s}[3], [%7]         \n"

                            "ld1    {v10.4s, v11.4s}, [%11], #32    \n"

                            "fmla   v8.4s, v12.4s, v0.s[1]  \n"
                            "fmla   v9.4s, v13.4s, v0.s[1]  \n"

                            "ld1    {v12.4s, v13.4s}, [%11], #32    \n"

                            "fmla   v14.4s, v10.4s, v0.s[2] \n"
                            "fmla   v15.4s, v11.4s, v0.s[2] \n"

                            "prfm   pldl1keep, [%9, #128]   \n"
                            "ld1    {v1.4s}, [%9]           \n"

                            "ld1    {v10.4s, v11.4s}, [%11], #32    \n"

                            "fmla   v8.4s, v12.4s, v1.s[0]  \n"
                            "fmla   v9.4s, v13.4s, v1.s[0]  \n"

                            "ld1    {v12.4s, v13.4s}, [%11], #32    \n"

                            "fmla   v14.4s, v10.4s, v1.s[1] \n"
                            "fmla   v15.4s, v11.4s, v1.s[1] \n"

                            "ld1    {v10.4s, v11.4s}, [%11], #32    \n"

                            "fmla   v8.4s, v12.4s, v1.s[2]  \n"
                            "fmla   v9.4s, v13.4s, v1.s[2]  \n"

                            "prfm   pldl1keep, [%10, #128]  \n"
                            "ld1    {v0.4s}, [%10]          \n"

                            "ld1    {v12.4s, v13.4s}, [%11], #32    \n"

                            "fmla   v14.4s, v10.4s, v0.s[0] \n"
                            "fmla   v15.4s, v11.4s, v0.s[0] \n"

                            "ld1    {v10.4s, v11.4s}, [%11], #32    \n"

                            "fmla   v8.4s, v12.4s, v0.s[1]  \n"
                            "fmla   v9.4s, v13.4s, v0.s[1]  \n"

                            "fmla   v14.4s, v10.4s, v0.s[2] \n"
                            "fmla   v15.4s, v11.4s, v0.s[2] \n"

                            "fadd   v8.4s, v8.4s, v14.4s    \n"
                            "fadd   v9.4s, v9.4s, v15.4s    \n"

                            "sub    %11, %11, #288          \n"

                            "st1    {v8.s}[0], [%0], #4     \n"
                            "st1    {v8.s}[1], [%1], #4     \n"
                            "st1    {v8.s}[2], [%2], #4     \n"
                            "st1    {v8.s}[3], [%3], #4     \n"

                            "st1    {v9.s}[0], [%4], #4     \n"
                            "st1    {v9.s}[1], [%5], #4     \n"
                            "st1    {v9.s}[2], [%6], #4     \n"
                            "st1    {v9.s}[3], [%7], #4     \n"

                            : "=r"(outptr0), // %0
                            "=r"(outptr1), // %1
                            "=r"(outptr2), // %2
                            "=r"(outptr3), // %3
                            "=r"(outptr4), // %4
                            "=r"(outptr5), // %5
                            "=r"(outptr6), // %6
                            "=r"(outptr7), // %7
                            "=r"(r0),      // %8
                            "=r"(r1),      // %9
                            "=r"(r2),      // %10
                            "=r"(ktmp)     // %11
                            : "0"(outptr0),
                            "1"(outptr1),
                            "2"(outptr2),
                            "3"(outptr3),
                            "4"(outptr4),
                            "5"(outptr5),
                            "6"(outptr6),
                            "7"(outptr7),
                            "8"(r0),
                            "9"(r1),
                            "10"(r2),
                            "11"(ktmp)
                            : "memory", "v0", "v1", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
    #else  // __aarch64__
                        asm volatile(
                            "vld1.f32   {d20-d23}, [%11 :128]! \n"

                            "pld        [%8, #128]      \n"
                            "vld1.f32   {d0-d1}, [%8]   \n"

                            "vld1.f32   {d24-d27}, [%11 :128]! \n"

                            "vld1.f32   {d16[0]}, [%0]  \n"
                            "vld1.f32   {d16[1]}, [%1]  \n"
                            "vld1.f32   {d17[0]}, [%2]  \n"
                            "vld1.f32   {d17[1]}, [%3]  \n"

                            "vmul.f32   q14, q10, d0[0] \n"
                            "vmul.f32   q15, q11, d0[0] \n"

                            "vld1.f32   {d18[0]}, [%4]  \n"
                            "vld1.f32   {d18[1]}, [%5]  \n"
                            "vld1.f32   {d19[0]}, [%6]  \n"
                            "vld1.f32   {d19[1]}, [%7]  \n"

                            "vld1.f32   {d20-d23}, [%11 :128]! \n"

                            "vmla.f32   q8, q12, d0[1]  \n"
                            "vmla.f32   q9, q13, d0[1]  \n"

                            "vld1.f32   {d24-d27}, [%11 :128]! \n"

                            "vmla.f32   q14, q10, d1[0] \n"
                            "vmla.f32   q15, q11, d1[0] \n"

                            "pld        [%9, #128]      \n"
                            "vld1.f32   {d2-d3}, [%9]   \n"

                            "vld1.f32   {d20-d23}, [%11 :128]! \n"

                            "vmla.f32   q8, q12, d2[0]  \n"
                            "vmla.f32   q9, q13, d2[0]  \n"

                            "vld1.f32   {d24-d27}, [%11 :128]! \n"

                            "vmla.f32   q14, q10, d2[1] \n"
                            "vmla.f32   q15, q11, d2[1] \n"

                            "vld1.f32   {d20-d23}, [%11 :128]! \n"

                            "vmla.f32   q8, q12, d3[0]  \n"
                            "vmla.f32   q9, q13, d3[0]  \n"

                            "pld        [%10, #128]     \n"
                            "vld1.f32   {d0-d1}, [%10]  \n"

                            "vld1.f32   {d24-d27}, [%11 :128]! \n"

                            "vmla.f32   q14, q10, d0[0] \n"
                            "vmla.f32   q15, q11, d0[0] \n"

                            "vld1.f32   {d20-d23}, [%11 :128]! \n"

                            "vmla.f32   q8, q12, d0[1]  \n"
                            "vmla.f32   q9, q13, d0[1]  \n"

                            "vmla.f32   q14, q10, d1[0] \n"
                            "vmla.f32   q15, q11, d1[0] \n"

                            "vadd.f32   q8, q8, q14     \n"
                            "vadd.f32   q9, q9, q15     \n"

                            "sub        %11, %11, #288  \n"

                            "vst1.f32   {d16[0]}, [%0]! \n"
                            "vst1.f32   {d16[1]}, [%1]! \n"
                            "vst1.f32   {d17[0]}, [%2]! \n"
                            "vst1.f32   {d17[1]}, [%3]! \n"

                            "vst1.f32   {d18[0]}, [%4]! \n"
                            "vst1.f32   {d18[1]}, [%5]! \n"
                            "vst1.f32   {d19[0]}, [%6]! \n"
                            "vst1.f32   {d19[1]}, [%7]! \n"

                            : "=r"(outptr0), // %0
                            "=r"(outptr1), // %1
                            "=r"(outptr2), // %2
                            "=r"(outptr3), // %3
                            "=r"(outptr4), // %4
                            "=r"(outptr5), // %5
                            "=r"(outptr6), // %6
                            "=r"(outptr7), // %7
                            "=r"(r0),      // %8
                            "=r"(r1),      // %9
                            "=r"(r2),      // %10
                            "=r"(ktmp)     // %11
                            : "0"(outptr0),
                            "1"(outptr1),
                            "2"(outptr2),
                            "3"(outptr3),
                            "4"(outptr4),
                            "5"(outptr5),
                            "6"(outptr6),
                            "7"(outptr7),
                            "8"(r0),
                            "9"(r1),
                            "10"(r2),
                            "11"(ktmp)
                            : "memory", "q0", "q1", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
    #endif // __aarch64__
    #else  // __ARM_NEON
                        float sum0 = 0.f;
                        float sum1 = 0.f;
                        float sum2 = 0.f;
                        float sum3 = 0.f;
                        float sum4 = 0.f;
                        float sum5 = 0.f;
                        float sum6 = 0.f;
                        float sum7 = 0.f;

                        sum0 += r0[0] * ktmp[0];
                        sum1 += r0[0] * ktmp[1];
                        sum2 += r0[0] * ktmp[2];
                        sum3 += r0[0] * ktmp[3];
                        sum4 += r0[0] * ktmp[4];
                        sum5 += r0[0] * ktmp[5];
                        sum6 += r0[0] * ktmp[6];
                        sum7 += r0[0] * ktmp[7];
                        ktmp += 8;

                        sum0 += r0[1] * ktmp[0];
                        sum1 += r0[1] * ktmp[1];
                        sum2 += r0[1] * ktmp[2];
                        sum3 += r0[1] * ktmp[3];
                        sum4 += r0[1] * ktmp[4];
                        sum5 += r0[1] * ktmp[5];
                        sum6 += r0[1] * ktmp[6];
                        sum7 += r0[1] * ktmp[7];
                        ktmp += 8;

                        sum0 += r0[2] * ktmp[0];
                        sum1 += r0[2] * ktmp[1];
                        sum2 += r0[2] * ktmp[2];
                        sum3 += r0[2] * ktmp[3];
                        sum4 += r0[2] * ktmp[4];
                        sum5 += r0[2] * ktmp[5];
                        sum6 += r0[2] * ktmp[6];
                        sum7 += r0[2] * ktmp[7];
                        ktmp += 8;

                        sum0 += r1[0] * ktmp[0];
                        sum1 += r1[0] * ktmp[1];
                        sum2 += r1[0] * ktmp[2];
                        sum3 += r1[0] * ktmp[3];
                        sum4 += r1[0] * ktmp[4];
                        sum5 += r1[0] * ktmp[5];
                        sum6 += r1[0] * ktmp[6];
                        sum7 += r1[0] * ktmp[7];
                        ktmp += 8;

                        sum0 += r1[1] * ktmp[0];
                        sum1 += r1[1] * ktmp[1];
                        sum2 += r1[1] * ktmp[2];
                        sum3 += r1[1] * ktmp[3];
                        sum4 += r1[1] * ktmp[4];
                        sum5 += r1[1] * ktmp[5];
                        sum6 += r1[1] * ktmp[6];
                        sum7 += r1[1] * ktmp[7];
                        ktmp += 8;

                        sum0 += r1[2] * ktmp[0];
                        sum1 += r1[2] * ktmp[1];
                        sum2 += r1[2] * ktmp[2];
                        sum3 += r1[2] * ktmp[3];
                        sum4 += r1[2] * ktmp[4];
                        sum5 += r1[2] * ktmp[5];
                        sum6 += r1[2] * ktmp[6];
                        sum7 += r1[2] * ktmp[7];
                        ktmp += 8;

                        sum0 += r2[0] * ktmp[0];
                        sum1 += r2[0] * ktmp[1];
                        sum2 += r2[0] * ktmp[2];
                        sum3 += r2[0] * ktmp[3];
                        sum4 += r2[0] * ktmp[4];
                        sum5 += r2[0] * ktmp[5];
                        sum6 += r2[0] * ktmp[6];
                        sum7 += r2[0] * ktmp[7];
                        ktmp += 8;

                        sum0 += r2[1] * ktmp[0];
                        sum1 += r2[1] * ktmp[1];
                        sum2 += r2[1] * ktmp[2];
                        sum3 += r2[1] * ktmp[3];
                        sum4 += r2[1] * ktmp[4];
                        sum5 += r2[1] * ktmp[5];
                        sum6 += r2[1] * ktmp[6];
                        sum7 += r2[1] * ktmp[7];
                        ktmp += 8;

                        sum0 += r2[2] * ktmp[0];
                        sum1 += r2[2] * ktmp[1];
                        sum2 += r2[2] * ktmp[2];
                        sum3 += r2[2] * ktmp[3];
                        sum4 += r2[2] * ktmp[4];
                        sum5 += r2[2] * ktmp[5];
                        sum6 += r2[2] * ktmp[6];
                        sum7 += r2[2] * ktmp[7];
                        ktmp += 8;

                        *outptr0 += sum0;
                        *outptr1 += sum1;
                        *outptr2 += sum2;
                        *outptr3 += sum3;
                        *outptr4 += sum4;
                        *outptr5 += sum5;
                        *outptr6 += sum6;
                        *outptr7 += sum7;

                        ktmp -= 8 * 9;

                        outptr0++;
                        outptr1++;
                        outptr2++;
                        outptr3++;
                        outptr4++;
                        outptr5++;
                        outptr6++;
                        outptr7++;
    #endif // __ARM_NEON
                        r0 += 2;
                        r1 += 2;
                        r2 += 2;
                    }

                    r0 += tailstep;
                    r1 += tailstep;
                    r2 += tailstep;
                }

                ktmp += 8 * 9;
            }
        }
    });

    otter::parallel_for(remain_outch_start, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end))
        {
            auto out = output_t[p];

            const float bias0 = bias ? bias[p] : 0.f;

            out.fill_(bias0);

            const float* ktmp = kernel_a[p / 8 + p % 8].data();

            for (int q = 0; q < inch; q++)
            {
                float* outptr = out.data_ptr<float>();

                const float* img0 = input_a[q].data();

                const float* r0 = img0;
                const float* r1 = img0 + w;
                const float* r2 = img0 + w * 2;

                const float* k0 = ktmp;
                const float* k1 = ktmp + 3;
                const float* k2 = ktmp + 6;

    #if __ARM_NEON
                float32x4_t _k0123 = vld1q_f32(k0);
                float32x4_t _k3456 = vld1q_f32(k1);
                float32x4_t _k6789 = vld1q_f32(k2);
    #endif // __ARM_NEON

                int i = 0;

                for (; i < outh; i++)
                {
    #if __ARM_NEON
                    int nn = outw >> 2;
                    int remain = outw & 3;
    #else
                    int remain = outw;
    #endif // __ARM_NEON

    #if __ARM_NEON
    #if __aarch64__
                    if (nn > 0)
                    {
                        asm volatile(
                            "prfm       pldl1keep, [%2, #256]          \n"
                            "ld2        {v2.4s, v3.4s}, [%2], #32      \n"
                            "0:                                        \n"

                            "prfm       pldl1keep, [%1, #128]          \n"
                            "ld1        {v0.4s}, [%1]                  \n"

                            "fmla       v0.4s,  v2.4s, %10.s[0]        \n"
                            "fmul       v10.4s, v3.4s, %10.s[1]        \n"

                            "prfm       pldl1keep, [%2, #256]          \n"
                            "ld2        {v8.4s, v9.4s}, [%2]           \n"
                            "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                            "fmul       v11.4s, v1.4s, %10.s[2]        \n"

                            "prfm       pldl1keep, [%3, #256]          \n"
                            "ld2        {v2.4s, v3.4s}, [%3], #32      \n"

                            "fmla       v0.4s,  v2.4s, %11.s[0]        \n"
                            "fmla       v10.4s, v3.4s, %11.s[1]        \n"

                            "prfm       pldl1keep, [%3, #256]          \n"
                            "ld2        {v8.4s, v9.4s}, [%3]           \n"
                            "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                            "fmla       v11.4s, v1.4s, %11.s[2]        \n"

                            "prfm       pldl1keep, [%4, #256]          \n"
                            "ld2        {v2.4s, v3.4s}, [%4], #32      \n"

                            "fmla       v0.4s,  v2.4s, %12.s[0]        \n"
                            "fmla       v10.4s, v3.4s, %12.s[1]        \n"

                            "prfm       pldl1keep, [%4, #256]          \n"
                            "ld2        {v8.4s, v9.4s}, [%4]           \n"
                            "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                            "fmla       v11.4s, v1.4s, %12.s[2]        \n"

                            "prfm       pldl1keep, [%2, #256]          \n"
                            "ld2        {v2.4s, v3.4s}, [%2], #32      \n"

                            "fadd       v0.4s, v0.4s, v10.4s           \n"
                            "fadd       v0.4s, v0.4s, v11.4s           \n"

                            "subs       %w0, %w0, #1                   \n"
                            "st1        {v0.4s}, [%1], #16             \n"
                            "bne        0b                             \n"
                            "sub        %2, %2, #32                    \n"
                            : "=r"(nn),     // %0
                            "=r"(outptr), // %1
                            "=r"(r0),     // %2
                            "=r"(r1),     // %3
                            "=r"(r2)      // %4
                            : "0"(nn),
                            "1"(outptr),
                            "2"(r0),
                            "3"(r1),
                            "4"(r2),
                            "w"(_k0123), // %10
                            "w"(_k3456), // %11
                            "w"(_k6789)  // %12
                            : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                    }
    #else
                    if (nn > 0)
                    {
                        asm volatile(
                            "pld        [%2, #256]          \n"
                            "vld2.f32   {d4-d7}, [%2]!      \n"

                            "0:                             \n"
                            "pld        [%1, #128]          \n"
                            "vld1.f32   {d0-d1}, [%1]       \n"

                            "vmla.f32   q0, q2, %e10[0]     \n"
                            "vmul.f32   q10, q3, %e10[1]    \n"

                            "pld        [%2, #128]          \n"
                            "vld2.f32   {d16-d17}, [%2]     \n"
                            "vext.32    q1, q2, q8, #1      \n"

                            "vmul.f32   q11, q1, %f10[0]    \n"

                            "pld        [%3, #256]          \n"
                            "vld2.f32   {d4-d7}, [%3]!      \n"

                            "vmla.f32   q0, q2, %e11[0]     \n"
                            "vmla.f32   q10, q3, %e11[1]    \n"

                            "pld        [%3, #128]          \n"
                            "vld2.f32   {d16-d17}, [%3]     \n"
                            "vext.32    q1, q2, q8, #1      \n"

                            "vmla.f32   q11, q1, %f11[0]    \n"

                            "pld        [%4, #256]          \n"
                            "vld2.f32   {d4-d7}, [%4]!      \n"

                            "vmla.f32   q0, q2, %e12[0]     \n"
                            "vmla.f32   q10, q3, %e12[1]    \n"

                            "pld        [%4, #128]          \n"
                            "vld2.f32   {d16-d17}, [%4]     \n"
                            "vext.32    q1, q2, q8, #1      \n"

                            "vmla.f32   q11, q1, %f12[0]    \n"

                            "pld        [%2, #256]          \n"
                            "vld2.f32   {d4-d7}, [%2]!      \n"

                            "vadd.f32   q0, q0, q10         \n"
                            "vadd.f32   q0, q0, q11         \n"

                            "subs       %0, #1              \n"
                            "vst1.f32   {d0-d1}, [%1]!      \n"
                            "bne        0b                  \n"
                            "sub        %2, #32             \n"
                            : "=r"(nn),     // %0
                            "=r"(outptr), // %1
                            "=r"(r0),     // %2
                            "=r"(r1),     // %3
                            "=r"(r2)      // %4
                            : "0"(nn),
                            "1"(outptr),
                            "2"(r0),
                            "3"(r1),
                            "4"(r2),
                            "w"(_k0123), // %10
                            "w"(_k3456), // %11
                            "w"(_k6789)  // %12
                            : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                    }
    #endif // __aarch64__
    #endif // __ARM_NEON
                    for (; remain > 0; remain--)
                    {
    #if __ARM_NEON
                        float32x4_t _r00 = vld1q_f32(r0);
                        float32x4_t _r10 = vld1q_f32(r1);
                        float32x4_t _r20 = vld1q_f32(r2);

                        float32x4_t _sum = vmulq_f32(_r00, _k0123);
                        _sum = vmlaq_f32(_sum, _r10, _k3456);
                        _sum = vmlaq_f32(_sum, _r20, _k6789);

                        _sum = vsetq_lane_f32(*outptr, _sum, 3);

    #if __aarch64__
                        *outptr = vaddvq_f32(_sum);
    #else
                        float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                        _ss = vpadd_f32(_ss, _ss);

                        *outptr = vget_lane_f32(_ss, 0);
    #endif // __aarch64__
    #else
                        float sum = 0;

                        sum += r0[0] * ktmp[0];
                        sum += r0[1] * ktmp[1];
                        sum += r0[2] * ktmp[2];
                        sum += r1[0] * ktmp[3];
                        sum += r1[1] * ktmp[4];
                        sum += r1[2] * ktmp[5];
                        sum += r2[0] * ktmp[6];
                        sum += r2[1] * ktmp[7];
                        sum += r2[2] * ktmp[8];

                        *outptr += sum;
    #endif // __ARM_NEON

                        r0 += 2;
                        r1 += 2;
                        r2 += 2;
                        outptr++;
                    }

                    r0 += tailstep;
                    r1 += tailstep;
                    r2 += tailstep;
                }

                ktmp += 9;
            }
        }
    });
    
    return output;
}

Tensor conv2d_3x3s2_packed_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, self.options());
    
    return conv2d_3x3s2_packed_neon_out(self, weight, weight_o, bias, kernel_size, stride, padding, output);
}



}   // end namespace otter
