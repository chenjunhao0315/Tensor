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
static void convolution_im2col_sgemm_transform_kernel_neon(const Tensor& kernel_, Tensor& kernel_tf, int64_t input_channels, int64_t output_channels, int64_t kernel_width, int64_t kernel_height) {
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
static void convolution_im2col_sgemm_transform_kernel_neon(const Tensor& /*_kernel*/, Tensor& /*kernel_tf*/, int64_t /*input_channels*/, int64_t /*out_chnnels*/, int64_t /*kernel_width*/, int64_t /*kernel_height*/) {
}
#endif

Tensor& sgemm_conv2d_1x1s1_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef /*kernel_size*/,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_(output_size);
    
    const Tensor input = self.contiguous();
    const int64_t input_channels  = input.size(1);
    const int64_t output_channels = weight.size(0);
    
    Tensor im2col = self.view({self.size(0), self.size(1), -1});
    Tensor kernel_pack4x4;
    otter::convolution_im2col_sgemm_transform_kernel_neon(weight, kernel_pack4x4, input_channels, output_channels, 1, 1);
    otter::im2col_sgemm_conv2d_impl_neon(im2col, kernel_pack4x4, bias, input_channels, output_channels, output);
    
    return output;
}

Tensor sgemm_conv2d_1x1s1_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, self.options());
    
    return sgemm_conv2d_1x1s1_neon_out(self, weight, bias, kernel_size, stride, padding, output);
}

Tensor& sgemm_conv2d_1x1s2_neon_out(
    const Tensor& self,
    const Tensor& weight,
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
    
    return otter::sgemm_conv2d_1x1s1_neon_out(self_shrink, weight, bias, kernel_size, stride, padding, output);
}

Tensor sgemm_conv2d_1x1s2_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {

    auto output = otter::empty({}, self.options());
    
    return sgemm_conv2d_1x1s2_neon_out(self, weight, bias, kernel_size, stride, padding, output);
}

Tensor& sgemm_conv2d_neon_out(
    const Tensor& self,
    const Tensor& weight,
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
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {

    auto output = otter::empty({}, self.options());
    
    return sgemm_conv2d_neon_out(self, weight, bias, kernel_size, stride, padding, output);
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
    
    int64_t inch = self.size(1);

    int64_t outw  = output_shape[3];
    int64_t outh  = output_shape[2];
    int64_t outch = output_shape[1];

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
        for (const auto pp : otter::irange(begin, end))
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
            int p = pp * 6;

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
                float* outptr0 = out0.accessor<float, 3>().data();
                float* outptr1 = out1.accessor<float, 3>().data();
                float* outptr2 = out2.accessor<float, 3>().data();
                float* outptr3 = out3.accessor<float, 3>().data();
                float* outptr4 = out4.accessor<float, 3>().data();
                float* outptr5 = out5.accessor<float, 3>().data();

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

                if (nn > 0)
                {
                    asm volatile(
                        "pld        [%7, #128]              \n"
                        "vld1.f32   {d24-d25}, [%7 :128]!   \n" // q12 = r0

                        "pld        [%1, #128]              \n"
                        "vld1.f32   {d12-d13}, [%1 :128]    \n" // q6 = outptr0

                        "pld        [%2, #128]              \n"
                        "vld1.f32   {d14-d15}, [%2 :128]    \n" // q7 = outptr1

                        "vmla.f32   q6, q12, %e22[0]        \n"

                        "0:                                 \n"

                        "pld        [%3, #128]              \n"
                        "vld1.f32   {d16-d17}, [%3 :128]    \n" // q8 = outptr2

                        "vmla.f32   q7, q12, %e23[0]        \n"

                        "pld        [%4, #128]              \n"
                        "vld1.f32   {d18-d19}, [%4 :128]    \n" // q9 = outptr3

                        "vmla.f32   q8, q12, %e24[0]        \n"

                        "pld        [%8, #128]              \n"
                        "vld1.f32   {d26-d27}, [%8 :128]!   \n" // q13 = r1

                        "vmla.f32   q9, q12, %e25[0]        \n"

                        "pld        [%5, #128]              \n"
                        "vld1.f32   {d20-d21}, [%5 :128]    \n" // q10 = outptr4

                        "vmla.f32   q6, q13, %e22[1]        \n"
                        "vmla.f32   q7, q13, %e23[1]        \n"

                        "pld        [%6, #128]              \n"
                        "vld1.f32   {d22-d23}, [%6 :128]    \n" // q11 = outptr5

                        "vmla.f32   q10, q12, %e26[0]       \n"
                        "vmla.f32   q11, q12, %e27[0]       \n"

                        "vmla.f32   q8, q13, %e24[1]        \n"
                        "vmla.f32   q9, q13, %e25[1]        \n"

                        "pld        [%9, #128]              \n"
                        "vld1.f32   {d28-d29}, [%9 :128]!   \n" // q14 = r2

                        "vmla.f32   q10, q13, %e26[1]       \n"
                        "vmla.f32   q11, q13, %e27[1]       \n"

                        "vmla.f32   q6, q14, %f22[0]        \n"
                        "vmla.f32   q7, q14, %f23[0]        \n"
                        "vmla.f32   q8, q14, %f24[0]        \n"
                        "vmla.f32   q9, q14, %f25[0]        \n"

                        "pld        [%10, #128]             \n"
                        "vld1.f32   {d30-d31}, [%10 :128]!  \n" // q15 = r3

                        "vmla.f32   q10, q14, %f26[0]       \n"
                        "vmla.f32   q11, q14, %f27[0]       \n"

                        "vmla.f32   q6, q15, %f22[1]        \n"
                        "vmla.f32   q7, q15, %f23[1]        \n"
                        "vmla.f32   q8, q15, %f24[1]        \n"
                        "vmla.f32   q9, q15, %f25[1]        \n"

                        "pld        [%7, #128]              \n"
                        "vld1.f32   {d24-d25}, [%7 :128]!   \n" // q12 = r0

                        "vmla.f32   q10, q15, %f26[1]       \n"
                        "vmla.f32   q11, q15, %f27[1]       \n"

                        "vst1.f32   {d12-d13}, [%1 :128]!   \n"
                        "vst1.f32   {d14-d15}, [%2 :128]!   \n"

                        "pld        [%1, #128]              \n"
                        "vld1.f32   {d12-d13}, [%1 :128]    \n" // q6 = outptr0

                        "vst1.f32   {d16-d17}, [%3 :128]!   \n"
                        "vst1.f32   {d18-d19}, [%4 :128]!   \n"

                        "vmla.f32   q6, q12, %e22[0]        \n"

                        "pld        [%2, #128]              \n"
                        "vld1.f32   {d14-d15}, [%2 :128]    \n" // q7 = outptr1

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
                        "=r"(r0),      // %7
                        "=r"(r1),      // %8
                        "=r"(r2),      // %9
                        "=r"(r3)       // %10
                        : "0"(nn),
                        "1"(outptr0),
                        "2"(outptr1),
                        "3"(outptr2),
                        "4"(outptr3),
                        "5"(outptr4),
                        "6"(outptr5),
                        "7"(r0),
                        "8"(r1),
                        "9"(r2),
                        "10"(r3),
                        "w"(_k0), // %22
                        "w"(_k1), // %23
                        "w"(_k2), // %24
                        "w"(_k3), // %25
                        "w"(_k4), // %26
                        "w"(_k5)  // %27
                        : "cc", "memory", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
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
                float* outptr0 = out0.accessor<float, 3>().data();
                float* outptr1 = out1.accessor<float, 3>().data();
                float* outptr2 = out2.accessor<float, 3>().data();
                float* outptr3 = out3.accessor<float, 3>().data();
                float* outptr4 = out4.accessor<float, 3>().data();
                float* outptr5 = out5.accessor<float, 3>().data();

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
            int p = remain_outch_start + pp * 4;

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
                float* outptr0 = out0.accessor<float, 3>().data();
                float* outptr1 = out1.accessor<float, 3>().data();
                float* outptr2 = out2.accessor<float, 3>().data();
                float* outptr3 = out3.accessor<float, 3>().data();

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
                float* outptr0 = out0.accessor<float, 3>().data();
                float* outptr1 = out1.accessor<float, 3>().data();
                float* outptr2 = out2.accessor<float, 3>().data();
                float* outptr3 = out3.accessor<float, 3>().data();

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
                float* outptr = out.accessor<float, 3>().data();

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
                float* outptr = out.accessor<float, 3>().data();

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



}   // end namespace otter
