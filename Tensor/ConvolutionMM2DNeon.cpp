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

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

namespace otter {

#ifdef __ARM_NEON__
void im2col_sgemm_conv2d_impl(
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
    tmp = otter::full({packChannel, packHeight, packWidth}, -1, ScalarType::Float);
    
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
void im2col_sgemm_conv2d_impl(const Tensor& im2col_, const Tensor& kernel_pack4x4_, const Tensor& bias_, int64_t input_channels, int64_t output_channels, Tensor& output) {}
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
static void convolution_im2col_sgemm_transform_kernel_neon(const Tensor& _kernel, Tensor& kernel_tf, int64_t input_channels, int64_t out_chnnels, int64_t kernel_width, int64_t kernel_height) {
}
#endif

Tensor& sgemm_conv2d_1x1s1_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    if (!output.defined()) {
        auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
        output = otter::empty(output_size, self.options());
    }
    
    const Tensor input = self.contiguous();
    const int64_t input_channels  = input.size(1);
    const int64_t output_channels = weight.size(0);
    
    Tensor im2col = self.view({self.size(0), self.size(1), -1});
    Tensor kernel_pack4x4;
    otter::convolution_im2col_sgemm_transform_kernel_neon(weight, kernel_pack4x4, input_channels, output_channels, 1, 1);
    otter::im2col_sgemm_conv2d_impl(im2col, kernel_pack4x4, bias, input_channels, output_channels, output);
    
    return output;
}

Tensor sgemm_conv2d_1x1s1_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    auto out = otter::empty(output_size, self.options());
    
    sgemm_conv2d_1x1s1_neon_out(self, weight, bias, kernel_size, stride, padding, out);
    
    return out;
}

Tensor& sgemm_conv2d_1x1s2_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    if (!output.defined()) {
        auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
        output = otter::empty(output_size, self.options());
    }
    
    int64_t batch_size = self.size(0);
    int64_t input_channels = self.size(1);
    int64_t input_width = self.size(3);
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
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
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    auto out = otter::empty(output_size, self.options());
    
    return sgemm_conv2d_1x1s2_neon_out(self, weight, bias, kernel_size, stride, padding, out);
}

Tensor& sgemm_conv2d_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    if (!output.defined()) {
        auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
        output = otter::empty(output_size, self.options());
    }
    
    const int64_t kernel_height = kernel_size[0];
    const int64_t kernel_width  = kernel_size[1];
    
    const Tensor input = self.contiguous();
    const int64_t input_channels  = input.size(1);
    const int64_t output_channels = weight.size(0);
    
    Tensor im2col = otter::im2col_cpu(input, kernel_size, stride, padding, {1, 1});
    Tensor kernel_pack4x4;
    otter::convolution_im2col_sgemm_transform_kernel_neon(weight, kernel_pack4x4, input_channels, output_channels, kernel_width, kernel_height);
    
    im2col_sgemm_conv2d_impl(
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
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    auto out = otter::empty(output_size, self.options());
    
    sgemm_conv2d_neon_out(self, weight, bias, kernel_size, stride, padding, out);
    
    return out;
}

}
