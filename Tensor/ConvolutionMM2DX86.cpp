//
//  ConvolutionMM2DX86.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/2.
//

// https://github.com/Tencent/ncnn/blob/master/src/layer/x86/convolution_sgemm.h
// https://github.com/Tencent/ncnn/blob/master/src/layer/x86/convolution_1x1.h
// https://github.com/Tencent/ncnn/blob/master/src/layer/x86/convolution_3x3.h

#include "ConvolutionMM2DX86.hpp"
#include "Tensor.hpp"
#include "TensorFactory.hpp"
#include "im2col.hpp"
#include "Parallel.hpp"
#include "Padding.hpp"
#include "VecIntrinsic.hpp"
#include "TensorTransform.hpp"

namespace otter {

#if __SSE2__
static void im2col_sgemm_conv2d_impl_x86(
    const Tensor& im2col_,
    const Tensor& kernel_packed_,
    const Tensor& bias_,
    int64_t input_channels,
    int64_t output_channels,
    Tensor& output) {
    
    Tensor im2col = im2col_.view({input_channels, -1, im2col_.size(2)});
    const float* bias = (bias_.defined()) ? bias_.data_ptr<float>() : nullptr;
    
    const int64_t outSize = im2col.size(2); // im2col width = output_height * output_width
    const int64_t kernelSize = im2col.size(1); // im2col height = kernel_height * kernel_width
    
    Tensor tmp;
    int64_t packChannel = outSize;
    int64_t packHeight  = input_channels;
    int64_t packWidth   = kernelSize;
#if __SSE2__
#if __AVX__
    if (outSize >= 8) {
        packChannel = outSize / 8 + (outSize % 8) / 4 + outSize % 4;
        packHeight  = input_channels;
        packWidth   = 8 * kernelSize;
    } else if (outSize >= 4) {
        packChannel = outSize / 4 + outSize % 4;
        packHeight  = input_channels;
        packWidth   = 4 * kernelSize;
    }
#else
    if (outSize >= 4) {
        packChannel = outSize / 4 + outSize % 4;
        packHeight  = input_channels;
        packWidth   = 4 * kernelSize;
    }
#endif
    tmp = otter::empty({packChannel, packHeight, packWidth}, ScalarType::Float);
    
    auto tmp_a = tmp.accessor<float, 3>();
    auto im2col_a = im2col.accessor<float, 3>();
    
    {
#if __AVX__
        int nn_size = (int)outSize / 8;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = (int)ii * 8;

                float* tmpptr = tmp_a[i / 8].data();

                for (int q = 0; q < input_channels; q++)
                {
                    const float* img0 = (const float*)im2col_a[q].data() + i;

                    for (int k = 0; k < kernelSize; k++)
                    {
                        __m256 _r0 = _mm256_loadu_ps(img0);
                        _mm256_storeu_ps(tmpptr, _r0);
                        img0 += outSize;
                        tmpptr += 8;
                    }
                }
            }
        });

        int remain_size_start = nn_size * 8;
        nn_size = ((int)outSize - remain_size_start) / 4;
#else
        int nn_size = outSize / 4;
        int remain_size_start = 0;
#endif

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + (int)ii * 4;

#if __AVX__
                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();
#else
                float* tmpptr = tmp_a[i / 4].data();
#endif

                for (int q = 0; q < input_channels; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i;

                    for (int k = 0; k < kernelSize; k++)
                    {
                        __m128 _r0 = _mm_loadu_ps(img0);
                        _mm_storeu_ps(tmpptr, _r0);
                        img0 += outSize;
                        tmpptr += 4;
                    }
                }
            }
        });

        remain_size_start += nn_size * 4;

        otter::parallel_for(remain_size_start, outSize, 0, [&](int64_t begin, int64_t end) {
            for (const auto i : otter::irange(begin, end)) {
#if __AVX__
                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();
#else
                float* tmpptr = tmp_a[i / 4 + i % 4].data();
#endif

                for (int q = 0; q < input_channels; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i;

                    for (int k = 0; k < kernelSize; k++) {
                        tmpptr[0] = img0[0];
                        img0 += outSize;
                        tmpptr += 1;
                    }
                }
            }
        });
    }
#else // __SSE2__
    tmp = otter::empty({packChannel, packHeight, packWidth}, ScalarType::Float);
    {
        otter::parallel_for(0, outSize, 0, [&](int64_t begin, int64_t end) {
            for (const auto i : otter::irange(begin, end)) {
                float* tmpptr = tmp_a[i].data();

                for (int q = 0; q < input_channels; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i;

                    for (int k = 0; k < kernelSize; k++) {
                        tmpptr[0] = img0[0];
                        img0 += size;
                        tmpptr += 1;
                    }
                }
            }
        });
    }
#endif // __SSE2__
    
    auto output_a = output.accessor<float, 4>()[0];
    auto kernel_a = kernel_packed_.accessor<float, 3>();
    
#if __SSE2__
    int nn_outch = output_channels >> 3;
    int remain_outch_start = nn_outch << 3;

    otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto pp : otter::irange(begin, end)) {
            int p = (int)pp * 8;

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
    #if __AVX__
            for (; i + 7 < outSize; i += 8)
            {
                const float* tmpptr = tmp_a[i / 8].data();
                const float* kptr = kernel_a[p / 8].data();

                int nn = input_channels * kernelSize; // input_channels always > 0

                __m256 _sum0 = _mm256_broadcast_ss(biasptr);
                __m256 _sum1 = _mm256_broadcast_ss(biasptr + 1);
                __m256 _sum2 = _mm256_broadcast_ss(biasptr + 2);
                __m256 _sum3 = _mm256_broadcast_ss(biasptr + 3);
                __m256 _sum4 = _mm256_broadcast_ss(biasptr + 4);
                __m256 _sum5 = _mm256_broadcast_ss(biasptr + 5);
                __m256 _sum6 = _mm256_broadcast_ss(biasptr + 6);
                __m256 _sum7 = _mm256_broadcast_ss(biasptr + 7);

                int j = 0;
                for (; j + 3 < nn; j += 4)
                {
                    __m256 _val = _mm256_loadu_ps(tmpptr);

                    __m256 _w0 = _mm256_broadcast_ss(kptr);
                    __m256 _w1 = _mm256_broadcast_ss(kptr + 1);
                    _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                    __m256 _w2 = _mm256_broadcast_ss(kptr + 2);
                    __m256 _w3 = _mm256_broadcast_ss(kptr + 3);
                    _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                    __m256 _w4 = _mm256_broadcast_ss(kptr + 4);
                    __m256 _w5 = _mm256_broadcast_ss(kptr + 5);
                    _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                    __m256 _w6 = _mm256_broadcast_ss(kptr + 6);
                    __m256 _w7 = _mm256_broadcast_ss(kptr + 7);
                    _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                    tmpptr += 8;
                    kptr += 8;

                    _val = _mm256_loadu_ps(tmpptr);

                    _w0 = _mm256_broadcast_ss(kptr);
                    _w1 = _mm256_broadcast_ss(kptr + 1);
                    _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                    _w2 = _mm256_broadcast_ss(kptr + 2);
                    _w3 = _mm256_broadcast_ss(kptr + 3);
                    _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                    _w4 = _mm256_broadcast_ss(kptr + 4);
                    _w5 = _mm256_broadcast_ss(kptr + 5);
                    _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                    _w6 = _mm256_broadcast_ss(kptr + 6);
                    _w7 = _mm256_broadcast_ss(kptr + 7);
                    _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                    tmpptr += 8;
                    kptr += 8;

                    _val = _mm256_loadu_ps(tmpptr);

                    _w0 = _mm256_broadcast_ss(kptr);
                    _w1 = _mm256_broadcast_ss(kptr + 1);
                    _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                    _w2 = _mm256_broadcast_ss(kptr + 2);
                    _w3 = _mm256_broadcast_ss(kptr + 3);
                    _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                    _w4 = _mm256_broadcast_ss(kptr + 4);
                    _w5 = _mm256_broadcast_ss(kptr + 5);
                    _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                    _w6 = _mm256_broadcast_ss(kptr + 6);
                    _w7 = _mm256_broadcast_ss(kptr + 7);
                    _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                    tmpptr += 8;
                    kptr += 8;

                    _val = _mm256_loadu_ps(tmpptr);

                    _w0 = _mm256_broadcast_ss(kptr);
                    _w1 = _mm256_broadcast_ss(kptr + 1);
                    _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                    _w2 = _mm256_broadcast_ss(kptr + 2);
                    _w3 = _mm256_broadcast_ss(kptr + 3);
                    _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                    _w4 = _mm256_broadcast_ss(kptr + 4);
                    _w5 = _mm256_broadcast_ss(kptr + 5);
                    _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                    _w6 = _mm256_broadcast_ss(kptr + 6);
                    _w7 = _mm256_broadcast_ss(kptr + 7);
                    _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                    tmpptr += 8;
                    kptr += 8;
                }
                for (; j < nn; j++)
                {
                    __m256 _val = _mm256_loadu_ps(tmpptr);

                    __m256 _w0 = _mm256_broadcast_ss(kptr);
                    __m256 _w1 = _mm256_broadcast_ss(kptr + 1);
                    _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                    __m256 _w2 = _mm256_broadcast_ss(kptr + 2);
                    __m256 _w3 = _mm256_broadcast_ss(kptr + 3);
                    _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                    __m256 _w4 = _mm256_broadcast_ss(kptr + 4);
                    __m256 _w5 = _mm256_broadcast_ss(kptr + 5);
                    _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                    __m256 _w6 = _mm256_broadcast_ss(kptr + 6);
                    __m256 _w7 = _mm256_broadcast_ss(kptr + 7);
                    _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                    tmpptr += 8;
                    kptr += 8;
                }

                _mm256_storeu_ps(outptr0, _sum0);
                _mm256_storeu_ps(outptr1, _sum1);
                _mm256_storeu_ps(outptr2, _sum2);
                _mm256_storeu_ps(outptr3, _sum3);
                _mm256_storeu_ps(outptr4, _sum4);
                _mm256_storeu_ps(outptr5, _sum5);
                _mm256_storeu_ps(outptr6, _sum6);
                _mm256_storeu_ps(outptr7, _sum7);

                outptr0 += 8;
                outptr1 += 8;
                outptr2 += 8;
                outptr3 += 8;
                outptr4 += 8;
                outptr5 += 8;
                outptr6 += 8;
                outptr7 += 8;
            }
    #endif
            for (; i + 3 < outSize; i += 4)
            {
    #if __AVX__
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();
    #else
                const float* tmpptr = tmp_a[i / 4].data();
    #endif
                const float* kptr = kernel_a[p / 8].data();

                int nn = (int)input_channels * kernelSize; // input_channels always > 0

                __m128 _sum0 = _mm_set1_ps(biasptr[0]);
                __m128 _sum1 = _mm_set1_ps(biasptr[1]);
                __m128 _sum2 = _mm_set1_ps(biasptr[2]);
                __m128 _sum3 = _mm_set1_ps(biasptr[3]);
                __m128 _sum4 = _mm_set1_ps(biasptr[4]);
                __m128 _sum5 = _mm_set1_ps(biasptr[5]);
                __m128 _sum6 = _mm_set1_ps(biasptr[6]);
                __m128 _sum7 = _mm_set1_ps(biasptr[7]);

                int j = 0;
                for (; j + 3 < nn; j += 4)
                {
                    __m128 _val = _mm_loadu_ps(tmpptr);

                    __m128 _w0 = _mm_load1_ps(kptr);
                    __m128 _w1 = _mm_load1_ps(kptr + 1);
                    _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                    __m128 _w2 = _mm_load1_ps(kptr + 2);
                    __m128 _w3 = _mm_load1_ps(kptr + 3);
                    _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                    __m128 _w4 = _mm_load1_ps(kptr + 4);
                    __m128 _w5 = _mm_load1_ps(kptr + 5);
                    _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                    _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                    __m128 _w6 = _mm_load1_ps(kptr + 6);
                    __m128 _w7 = _mm_load1_ps(kptr + 7);
                    _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                    _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                    tmpptr += 4;
                    kptr += 8;

                    _val = _mm_loadu_ps(tmpptr);

                    _w0 = _mm_load1_ps(kptr);
                    _w1 = _mm_load1_ps(kptr + 1);
                    _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                    _w2 = _mm_load1_ps(kptr + 2);
                    _w3 = _mm_load1_ps(kptr + 3);
                    _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                    _w4 = _mm_load1_ps(kptr + 4);
                    _w5 = _mm_load1_ps(kptr + 5);
                    _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                    _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                    _w6 = _mm_load1_ps(kptr + 6);
                    _w7 = _mm_load1_ps(kptr + 7);
                    _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                    _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                    tmpptr += 4;
                    kptr += 8;

                    _val = _mm_loadu_ps(tmpptr);

                    _w0 = _mm_load1_ps(kptr);
                    _w1 = _mm_load1_ps(kptr + 1);
                    _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                    _w2 = _mm_load1_ps(kptr + 2);
                    _w3 = _mm_load1_ps(kptr + 3);
                    _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                    _w4 = _mm_load1_ps(kptr + 4);
                    _w5 = _mm_load1_ps(kptr + 5);
                    _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                    _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                    _w6 = _mm_load1_ps(kptr + 6);
                    _w7 = _mm_load1_ps(kptr + 7);
                    _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                    _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                    tmpptr += 4;
                    kptr += 8;

                    _val = _mm_loadu_ps(tmpptr);

                    _w0 = _mm_load1_ps(kptr);
                    _w1 = _mm_load1_ps(kptr + 1);
                    _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                    _w2 = _mm_load1_ps(kptr + 2);
                    _w3 = _mm_load1_ps(kptr + 3);
                    _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                    _w4 = _mm_load1_ps(kptr + 4);
                    _w5 = _mm_load1_ps(kptr + 5);
                    _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                    _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                    _w6 = _mm_load1_ps(kptr + 6);
                    _w7 = _mm_load1_ps(kptr + 7);
                    _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                    _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                    tmpptr += 4;
                    kptr += 8;
                }
                for (; j < nn; j++)
                {
                    __m128 _val = _mm_loadu_ps(tmpptr);

                    __m128 _w0 = _mm_load1_ps(kptr);
                    __m128 _w1 = _mm_load1_ps(kptr + 1);
                    _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                    __m128 _w2 = _mm_load1_ps(kptr + 2);
                    __m128 _w3 = _mm_load1_ps(kptr + 3);
                    _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                    __m128 _w4 = _mm_load1_ps(kptr + 4);
                    __m128 _w5 = _mm_load1_ps(kptr + 5);
                    _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                    _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                    __m128 _w6 = _mm_load1_ps(kptr + 6);
                    __m128 _w7 = _mm_load1_ps(kptr + 7);
                    _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                    _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                    tmpptr += 4;
                    kptr += 8;
                }

                _mm_storeu_ps(outptr0, _sum0);
                _mm_storeu_ps(outptr1, _sum1);
                _mm_storeu_ps(outptr2, _sum2);
                _mm_storeu_ps(outptr3, _sum3);
                _mm_storeu_ps(outptr4, _sum4);
                _mm_storeu_ps(outptr5, _sum5);
                _mm_storeu_ps(outptr6, _sum6);
                _mm_storeu_ps(outptr7, _sum7);

                outptr0 += 4;
                outptr1 += 4;
                outptr2 += 4;
                outptr3 += 4;
                outptr4 += 4;
                outptr5 += 4;
                outptr6 += 4;
                outptr7 += 4;
            }
            for (; i < outSize; i++)
            {
    #if __AVX__
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();
    #else
                const float* tmpptr = tmp_a[i / 4 + i % 4].data();
    #endif
                const float* kptr = kernel_a[p / 8].data();

                int nn = (int)input_channels * kernelSize; // input_channels always > 0

    #if __AVX__
                __m256 _sum = _mm256_loadu_ps(biasptr);
    #else
                __m128 _sum0 = _mm_loadu_ps(biasptr);
                __m128 _sum1 = _mm_loadu_ps(biasptr + 4);
    #endif

                int j = 0;
                for (; j + 3 < nn; j += 4)
                {
    #if __AVX__
                    __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                    __m256 _w0 = _mm256_loadu_ps(kptr);
                    _sum = _mm256_comp_fmadd_ps(_val0, _w0, _sum);

                    __m256 _val1 = _mm256_broadcast_ss(tmpptr + 1);
                    __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                    _sum = _mm256_comp_fmadd_ps(_val1, _w1, _sum);

                    __m256 _val2 = _mm256_broadcast_ss(tmpptr + 2);
                    __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                    _sum = _mm256_comp_fmadd_ps(_val2, _w2, _sum);

                    __m256 _val3 = _mm256_broadcast_ss(tmpptr + 3);
                    __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                    _sum = _mm256_comp_fmadd_ps(_val3, _w3, _sum);
    #else
                    __m128 _val0 = _mm_load1_ps(tmpptr);
                    __m128 _w00 = _mm_loadu_ps(kptr);
                    __m128 _w01 = _mm_loadu_ps(kptr + 4);
                    _sum0 = _mm_comp_fmadd_ps(_val0, _w00, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val0, _w01, _sum1);

                    __m128 _val1 = _mm_load1_ps(tmpptr + 1);
                    __m128 _w10 = _mm_loadu_ps(kptr + 8);
                    __m128 _w11 = _mm_loadu_ps(kptr + 12);
                    _sum0 = _mm_comp_fmadd_ps(_val1, _w10, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val1, _w11, _sum1);

                    __m128 _val2 = _mm_load1_ps(tmpptr + 2);
                    __m128 _w20 = _mm_loadu_ps(kptr + 16);
                    __m128 _w21 = _mm_loadu_ps(kptr + 20);
                    _sum0 = _mm_comp_fmadd_ps(_val2, _w20, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val2, _w21, _sum1);

                    __m128 _val3 = _mm_load1_ps(tmpptr + 3);
                    __m128 _w30 = _mm_loadu_ps(kptr + 24);
                    __m128 _w31 = _mm_loadu_ps(kptr + 28);
                    _sum0 = _mm_comp_fmadd_ps(_val3, _w30, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val3, _w31, _sum1);
    #endif
                    tmpptr += 4;
                    kptr += 32;
                }
                for (; j < nn; j++)
                {
    #if __AVX__
                    __m256 _val = _mm256_broadcast_ss(tmpptr);
                    __m256 _w = _mm256_loadu_ps(kptr);
                    _sum = _mm256_comp_fmadd_ps(_val, _w, _sum);
    #else
                    __m128 _val = _mm_load1_ps(tmpptr);
                    __m128 _w0 = _mm_loadu_ps(kptr);
                    __m128 _w1 = _mm_loadu_ps(kptr + 4);
                    _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
    #endif
                    tmpptr += 1;
                    kptr += 8;
                }

                float sum[8];
    #if __AVX__
                _mm256_storeu_ps(sum, _sum);
    #else
                _mm_storeu_ps(sum, _sum0);
                _mm_storeu_ps(sum + 4, _sum1);
    #endif

                outptr0[0] = sum[0];
                outptr1[0] = sum[1];
                outptr2[0] = sum[2];
                outptr3[0] = sum[3];
                outptr4[0] = sum[4];
                outptr5[0] = sum[5];
                outptr6[0] = sum[6];
                outptr7[0] = sum[7];

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
    });

    nn_outch = (output_channels - remain_outch_start) >> 2;

    otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto pp : otter::irange(begin, end)) {
            int p = remain_outch_start + (int)pp * 4;

            float* outptr0 = output_a[p + 0].data();
            float* outptr1 = output_a[p + 1].data();
            float* outptr2 = output_a[p + 2].data();
            float* outptr3 = output_a[p + 3].data();

            const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + p : zeros;

            int i = 0;
    #if __AVX__
            for (; i + 7 < outSize; i += 8)
            {
                const float* tmpptr = tmp_a[i / 8].data();
                const float* kptr = kernel_a[p / 8 + (p % 8) / 4].data();

                int nn = (int)input_channels * kernelSize; // input_channels always > 0

                __m256 _sum0 = _mm256_broadcast_ss(biasptr);
                __m256 _sum1 = _mm256_broadcast_ss(biasptr + 1);
                __m256 _sum2 = _mm256_broadcast_ss(biasptr + 2);
                __m256 _sum3 = _mm256_broadcast_ss(biasptr + 3);

                int j = 0;
                for (; j + 3 < nn; j += 4)
                {
                    __m256 _val = _mm256_loadu_ps(tmpptr);
                    __m256 _w0 = _mm256_broadcast_ss(kptr);
                    __m256 _w1 = _mm256_broadcast_ss(kptr + 1);
                    __m256 _w2 = _mm256_broadcast_ss(kptr + 2);
                    __m256 _w3 = _mm256_broadcast_ss(kptr + 3);
                    _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                    tmpptr += 8;
                    kptr += 4;

                    _val = _mm256_loadu_ps(tmpptr);
                    _w0 = _mm256_broadcast_ss(kptr);
                    _w1 = _mm256_broadcast_ss(kptr + 1);
                    _w2 = _mm256_broadcast_ss(kptr + 2);
                    _w3 = _mm256_broadcast_ss(kptr + 3);
                    _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                    tmpptr += 8;
                    kptr += 4;

                    _val = _mm256_loadu_ps(tmpptr);
                    _w0 = _mm256_broadcast_ss(kptr);
                    _w1 = _mm256_broadcast_ss(kptr + 1);
                    _w2 = _mm256_broadcast_ss(kptr + 2);
                    _w3 = _mm256_broadcast_ss(kptr + 3);
                    _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                    tmpptr += 8;
                    kptr += 4;

                    _val = _mm256_loadu_ps(tmpptr);
                    _w0 = _mm256_broadcast_ss(kptr);
                    _w1 = _mm256_broadcast_ss(kptr + 1);
                    _w2 = _mm256_broadcast_ss(kptr + 2);
                    _w3 = _mm256_broadcast_ss(kptr + 3);
                    _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                    tmpptr += 8;
                    kptr += 4;
                }
                for (; j < nn; j++)
                {
                    __m256 _val = _mm256_loadu_ps(tmpptr);
                    __m256 _w0 = _mm256_broadcast_ss(kptr);
                    __m256 _w1 = _mm256_broadcast_ss(kptr + 1);
                    __m256 _w2 = _mm256_broadcast_ss(kptr + 2);
                    __m256 _w3 = _mm256_broadcast_ss(kptr + 3);
                    _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                    tmpptr += 8;
                    kptr += 4;
                }

                _mm256_storeu_ps(outptr0, _sum0);
                _mm256_storeu_ps(outptr1, _sum1);
                _mm256_storeu_ps(outptr2, _sum2);
                _mm256_storeu_ps(outptr3, _sum3);

                outptr0 += 8;
                outptr1 += 8;
                outptr2 += 8;
                outptr3 += 8;
            }
    #endif
            for (; i + 3 < outSize; i += 4)
            {
    #if __AVX__
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();
    #else
                const float* tmpptr = tmp_a[i / 4].data();
    #endif
                const float* kptr = kernel_a[p / 8 + (p % 8) / 4].data();

                int nn = (int)input_channels * kernelSize; // input_channels always > 0

                __m128 _sum0 = _mm_set1_ps(biasptr[0]);
                __m128 _sum1 = _mm_set1_ps(biasptr[1]);
                __m128 _sum2 = _mm_set1_ps(biasptr[2]);
                __m128 _sum3 = _mm_set1_ps(biasptr[3]);

                int j = 0;
                for (; j + 3 < nn; j += 4)
                {
                    __m128 _val = _mm_loadu_ps(tmpptr);
                    __m128 _w0 = _mm_load1_ps(kptr);
                    __m128 _w1 = _mm_load1_ps(kptr + 1);
                    __m128 _w2 = _mm_load1_ps(kptr + 2);
                    __m128 _w3 = _mm_load1_ps(kptr + 3);
                    _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                    tmpptr += 4;
                    kptr += 4;

                    _val = _mm_loadu_ps(tmpptr);
                    _w0 = _mm_load1_ps(kptr);
                    _w1 = _mm_load1_ps(kptr + 1);
                    _w2 = _mm_load1_ps(kptr + 2);
                    _w3 = _mm_load1_ps(kptr + 3);
                    _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                    tmpptr += 4;
                    kptr += 4;

                    _val = _mm_loadu_ps(tmpptr);
                    _w0 = _mm_load1_ps(kptr);
                    _w1 = _mm_load1_ps(kptr + 1);
                    _w2 = _mm_load1_ps(kptr + 2);
                    _w3 = _mm_load1_ps(kptr + 3);
                    _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                    tmpptr += 4;
                    kptr += 4;

                    _val = _mm_loadu_ps(tmpptr);
                    _w0 = _mm_load1_ps(kptr);
                    _w1 = _mm_load1_ps(kptr + 1);
                    _w2 = _mm_load1_ps(kptr + 2);
                    _w3 = _mm_load1_ps(kptr + 3);
                    _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                    tmpptr += 4;
                    kptr += 4;
                }
                for (; j < nn; j++)
                {
                    __m128 _val = _mm_loadu_ps(tmpptr);
                    __m128 _w0 = _mm_load1_ps(kptr);
                    __m128 _w1 = _mm_load1_ps(kptr + 1);
                    __m128 _w2 = _mm_load1_ps(kptr + 2);
                    __m128 _w3 = _mm_load1_ps(kptr + 3);
                    _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                    tmpptr += 4;
                    kptr += 4;
                }

                _mm_storeu_ps(outptr0, _sum0);
                _mm_storeu_ps(outptr1, _sum1);
                _mm_storeu_ps(outptr2, _sum2);
                _mm_storeu_ps(outptr3, _sum3);

                outptr0 += 4;
                outptr1 += 4;
                outptr2 += 4;
                outptr3 += 4;
            }
            for (; i < outSize; i++)
            {
    #if __AVX__
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();
    #else
                const float* tmpptr = tmp_a[i / 4 + i % 4].data();
    #endif
                const float* kptr = kernel_a[p / 8 + (p % 8) / 4].data();

                int nn = (int)input_channels * kernelSize; // input_channels always > 0

                __m128 _sum = _mm_loadu_ps(biasptr);

                int j = 0;
                for (; j + 3 < nn; j += 4)
                {
                    __m128 _val0 = _mm_load1_ps(tmpptr);
                    __m128 _w0 = _mm_loadu_ps(kptr);
                    _sum = _mm_comp_fmadd_ps(_val0, _w0, _sum);

                    __m128 _val1 = _mm_load1_ps(tmpptr + 1);
                    __m128 _w1 = _mm_loadu_ps(kptr + 4);
                    _sum = _mm_comp_fmadd_ps(_val1, _w1, _sum);

                    __m128 _val2 = _mm_load1_ps(tmpptr + 2);
                    __m128 _w2 = _mm_loadu_ps(kptr + 8);
                    _sum = _mm_comp_fmadd_ps(_val2, _w2, _sum);

                    __m128 _val3 = _mm_load1_ps(tmpptr + 3);
                    __m128 _w3 = _mm_loadu_ps(kptr + 12);
                    _sum = _mm_comp_fmadd_ps(_val3, _w3, _sum);

                    tmpptr += 4;
                    kptr += 16;
                }
                for (; j < nn; j++)
                {
                    __m128 _val = _mm_load1_ps(tmpptr);
                    __m128 _w0 = _mm_loadu_ps(kptr);
                    _sum = _mm_comp_fmadd_ps(_val, _w0, _sum);

                    tmpptr += 1;
                    kptr += 4;
                }

                float sum[4];
                _mm_storeu_ps(sum, _sum);

                outptr0[0] = sum[0];
                outptr1[0] = sum[1];
                outptr2[0] = sum[2];
                outptr3[0] = sum[3];

                outptr0++;
                outptr1++;
                outptr2++;
                outptr3++;
            }
        }
    });

    remain_outch_start += nn_outch << 2;

    otter::parallel_for(remain_outch_start, output_channels, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            float* outptr0 = output_a[p].data();

            const float bias0 = bias ? bias[p] : 0.f;

            int i = 0;
    #if __AVX__
            for (; i + 7 < outSize; i += 8)
            {
                const float* tmpptr = tmp_a[i / 8].data();
                const float* kptr = kernel_a[p / 8 + (p % 8) / 4 + p % 4].data();

                int nn = (int)input_channels * kernelSize; // input_channels always > 0

                __m256 _sum0 = _mm256_set1_ps(bias0);

                int j = 0;
                for (; j + 3 < nn; j += 4)
                {
                    __m256 _val0 = _mm256_loadu_ps(tmpptr);
                    __m256 _w0 = _mm256_broadcast_ss(kptr);
                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);

                    __m256 _val1 = _mm256_loadu_ps(tmpptr + 8);
                    __m256 _w1 = _mm256_broadcast_ss(kptr + 1);
                    _sum0 = _mm256_comp_fmadd_ps(_val1, _w1, _sum0);

                    __m256 _val2 = _mm256_loadu_ps(tmpptr + 16);
                    __m256 _w2 = _mm256_broadcast_ss(kptr + 2);
                    _sum0 = _mm256_comp_fmadd_ps(_val2, _w2, _sum0);

                    __m256 _val3 = _mm256_loadu_ps(tmpptr + 24);
                    __m256 _w3 = _mm256_broadcast_ss(kptr + 3);
                    _sum0 = _mm256_comp_fmadd_ps(_val3, _w3, _sum0);

                    tmpptr += 32;
                    kptr += 4;
                }
                for (; j < nn; j++)
                {
                    __m256 _val = _mm256_loadu_ps(tmpptr);
                    __m256 _w0 = _mm256_broadcast_ss(kptr);
                    _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                    tmpptr += 8;
                    kptr++;
                }

                _mm256_storeu_ps(outptr0, _sum0);

                outptr0 += 8;
            }
    #endif
            for (; i + 3 < outSize; i += 4)
            {
    #if __AVX__
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();
    #else
                const float* tmpptr = tmp_a[i / 4].data();
    #endif
                const float* kptr = kernel_a[p / 8 + (p % 8) / 4 + p % 4].data();

                int nn = (int)input_channels * kernelSize; // input_channels always > 0

                __m128 _sum0 = _mm_set1_ps(bias0);

                int j = 0;
                for (; j + 3 < nn; j += 4)
                {
                    __m128 _val0 = _mm_loadu_ps(tmpptr);
                    __m128 _w0 = _mm_load1_ps(kptr);
                    _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);

                    __m128 _val1 = _mm_loadu_ps(tmpptr + 4);
                    __m128 _w1 = _mm_load1_ps(kptr + 1);
                    _sum0 = _mm_comp_fmadd_ps(_val1, _w1, _sum0);

                    __m128 _val2 = _mm_loadu_ps(tmpptr + 8);
                    __m128 _w2 = _mm_load1_ps(kptr + 2);
                    _sum0 = _mm_comp_fmadd_ps(_val2, _w2, _sum0);

                    __m128 _val3 = _mm_loadu_ps(tmpptr + 12);
                    __m128 _w3 = _mm_load1_ps(kptr + 3);
                    _sum0 = _mm_comp_fmadd_ps(_val3, _w3, _sum0);

                    tmpptr += 16;
                    kptr += 4;
                }
                for (; j < nn; j++)
                {
                    __m128 _val = _mm_loadu_ps(tmpptr);
                    __m128 _w0 = _mm_load1_ps(kptr);
                    _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                    tmpptr += 4;
                    kptr++;
                }

                _mm_storeu_ps(outptr0, _sum0);

                outptr0 += 4;
            }
            for (; i < outSize; i++)
            {
    #if __AVX__
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();
    #else
                const float* tmpptr = tmp_a[i / 4 + i % 4].data();
    #endif
                const float* kptr = kernel_a[p / 8 + (p % 8) / 4 + p % 4].data();

                int nn = (int)input_channels * kernelSize; // input_channels always > 0

                float sum0 = bias0;

                for (int j = 0; j < nn; j++)
                {
                    sum0 += tmpptr[0] * kptr[0];
                    tmpptr++;
                    kptr++;
                }

                outptr0[0] = sum0;

                outptr0++;
            }
        }
    });
#else // __SSE2__
    otter::parallel_for(0, output_channels, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            float* outptr0 = output_a[p].data();

            const float bias0 = bias ? bias[p] : 0.f;

            for (int i = 0; i < size; i++)
            {
                const float* tmpptr = tmp_a[i].data();
                const float* kptr = kernel_a[p].data();

                int nn = input_channels * kernelSize; // input_channels always > 0

                float sum0 = bias0;

                for (int j = 0; j < nn; j++)
                {
                    sum0 += tmpptr[0] * kptr[0];
                    tmpptr++;
                    kptr++;
                }

                outptr0[0] = sum0;

                outptr0++;
            }
        }
    });
#endif // __SSE2__
}
#else
static void im2col_sgemm_conv2d_impl_x86(
    const Tensor& /*im2col_*/,
    const Tensor& /*kernel_packed_*/,
    const Tensor& /*bias_*/,
    int64_t /*input_channels*/,
    int64_t /*output_channels*/,
    Tensor& /*output*/) {
}
#endif

#ifdef __SSE2__
void convolution_im2col_sgemm_transform_kernel_x86(const Tensor& kernel_, Tensor& kernel_tf, int64_t input_channels, int64_t output_channels, int64_t kernel_width, int64_t kernel_height) {
    const int64_t kernelSize = kernel_width * kernel_height;
    
    auto kernel = kernel_.view({output_channels, input_channels, kernelSize});

    // interleave
    // src = maxk-inch-outch
    // dst = 8b-maxk-inch-outch/8b
#if __SSE2__
    kernel_tf = otter::empty({output_channels / 8 + (output_channels % 8) / 4 + output_channels % 4, input_channels, 8 * kernelSize}, otter::ScalarType::Float);
    
    auto kernel_a = kernel.accessor<float, 3>();
    auto kernel_tf_a = kernel_tf.accessor<float, 3>();

    int q = 0;
    for (; q + 7 < output_channels; q += 8)
    {
        const auto k0 = kernel_a[q + 0];
        const auto k1 = kernel_a[q + 1];
        const auto k2 = kernel_a[q + 2];
        const auto k3 = kernel_a[q + 3];
        const auto k4 = kernel_a[q + 4];
        const auto k5 = kernel_a[q + 5];
        const auto k6 = kernel_a[q + 6];
        const auto k7 = kernel_a[q + 7];

        float* g00 = kernel_tf_a[q / 8].data();

        for (int p = 0; p < input_channels; p++)
        {
            const float* k00 = k0[p].data();
            const float* k10 = k1[p].data();
            const float* k20 = k2[p].data();
            const float* k30 = k3[p].data();
            const float* k40 = k4[p].data();
            const float* k50 = k5[p].data();
            const float* k60 = k6[p].data();
            const float* k70 = k7[p].data();

            for (int k = 0; k < kernelSize; k++)
            {
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
    for (; q + 3 < output_channels; q += 4)
    {
        const auto k0 = kernel_a[q + 0];
        const auto k1 = kernel_a[q + 1];
        const auto k2 = kernel_a[q + 2];
        const auto k3 = kernel_a[q + 3];

        float* g00 = kernel_tf_a[q / 8 + (q % 8) / 4].data();

        for (int p = 0; p < input_channels; p++)
        {
            const float* k00 = k0[p].data();
            const float* k10 = k1[p].data();
            const float* k20 = k2[p].data();
            const float* k30 = k3[p].data();

            for (int k = 0; k < kernelSize; k++)
            {
                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];

                g00 += 4;
            }
        }
    }
    for (; q < output_channels; q++)
    {
        const auto k0 = kernel_a[q];

        float* g00 = kernel_tf_a[q / 8 + (q % 8) / 4 + q % 4].data();

        for (int p = 0; p < input_channels; p++)
        {
            const float* k00 = k0[p].data();

            for (int k = 0; k < kernelSize; k++)
            {
                g00[0] = k00[k];

                g00 += 1;
            }
        }
    }
#else
    kernel_tm = kernel;
#endif // __SSE2__
}
#else
void convolution_im2col_sgemm_transform_kernel_x86(const Tensor& /*_kernel*/, Tensor& /*kernel_tf*/, int64_t /*input_channels*/, int64_t /*out_chnnels*/, int64_t /*kernel_width*/, int64_t /*kernel_height*/) {
}
#endif

Tensor& sgemm_conv2d_x86_out(
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
    
    const int64_t input_channels  = self.size(1);
    const int64_t output_channels = weight.size(0);
    
    Tensor im2col = otter::im2col_cpu(self, kernel_size, stride, padding, {1, 1});
    Tensor kernel_packed;
    if (weight_o.defined())
        kernel_packed = weight_o;
    else
        otter::convolution_im2col_sgemm_transform_kernel_x86(weight, kernel_packed, input_channels, output_channels, kernel_width, kernel_height);
    otter::im2col_sgemm_conv2d_impl_x86(im2col, kernel_packed, bias, input_channels, output_channels, output);
    
    return output;
}
    
Tensor sgemm_conv2d_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, self.options());
    
    return sgemm_conv2d_x86_out(self, weight, weight_o, bias, kernel_size, stride, padding, output);
}

void conv3x3s1_winograd23_transform_kernel_x86(
    const Tensor& kernel_,
    Tensor& kernel_tf,
    int64_t input_channels,
    int64_t output_channels) {
    
    const int64_t kernelSize = 3 * 3;
    
    float* kernel = kernel_.view({output_channels, input_channels, kernelSize}).data_ptr<float>();
    
    kernel_tf = otter::empty({output_channels, input_channels, 4 * 4}, otter::ScalarType::Float);

    // G
    const float ktm[4][3] = {
        {1.0f, 0.0f, 0.0f},
        {1.0f / 2, 1.0f / 2, 1.0f / 2},
        {1.0f / 2, -1.0f / 2, 1.0f / 2},
        {0.0f, 0.0f, 1.0f}
    };
    
    auto kernel_tf_a = kernel_tf.accessor<float, 3>();

    otter::parallel_for(0, output_channels, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            for (int q = 0; q < input_channels; q++) {
                const float* kernel0 = (const float*)kernel + p * input_channels * 9 + q * 9;
                float* kernel_tm0 = kernel_tf_a[p][q].data();

                // transform kernel
                const float* k0 = kernel0;
                const float* k1 = kernel0 + 3;
                const float* k2 = kernel0 + 6;

                // h
                float tmp[4][3];
                for (int i = 0; i < 4; i++)
                {
                    tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                    tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                    tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
                }

                // U
                for (int j = 0; j < 4; j++)
                {
                    float* tmpp = &tmp[j][0];

                    for (int i = 0; i < 4; i++)
                    {
                        kernel_tm0[j * 4 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                    }
                }
            }
        }
    });
}

Tensor& conv2d_3x3s1_winograd23_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias_,
    IntArrayRef /*kernel_size*/,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
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
    
    outw = (outw + 1) / 2 * 2;
    outh = (outh + 1) / 2 * 2;

    w = outw + 2;
    h = outh + 2;
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1] + w - origin_w, padding[0], padding[0] + h - origin_h}, 0);
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        otter::conv3x3s1_winograd23_transform_kernel_x86(weight, kernel_tf, inch, outch);
    auto kernel_tf_a = kernel_tf.accessor<float, 3>();
    
    const float* bias = (bias_.defined()) ? bias_.data_ptr<float>() : nullptr;
    
    auto input_a = input.accessor<float, 4>()[0];
    
    Tensor input_tf;
    {
        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm / 4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 4;

        const int tiles = nColBlocks * nRowBlocks;
        
        input_tf = otter::empty({inch, tiles, 4 * 4}, otter::ScalarType::Float);
        
        auto input_tf_a = input_tf.accessor<float, 3>();

        // BT
        // const float itm[4][4] = {
        //     {1.0f,  0.0f, -1.0f,  0.0f},
        //     {0.0f,  1.0f,  1.00f, 0.0f},
        //     {0.0f, -1.0f,  1.00f, 0.0f},
        //     {0.0f, -1.0f,  0.00f, 1.0f}
        // };
        otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
            for (const auto q : otter::irange(begin, end)) {
                const float* img = input_a[q].data();
                float* out_tm0 = input_tf_a[q].data();

                for (int j = 0; j < nColBlocks; j++) {
                    const float* r0 = img + w * j * 2;
                    const float* r1 = r0 + w;
                    const float* r2 = r1 + w;
                    const float* r3 = r2 + w;

                    for (int i = 0; i < nRowBlocks; i++) {
    #if __AVX__
                        __m128 _d0, _d1, _d2, _d3;
                        __m128 _w0, _w1, _w2, _w3;

                        // load
                        _d0 = _mm_loadu_ps(r0);
                        _d1 = _mm_loadu_ps(r1);
                        _d2 = _mm_loadu_ps(r2);
                        _d3 = _mm_loadu_ps(r3);

                        // w = B_t * d
                        _w0 = _mm_sub_ps(_d0, _d2);
                        _w1 = _mm_add_ps(_d1, _d2);
                        _w2 = _mm_sub_ps(_d2, _d1);
                        _w3 = _mm_sub_ps(_d3, _d1);

                        // transpose d to d_t
                        _MM_TRANSPOSE4_PS(_w0, _w1, _w2, _w3);

                        // d = B_t * d_t
                        _d0 = _mm_sub_ps(_w0, _w2);
                        _d1 = _mm_add_ps(_w1, _w2);
                        _d2 = _mm_sub_ps(_w2, _w1);
                        _d3 = _mm_sub_ps(_w3, _w1);

                        // save to out_tm
                        _mm_storeu_ps(out_tm0, _d0);
                        _mm_storeu_ps(out_tm0 + 4, _d1);
                        _mm_storeu_ps(out_tm0 + 8, _d2);
                        _mm_storeu_ps(out_tm0 + 12, _d3);
    #else
                        float d0[4], d1[4], d2[4], d3[4];
                        float w0[4], w1[4], w2[4], w3[4];
                        float t0[4], t1[4], t2[4], t3[4];
                        // load
                        for (int n = 0; n < 4; n++)
                        {
                            d0[n] = r0[n];
                            d1[n] = r1[n];
                            d2[n] = r2[n];
                            d3[n] = r3[n];
                        }
                        // w = B_t * d
                        for (int n = 0; n < 4; n++)
                        {
                            w0[n] = d0[n] - d2[n];
                            w1[n] = d1[n] + d2[n];
                            w2[n] = d2[n] - d1[n];
                            w3[n] = d3[n] - d1[n];
                        }
                        // transpose d to d_t
                        {
                            t0[0] = w0[0];
                            t1[0] = w0[1];
                            t2[0] = w0[2];
                            t3[0] = w0[3];
                            t0[1] = w1[0];
                            t1[1] = w1[1];
                            t2[1] = w1[2];
                            t3[1] = w1[3];
                            t0[2] = w2[0];
                            t1[2] = w2[1];
                            t2[2] = w2[2];
                            t3[2] = w2[3];
                            t0[3] = w3[0];
                            t1[3] = w3[1];
                            t2[3] = w3[2];
                            t3[3] = w3[3];
                        }
                        // d = B_t * d_t
                        for (int n = 0; n < 4; n++)
                        {
                            d0[n] = t0[n] - t2[n];
                            d1[n] = t1[n] + t2[n];
                            d2[n] = t2[n] - t1[n];
                            d3[n] = t3[n] - t1[n];
                        }
                        // save to out_tm
                        for (int n = 0; n < 4; n++)
                        {
                            out_tm0[n] = d0[n];
                            out_tm0[n + 4] = d1[n];
                            out_tm0[n + 8] = d2[n];
                            out_tm0[n + 12] = d3[n];
                        }
    #endif
                        r0 += 2;
                        r1 += 2;
                        r2 += 2;
                        r3 += 2;

                        out_tm0 += 16;
                    }
                }
            }
        });
    }
    input.reset();
    
    Tensor output_tf;
    {
        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm / 4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 4;

        const int tiles = nColBlocks * nRowBlocks;
        
        output_tf = otter::empty({outch, tiles, 16}, otter::ScalarType::Float);
        
        auto output_tf_a = output_tf.accessor<float, 3>();

        int nn_outch = outch >> 2;
        int remain_outch_start = nn_outch << 2;
        
        auto input_tf_a = input_tf.accessor<float, 3>();

        otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
            for (const auto pp : otter::irange(begin, end)) {
                int p = pp * 4;

                auto out0_tm = output_tf_a[p + 0];
                auto out1_tm = output_tf_a[p + 1];
                auto out2_tm = output_tf_a[p + 2];
                auto out3_tm = output_tf_a[p + 3];

                const auto kernel0_tm = kernel_tf_a[p + 0];
                const auto kernel1_tm = kernel_tf_a[p + 1];
                const auto kernel2_tm = kernel_tf_a[p + 2];
                const auto kernel3_tm = kernel_tf_a[p + 3];

                for (int i = 0; i < tiles; i++)
                {
                    float* output0_tm = out0_tm[i].data();
                    float* output1_tm = out1_tm[i].data();
                    float* output2_tm = out2_tm[i].data();
                    float* output3_tm = out3_tm[i].data();

    #if __AVX__
                    float zero_val = 0.f;

                    __m256 _sum0 = _mm256_broadcast_ss(&zero_val);
                    __m256 _sum0n = _mm256_broadcast_ss(&zero_val);
                    __m256 _sum1 = _mm256_broadcast_ss(&zero_val);
                    __m256 _sum1n = _mm256_broadcast_ss(&zero_val);
                    __m256 _sum2 = _mm256_broadcast_ss(&zero_val);
                    __m256 _sum2n = _mm256_broadcast_ss(&zero_val);
                    __m256 _sum3 = _mm256_broadcast_ss(&zero_val);
                    __m256 _sum3n = _mm256_broadcast_ss(&zero_val);

                    int q = 0;

                    for (; q + 3 < inch; q += 4)
                    {
                        const float* r0 = input_tf_a[q + 0][i].data();
                        const float* r1 = input_tf_a[q + 1][i].data();
                        const float* r2 = input_tf_a[q + 2][i].data();
                        const float* r3 = input_tf_a[q + 3][i].data();

                        const float* k0 = kernel0_tm[q].data();
                        const float* k1 = kernel1_tm[q].data();
                        const float* k2 = kernel2_tm[q].data();
                        const float* k3 = kernel3_tm[q].data();

                        __m256 _r0 = _mm256_loadu_ps(r0);
                        __m256 _r0n = _mm256_loadu_ps(r0 + 8);
                        // k0
                        __m256 _k0 = _mm256_loadu_ps(k0);
                        __m256 _k0n = _mm256_loadu_ps(k0 + 8);
                        __m256 _k1 = _mm256_loadu_ps(k1);
                        __m256 _k1n = _mm256_loadu_ps(k1 + 8);
                        __m256 _k2 = _mm256_loadu_ps(k2);
                        __m256 _k2n = _mm256_loadu_ps(k2 + 8);
                        __m256 _k3 = _mm256_loadu_ps(k3);
                        __m256 _k3n = _mm256_loadu_ps(k3 + 8);
                        _sum0 = _mm256_comp_fmadd_ps(_r0, _k0, _sum0);
                        _sum0n = _mm256_comp_fmadd_ps(_r0n, _k0n, _sum0n);
                        _sum1 = _mm256_comp_fmadd_ps(_r0, _k1, _sum1);
                        _sum1n = _mm256_comp_fmadd_ps(_r0n, _k1n, _sum1n);
                        _sum2 = _mm256_comp_fmadd_ps(_r0, _k2, _sum2);
                        _sum2n = _mm256_comp_fmadd_ps(_r0n, _k2n, _sum2n);
                        _sum3 = _mm256_comp_fmadd_ps(_r0, _k3, _sum3);
                        _sum3n = _mm256_comp_fmadd_ps(_r0n, _k3n, _sum3n);

                        // k1
                        _r0 = _mm256_loadu_ps(r1);
                        _r0n = _mm256_loadu_ps(r1 + 8);
                        _k0 = _mm256_loadu_ps(k0 + 16);
                        _k0n = _mm256_loadu_ps(k0 + 24);
                        _k1 = _mm256_loadu_ps(k1 + 16);
                        _k1n = _mm256_loadu_ps(k1 + 24);
                        _k2 = _mm256_loadu_ps(k2 + 16);
                        _k2n = _mm256_loadu_ps(k2 + 24);
                        _k3 = _mm256_loadu_ps(k3 + 16);
                        _k3n = _mm256_loadu_ps(k3 + 24);
                        _sum0 = _mm256_comp_fmadd_ps(_r0, _k0, _sum0);
                        _sum0n = _mm256_comp_fmadd_ps(_r0n, _k0n, _sum0n);
                        _sum1 = _mm256_comp_fmadd_ps(_r0, _k1, _sum1);
                        _sum1n = _mm256_comp_fmadd_ps(_r0n, _k1n, _sum1n);
                        _sum2 = _mm256_comp_fmadd_ps(_r0, _k2, _sum2);
                        _sum2n = _mm256_comp_fmadd_ps(_r0n, _k2n, _sum2n);
                        _sum3 = _mm256_comp_fmadd_ps(_r0, _k3, _sum3);
                        _sum3n = _mm256_comp_fmadd_ps(_r0n, _k3n, _sum3n);
                        // k2
                        _r0 = _mm256_loadu_ps(r2);
                        _r0n = _mm256_loadu_ps(r2 + 8);
                        _k0 = _mm256_loadu_ps(k0 + 32);
                        _k0n = _mm256_loadu_ps(k0 + 40);
                        _k1 = _mm256_loadu_ps(k1 + 32);
                        _k1n = _mm256_loadu_ps(k1 + 40);
                        _k2 = _mm256_loadu_ps(k2 + 32);
                        _k2n = _mm256_loadu_ps(k2 + 40);
                        _k3 = _mm256_loadu_ps(k3 + 32);
                        _k3n = _mm256_loadu_ps(k3 + 40);
                        _sum0 = _mm256_comp_fmadd_ps(_r0, _k0, _sum0);
                        _sum0n = _mm256_comp_fmadd_ps(_r0n, _k0n, _sum0n);
                        _sum1 = _mm256_comp_fmadd_ps(_r0, _k1, _sum1);
                        _sum1n = _mm256_comp_fmadd_ps(_r0n, _k1n, _sum1n);
                        _sum2 = _mm256_comp_fmadd_ps(_r0, _k2, _sum2);
                        _sum2n = _mm256_comp_fmadd_ps(_r0n, _k2n, _sum2n);
                        _sum3 = _mm256_comp_fmadd_ps(_r0, _k3, _sum3);
                        _sum3n = _mm256_comp_fmadd_ps(_r0n, _k3n, _sum3n);
                        // k3
                        _r0 = _mm256_loadu_ps(r3);
                        _r0n = _mm256_loadu_ps(r3 + 8);
                        _k0 = _mm256_loadu_ps(k0 + 48);
                        _k0n = _mm256_loadu_ps(k0 + 56);
                        _k1 = _mm256_loadu_ps(k1 + 48);
                        _k1n = _mm256_loadu_ps(k1 + 56);
                        _k2 = _mm256_loadu_ps(k2 + 48);
                        _k2n = _mm256_loadu_ps(k2 + 56);
                        _k3 = _mm256_loadu_ps(k3 + 48);
                        _k3n = _mm256_loadu_ps(k3 + 56);
                        _sum0 = _mm256_comp_fmadd_ps(_r0, _k0, _sum0);
                        _sum0n = _mm256_comp_fmadd_ps(_r0n, _k0n, _sum0n);
                        _sum1 = _mm256_comp_fmadd_ps(_r0, _k1, _sum1);
                        _sum1n = _mm256_comp_fmadd_ps(_r0n, _k1n, _sum1n);
                        _sum2 = _mm256_comp_fmadd_ps(_r0, _k2, _sum2);
                        _sum2n = _mm256_comp_fmadd_ps(_r0n, _k2n, _sum2n);
                        _sum3 = _mm256_comp_fmadd_ps(_r0, _k3, _sum3);
                        _sum3n = _mm256_comp_fmadd_ps(_r0n, _k3n, _sum3n);
                    }

                    for (; q < inch; q++)
                    {
                        const float* r0 = input_tf_a[q][i].data();

                        const float* k0 = kernel0_tm[q].data();
                        const float* k1 = kernel1_tm[q].data();
                        const float* k2 = kernel2_tm[q].data();
                        const float* k3 = kernel3_tm[q].data();

                        __m256 _r0 = _mm256_loadu_ps(r0);
                        __m256 _r0n = _mm256_loadu_ps(r0 + 8);
                        __m256 _k0 = _mm256_loadu_ps(k0);
                        __m256 _k0n = _mm256_loadu_ps(k0 + 8);
                        __m256 _k1 = _mm256_loadu_ps(k1);
                        __m256 _k1n = _mm256_loadu_ps(k1 + 8);
                        __m256 _k2 = _mm256_loadu_ps(k2);
                        __m256 _k2n = _mm256_loadu_ps(k2 + 8);
                        __m256 _k3 = _mm256_loadu_ps(k3);
                        __m256 _k3n = _mm256_loadu_ps(k3 + 8);

                        _sum0 = _mm256_comp_fmadd_ps(_r0, _k0, _sum0);
                        _sum0n = _mm256_comp_fmadd_ps(_r0n, _k0n, _sum0n);
                        _sum1 = _mm256_comp_fmadd_ps(_r0, _k1, _sum1);
                        _sum1n = _mm256_comp_fmadd_ps(_r0n, _k1n, _sum1n);
                        _sum2 = _mm256_comp_fmadd_ps(_r0, _k2, _sum2);
                        _sum2n = _mm256_comp_fmadd_ps(_r0n, _k2n, _sum2n);
                        _sum3 = _mm256_comp_fmadd_ps(_r0, _k3, _sum3);
                        _sum3n = _mm256_comp_fmadd_ps(_r0n, _k3n, _sum3n);
                    }

                    _mm256_storeu_ps(output0_tm, _sum0);
                    _mm256_storeu_ps(output0_tm + 8, _sum0n);
                    _mm256_storeu_ps(output1_tm, _sum1);
                    _mm256_storeu_ps(output1_tm + 8, _sum1n);
                    _mm256_storeu_ps(output2_tm, _sum2);
                    _mm256_storeu_ps(output2_tm + 8, _sum2n);
                    _mm256_storeu_ps(output3_tm, _sum3);
                    _mm256_storeu_ps(output3_tm + 8, _sum3n);
    #else
                    float sum0[16] = {0.0f};
                    float sum1[16] = {0.0f};
                    float sum2[16] = {0.0f};
                    float sum3[16] = {0.0f};

                    int q = 0;
                    for (; q + 3 < inch; q += 4)
                    {
                        const float* r0 = input_tf_a[q + 0][i].data();
                        const float* r1 = input_tf_a[q + 1][i].data();
                        const float* r2 = input_tf_a[q + 2][i].data();
                        const float* r3 = input_tf_a[q + 3][i].data();

                        const float* k0 = kernel0_tm[q].data();
                        const float* k1 = kernel1_tm[q].data();
                        const float* k2 = kernel2_tm[q].data();
                        const float* k3 = kernel3_tm[q].data();

                        for (int n = 0; n < 16; n++)
                        {
                            sum0[n] += r0[n] * k0[n];
                            k0 += 16;
                            sum0[n] += r1[n] * k0[n];
                            k0 += 16;
                            sum0[n] += r2[n] * k0[n];
                            k0 += 16;
                            sum0[n] += r3[n] * k0[n];
                            k0 -= 16 * 3;

                            sum1[n] += r0[n] * k1[n];
                            k1 += 16;
                            sum1[n] += r1[n] * k1[n];
                            k1 += 16;
                            sum1[n] += r2[n] * k1[n];
                            k1 += 16;
                            sum1[n] += r3[n] * k1[n];
                            k1 -= 16 * 3;

                            sum2[n] += r0[n] * k2[n];
                            k2 += 16;
                            sum2[n] += r1[n] * k2[n];
                            k2 += 16;
                            sum2[n] += r2[n] * k2[n];
                            k2 += 16;
                            sum2[n] += r3[n] * k2[n];
                            k2 -= 16 * 3;

                            sum3[n] += r0[n] * k3[n];
                            k3 += 16;
                            sum3[n] += r1[n] * k3[n];
                            k3 += 16;
                            sum3[n] += r2[n] * k3[n];
                            k3 += 16;
                            sum3[n] += r3[n] * k3[n];
                            k3 -= 16 * 3;
                        }
                    }

                    for (; q < inch; q++)
                    {
                        const float* r0 = input_tf_a[q][i].data();

                        const float* k0 = kernel0_tm[q].data();
                        const float* k1 = kernel1_tm[q].data();
                        const float* k2 = kernel2_tm[q].data();
                        const float* k3 = kernel3_tm[q].data();

                        for (int n = 0; n < 16; n++)
                        {
                            sum0[n] += r0[n] * k0[n];
                            sum1[n] += r0[n] * k1[n];
                            sum2[n] += r0[n] * k2[n];
                            sum3[n] += r0[n] * k3[n];
                        }
                    }

                    for (int n = 0; n < 16; n++)
                    {
                        output0_tm[n] = sum0[n];
                        output1_tm[n] = sum1[n];
                        output2_tm[n] = sum2[n];
                        output3_tm[n] = sum3[n];
                    }
    #endif
                }
            }
        });

        otter::parallel_for(remain_outch_start, outch, 0, [&](int64_t begin, int64_t end) {
            for (const auto p : otter::irange(begin, end)) {
                auto out0_tm = output_tf_a[p];
                const auto kernel0_tm = kernel_tf_a[p];

                for (int i = 0; i < tiles; i++)
                {
                    float* output0_tm = out0_tm[i].data();

                    float sum0[16] = {0.0f};

                    int q = 0;
                    for (; q + 3 < inch; q += 4)
                    {
                        const float* r0 = input_tf_a[q + 0][i].data();
                        const float* r1 = input_tf_a[q + 1][i].data();
                        const float* r2 = input_tf_a[q + 2][i].data();
                        const float* r3 = input_tf_a[q + 3][i].data();

                        const float* k0 = kernel0_tm[q + 0].data();
                        const float* k1 = kernel0_tm[q + 1].data();
                        const float* k2 = kernel0_tm[q + 2].data();
                        const float* k3 = kernel0_tm[q + 3].data();

                        for (int n = 0; n < 16; n++)
                        {
                            sum0[n] += r0[n] * k0[n];
                            sum0[n] += r1[n] * k1[n];
                            sum0[n] += r2[n] * k2[n];
                            sum0[n] += r3[n] * k3[n];
                        }
                    }

                    for (; q < inch; q++)
                    {
                        const float* r0 = input_tf_a[q][i].data();
                        const float* k0 = kernel0_tm[q].data();

                        for (int n = 0; n < 16; n++)
                        {
                            sum0[n] += r0[n] * k0[n];
                        }
                    }

                    for (int n = 0; n < 16; n++)
                    {
                        output0_tm[n] = sum0[n];
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
        // AT
        // const float itm[2][4] = {
        //     {1.0f,  1.0f,  1.0f,  0.0f},
        //     {0.0f,  1.0f, -1.0f,  1.0f}
        // };

        int w_tm = outw / 2 * 4;
        int h_tm = outh / 2 * 4;

        int nColBlocks = h_tm / 4; // may be the block num in Feathercnn
        int nRowBlocks = w_tm / 4;

        otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
            for (const auto p : otter::irange(begin, end)) {
                auto out_tm = output_tf_a[p];
                auto out = output_bordered_a[p];

                const float bias0 = bias ? bias[p] : 0.f;

                for (int j = 0; j < nColBlocks; j++)
                {
                    float* outRow0 = out[j * 2 + 0].data();
                    float* outRow1 = out[j * 2 + 1].data();

                    for (int i = 0; i < nRowBlocks; i++)
                    {
                        float* out_tile = out_tm[j * nRowBlocks + i].data();

                        float s0[4], s1[4], s2[4], s3[4];
                        float w0[4], w1[4];
                        float d0[2], d1[2], d2[2], d3[2];
                        float o0[2], o1[2];
                        // load
                        for (int n = 0; n < 4; n++)
                        {
                            s0[n] = out_tile[n];
                            s1[n] = out_tile[n + 4];
                            s2[n] = out_tile[n + 8];
                            s3[n] = out_tile[n + 12];
                        }
                        // w = A_T * W
                        for (int n = 0; n < 4; n++)
                        {
                            w0[n] = s0[n] + s1[n] + s2[n];
                            w1[n] = s1[n] - s2[n] + s3[n];
                        }
                        // transpose w to w_t
                        {
                            d0[0] = w0[0];
                            d0[1] = w1[0];
                            d1[0] = w0[1];
                            d1[1] = w1[1];
                            d2[0] = w0[2];
                            d2[1] = w1[2];
                            d3[0] = w0[3];
                            d3[1] = w1[3];
                        }
                        // Y = A_T * w_t
                        for (int n = 0; n < 2; n++)
                        {
                            o0[n] = d0[n] + d1[n] + d2[n] + bias0;
                            o1[n] = d1[n] - d2[n] + d3[n] + bias0;
                        }
                        // save to top blob tm
                        outRow0[0] = o0[0];
                        outRow0[1] = o0[1];
                        outRow1[0] = o1[0];
                        outRow1[1] = o1[1];

                        outRow0 += 2;
                        outRow1 += 2;
                    }
                }
            }
        });
    }
    
    if (output_bordered.size(3) != output_shape[3] || output.size(2) != output_shape[2])
        otter::crop_(output_bordered, {0, output_bordered.size(3) - output_shape[3], 0, output_bordered.size(2) - output_shape[2]}, output);
    
    return output;
}

Tensor conv2d_3x3s1_winograd23_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, self.options());
    
    return conv2d_3x3s1_winograd23_x86_out(self, weight, weight_o, bias, kernel_size, stride, padding, output);
}

}   // end namespace otter
