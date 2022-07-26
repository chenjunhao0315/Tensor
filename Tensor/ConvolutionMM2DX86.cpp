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

void convolution_winograd_dot_sse(Tensor& bottom_blob_tm, int outch, const Tensor& kernel_tm, Tensor& top_blob_tm)
{
    // Tensor bottom_blob_tm(tiles, 16/36/64, inch, 4u, 1, opt.workspace_allocator);

    const int tiles = bottom_blob_tm.size(2);
    const int batch = bottom_blob_tm.size(1);
    const int inch = bottom_blob_tm.size(0);

    // permute
    Tensor bottom_blob_tm2;
#if __AVX__
    if (tiles >= 8)
        bottom_blob_tm2 = otter::empty({batch, tiles / 8 + (tiles % 8) / 4 + tiles % 4, 8 * inch}, otter::ScalarType::Float);
    else if (tiles >= 4)
        bottom_blob_tm2 = otter::empty({batch, tiles / 4 + tiles % 4, 4 * inch}, otter::ScalarType::Float);
    else
        bottom_blob_tm2 = otter::empty({batch, tiles, 1 * inch}, otter::ScalarType::Float);
#elif __SSE2__
    if (tiles >= 4)
        bottom_blob_tm2 = otter::empty({batch, tiles / 4 + tiles % 4, 4 * inch}, otter::ScalarType::Float);
    else
        bottom_blob_tm2 = otter::empty({batch, tiles, 1 * inch}, otter::ScalarType::Float);
#else
    bottom_blob_tm2 = otter::empty({batch, tiles, 1 * inch}, otter::ScalarType::Float);
#endif
    
    auto bottom_blob_tm_a = bottom_blob_tm.accessor<float, 3>();
    auto bottom_blob_tm2_a = bottom_blob_tm2.accessor<float, 3>();
    int bottom_blob_tm_cstep = tiles * batch;

    otter::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
        for (const auto r : otter::irange(begin, end)) {
            auto tm2 = bottom_blob_tm2_a[r];

            // tile
            int i = 0;
    #if __SSE2__
    #if __AVX__
            for (; i + 7 < tiles; i += 8)
            {
                float* tmpptr = tm2[i / 8].data();

                const float* r0 = bottom_blob_tm_a.data();

                r0 += (r * tiles + i);

                for (int q = 0; q < inch; q++)
                {
                    __m256 _r0 = _mm256_loadu_ps(r0);
                    _mm256_storeu_ps(tmpptr, _r0);

                    r0 += bottom_blob_tm_cstep;
                    tmpptr += 8;
                }
            }
    #endif // __AVX__
            for (; i + 3 < tiles; i += 4)
            {
    #if __AVX__
                float* tmpptr = tm2[i / 8 + (i % 8) / 4].data();
    #else
                float* tmpptr = tm2[i / 4].data();
    #endif

                const float* r0 = bottom_blob_tm_a.data();

                r0 += (r * tiles + i);

                for (int q = 0; q < inch; q++)
                {
                    __m128 _r0 = _mm_loadu_ps(r0);
                    _mm_storeu_ps(tmpptr, _r0);

                    r0 += bottom_blob_tm_cstep;
                    tmpptr += 4;
                }
            }
    #endif // __SSE2__
            for (; i < tiles; i++)
            {
    #if __AVX__
                float* tmpptr = tm2[i / 8 + (i % 8) / 4 + i % 4].data();
    #elif __SSE2__
                float* tmpptr = tm2[i / 4 + i % 4].data();
    #else
                float* tmpptr = tm2[i].data();
    #endif

                const float* r0 = bottom_blob_tm_a.data();

                r0 += (r * tiles + i);

                for (int q = 0; q < inch; q++)
                {
                    tmpptr[0] = r0[0];

                    r0 += bottom_blob_tm_cstep;
                    tmpptr += 1;
                }
            }
        }
    });

    bottom_blob_tm.reset();
    // permute end

    top_blob_tm = otter::empty({outch, batch, tiles}, otter::ScalarType::Float);
    auto top_blob_tm_a = top_blob_tm.accessor<float, 3>();
    auto kernel_tm_a = kernel_tm.accessor<float, 3>();

#if __SSE2__
    int nn_outch = outch >> 3;
    int remain_outch_start = nn_outch << 3;

    otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto pp : otter::irange(begin, end))
        {
            int p = pp * 8;

            float* output0_tm = top_blob_tm_a[p + 0].data();
            float* output1_tm = top_blob_tm_a[p + 1].data();
            float* output2_tm = top_blob_tm_a[p + 2].data();
            float* output3_tm = top_blob_tm_a[p + 3].data();
            float* output4_tm = top_blob_tm_a[p + 4].data();
            float* output5_tm = top_blob_tm_a[p + 5].data();
            float* output6_tm = top_blob_tm_a[p + 6].data();
            float* output7_tm = top_blob_tm_a[p + 7].data();

            const auto kernel0_tm = kernel_tm_a[p / 8];

            for (int r = 0; r < batch; r++)
            {
                const auto bb2 = bottom_blob_tm2_a[r];

                int i = 0;
    #if __AVX__
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2[i / 8].data();
                    const float* k0 = kernel0_tm[r].data();

                    int nn = inch; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    __m256 _sum4 = _mm256_setzero_ps();
                    __m256 _sum5 = _mm256_setzero_ps();
                    __m256 _sum6 = _mm256_setzero_ps();
                    __m256 _sum7 = _mm256_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m256 _val = _mm256_loadu_ps(r0);

                        __m256 _w0 = _mm256_broadcast_ss(k0);
                        __m256 _w1 = _mm256_broadcast_ss(k0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        __m256 _w2 = _mm256_broadcast_ss(k0 + 2);
                        __m256 _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                        __m256 _w4 = _mm256_broadcast_ss(k0 + 4);
                        __m256 _w5 = _mm256_broadcast_ss(k0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                        __m256 _w6 = _mm256_broadcast_ss(k0 + 6);
                        __m256 _w7 = _mm256_broadcast_ss(k0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 8;
                        k0 += 8;

                        _val = _mm256_loadu_ps(r0);

                        _w0 = _mm256_broadcast_ss(k0);
                        _w1 = _mm256_broadcast_ss(k0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm256_broadcast_ss(k0 + 2);
                        _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm256_broadcast_ss(k0 + 4);
                        _w5 = _mm256_broadcast_ss(k0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm256_broadcast_ss(k0 + 6);
                        _w7 = _mm256_broadcast_ss(k0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 8;
                        k0 += 8;

                        _val = _mm256_loadu_ps(r0);

                        _w0 = _mm256_broadcast_ss(k0);
                        _w1 = _mm256_broadcast_ss(k0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm256_broadcast_ss(k0 + 2);
                        _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm256_broadcast_ss(k0 + 4);
                        _w5 = _mm256_broadcast_ss(k0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm256_broadcast_ss(k0 + 6);
                        _w7 = _mm256_broadcast_ss(k0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 8;
                        k0 += 8;

                        _val = _mm256_loadu_ps(r0);

                        _w0 = _mm256_broadcast_ss(k0);
                        _w1 = _mm256_broadcast_ss(k0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm256_broadcast_ss(k0 + 2);
                        _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm256_broadcast_ss(k0 + 4);
                        _w5 = _mm256_broadcast_ss(k0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm256_broadcast_ss(k0 + 6);
                        _w7 = _mm256_broadcast_ss(k0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 8;
                        k0 += 8;
                    }
                    for (; j < nn; j++)
                    {
                        __m256 _val = _mm256_loadu_ps(r0);

                        __m256 _w0 = _mm256_broadcast_ss(k0);
                        __m256 _w1 = _mm256_broadcast_ss(k0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        __m256 _w2 = _mm256_broadcast_ss(k0 + 2);
                        __m256 _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);
                        __m256 _w4 = _mm256_broadcast_ss(k0 + 4);
                        __m256 _w5 = _mm256_broadcast_ss(k0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val, _w5, _sum5);
                        __m256 _w6 = _mm256_broadcast_ss(k0 + 6);
                        __m256 _w7 = _mm256_broadcast_ss(k0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 8;
                        k0 += 8;
                    }

                    _mm256_storeu_ps(output0_tm, _sum0);
                    _mm256_storeu_ps(output1_tm, _sum1);
                    _mm256_storeu_ps(output2_tm, _sum2);
                    _mm256_storeu_ps(output3_tm, _sum3);
                    _mm256_storeu_ps(output4_tm, _sum4);
                    _mm256_storeu_ps(output5_tm, _sum5);
                    _mm256_storeu_ps(output6_tm, _sum6);
                    _mm256_storeu_ps(output7_tm, _sum7);

                    output0_tm += 8;
                    output1_tm += 8;
                    output2_tm += 8;
                    output3_tm += 8;
                    output4_tm += 8;
                    output5_tm += 8;
                    output6_tm += 8;
                    output7_tm += 8;
                }
    #endif // __AVX__
                for (; i + 3 < tiles; i += 4)
                {
    #if __AVX__
                    const float* r0 = bb2[i / 8 + (i % 8) / 4].data();
    #else
                    const float* r0 = bb2[i / 4].data();
    #endif
                    const float* k0 = kernel0_tm[r].data();

                    int nn = inch; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();
                    __m128 _sum4 = _mm_setzero_ps();
                    __m128 _sum5 = _mm_setzero_ps();
                    __m128 _sum6 = _mm_setzero_ps();
                    __m128 _sum7 = _mm_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m128 _val = _mm_loadu_ps(r0);

                        __m128 _w0 = _mm_load1_ps(k0);
                        __m128 _w1 = _mm_load1_ps(k0 + 1);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        __m128 _w2 = _mm_load1_ps(k0 + 2);
                        __m128 _w3 = _mm_load1_ps(k0 + 3);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                        __m128 _w4 = _mm_load1_ps(k0 + 4);
                        __m128 _w5 = _mm_load1_ps(k0 + 5);
                        _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                        __m128 _w6 = _mm_load1_ps(k0 + 6);
                        __m128 _w7 = _mm_load1_ps(k0 + 7);
                        _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 4;
                        k0 += 8;

                        _val = _mm_loadu_ps(r0);

                        _w0 = _mm_load1_ps(k0);
                        _w1 = _mm_load1_ps(k0 + 1);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm_load1_ps(k0 + 2);
                        _w3 = _mm_load1_ps(k0 + 3);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm_load1_ps(k0 + 4);
                        _w5 = _mm_load1_ps(k0 + 5);
                        _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm_load1_ps(k0 + 6);
                        _w7 = _mm_load1_ps(k0 + 7);
                        _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 4;
                        k0 += 8;

                        _val = _mm_loadu_ps(r0);

                        _w0 = _mm_load1_ps(k0);
                        _w1 = _mm_load1_ps(k0 + 1);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm_load1_ps(k0 + 2);
                        _w3 = _mm_load1_ps(k0 + 3);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm_load1_ps(k0 + 4);
                        _w5 = _mm_load1_ps(k0 + 5);
                        _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm_load1_ps(k0 + 6);
                        _w7 = _mm_load1_ps(k0 + 7);
                        _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 4;
                        k0 += 8;

                        _val = _mm_loadu_ps(r0);

                        _w0 = _mm_load1_ps(k0);
                        _w1 = _mm_load1_ps(k0 + 1);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _w2 = _mm_load1_ps(k0 + 2);
                        _w3 = _mm_load1_ps(k0 + 3);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                        _w4 = _mm_load1_ps(k0 + 4);
                        _w5 = _mm_load1_ps(k0 + 5);
                        _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                        _w6 = _mm_load1_ps(k0 + 6);
                        _w7 = _mm_load1_ps(k0 + 7);
                        _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 4;
                        k0 += 8;
                    }
                    for (; j < nn; j++)
                    {
                        __m128 _val = _mm_loadu_ps(r0);

                        __m128 _w0 = _mm_load1_ps(k0);
                        __m128 _w1 = _mm_load1_ps(k0 + 1);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        __m128 _w2 = _mm_load1_ps(k0 + 2);
                        __m128 _w3 = _mm_load1_ps(k0 + 3);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);
                        __m128 _w4 = _mm_load1_ps(k0 + 4);
                        __m128 _w5 = _mm_load1_ps(k0 + 5);
                        _sum4 = _mm_comp_fmadd_ps(_val, _w4, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val, _w5, _sum5);
                        __m128 _w6 = _mm_load1_ps(k0 + 6);
                        __m128 _w7 = _mm_load1_ps(k0 + 7);
                        _sum6 = _mm_comp_fmadd_ps(_val, _w6, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val, _w7, _sum7);

                        r0 += 4;
                        k0 += 8;
                    }

                    _mm_storeu_ps(output0_tm, _sum0);
                    _mm_storeu_ps(output1_tm, _sum1);
                    _mm_storeu_ps(output2_tm, _sum2);
                    _mm_storeu_ps(output3_tm, _sum3);
                    _mm_storeu_ps(output4_tm, _sum4);
                    _mm_storeu_ps(output5_tm, _sum5);
                    _mm_storeu_ps(output6_tm, _sum6);
                    _mm_storeu_ps(output7_tm, _sum7);

                    output0_tm += 4;
                    output1_tm += 4;
                    output2_tm += 4;
                    output3_tm += 4;
                    output4_tm += 4;
                    output5_tm += 4;
                    output6_tm += 4;
                    output7_tm += 4;
                }
                for (; i < tiles; i++)
                {
    #if __AVX__
                    const float* r0 = bb2[i / 8 + (i % 8) / 4 + i % 4].data();
    #else
                    const float* r0 = bb2[i / 4 + i % 4].data();
    #endif
                    const float* k0 = kernel0_tm[r].data();

                    int nn = inch; // inch always > 0

    #if __AVX__
                    __m256 _sum = _mm256_setzero_ps();
    #else
                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
    #endif

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
    #if __AVX__
                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        __m256 _w0 = _mm256_loadu_ps(k0);
                        _sum = _mm256_comp_fmadd_ps(_val0, _w0, _sum);

                        __m256 _val1 = _mm256_broadcast_ss(r0 + 1);
                        __m256 _w1 = _mm256_loadu_ps(k0 + 8);
                        _sum = _mm256_comp_fmadd_ps(_val1, _w1, _sum);

                        __m256 _val2 = _mm256_broadcast_ss(r0 + 2);
                        __m256 _w2 = _mm256_loadu_ps(k0 + 16);
                        _sum = _mm256_comp_fmadd_ps(_val2, _w2, _sum);

                        __m256 _val3 = _mm256_broadcast_ss(r0 + 3);
                        __m256 _w3 = _mm256_loadu_ps(k0 + 24);
                        _sum = _mm256_comp_fmadd_ps(_val3, _w3, _sum);
    #else
                        __m128 _val0 = _mm_load1_ps(r0);
                        __m128 _w00 = _mm_loadu_ps(k0);
                        __m128 _w01 = _mm_loadu_ps(k0 + 4);
                        _sum0 = _mm_comp_fmadd_ps(_val0, _w00, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val0, _w01, _sum1);

                        __m128 _val1 = _mm_load1_ps(r0 + 1);
                        __m128 _w10 = _mm_loadu_ps(k0 + 8);
                        __m128 _w11 = _mm_loadu_ps(k0 + 12);
                        _sum0 = _mm_comp_fmadd_ps(_val1, _w10, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val1, _w11, _sum1);

                        __m128 _val2 = _mm_load1_ps(r0 + 2);
                        __m128 _w20 = _mm_loadu_ps(k0 + 16);
                        __m128 _w21 = _mm_loadu_ps(k0 + 20);
                        _sum0 = _mm_comp_fmadd_ps(_val2, _w20, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val2, _w21, _sum1);

                        __m128 _val3 = _mm_load1_ps(r0 + 3);
                        __m128 _w30 = _mm_loadu_ps(k0 + 24);
                        __m128 _w31 = _mm_loadu_ps(k0 + 28);
                        _sum0 = _mm_comp_fmadd_ps(_val3, _w30, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val3, _w31, _sum1);
    #endif
                        r0 += 4;
                        k0 += 32;
                    }
                    for (; j < nn; j++)
                    {
    #if __AVX__
                        __m256 _val = _mm256_broadcast_ss(r0);
                        __m256 _w = _mm256_loadu_ps(k0);
                        _sum = _mm256_comp_fmadd_ps(_val, _w, _sum);
    #else
                        __m128 _val = _mm_load1_ps(r0);
                        __m128 _w0 = _mm_loadu_ps(k0);
                        __m128 _w1 = _mm_loadu_ps(k0 + 4);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
    #endif
                        r0 += 1;
                        k0 += 8;
                    }

                    float sum[8];
    #if __AVX__
                    _mm256_storeu_ps(sum, _sum);
    #else
                    _mm_storeu_ps(sum, _sum0);
                    _mm_storeu_ps(sum + 4, _sum1);
    #endif

                    output0_tm[0] = sum[0];
                    output1_tm[0] = sum[1];
                    output2_tm[0] = sum[2];
                    output3_tm[0] = sum[3];
                    output4_tm[0] = sum[4];
                    output5_tm[0] = sum[5];
                    output6_tm[0] = sum[6];
                    output7_tm[0] = sum[7];

                    output0_tm++;
                    output1_tm++;
                    output2_tm++;
                    output3_tm++;
                    output4_tm++;
                    output5_tm++;
                    output6_tm++;
                    output7_tm++;
                }
            }
        }
    });

    nn_outch = (outch - remain_outch_start) >> 2;

    otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto pp : otter::irange(begin, end))
        {
            int p = remain_outch_start + pp * 4;

            float* output0_tm = top_blob_tm_a[p + 0].data();
            float* output1_tm = top_blob_tm_a[p + 1].data();
            float* output2_tm = top_blob_tm_a[p + 2].data();
            float* output3_tm = top_blob_tm_a[p + 3].data();

            const auto kernel0_tm = kernel_tm_a[p / 8 + (p % 8) / 4];

            for (int r = 0; r < batch; r++)
            {
                const auto bb2 = bottom_blob_tm2_a[r];

                int i = 0;
    #if __AVX__
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2[i / 8].data();
                    const float* k0 = kernel0_tm[r].data();

                    int nn = inch; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m256 _val = _mm256_loadu_ps(r0);
                        __m256 _w0 = _mm256_broadcast_ss(k0);
                        __m256 _w1 = _mm256_broadcast_ss(k0 + 1);
                        __m256 _w2 = _mm256_broadcast_ss(k0 + 2);
                        __m256 _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 8;
                        k0 += 4;

                        _val = _mm256_loadu_ps(r0);
                        _w0 = _mm256_broadcast_ss(k0);
                        _w1 = _mm256_broadcast_ss(k0 + 1);
                        _w2 = _mm256_broadcast_ss(k0 + 2);
                        _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 8;
                        k0 += 4;

                        _val = _mm256_loadu_ps(r0);
                        _w0 = _mm256_broadcast_ss(k0);
                        _w1 = _mm256_broadcast_ss(k0 + 1);
                        _w2 = _mm256_broadcast_ss(k0 + 2);
                        _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 8;
                        k0 += 4;

                        _val = _mm256_loadu_ps(r0);
                        _w0 = _mm256_broadcast_ss(k0);
                        _w1 = _mm256_broadcast_ss(k0 + 1);
                        _w2 = _mm256_broadcast_ss(k0 + 2);
                        _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 8;
                        k0 += 4;
                    }
                    for (; j < nn; j++)
                    {
                        __m256 _val = _mm256_loadu_ps(r0);
                        __m256 _w0 = _mm256_broadcast_ss(k0);
                        __m256 _w1 = _mm256_broadcast_ss(k0 + 1);
                        __m256 _w2 = _mm256_broadcast_ss(k0 + 2);
                        __m256 _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm256_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 8;
                        k0 += 4;
                    }

                    _mm256_storeu_ps(output0_tm, _sum0);
                    _mm256_storeu_ps(output1_tm, _sum1);
                    _mm256_storeu_ps(output2_tm, _sum2);
                    _mm256_storeu_ps(output3_tm, _sum3);

                    output0_tm += 8;
                    output1_tm += 8;
                    output2_tm += 8;
                    output3_tm += 8;
                }
    #endif // __AVX__
                for (; i + 3 < tiles; i += 4)
                {
    #if __AVX__
                    const float* r0 = bb2[i / 8 + (i % 8) / 4].data();
    #else
                    const float* r0 = bb2[i / 4].data();
    #endif
                    const float* k0 = kernel0_tm[r].data();

                    int nn = inch; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m128 _val = _mm_loadu_ps(r0);
                        __m128 _w0 = _mm_load1_ps(k0);
                        __m128 _w1 = _mm_load1_ps(k0 + 1);
                        __m128 _w2 = _mm_load1_ps(k0 + 2);
                        __m128 _w3 = _mm_load1_ps(k0 + 3);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 4;
                        k0 += 4;

                        _val = _mm_loadu_ps(r0);
                        _w0 = _mm_load1_ps(k0);
                        _w1 = _mm_load1_ps(k0 + 1);
                        _w2 = _mm_load1_ps(k0 + 2);
                        _w3 = _mm_load1_ps(k0 + 3);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 4;
                        k0 += 4;

                        _val = _mm_loadu_ps(r0);
                        _w0 = _mm_load1_ps(k0);
                        _w1 = _mm_load1_ps(k0 + 1);
                        _w2 = _mm_load1_ps(k0 + 2);
                        _w3 = _mm_load1_ps(k0 + 3);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 4;
                        k0 += 4;

                        _val = _mm_loadu_ps(r0);
                        _w0 = _mm_load1_ps(k0);
                        _w1 = _mm_load1_ps(k0 + 1);
                        _w2 = _mm_load1_ps(k0 + 2);
                        _w3 = _mm_load1_ps(k0 + 3);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 4;
                        k0 += 4;
                    }
                    for (; j < nn; j++)
                    {
                        __m128 _val = _mm_loadu_ps(r0);
                        __m128 _w0 = _mm_load1_ps(k0);
                        __m128 _w1 = _mm_load1_ps(k0 + 1);
                        __m128 _w2 = _mm_load1_ps(k0 + 2);
                        __m128 _w3 = _mm_load1_ps(k0 + 3);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val, _w1, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val, _w2, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val, _w3, _sum3);

                        r0 += 4;
                        k0 += 4;
                    }

                    _mm_storeu_ps(output0_tm, _sum0);
                    _mm_storeu_ps(output1_tm, _sum1);
                    _mm_storeu_ps(output2_tm, _sum2);
                    _mm_storeu_ps(output3_tm, _sum3);

                    output0_tm += 4;
                    output1_tm += 4;
                    output2_tm += 4;
                    output3_tm += 4;
                }
                for (; i < tiles; i++)
                {
    #if __AVX__
                    const float* r0 = bb2[i / 8 + (i % 8) / 4 + i % 4].data();
    #else
                    const float* r0 = bb2[i / 4 + i % 4].data();
    #endif
                    const float* k0 = kernel0_tm[r].data();

                    int nn = inch; // inch always > 0

                    __m128 _sum = _mm_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m128 _val0 = _mm_load1_ps(r0);
                        __m128 _w0 = _mm_loadu_ps(k0);
                        _sum = _mm_comp_fmadd_ps(_val0, _w0, _sum);

                        __m128 _val1 = _mm_load1_ps(r0 + 1);
                        __m128 _w1 = _mm_loadu_ps(k0 + 4);
                        _sum = _mm_comp_fmadd_ps(_val1, _w1, _sum);

                        __m128 _val2 = _mm_load1_ps(r0 + 2);
                        __m128 _w2 = _mm_loadu_ps(k0 + 8);
                        _sum = _mm_comp_fmadd_ps(_val2, _w2, _sum);

                        __m128 _val3 = _mm_load1_ps(r0 + 3);
                        __m128 _w3 = _mm_loadu_ps(k0 + 12);
                        _sum = _mm_comp_fmadd_ps(_val3, _w3, _sum);

                        r0 += 4;
                        k0 += 16;
                    }
                    for (; j < nn; j++)
                    {
                        __m128 _val = _mm_load1_ps(r0);
                        __m128 _w0 = _mm_loadu_ps(k0);
                        _sum = _mm_comp_fmadd_ps(_val, _w0, _sum);

                        r0 += 1;
                        k0 += 4;
                    }

                    float sum[4];
                    _mm_storeu_ps(sum, _sum);

                    output0_tm[0] = sum[0];
                    output1_tm[0] = sum[1];
                    output2_tm[0] = sum[2];
                    output3_tm[0] = sum[3];

                    output0_tm++;
                    output1_tm++;
                    output2_tm++;
                    output3_tm++;
                }
            }
        }
    });

    remain_outch_start += nn_outch << 2;
#else
    int remain_outch_start = 0;
#endif

    otter::parallel_for(remain_outch_start, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            float* output0_tm = top_blob_tm_a[p].data();

    #if __SSE2__
            const auto kernel0_tm = kernel_tm_a[p / 8 + (p % 8) / 4 + p % 4];
    #else
            const auto kernel0_tm = kernel_tm_a[p];
    #endif

            for (int r = 0; r < batch; r++)
            {
                const auto bb2 = bottom_blob_tm2_a[r];

                int i = 0;
    #if __SSE2__
    #if __AVX__
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2[i / 8].data();
                    const float* k0 = kernel0_tm[r].data();

                    int nn = inch; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m256 _val0 = _mm256_loadu_ps(r0);
                        __m256 _w0 = _mm256_broadcast_ss(k0);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);

                        __m256 _val1 = _mm256_loadu_ps(r0 + 8);
                        __m256 _w1 = _mm256_broadcast_ss(k0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val1, _w1, _sum0);

                        __m256 _val2 = _mm256_loadu_ps(r0 + 16);
                        __m256 _w2 = _mm256_broadcast_ss(k0 + 2);
                        _sum0 = _mm256_comp_fmadd_ps(_val2, _w2, _sum0);

                        __m256 _val3 = _mm256_loadu_ps(r0 + 24);
                        __m256 _w3 = _mm256_broadcast_ss(k0 + 3);
                        _sum0 = _mm256_comp_fmadd_ps(_val3, _w3, _sum0);

                        r0 += 32;
                        k0 += 4;
                    }
                    for (; j < nn; j++)
                    {
                        __m256 _val = _mm256_loadu_ps(r0);
                        __m256 _w0 = _mm256_broadcast_ss(k0);
                        _sum0 = _mm256_comp_fmadd_ps(_val, _w0, _sum0);
                        r0 += 8;
                        k0++;
                    }

                    _mm256_storeu_ps(output0_tm, _sum0);
                    output0_tm += 8;
                }
    #endif // __AVX__
                for (; i + 3 < tiles; i += 4)
                {
    #if __AVX__
                    const float* r0 = bb2[i / 8 + (i % 8) / 4].data();
    #else
                    const float* r0 = bb2[i / 4].data();
    #endif
                    const float* k0 = kernel0_tm[r].data();

                    int nn = inch; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();

                    int j = 0;
                    for (; j + 3 < nn; j += 4)
                    {
                        __m128 _val0 = _mm_loadu_ps(r0);
                        __m128 _w0 = _mm_load1_ps(k0);
                        _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);

                        __m128 _val1 = _mm_loadu_ps(r0 + 4);
                        __m128 _w1 = _mm_load1_ps(k0 + 1);
                        _sum0 = _mm_comp_fmadd_ps(_val1, _w1, _sum0);

                        __m128 _val2 = _mm_loadu_ps(r0 + 8);
                        __m128 _w2 = _mm_load1_ps(k0 + 2);
                        _sum0 = _mm_comp_fmadd_ps(_val2, _w2, _sum0);

                        __m128 _val3 = _mm_loadu_ps(r0 + 12);
                        __m128 _w3 = _mm_load1_ps(k0 + 3);
                        _sum0 = _mm_comp_fmadd_ps(_val3, _w3, _sum0);

                        r0 += 16;
                        k0 += 4;
                    }
                    for (; j < nn; j++)
                    {
                        __m128 _val = _mm_loadu_ps(r0);
                        __m128 _w0 = _mm_load1_ps(k0);
                        _sum0 = _mm_comp_fmadd_ps(_val, _w0, _sum0);
                        r0 += 4;
                        k0++;
                    }

                    _mm_storeu_ps(output0_tm, _sum0);
                    output0_tm += 4;
                }
    #endif // __SSE2__
                for (; i < tiles; i++)
                {
    #if __AVX__
                    const float* r0 = bb2[i / 8 + (i % 8) / 4 + i % 4].data();
    #elif __SSE2__
                    const float* r0 = bb2[i / 4 + i % 4].data();
    #else
                    const float* r0 = bb2[i].data();
    #endif
                    const float* k0 = kernel0_tm[r].data();

                    int nn = inch; // inch always > 0

                    float sum = 0.f;

                    for (int j = 0; j < nn; j++)
                    {
                        float w0 = k0[0];
                        float val0 = r0[0];
                        sum += val0 * w0;

                        r0 += 1;
                        k0 += 1;
                    }

                    output0_tm[0] = sum;
                    output0_tm += 1;
                }
            }
        }
    });
}

void conv3x3s1_winograd23_transform_kernel_sse(const Tensor& kernel, Tensor& kernel_tm2, int inch, int outch)
{
    Tensor kernel_tm = otter::empty({outch, inch, 4 * 4}, otter::ScalarType::Float);

    // G
    const float ktm[4][3] = {
        {1.0f, 0.0f, 0.0f},
        {1.0f / 2, 1.0f / 2, 1.0f / 2},
        {1.0f / 2, -1.0f / 2, 1.0f / 2},
        {0.0f, 0.0f, 1.0f}
    };
    
    const float* kernel_ptr = (const float*)kernel.data_ptr();
    auto kernel_tm_a = kernel_tm.accessor<float, 3>();

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            for (int q = 0; q < inch; q++) {
                const float* kernel0 = (const float*)kernel_ptr + p * inch * 9 + q * 9;
                float* kernel_tm0 = kernel_tm_a[p][q].data();

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

    // interleave
    // src = 16-inch-outch
    // dst = inch-16-outch
#if __SSE2__
    kernel_tm2 = otter::empty({outch / 8 + (outch % 8) / 4 + outch % 4, 16, 8 * inch}, otter::ScalarType::Float);
#else
    kernel_tm2 = otter::empty({outch, 16m inch}, otter::ScalarType::Float);
#endif
    
    auto kernel_tm2_a = kernel_tm2.accessor<float, 3>();

    int q = 0;
#if __SSE2__
    for (; q + 7 < outch; q += 8)
    {
        auto g0 = kernel_tm2_a[q / 8];

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0[k].data();

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 8; i++)
                {
                    const float* k00 = kernel_tm_a[q + i][p].data();
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        auto g0 = kernel_tm2_a[q / 8 + (q % 8) / 4];

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0[k].data();

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 4; i++)
                {
                    const float* k00 = kernel_tm_a[q + i][p].data();
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#endif
    for (; q < outch; q++)
    {
#if __SSE2__
        auto g0 = kernel_tm2_a[q / 8 + (q % 8) / 4 + q % 4];
#else
        auto g0 = kernel_tm2_a[q];
#endif

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0[k].data();

            for (int p = 0; p < inch; p++)
            {
                const float* k00 = kernel_tm_a[q][p].data();
                g00[0] = k00[k];
                g00++;
            }
        }
    }
}

void conv3x3s1_winograd43_transform_kernel_sse(const Tensor& kernel, Tensor& kernel_tm2, int inch, int outch) {
    Tensor kernel_tm = otter::empty({outch, inch, 6 * 6}, otter::ScalarType::Float);

    // G
    const float ktm[6][3] = {
        {1.0f / 4, 0.0f, 0.0f},
        {-1.0f / 6, -1.0f / 6, -1.0f / 6},
        {-1.0f / 6, 1.0f / 6, -1.0f / 6},
        {1.0f / 24, 1.0f / 12, 1.0f / 6},
        {1.0f / 24, -1.0f / 12, 1.0f / 6},
        {0.0f, 0.0f, 1.0f}
    };
    
    const float* kernel_ptr = (const float*)kernel.data_ptr();
    auto kernel_tm_a = kernel_tm.accessor<float, 3>();

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            for (int q = 0; q < inch; q++)
            {
                const float* kernel0 = (const float*)kernel_ptr + p * inch * 9 + q * 9;
                float* kernel_tm0 = kernel_tm_a[p][q].data();

                // transform kernel
                const float* k0 = kernel0;
                const float* k1 = kernel0 + 3;
                const float* k2 = kernel0 + 6;

                // h
                float tmp[6][3];
                for (int i = 0; i < 6; i++)
                {
                    tmp[i][0] = k0[0] * ktm[i][0] + k0[1] * ktm[i][1] + k0[2] * ktm[i][2];
                    tmp[i][1] = k1[0] * ktm[i][0] + k1[1] * ktm[i][1] + k1[2] * ktm[i][2];
                    tmp[i][2] = k2[0] * ktm[i][0] + k2[1] * ktm[i][1] + k2[2] * ktm[i][2];
                }

                // U
                for (int j = 0; j < 6; j++)
                {
                    float* tmpp = &tmp[j][0];

                    for (int i = 0; i < 6; i++)
                    {
                        kernel_tm0[j * 6 + i] = tmpp[0] * ktm[i][0] + tmpp[1] * ktm[i][1] + tmpp[2] * ktm[i][2];
                    }
                }
            }
        }
    });

    // interleave
    // src = 36-inch-outch
    // dst = inch-36-outch
#if __SSE2__
    kernel_tm2 = otter::empty({outch / 8 + (outch % 8) / 4 + outch % 4, 36, 8 * inch}, otter::ScalarType::Float);
#else
    kernel_tm2 = otter::empty({outch, 36, inch}, otter::ScalarType::Float);
#endif
    
    auto kernel_tm2_a = kernel_tm2.accessor<float, 3>();

    int q = 0;
#if __SSE2__
    for (; q + 7 < outch; q += 8)
    {
        auto g0 = kernel_tm2_a[q / 8];

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0[k].data();

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 8; i++)
                {
                    const float* k00 = kernel_tm_a[q + i][p].data();
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        auto g0 = kernel_tm2_a[q / 8 + (q % 8) / 4];

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0[k].data();

            for (int p = 0; p < inch; p++)
            {
                for (int i = 0; i < 4; i++)
                {
                    const float* k00 = kernel_tm_a[q + i][p].data();
                    g00[0] = k00[k];
                    g00++;
                }
            }
        }
    }
#endif
    for (; q < outch; q++)
    {
#if __SSE2__
        auto g0 = kernel_tm2_a[q / 8 + (q % 8) / 4 + q % 4];
#else
        auto g0 = kernel_tm2_a[q];
#endif

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0[k].data();

            for (int p = 0; p < inch; p++)
            {
                const float* k00 = kernel_tm_a[q][p].data();
                g00[0] = k00[k];
                g00++;
            }
        }
    }
}

static void conv3x3s1_winograd43_transform_input_sse(const Tensor& bottom_blob, Tensor& bottom_blob_tm)
{
    const int w = bottom_blob.size(2);
    const int h = bottom_blob.size(1);
    const int inch = bottom_blob.size(0);

    const int w_tiles = (w - 2) / 4;
    const int h_tiles = (h - 2) / 4;
    const int tiles = w_tiles * h_tiles;

    // const float itm[6][6] = {
    //     {4.0f, 0.0f, -5.0f, 0.0f, 1.0f, 0.0f},
    //     {0.0f,-4.0f, -4.0f, 1.0f, 1.0f, 0.0f},
    //     {0.0f, 4.0f, -4.0f,-1.0f, 1.0f, 0.0f},
    //     {0.0f,-2.0f, -1.0f, 2.0f, 1.0f, 0.0f},
    //     {0.0f, 2.0f, -1.0f,-2.0f, 1.0f, 0.0f},
    //     {0.0f, 4.0f,  0.0f,-5.0f, 0.0f, 1.0f}
    // };

    // 0 =  4 * r00 - 5 * r02 + r04
    // 1 = -4 * (r01 + r02) + r04 + r03
    // 2 =  4 * (r01 - r02) + r04 - r03
    // 3 = -2 * (r01 - r03) + r04 - r02
    // 4 =  2 * (r01 - r03) + r04 - r02
    // 5 =  4 * r01 - 5 * r03 + r05
    
    auto bottom_blob_a = bottom_blob.accessor<float, 3>();
    auto bottom_blob_tm_a = bottom_blob_tm.accessor<float, 3>();

    otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
        for (const auto q : otter::irange(begin, end)) {
            const auto img0 = bottom_blob_a[q];
            auto img0_tm = bottom_blob_tm_a[q];

            float tmp[6][6];

            // tile
            for (int i = 0; i < h_tiles; i++)
            {
                for (int j = 0; j < w_tiles; j++)
                {
                    const float* r0 = img0[i * 4].data() + (j * 4);

                    for (int m = 0; m < 6; m++)
                    {
                        float r00 = r0[0];
                        float r01 = r0[1];
                        float r02 = r0[2];
                        float r03 = r0[3];
                        float r04 = r0[4];
                        float r05 = r0[5];

                        float tmp0m = 4 * r00 - 5 * r02 + r04;
                        float tmp1m = -4 * (r01 + r02) + r04 + r03;
                        float tmp2m = 4 * (r01 - r02) + r04 - r03;
                        float tmp3m = -2 * (r01 - r03) + r04 - r02;
                        float tmp4m = 2 * (r01 - r03) + r04 - r02;
                        float tmp5m = 4 * r01 - 5 * r03 + r05;

                        tmp[0][m] = tmp0m;
                        tmp[1][m] = tmp1m;
                        tmp[2][m] = tmp2m;
                        tmp[3][m] = tmp3m;
                        tmp[4][m] = tmp4m;
                        tmp[5][m] = tmp5m;

                        r0 += w;
                    }

                    float* r0_tm_0 = (float*)img0_tm.data() + (i * w_tiles + j);
                    float* r0_tm_1 = r0_tm_0 + tiles;
                    float* r0_tm_2 = r0_tm_0 + tiles * 2;
                    float* r0_tm_3 = r0_tm_0 + tiles * 3;
                    float* r0_tm_4 = r0_tm_0 + tiles * 4;
                    float* r0_tm_5 = r0_tm_0 + tiles * 5;

                    for (int m = 0; m < 6; m++)
                    {
                        float tmp00 = tmp[m][0];
                        float tmp01 = tmp[m][1];
                        float tmp02 = tmp[m][2];
                        float tmp03 = tmp[m][3];
                        float tmp04 = tmp[m][4];
                        float tmp05 = tmp[m][5];

                        float r0tm0 = 4 * tmp00 - 5 * tmp02 + tmp04;
                        float r0tm1 = -4 * (tmp01 + tmp02) + tmp04 + tmp03;
                        float r0tm2 = 4 * (tmp01 - tmp02) + tmp04 - tmp03;
                        float r0tm3 = -2 * (tmp01 - tmp03) + tmp04 - tmp02;
                        float r0tm4 = 2 * (tmp01 - tmp03) + tmp04 - tmp02;
                        float r0tm5 = 4 * tmp01 - 5 * tmp03 + tmp05;

                        r0_tm_0[0] = r0tm0;
                        r0_tm_1[0] = r0tm1;
                        r0_tm_2[0] = r0tm2;
                        r0_tm_3[0] = r0tm3;
                        r0_tm_4[0] = r0tm4;
                        r0_tm_5[0] = r0tm5;

                        r0_tm_0 += tiles * 6;
                        r0_tm_1 += tiles * 6;
                        r0_tm_2 += tiles * 6;
                        r0_tm_3 += tiles * 6;
                        r0_tm_4 += tiles * 6;
                        r0_tm_5 += tiles * 6;
                    }
                }
            }
        }
    });
}

static void conv3x3s1_winograd43_transform_output_sse(const Tensor& top_blob_tm, Tensor& top_blob, const Tensor& bias)
{
    const int outw = top_blob.size(2);
    const int outh = top_blob.size(1);
    const int outch = top_blob.size(0);

    const int w_tiles = outw / 4;
    const int h_tiles = outh / 4;
    const int tiles = w_tiles * h_tiles;

    const float* biasptr = bias.data_ptr<float>();

    // const float otm[4][6] = {
    //     {1.0f, 1.0f,  1.0f, 1.0f,  1.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.0f},
    //     {0.0f, 1.0f,  1.0f, 4.0f,  4.0f, 0.0f},
    //     {0.0f, 1.0f, -1.0f, 8.0f, -8.0f, 1.0f}
    // };

    // 0 = r00 + (r01 + r02) + (r03 + r04)
    // 1 =       (r01 - r02) + (r03 - r04) * 2
    // 2 =       (r01 + r02) + (r03 + r04) * 4
    // 3 = r05 + (r01 - r02) + (r03 - r04) * 8
    
    auto top_blob_a = top_blob.accessor<float, 3>();
    auto top_blob_tm_a = top_blob_tm.accessor<float, 3>();

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end))
        {
            const auto out0_tm = top_blob_tm_a[p];
            auto out0 = top_blob_a[p];

            float bias0 = biasptr ? biasptr[p] : 0.f;

            float tmp[4][6];

            // tile
            for (int i = 0; i < h_tiles; i++)
            {
                for (int j = 0; j < w_tiles; j++)
                {
                    const float* output0_tm_0 = (const float*)out0_tm.data() + (i * w_tiles + j);
                    const float* output0_tm_1 = output0_tm_0 + tiles;
                    const float* output0_tm_2 = output0_tm_0 + tiles * 2;
                    const float* output0_tm_3 = output0_tm_0 + tiles * 3;
                    const float* output0_tm_4 = output0_tm_0 + tiles * 4;
                    const float* output0_tm_5 = output0_tm_0 + tiles * 5;

                    float* output0 = out0[i * 4].data() + (j * 4);

                    for (int m = 0; m < 6; m++)
                    {
                        float out0tm0 = output0_tm_0[0];
                        float out0tm1 = output0_tm_1[0];
                        float out0tm2 = output0_tm_2[0];
                        float out0tm3 = output0_tm_3[0];
                        float out0tm4 = output0_tm_4[0];
                        float out0tm5 = output0_tm_5[0];

                        float tmp02a = out0tm1 + out0tm2;
                        float tmp13a = out0tm1 - out0tm2;

                        float tmp02b = out0tm3 + out0tm4;
                        float tmp13b = out0tm3 - out0tm4;

                        float tmp0m = out0tm0 + tmp02a + tmp02b;
                        float tmp1m = tmp13a + tmp13b * 2;
                        float tmp2m = tmp02a + tmp02b * 4;
                        float tmp3m = out0tm5 + tmp13a + tmp13b * 8;

                        tmp[0][m] = tmp0m;
                        tmp[1][m] = tmp1m;
                        tmp[2][m] = tmp2m;
                        tmp[3][m] = tmp3m;

                        output0_tm_0 += tiles * 6;
                        output0_tm_1 += tiles * 6;
                        output0_tm_2 += tiles * 6;
                        output0_tm_3 += tiles * 6;
                        output0_tm_4 += tiles * 6;
                        output0_tm_5 += tiles * 6;
                    }

                    for (int m = 0; m < 4; m++)
                    {
                        float tmp00 = tmp[m][0];
                        float tmp01 = tmp[m][1];
                        float tmp02 = tmp[m][2];
                        float tmp03 = tmp[m][3];
                        float tmp04 = tmp[m][4];
                        float tmp05 = tmp[m][5];

                        float tmp02a = tmp01 + tmp02;
                        float tmp13a = tmp01 - tmp02;

                        float tmp02b = tmp03 + tmp04;
                        float tmp13b = tmp03 - tmp04;

                        float out00 = bias0 + tmp00 + tmp02a + tmp02b;
                        float out01 = bias0 + tmp13a + tmp13b * 2;
                        float out02 = bias0 + tmp02a + tmp02b * 4;
                        float out03 = bias0 + tmp05 + tmp13a + tmp13b * 8;

                        output0[0] = out00;
                        output0[1] = out01;
                        output0[2] = out02;
                        output0[3] = out03;

                        output0 += outw;
                    }
                }
            }
        }
    });
}

static void conv3x3s1_winograd23_transform_input_sse(const Tensor& bottom_blob, Tensor& bottom_blob_tm)
{
    const int w = bottom_blob.size(2);
    const int h = bottom_blob.size(1);
    const int inch = bottom_blob.size(0);

    const int w_tiles = (w - 2) / 2;
    const int h_tiles = (h - 2) / 2;
    const int tiles = w_tiles * h_tiles;

    // const float itm[4][4] = {
    //     {1.0f,  0.0f, -1.0f,  0.0f},
    //     {0.0f,  1.0f,  1.00f, 0.0f},
    //     {0.0f, -1.0f,  1.00f, 0.0f},
    //     {0.0f, -1.0f,  0.00f, 1.0f}
    // };

    // 0 = r00 - r02
    // 1 = r01 + r02
    // 2 = r02 - r01
    // 3 = r03 - r01
    
    auto bottom_blob_a = bottom_blob.accessor<float, 3>();
    auto bottom_blob_tm_a = bottom_blob_tm.accessor<float, 3>();

    otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
        for (const auto q : otter::irange(begin, end)) {
            const auto img0 = bottom_blob_a[q];
            auto img0_tm = bottom_blob_tm_a[q];

            float tmp[4][4];

            // tile
            for (int i = 0; i < h_tiles; i++)
            {
                for (int j = 0; j < w_tiles; j++)
                {
                    const float* r0 = img0[i * 2].data() + (j * 2);

                    for (int m = 0; m < 4; m++)
                    {
                        float r00 = r0[0];
                        float r01 = r0[1];
                        float r02 = r0[2];
                        float r03 = r0[3];

                        float tmp0m = r00 - r02;
                        float tmp1m = r01 + r02;
                        float tmp2m = r02 - r01;
                        float tmp3m = r03 - r01;

                        tmp[0][m] = tmp0m;
                        tmp[1][m] = tmp1m;
                        tmp[2][m] = tmp2m;
                        tmp[3][m] = tmp3m;

                        r0 += w;
                    }

                    float* r0_tm_0 = (float*)img0_tm.data() + (i * w_tiles + j);
                    float* r0_tm_1 = r0_tm_0 + tiles;
                    float* r0_tm_2 = r0_tm_0 + tiles * 2;
                    float* r0_tm_3 = r0_tm_0 + tiles * 3;

                    for (int m = 0; m < 4; m++)
                    {
                        float tmp00 = tmp[m][0];
                        float tmp01 = tmp[m][1];
                        float tmp02 = tmp[m][2];
                        float tmp03 = tmp[m][3];

                        float r0tm0 = tmp00 - tmp02;
                        float r0tm1 = tmp01 + tmp02;
                        float r0tm2 = tmp02 - tmp01;
                        float r0tm3 = tmp03 - tmp01;

                        r0_tm_0[0] = r0tm0;
                        r0_tm_1[0] = r0tm1;
                        r0_tm_2[0] = r0tm2;
                        r0_tm_3[0] = r0tm3;

                        r0_tm_0 += tiles * 4;
                        r0_tm_1 += tiles * 4;
                        r0_tm_2 += tiles * 4;
                        r0_tm_3 += tiles * 4;
                    }
                }
            }
        }
    });
}

static void conv3x3s1_winograd23_transform_output_sse(const Tensor& top_blob_tm, Tensor& top_blob, const Tensor& bias)
{
    const int outw = top_blob.size(2);
    const int outh = top_blob.size(1);
    const int outch = top_blob.size(0);

    const int w_tiles = outw / 2;
    const int h_tiles = outh / 2;
    const int tiles = w_tiles * h_tiles;

    const float* biasptr = bias.data_ptr<float>();

    // const float otm[2][4] = {
    //     {1.0f,  1.0f,  1.0f,  0.0f},
    //     {0.0f,  1.0f, -1.0f,  1.0f}
    // };

    // 0 = r00 + r01 + r02
    // 1 = r01 - r02 + r03
    
    auto top_blob_a = top_blob.accessor<float, 3>();
    auto top_blob_tm_a = top_blob_tm.accessor<float, 3>();

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end))
        {
            const auto out0_tm = top_blob_tm_a[p];
            auto out0 = top_blob_a[p];

            float bias0 = biasptr ? biasptr[p] : 0.f;

            float tmp[2][4];

            // tile
            for (int i = 0; i < h_tiles; i++)
            {
                for (int j = 0; j < w_tiles; j++)
                {
                    const float* output0_tm_0 = (const float*)out0_tm.data() + (i * w_tiles + j);
                    const float* output0_tm_1 = output0_tm_0 + tiles;
                    const float* output0_tm_2 = output0_tm_0 + tiles * 2;
                    const float* output0_tm_3 = output0_tm_0 + tiles * 3;

                    float* output0 = out0[i * 2].data() + (j * 2);

                    for (int m = 0; m < 4; m++)
                    {
                        float out0tm0 = output0_tm_0[0];
                        float out0tm1 = output0_tm_1[0];
                        float out0tm2 = output0_tm_2[0];
                        float out0tm3 = output0_tm_3[0];

                        float tmp0m = out0tm0 + out0tm1 + out0tm2;
                        float tmp1m = out0tm1 - out0tm2 + out0tm3;

                        tmp[0][m] = tmp0m;
                        tmp[1][m] = tmp1m;

                        output0_tm_0 += tiles * 4;
                        output0_tm_1 += tiles * 4;
                        output0_tm_2 += tiles * 4;
                        output0_tm_3 += tiles * 4;
                    }

                    for (int m = 0; m < 2; m++)
                    {
                        float tmp00 = tmp[m][0];
                        float tmp01 = tmp[m][1];
                        float tmp02 = tmp[m][2];
                        float tmp03 = tmp[m][3];

                        float out00 = bias0 + tmp00 + tmp01 + tmp02;
                        float out01 = bias0 + tmp01 - tmp02 + tmp03;

                        output0[0] = out00;
                        output0[1] = out01;

                        output0 += outw;
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
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_shape = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
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
        otter::conv3x3s1_winograd23_transform_kernel_sse(weight, kernel_tf, inch, outch);
    
    // BEGIN transform input
    Tensor bottom_blob_tm;
    {
        int w_tiles = outw / 2;
        int h_tiles = outh / 2;
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm = otter::empty({inch, 16, tiles}, otter::ScalarType::Float);
        conv3x3s1_winograd23_transform_input_sse(input[0], bottom_blob_tm);
    }
    input.reset();
    // END transform input

    // BEGIN dot
    Tensor top_blob_tm;
    convolution_winograd_dot_sse(bottom_blob_tm, outch, kernel_tf, top_blob_tm);
    // END dot

    // BEGIN transform output
    Tensor top_blob_bordered;
    if (outw == output_shape[3] && outh == output_shape[2]) {
        top_blob_bordered = output;
    } else {
        top_blob_bordered = otter::empty({1, outch, outh, outw}, otter::ScalarType::Float);
    }
    {
        Tensor top_blob_bordered_t = top_blob_bordered[0];
        conv3x3s1_winograd23_transform_output_sse(top_blob_tm, top_blob_bordered_t, bias);
    }
    // END transform output
    
    otter::crop_(top_blob_bordered, {0, top_blob_bordered.size(3) - output_shape[3], 0, top_blob_bordered.size(2) - output_shape[2]}, output);
    
    return output;
}

Tensor conv2d_3x3s1_winograd23_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, self.options());
    
    return conv2d_3x3s1_winograd23_x86_out(self, weight, weight_o, bias, padding, output);
}

Tensor& conv2d_3x3s1_winograd43_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_shape = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
    output.resize_(output_shape);
    
    int origin_w = (int)self.size(3) + 2 * (int)padding[1];
    int origin_h = (int)self.size(2) + 2 * (int)padding[0];
    
    int w = origin_w;
    int h = origin_h;
    int inch  = (int)self.size(1);
    
    int outw  = (int)output_shape[3];
    int outh  = (int)output_shape[2];
    int outch = (int)output_shape[1];
    
    outw = (outw + 3) / 4 * 4;
    outh = (outh + 3) / 4 * 4;

    w = outw + 2;
    h = outh + 2;
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1] + w - origin_w, padding[0], padding[0] + h - origin_h}, 0);
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        otter::conv3x3s1_winograd43_transform_kernel_sse(weight, kernel_tf, inch, outch);
    
    // BEGIN transform input
    Tensor bottom_blob_tm;
    {
        int w_tiles = outw / 4;
        int h_tiles = outh / 4;
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm = otter::empty({inch, 36, tiles}, otter::ScalarType::Float);
        conv3x3s1_winograd43_transform_input_sse(input[0], bottom_blob_tm);
    }
    input.reset();
    // END transform input

    // BEGIN dot
    Tensor top_blob_tm;
    convolution_winograd_dot_sse(bottom_blob_tm, outch, kernel_tf, top_blob_tm);
    // END dot

    // BEGIN transform output
    Tensor top_blob_bordered;
    if (outw == output_shape[3] && outh == output_shape[2]) {
        top_blob_bordered = output;
    } else {
        top_blob_bordered = otter::empty({1, outch, outh, outw}, otter::ScalarType::Float);
    }
    {
        Tensor top_blob_bordered_t = top_blob_bordered[0];
        conv3x3s1_winograd43_transform_output_sse(top_blob_tm, top_blob_bordered_t, bias);
    }
    // END transform output
    
    otter::crop_(top_blob_bordered, {0, top_blob_bordered.size(3) - output_shape[3], 0, top_blob_bordered.size(2) - output_shape[2]}, output);
    
    return output;
}

Tensor conv2d_3x3s1_winograd43_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, self.options());
    
    return conv2d_3x3s1_winograd43_x86_out(self, weight, weight_o, bias, padding, output);
}

}   // end namespace otter
