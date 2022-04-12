//
//  ConvolutionMM2DX86.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/2.
//

#include "ConvolutionMM2DX86.hpp"
#include "Tensor.hpp"
#include "TensorFactory.hpp"
#include "im2col.hpp"
#include "Parallel.hpp"
#include "Padding.hpp"
#include "VecIntrinsic.hpp"

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
    const Tensor& im2col_,
    const Tensor& kernel_packed_,
    const Tensor& bias_,
    int64_t input_channels,
    int64_t output_channels,
    Tensor& output) {
}
#endif

#ifdef __SSE2__
static void convolution_im2col_sgemm_transform_kernel_x86(const Tensor& kernel_, Tensor& kernel_tf, int64_t input_channels, int64_t output_channels, int64_t kernel_width, int64_t kernel_height) {
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
static void convolution_im2col_sgemm_transform_kernel_x86(const Tensor& _kernel, Tensor& kernel_tf, int64_t input_channels, int64_t out_chnnels, int64_t kernel_width, int64_t kernel_height) {
}
#endif

Tensor& sgemm_conv2d_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    if (!output.defined()) {
        auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
        output.resize_(output_size);
    }
    
    const int64_t kernel_height = kernel_size[0];
    const int64_t kernel_width  = kernel_size[1];
    
    const Tensor input = self.contiguous();
    const int64_t input_channels  = input.size(1);
    const int64_t output_channels = weight.size(0);
    
    Tensor im2col = otter::im2col_cpu(input, kernel_size, stride, padding, {1, 1});
    Tensor kernel_packed;
    otter::convolution_im2col_sgemm_transform_kernel_x86(weight, kernel_packed, input_channels, output_channels, kernel_width, kernel_height);
    otter::im2col_sgemm_conv2d_impl_x86(im2col, kernel_packed, bias, input_channels, output_channels, output);
    
    return output;
}
    
Tensor sgemm_conv2d_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    auto output = otter::empty(output_size, self.options());
    
    return sgemm_conv2d_x86_out(self, weight, bias, kernel_size, stride, padding, output);
}

}   // end namespace otter
