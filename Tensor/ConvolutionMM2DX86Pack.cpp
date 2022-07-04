//
//  ConvolutionMM2DX86Pack.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/17.
//

// https://github.com/Tencent/ncnn/blob/master/src/layer/x86/convolution_sgemm_pack4to4.h
// https://github.com/Tencent/ncnn/blob/master/src/layer/x86/convolution_sgemm_pack1to4.h
// https://github.com/Tencent/ncnn/blob/master/src/layer/x86/convolution_sgemm_pack4to1.h

#include "ConvolutionMM2DX86Pack.hpp"
#include "TensorFactory.hpp"
#include "Parallel.hpp"
#include "VecIntrinsic.hpp"
#include "ConvolutionUtils.hpp"
#include "Padding.hpp"
#include "im2col.hpp"

namespace otter {

#if __SSE2__

void im2col_sgemm_conv2d_pack4_impl_x86(const Tensor& im2col, Tensor& output_, const Tensor& kernel, const Tensor& _bias) {
    const int size = im2col.size(2);
    const int maxk = im2col.size(1);
    const int inch = im2col.size(0);

    const int outch = output_.size(1);

    const float* bias = (_bias.defined()) ? _bias.data_ptr<float>() : nullptr;
    
    auto output_a = output_.accessor<float, 4, 4>()[0];
    auto im2col_a = im2col.accessor<float, 3, 4>();
    auto kernel_a = kernel.accessor<float, 3>();

    // permute
    Tensor tmp;
    if (size >= 12)
        tmp = otter::empty({size / 12 + (size % 12) / 8 + (size % 12 % 8) / 4 + (size % 12 % 4) / 2 + size % 12 % 2, inch, 12 * maxk}, otter::ScalarType::Float4);
    else if (size >= 8)
        tmp = otter::empty({size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, inch, 8 * maxk}, otter::ScalarType::Float4);
    else if (size >= 4)
        tmp = otter::empty({size / 4 + (size % 4) / 2 + size % 2, inch, 4 * maxk}, otter::ScalarType::Float4);
    else if (size >= 2)
        tmp = otter::empty({size / 2 + size % 2, inch, 2 * maxk}, otter::ScalarType::Float4);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Float4);
    
    auto tmp_a = tmp.accessor<float, 3, 4>();
    {
        int remain_size_start = 0;
        int nn_size = size / 12;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 12;

                float* tmpptr = (float*)tmp_a[i / 12].data();

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++) {
                        // transpose 4x12
                        __m128 _r0 = _mm_load_ps(img0);
                        __m128 _r1 = _mm_load_ps(img0 + 4);
                        __m128 _r2 = _mm_load_ps(img0 + 4 * 2);
                        __m128 _r3 = _mm_load_ps(img0 + 4 * 3);
                        __m128 _r4 = _mm_load_ps(img0 + 4 * 4);
                        __m128 _r5 = _mm_load_ps(img0 + 4 * 5);
                        __m128 _r6 = _mm_load_ps(img0 + 4 * 6);
                        __m128 _r7 = _mm_load_ps(img0 + 4 * 7);
                        __m128 _r8 = _mm_load_ps(img0 + 4 * 8);
                        __m128 _r9 = _mm_load_ps(img0 + 4 * 9);
                        __m128 _ra = _mm_load_ps(img0 + 4 * 10);
                        __m128 _rb = _mm_load_ps(img0 + 4 * 11);

                        _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                        _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                        _MM_TRANSPOSE4_PS(_r8, _r9, _ra, _rb);

                        _mm_store_ps(tmpptr, _r0);
                        _mm_store_ps(tmpptr + 4, _r4);
                        _mm_store_ps(tmpptr + 4 * 2, _r8);
                        _mm_store_ps(tmpptr + 4 * 3, _r1);
                        _mm_store_ps(tmpptr + 4 * 4, _r5);
                        _mm_store_ps(tmpptr + 4 * 5, _r9);
                        _mm_store_ps(tmpptr + 4 * 6, _r2);
                        _mm_store_ps(tmpptr + 4 * 7, _r6);
                        _mm_store_ps(tmpptr + 4 * 8, _ra);
                        _mm_store_ps(tmpptr + 4 * 9, _r3);
                        _mm_store_ps(tmpptr + 4 * 10, _r7);
                        _mm_store_ps(tmpptr + 4 * 11, _rb);

                        img0 += size * 4;
                        tmpptr += 48;
                    }
                }
            }
        });

        remain_size_start += nn_size * 12;
        nn_size = (size - remain_size_start) >> 3;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 8;

                float* tmpptr = (float*)tmp_a[i / 12 + (i % 12) / 8].data();

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++) {
                        // transpose 4x8
                        __m128 _r0 = _mm_load_ps(img0);
                        __m128 _r1 = _mm_load_ps(img0 + 4);
                        __m128 _r2 = _mm_load_ps(img0 + 4 * 2);
                        __m128 _r3 = _mm_load_ps(img0 + 4 * 3);
                        __m128 _r4 = _mm_load_ps(img0 + 4 * 4);
                        __m128 _r5 = _mm_load_ps(img0 + 4 * 5);
                        __m128 _r6 = _mm_load_ps(img0 + 4 * 6);
                        __m128 _r7 = _mm_load_ps(img0 + 4 * 7);

                        _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                        _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);

                        _mm_store_ps(tmpptr, _r0);
                        _mm_store_ps(tmpptr + 4, _r4);
                        _mm_store_ps(tmpptr + 4 * 2, _r1);
                        _mm_store_ps(tmpptr + 4 * 3, _r5);
                        _mm_store_ps(tmpptr + 4 * 4, _r2);
                        _mm_store_ps(tmpptr + 4 * 5, _r6);
                        _mm_store_ps(tmpptr + 4 * 6, _r3);
                        _mm_store_ps(tmpptr + 4 * 7, _r7);

                        img0 += size * 4;
                        tmpptr += 32;
                    }
                }
            }
        });

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 4;

                float* tmpptr = (float*)tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4].data();

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++) {
                        // transpose 4x4
                        __m128 _r0 = _mm_load_ps(img0);
                        __m128 _r1 = _mm_load_ps(img0 + 4);
                        __m128 _r2 = _mm_load_ps(img0 + 4 * 2);
                        __m128 _r3 = _mm_load_ps(img0 + 4 * 3);

                        _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

                        _mm_store_ps(tmpptr, _r0);
                        _mm_store_ps(tmpptr + 4, _r1);
                        _mm_store_ps(tmpptr + 4 * 2, _r2);
                        _mm_store_ps(tmpptr + 4 * 3, _r3);

                        img0 += size * 4;
                        tmpptr += 16;
                    }
                }
            }
        });

        remain_size_start += nn_size << 2;
        nn_size = (size - remain_size_start) >> 1;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 2;

                float* tmpptr = (float*)tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2].data();

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++) {
                        // transpose 4x2
                        __m128 _r0 = _mm_load_ps(img0);
                        __m128 _r1 = _mm_load_ps(img0 + 4);

                        __m128 _r01_0 = _mm_unpacklo_ps(_r0, _r1);
                        __m128 _r01_1 = _mm_unpackhi_ps(_r0, _r1);

                        _mm_store_ps(tmpptr, _r01_0);
                        _mm_store_ps(tmpptr + 4, _r01_1);

                        img0 += size * 4;
                        tmpptr += 8;
                    }
                }
            }
        });

        remain_size_start += nn_size << 1;

        otter::parallel_for(remain_size_start, size, 0, [&](int64_t begin, int64_t end) {
            for (const auto i : otter::irange(begin, end)) {
                float* tmpptr = (float*)tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2].data();

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++) {
                        __m128 _val = _mm_load_ps(img0);
                        _mm_store_ps(tmpptr, _val);

                        img0 += size * 4;
                        tmpptr += 4;
                    }
                }
            }
        });
    }

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            float* outptr0 = (float*)output_a[p].data();

            const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + p * 4 : zeros;

            int i = 0;
            for (; i + 11 < size; i += 12) {
                const float* tmpptr = (const float*)tmp_a[i / 12].data();
                const float* kptr0 = (const float*)kernel_a[p].data();

                int nn = inch * maxk * 4; // inch always > 0

                __m128 _sum0 = _mm_loadu_ps(biasptr);
                __m128 _sum1 = _sum0;
                __m128 _sum2 = _sum0;
                __m128 _sum3 = _sum0;
                __m128 _sum4 = _sum0;
                __m128 _sum5 = _sum0;
                __m128 _sum6 = _sum0;
                __m128 _sum7 = _sum0;
                __m128 _sum8 = _sum0;
                __m128 _sum9 = _sum0;
                __m128 _suma = _sum0;
                __m128 _sumb = _sum0;

                for (int j = 0; j < nn; j++) {
                    __m128 _w0 = _mm_load_ps(kptr0);

                    __m128 _val0 = _mm_load1_ps(tmpptr);
                    __m128 _val1 = _mm_load1_ps(tmpptr + 1);
                    __m128 _val2 = _mm_load1_ps(tmpptr + 2);
                    __m128 _val3 = _mm_load1_ps(tmpptr + 3);
                    __m128 _val4 = _mm_load1_ps(tmpptr + 4);
                    __m128 _val5 = _mm_load1_ps(tmpptr + 5);
                    __m128 _val6 = _mm_load1_ps(tmpptr + 6);
                    __m128 _val7 = _mm_load1_ps(tmpptr + 7);
                    __m128 _val8 = _mm_load1_ps(tmpptr + 8);
                    __m128 _val9 = _mm_load1_ps(tmpptr + 9);
                    __m128 _vala = _mm_load1_ps(tmpptr + 10);
                    __m128 _valb = _mm_load1_ps(tmpptr + 11);

                    _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val3, _w0, _sum3);
                    _sum4 = _mm_comp_fmadd_ps(_val4, _w0, _sum4);
                    _sum5 = _mm_comp_fmadd_ps(_val5, _w0, _sum5);
                    _sum6 = _mm_comp_fmadd_ps(_val6, _w0, _sum6);
                    _sum7 = _mm_comp_fmadd_ps(_val7, _w0, _sum7);
                    _sum8 = _mm_comp_fmadd_ps(_val8, _w0, _sum8);
                    _sum9 = _mm_comp_fmadd_ps(_val9, _w0, _sum9);
                    _suma = _mm_comp_fmadd_ps(_vala, _w0, _suma);
                    _sumb = _mm_comp_fmadd_ps(_valb, _w0, _sumb);

                    tmpptr += 12;
                    kptr0 += 4;
                }

                _mm_store_ps(outptr0, _sum0);
                _mm_store_ps(outptr0 + 4, _sum1);
                _mm_store_ps(outptr0 + 4 * 2, _sum2);
                _mm_store_ps(outptr0 + 4 * 3, _sum3);
                _mm_store_ps(outptr0 + 4 * 4, _sum4);
                _mm_store_ps(outptr0 + 4 * 5, _sum5);
                _mm_store_ps(outptr0 + 4 * 6, _sum6);
                _mm_store_ps(outptr0 + 4 * 7, _sum7);
                _mm_store_ps(outptr0 + 4 * 8, _sum8);
                _mm_store_ps(outptr0 + 4 * 9, _sum9);
                _mm_store_ps(outptr0 + 4 * 10, _suma);
                _mm_store_ps(outptr0 + 4 * 11, _sumb);

                outptr0 += 4 * 12;
            }
            for (; i + 7 < size; i += 8) {
                const float* tmpptr = (const float*)tmp_a[i / 12 + (i % 12) / 8].data();
                const float* kptr0 = (const float*)kernel_a[p].data();

                int nn = inch * maxk * 4; // inch always > 0

                __m128 _sum0 = _mm_loadu_ps(biasptr);
                __m128 _sum1 = _sum0;
                __m128 _sum2 = _sum0;
                __m128 _sum3 = _sum0;
                __m128 _sum4 = _sum0;
                __m128 _sum5 = _sum0;
                __m128 _sum6 = _sum0;
                __m128 _sum7 = _sum0;

                for (int j = 0; j < nn; j++) {
                    __m128 _w0 = _mm_load_ps(kptr0);

                    __m128 _val0 = _mm_load1_ps(tmpptr);
                    __m128 _val1 = _mm_load1_ps(tmpptr + 1);
                    __m128 _val2 = _mm_load1_ps(tmpptr + 2);
                    __m128 _val3 = _mm_load1_ps(tmpptr + 3);
                    __m128 _val4 = _mm_load1_ps(tmpptr + 4);
                    __m128 _val5 = _mm_load1_ps(tmpptr + 5);
                    __m128 _val6 = _mm_load1_ps(tmpptr + 6);
                    __m128 _val7 = _mm_load1_ps(tmpptr + 7);

                    _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val3, _w0, _sum3);
                    _sum4 = _mm_comp_fmadd_ps(_val4, _w0, _sum4);
                    _sum5 = _mm_comp_fmadd_ps(_val5, _w0, _sum5);
                    _sum6 = _mm_comp_fmadd_ps(_val6, _w0, _sum6);
                    _sum7 = _mm_comp_fmadd_ps(_val7, _w0, _sum7);

                    tmpptr += 8;
                    kptr0 += 4;
                }

                _mm_store_ps(outptr0, _sum0);
                _mm_store_ps(outptr0 + 4, _sum1);
                _mm_store_ps(outptr0 + 4 * 2, _sum2);
                _mm_store_ps(outptr0 + 4 * 3, _sum3);
                _mm_store_ps(outptr0 + 4 * 4, _sum4);
                _mm_store_ps(outptr0 + 4 * 5, _sum5);
                _mm_store_ps(outptr0 + 4 * 6, _sum6);
                _mm_store_ps(outptr0 + 4 * 7, _sum7);

                outptr0 += 4 * 8;
            }
            for (; i + 3 < size; i += 4) {
                const float* tmpptr = (const float*)tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4].data();
                const float* kptr0 = (const float*)kernel_a[p].data();

                int nn = inch * maxk * 4; // inch always > 0

                __m128 _sum0 = _mm_loadu_ps(biasptr);
                __m128 _sum1 = _sum0;
                __m128 _sum2 = _sum0;
                __m128 _sum3 = _sum0;

                for (int j = 0; j < nn; j++) {
                    __m128 _w0 = _mm_load_ps(kptr0);

                    __m128 _val0 = _mm_load1_ps(tmpptr);
                    __m128 _val1 = _mm_load1_ps(tmpptr + 1);
                    __m128 _val2 = _mm_load1_ps(tmpptr + 2);
                    __m128 _val3 = _mm_load1_ps(tmpptr + 3);

                    _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val3, _w0, _sum3);

                    tmpptr += 4;
                    kptr0 += 4;
                }

                _mm_store_ps(outptr0, _sum0);
                _mm_store_ps(outptr0 + 4, _sum1);
                _mm_store_ps(outptr0 + 4 * 2, _sum2);
                _mm_store_ps(outptr0 + 4 * 3, _sum3);

                outptr0 += 4 * 4;
            }
            for (; i + 1 < size; i += 2) {
                const float* tmpptr = (const float*)tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2].data();
                const float* kptr0 = (const float*)kernel_a[p].data();

                int nn = inch * maxk * 4; // inch always > 0

                __m128 _sum0 = _mm_loadu_ps(biasptr);
                __m128 _sum1 = _sum0;

                for (int j = 0; j < nn; j++) {
                    __m128 _w0 = _mm_load_ps(kptr0);

                    __m128 _val0 = _mm_load1_ps(tmpptr);
                    __m128 _val1 = _mm_load1_ps(tmpptr + 1);

                    _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);

                    tmpptr += 2;
                    kptr0 += 4;
                }

                _mm_store_ps(outptr0, _sum0);
                _mm_store_ps(outptr0 + 4, _sum1);

                outptr0 += 4 * 2;
            }
            for (; i < size; i++) {
                const float* tmpptr = (const float*)tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2].data();
                const float* kptr0 = (const float*)kernel_a[p].data();

                int nn = inch * maxk * 4; // inch always > 0

                __m128 _sum = _mm_loadu_ps(biasptr);

                for (int j = 0; j < nn; j++) {
                    __m128 _w0 = _mm_load_ps(kptr0);
                    __m128 _val0 = _mm_load1_ps(tmpptr);
                    _sum = _mm_comp_fmadd_ps(_val0, _w0, _sum);

                    tmpptr += 1;
                    kptr0 += 4;
                }

                _mm_store_ps(outptr0, _sum);

                outptr0 += 4;
            }
        }
    });
}

void im2col_sgemm_conv2d_pack4to1_impl_x86(const Tensor& im2col, Tensor& output_, const Tensor& kernel, const Tensor& _bias) {
    const int size = im2col.size(2);
    const int maxk = im2col.size(1);
    const int inch = im2col.size(0);

    const int outch = output_.size(1);

    const float* bias = (_bias.defined()) ? _bias.data_ptr<float>() : nullptr;
    
    auto output_a = output_.accessor<float, 4>()[0];
    auto im2col_a = im2col.accessor<float, 3, 4>();
    auto kernel_a = kernel.accessor<float, 3>();
    
    Tensor tmp;
    if (size >= 12)
        tmp = otter::empty({size / 12 + (size % 12) / 8 + (size % 12 % 8) / 4 + size % 12 % 4, inch, 12 * maxk}, otter::ScalarType::Float4);
    else if (size >= 8)
        tmp = otter::empty({size / 8 + (size % 8) / 4 + size % 4, inch, 8 * maxk}, otter::ScalarType::Float4);
    else if (size >= 4)
        tmp = otter::empty({size / 4 + size % 4, inch, 4 * maxk}, otter::ScalarType::Float4);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Float4);
    
    auto tmp_a = tmp.accessor<float, 3, 4>();
    {
        int remain_size_start = 0;
        int nn_size = size / 12;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 12;

                float* tmpptr = (float*)tmp_a[i / 12].data();

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++) {
                        // transpose 4x12
                        __m128 _r0 = _mm_load_ps(img0);
                        __m128 _r1 = _mm_load_ps(img0 + 4);
                        __m128 _r2 = _mm_load_ps(img0 + 4 * 2);
                        __m128 _r3 = _mm_load_ps(img0 + 4 * 3);
                        __m128 _r4 = _mm_load_ps(img0 + 4 * 4);
                        __m128 _r5 = _mm_load_ps(img0 + 4 * 5);
                        __m128 _r6 = _mm_load_ps(img0 + 4 * 6);
                        __m128 _r7 = _mm_load_ps(img0 + 4 * 7);
                        __m128 _r8 = _mm_load_ps(img0 + 4 * 8);
                        __m128 _r9 = _mm_load_ps(img0 + 4 * 9);
                        __m128 _ra = _mm_load_ps(img0 + 4 * 10);
                        __m128 _rb = _mm_load_ps(img0 + 4 * 11);

                        _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                        _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);
                        _MM_TRANSPOSE4_PS(_r8, _r9, _ra, _rb);

                        _mm_store_ps(tmpptr, _r0);
                        _mm_store_ps(tmpptr + 4, _r4);
                        _mm_store_ps(tmpptr + 4 * 2, _r8);
                        _mm_store_ps(tmpptr + 4 * 3, _r1);
                        _mm_store_ps(tmpptr + 4 * 4, _r5);
                        _mm_store_ps(tmpptr + 4 * 5, _r9);
                        _mm_store_ps(tmpptr + 4 * 6, _r2);
                        _mm_store_ps(tmpptr + 4 * 7, _r6);
                        _mm_store_ps(tmpptr + 4 * 8, _ra);
                        _mm_store_ps(tmpptr + 4 * 9, _r3);
                        _mm_store_ps(tmpptr + 4 * 10, _r7);
                        _mm_store_ps(tmpptr + 4 * 11, _rb);

                        img0 += size * 4;
                        tmpptr += 48;
                    }
                }
            }
        });

        remain_size_start += nn_size * 12;
        nn_size = (size - remain_size_start) >> 3;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 8;

                float* tmpptr = (float*)tmp_a[i / 12 + (i % 12) / 8].data();

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++) {
                        // transpose 4x8
                        __m128 _r0 = _mm_load_ps(img0);
                        __m128 _r1 = _mm_load_ps(img0 + 4);
                        __m128 _r2 = _mm_load_ps(img0 + 4 * 2);
                        __m128 _r3 = _mm_load_ps(img0 + 4 * 3);
                        __m128 _r4 = _mm_load_ps(img0 + 4 * 4);
                        __m128 _r5 = _mm_load_ps(img0 + 4 * 5);
                        __m128 _r6 = _mm_load_ps(img0 + 4 * 6);
                        __m128 _r7 = _mm_load_ps(img0 + 4 * 7);

                        _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);
                        _MM_TRANSPOSE4_PS(_r4, _r5, _r6, _r7);

                        _mm_store_ps(tmpptr, _r0);
                        _mm_store_ps(tmpptr + 4, _r4);
                        _mm_store_ps(tmpptr + 4 * 2, _r1);
                        _mm_store_ps(tmpptr + 4 * 3, _r5);
                        _mm_store_ps(tmpptr + 4 * 4, _r2);
                        _mm_store_ps(tmpptr + 4 * 5, _r6);
                        _mm_store_ps(tmpptr + 4 * 6, _r3);
                        _mm_store_ps(tmpptr + 4 * 7, _r7);

                        img0 += size * 4;
                        tmpptr += 32;
                    }
                }
            }
        });

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 4;

                float* tmpptr = (float*)tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4].data();

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++) {
                        // transpose 4x4
                        __m128 _r0 = _mm_load_ps(img0);
                        __m128 _r1 = _mm_load_ps(img0 + 4);
                        __m128 _r2 = _mm_load_ps(img0 + 4 * 2);
                        __m128 _r3 = _mm_load_ps(img0 + 4 * 3);

                        _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

                        _mm_store_ps(tmpptr, _r0);
                        _mm_store_ps(tmpptr + 4, _r1);
                        _mm_store_ps(tmpptr + 4 * 2, _r2);
                        _mm_store_ps(tmpptr + 4 * 3, _r3);

                        img0 += size * 4;
                        tmpptr += 16;
                    }
                }
            }
        });

        remain_size_start += nn_size << 2;

        otter::parallel_for(remain_size_start, size, 0, [&](int64_t begin, int64_t end) {
            for (const auto i : otter::irange(begin, end)) {
                float* tmpptr = (float*)tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4].data();

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++) {
                        __m128 _val = _mm_load_ps(img0);
                        _mm_store_ps(tmpptr, _val);

                        img0 += size * 4;
                        tmpptr += 4;
                    }
                }
            }
        });
    }

    int nn_outch = outch / 4;
    int remain_outch_start = nn_outch * 4;

    otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto pp : otter::irange(begin, end)) {
            int p = pp * 4;

            float* outptr0 = (float*)output_a[p + 0].data();
            float* outptr1 = (float*)output_a[p + 1].data();
            float* outptr2 = (float*)output_a[p + 2].data();
            float* outptr3 = (float*)output_a[p + 3].data();

            const float zeros[4] = {0.f};
            const float* biasptr = bias ? bias + p : zeros;

            int i = 0;
            for (; i + 11 < size; i += 12) {
                const float* tmpptr = (const float*)tmp_a[i / 12].data();
                const float* kptr0 = (const float*)kernel_a[p / 4].data();

                int nn = inch * maxk * 4; // inch always > 0

                __m128 _sum0 = _mm_load1_ps(biasptr);
                __m128 _sum1 = _mm_load1_ps(biasptr);
                __m128 _sum2 = _mm_load1_ps(biasptr);
                __m128 _sum3 = _mm_load1_ps(biasptr + 1);
                __m128 _sum4 = _mm_load1_ps(biasptr + 1);
                __m128 _sum5 = _mm_load1_ps(biasptr + 1);
                __m128 _sum6 = _mm_load1_ps(biasptr + 2);
                __m128 _sum7 = _mm_load1_ps(biasptr + 2);
                __m128 _sum8 = _mm_load1_ps(biasptr + 2);
                __m128 _sum9 = _mm_load1_ps(biasptr + 3);
                __m128 _suma = _mm_load1_ps(biasptr + 3);
                __m128 _sumb = _mm_load1_ps(biasptr + 3);

                for (int j = 0; j < nn; j++) {
                    __m128 _val0 = _mm_load_ps(tmpptr);
                    __m128 _val1 = _mm_load_ps(tmpptr + 4);
                    __m128 _val2 = _mm_load_ps(tmpptr + 8);

                    __m128 _w0 = _mm_load1_ps(kptr0);
                    __m128 _w1 = _mm_load1_ps(kptr0 + 1);
                    __m128 _w2 = _mm_load1_ps(kptr0 + 2);
                    __m128 _w3 = _mm_load1_ps(kptr0 + 3);

                    _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val0, _w1, _sum3);
                    _sum4 = _mm_comp_fmadd_ps(_val1, _w1, _sum4);
                    _sum5 = _mm_comp_fmadd_ps(_val2, _w1, _sum5);
                    _sum6 = _mm_comp_fmadd_ps(_val0, _w2, _sum6);
                    _sum7 = _mm_comp_fmadd_ps(_val1, _w2, _sum7);
                    _sum8 = _mm_comp_fmadd_ps(_val2, _w2, _sum8);
                    _sum9 = _mm_comp_fmadd_ps(_val0, _w3, _sum9);
                    _suma = _mm_comp_fmadd_ps(_val1, _w3, _suma);
                    _sumb = _mm_comp_fmadd_ps(_val2, _w3, _sumb);

                    tmpptr += 12;
                    kptr0 += 4;
                }

                _mm_store_ps(outptr0, _sum0);
                _mm_store_ps(outptr0 + 4, _sum1);
                _mm_store_ps(outptr0 + 8, _sum2);
                _mm_store_ps(outptr1, _sum3);
                _mm_store_ps(outptr1 + 4, _sum4);
                _mm_store_ps(outptr1 + 8, _sum5);
                _mm_store_ps(outptr2, _sum6);
                _mm_store_ps(outptr2 + 4, _sum7);
                _mm_store_ps(outptr2 + 8, _sum8);
                _mm_store_ps(outptr3, _sum9);
                _mm_store_ps(outptr3 + 4, _suma);
                _mm_store_ps(outptr3 + 8, _sumb);

                outptr0 += 12;
                outptr1 += 12;
                outptr2 += 12;
                outptr3 += 12;
            }
            for (; i + 7 < size; i += 8) {
                const float* tmpptr = (const float*)tmp_a[i / 12 + (i % 12) / 8].data();
                const float* kptr0 = (const float*)kernel_a[p / 4].data();

                int nn = inch * maxk * 4; // inch always > 0

                __m128 _sum0 = _mm_load1_ps(biasptr);
                __m128 _sum1 = _mm_load1_ps(biasptr);
                __m128 _sum2 = _mm_load1_ps(biasptr + 1);
                __m128 _sum3 = _mm_load1_ps(biasptr + 1);
                __m128 _sum4 = _mm_load1_ps(biasptr + 2);
                __m128 _sum5 = _mm_load1_ps(biasptr + 2);
                __m128 _sum6 = _mm_load1_ps(biasptr + 3);
                __m128 _sum7 = _mm_load1_ps(biasptr + 3);

                for (int j = 0; j < nn; j++) {
                    __m128 _val0 = _mm_load_ps(tmpptr);
                    __m128 _val1 = _mm_load_ps(tmpptr + 4);

                    __m128 _w0 = _mm_load1_ps(kptr0);
                    __m128 _w1 = _mm_load1_ps(kptr0 + 1);
                    __m128 _w2 = _mm_load1_ps(kptr0 + 2);
                    __m128 _w3 = _mm_load1_ps(kptr0 + 3);

                    _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_val0, _w1, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val1, _w1, _sum3);
                    _sum4 = _mm_comp_fmadd_ps(_val0, _w2, _sum4);
                    _sum5 = _mm_comp_fmadd_ps(_val1, _w2, _sum5);
                    _sum6 = _mm_comp_fmadd_ps(_val0, _w3, _sum6);
                    _sum7 = _mm_comp_fmadd_ps(_val1, _w3, _sum7);

                    tmpptr += 8;
                    kptr0 += 4;
                }

                _mm_store_ps(outptr0, _sum0);
                _mm_store_ps(outptr0 + 4, _sum1);
                _mm_store_ps(outptr1, _sum2);
                _mm_store_ps(outptr1 + 4, _sum3);
                _mm_store_ps(outptr2, _sum4);
                _mm_store_ps(outptr2 + 4, _sum5);
                _mm_store_ps(outptr3, _sum6);
                _mm_store_ps(outptr3 + 4, _sum7);

                outptr0 += 8;
                outptr1 += 8;
                outptr2 += 8;
                outptr3 += 8;
            }
            for (; i + 3 < size; i += 4) {
                const float* tmpptr = (const float*)tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4].data();
                const float* kptr0 = (const float*)kernel_a[p / 4].data();

                int nn = inch * maxk * 4; // inch always > 0

                __m128 _sum0 = _mm_load1_ps(biasptr);
                __m128 _sum1 = _mm_load1_ps(biasptr + 1);
                __m128 _sum2 = _mm_load1_ps(biasptr + 2);
                __m128 _sum3 = _mm_load1_ps(biasptr + 3);

                for (int j = 0; j < nn; j++) {
                    __m128 _val0 = _mm_load_ps(tmpptr);

                    __m128 _w0 = _mm_load1_ps(kptr0);
                    __m128 _w1 = _mm_load1_ps(kptr0 + 1);
                    __m128 _w2 = _mm_load1_ps(kptr0 + 2);
                    __m128 _w3 = _mm_load1_ps(kptr0 + 3);

                    _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val0, _w1, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_val0, _w2, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val0, _w3, _sum3);

                    tmpptr += 4;
                    kptr0 += 4;
                }

                _mm_store_ps(outptr0, _sum0);
                _mm_store_ps(outptr1, _sum1);
                _mm_store_ps(outptr2, _sum2);
                _mm_store_ps(outptr3, _sum3);

                outptr0 += 4;
                outptr1 += 4;
                outptr2 += 4;
                outptr3 += 4;
            }
            for (; i < size; i++) {
                const float* tmpptr = (const float*)tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4].data();
                const float* kptr0 = (const float*)kernel_a[p / 4].data();

                int nn = inch * maxk * 4; // inch always > 0

                __m128 _sum = _mm_loadu_ps(biasptr);

                for (int j = 0; j < nn; j++)
                {
                    __m128 _val0 = _mm_load1_ps(tmpptr);
                    __m128 _w0 = _mm_load_ps(kptr0);
                    _sum = _mm_comp_fmadd_ps(_val0, _w0, _sum);

                    tmpptr += 1;
                    kptr0 += 4;
                }

                float sum[4];
                _mm_storeu_ps(sum, _sum);

                outptr0[0] = sum[0];
                outptr1[0] = sum[1];
                outptr2[0] = sum[2];
                outptr3[0] = sum[3];

                outptr0 += 1;
                outptr1 += 1;
                outptr2 += 1;
                outptr3 += 1;
            }
        }
    });

    otter::parallel_for(remain_outch_start, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            float* outptr0 = (float*)output_a[p].data();

            const float bias0 = bias ? bias[p] : 0.f;

            int i = 0;
            for (; i + 11 < size; i += 12) {
                const float* tmpptr = (const float*)tmp_a[i / 12].data();
                const float* kptr0 = (const float*)kernel_a[p / 4 + p % 4].data();

                int nn = inch * maxk * 4; // inch always > 0

                __m128 _sum0 = _mm_set1_ps(bias0);
                __m128 _sum1 = _mm_set1_ps(bias0);
                __m128 _sum2 = _mm_set1_ps(bias0);

                for (int j = 0; j < nn; j++) {
                    __m128 _val0 = _mm_load_ps(tmpptr);
                    __m128 _val1 = _mm_load_ps(tmpptr + 4);
                    __m128 _val2 = _mm_load_ps(tmpptr + 8);
                    __m128 _w0 = _mm_load1_ps(kptr0);
                    _sum0 = _mm_comp_fmadd_ps(_w0, _val0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_w0, _val1, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_w0, _val2, _sum2);

                    tmpptr += 12;
                    kptr0 += 1;
                }

                _mm_storeu_ps(outptr0, _sum0);
                _mm_storeu_ps(outptr0 + 4, _sum1);
                _mm_storeu_ps(outptr0 + 8, _sum2);

                outptr0 += 12;
            }
            for (; i + 7 < size; i += 8) {
                const float* tmpptr = (const float*)tmp_a[i / 12 + (i % 12) / 8].data();
                const float* kptr0 = (const float*)kernel_a[p / 4 + p % 4].data();

                int nn = inch * maxk * 4; // inch always > 0

                __m128 _sum0 = _mm_set1_ps(bias0);
                __m128 _sum1 = _mm_set1_ps(bias0);

                for (int j = 0; j < nn; j++) {
                    __m128 _val0 = _mm_load_ps(tmpptr);
                    __m128 _val1 = _mm_load_ps(tmpptr + 4);
                    __m128 _w0 = _mm_load1_ps(kptr0);
                    _sum0 = _mm_comp_fmadd_ps(_w0, _val0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_w0, _val1, _sum1);

                    tmpptr += 8;
                    kptr0 += 1;
                }

                _mm_storeu_ps(outptr0, _sum0);
                _mm_storeu_ps(outptr0 + 4, _sum1);

                outptr0 += 8;
            }
            for (; i + 3 < size; i += 4) {
                const float* tmpptr = (const float*)tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4].data();
                const float* kptr0 = (const float*)kernel_a[p / 4 + p % 4].data();

                int nn = inch * maxk * 4; // inch always > 0

                __m128 _sum0 = _mm_set1_ps(bias0);

                for (int j = 0; j < nn; j++) {
                    __m128 _val0 = _mm_load_ps(tmpptr);
                    __m128 _w0 = _mm_load1_ps(kptr0);
                    _sum0 = _mm_comp_fmadd_ps(_w0, _val0, _sum0);

                    tmpptr += 4;
                    kptr0 += 1;
                }

                _mm_storeu_ps(outptr0, _sum0);

                outptr0 += 4;
            }
            for (; i < size; i++) {
                const float* tmpptr = (const float*)tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4].data();
                const float* kptr0 = (const float*)kernel_a[p / 4 + p % 4].data();

                int nn = inch * maxk; // inch always > 0

                float sum0 = bias0;

                __m128 _sum0 = _mm_setzero_ps();

                for (int j = 0; j < nn; j++) {
                    __m128 _val0 = _mm_load_ps(tmpptr);
                    __m128 _w0 = _mm_load_ps(kptr0);
                    _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);

                    tmpptr += 4;
                    kptr0 += 4;
                }

                sum0 += _mm_reduce_add_ps(_sum0);

                outptr0[0] = sum0;

                outptr0 += 1;
            }
        }
    });
}

void im2col_sgemm_conv2d_pack1to4_impl_x86(const Tensor& im2col, Tensor& output_, const Tensor& kernel, const Tensor& _bias) {
    const int size = im2col.size(2);
    const int maxk = im2col.size(1);
    const int inch = im2col.size(0);

    const int outch = output_.size(1);

    const float* bias = (_bias.defined()) ? _bias.data_ptr<float>() : nullptr;
    
    auto output_a = output_.accessor<float, 4, 4>()[0];
    auto im2col_a = im2col.accessor<float, 3>();
    auto kernel_a = kernel.accessor<float, 3>();
    
    Tensor tmp;
    if (size >= 8)
        tmp = otter::empty({size / 8 + (size % 8) / 4 + size % 4, inch, 8 * maxk}, otter::ScalarType::Float);
    else if (size >= 4)
        tmp = otter::empty({size / 4 + size % 4, inch, 4 * maxk}, otter::ScalarType::Float);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Float);
    
    auto tmp_a = tmp.accessor<float, 3>();
    {
        int nn_size = size >> 3;
        int remain_size_start = 0;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 8;

                float* tmpptr = tmp_a[i / 8].data();

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i;

                    for (int k = 0; k < maxk; k++) {
                        __m128 _r0 = _mm_loadu_ps(img0);
                        __m128 _r1 = _mm_loadu_ps(img0 + 4);
                        _mm_store_ps(tmpptr, _r0);
                        _mm_store_ps(tmpptr + 4, _r1);

                        img0 += size;
                        tmpptr += 8;
                    }
                }
            }
        });

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 4;

                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i;

                    for (int k = 0; k < maxk; k++) {
                        __m128 _r0 = _mm_loadu_ps(img0);
                        _mm_store_ps(tmpptr, _r0);

                        img0 += size;
                        tmpptr += 4;
                    }
                }
            }
        });

        remain_size_start += nn_size << 2;

        otter::parallel_for(remain_size_start, size, 0, [&](int64_t begin, int64_t end) {
            for (const auto i : otter::irange(begin, end)) {
                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();

                for (int q = 0; q < inch; q++)  {
                    const float* img0 = (const float*)im2col_a[q].data() + i;

                    for (int k = 0; k < maxk; k++) {
                        tmpptr[0] = img0[0];
                        img0 += size;
                        tmpptr += 1;
                    }
                }
            }
        });
    }

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            float* outptr0 = output_a[p].data();

            const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + p * 4 : zeros;

            int i = 0;
            for (; i + 7 < size; i += 8) {
                const float* tmpptr = tmp_a[i / 8].data();
                const float* kptr0 = kernel_a[p].data();

                int nn = inch * maxk; // inch always > 0

                __m128 _sum0 = _mm_loadu_ps(biasptr);
                __m128 _sum1 = _sum0;
                __m128 _sum2 = _sum0;
                __m128 _sum3 = _sum0;
                __m128 _sum4 = _sum0;
                __m128 _sum5 = _sum0;
                __m128 _sum6 = _sum0;
                __m128 _sum7 = _sum0;

                for (int j = 0; j < nn; j++) {
                    __m128 _w0 = _mm_load_ps(kptr0);

                    __m128 _val0 = _mm_load1_ps(tmpptr);
                    __m128 _val1 = _mm_load1_ps(tmpptr + 1);
                    __m128 _val2 = _mm_load1_ps(tmpptr + 2);
                    __m128 _val3 = _mm_load1_ps(tmpptr + 3);
                    __m128 _val4 = _mm_load1_ps(tmpptr + 4);
                    __m128 _val5 = _mm_load1_ps(tmpptr + 5);
                    __m128 _val6 = _mm_load1_ps(tmpptr + 6);
                    __m128 _val7 = _mm_load1_ps(tmpptr + 7);

                    _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val3, _w0, _sum3);
                    _sum4 = _mm_comp_fmadd_ps(_val4, _w0, _sum4);
                    _sum5 = _mm_comp_fmadd_ps(_val5, _w0, _sum5);
                    _sum6 = _mm_comp_fmadd_ps(_val6, _w0, _sum6);
                    _sum7 = _mm_comp_fmadd_ps(_val7, _w0, _sum7);

                    tmpptr += 8;
                    kptr0 += 4;
                }

                _mm_store_ps(outptr0, _sum0);
                _mm_store_ps(outptr0 + 4, _sum1);
                _mm_store_ps(outptr0 + 8, _sum2);
                _mm_store_ps(outptr0 + 12, _sum3);
                _mm_store_ps(outptr0 + 16, _sum4);
                _mm_store_ps(outptr0 + 20, _sum5);
                _mm_store_ps(outptr0 + 24, _sum6);
                _mm_store_ps(outptr0 + 28, _sum7);
                outptr0 += 32;
            }
            for (; i + 3 < size; i += 4) {
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();
                const float* kptr0 = kernel_a[p].data();

                int nn = inch * maxk; // inch always > 0

                __m128 _sum0 = _mm_loadu_ps(biasptr);
                __m128 _sum1 = _sum0;
                __m128 _sum2 = _sum0;
                __m128 _sum3 = _sum0;

                for (int j = 0; j < nn; j++) {
                    __m128 _w0 = _mm_load_ps(kptr0);

                    __m128 _val0 = _mm_load1_ps(tmpptr);
                    __m128 _val1 = _mm_load1_ps(tmpptr + 1);
                    __m128 _val2 = _mm_load1_ps(tmpptr + 2);
                    __m128 _val3 = _mm_load1_ps(tmpptr + 3);

                    _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                    _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val3, _w0, _sum3);

                    tmpptr += 4;
                    kptr0 += 4;
                }

                _mm_store_ps(outptr0, _sum0);
                _mm_store_ps(outptr0 + 4, _sum1);
                _mm_store_ps(outptr0 + 8, _sum2);
                _mm_store_ps(outptr0 + 12, _sum3);
                outptr0 += 16;
            }
            for (; i < size; i++) {
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();
                const float* kptr0 = kernel_a[p].data();

                int nn = inch * maxk; // inch always > 0

                __m128 _sum = _mm_loadu_ps(biasptr);

                for (int j = 0; j < nn; j++) {
                    __m128 _w0 = _mm_load_ps(kptr0);
                    __m128 _val = _mm_load1_ps(tmpptr);
                    _sum = _mm_comp_fmadd_ps(_w0, _val, _sum);

                    tmpptr += 1;
                    kptr0 += 4;
                }

                _mm_store_ps(outptr0, _sum);
                outptr0 += 4;
            }
        }
    });
}

void convolution_im2col_sgemm_transform_kernel_pack4_sse(const Tensor& kernel_, Tensor& kernel_tf, int inch, int outch, int kernel_w, int kernel_h) {
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 4b-4a-maxk-inch/4a-outch/4b
    Tensor kernel = kernel_.view({outch, inch, maxk});
    kernel_tf = otter::empty({outch / 4, inch / 4, 16 * maxk}, otter::ScalarType::Float);
    
    auto kernel_a = kernel.accessor<float, 3>();
    auto kernel_tf_a = kernel_tf.accessor<float, 3>();

    int q = 0;
    for (; q + 3 < outch; q += 4) {
        const auto k0 = kernel_a[q + 0];
        const auto k1 = kernel_a[q + 1];
        const auto k2 = kernel_a[q + 2];
        const auto k3 = kernel_a[q + 3];

        float* g00 = kernel_tf_a[q / 4].data();

        for (int p = 0; p + 3 < inch; p += 4) {
            const float* k00 = k0[p + 0].data();
            const float* k01 = k0[p + 1].data();
            const float* k02 = k0[p + 2].data();
            const float* k03 = k0[p + 3].data();

            const float* k10 = k1[p + 0].data();
            const float* k11 = k1[p + 1].data();
            const float* k12 = k1[p + 2].data();
            const float* k13 = k1[p + 3].data();

            const float* k20 = k2[p + 0].data();
            const float* k21 = k2[p + 1].data();
            const float* k22 = k2[p + 2].data();
            const float* k23 = k2[p + 3].data();

            const float* k30 = k3[p + 0].data();
            const float* k31 = k3[p + 1].data();
            const float* k32 = k3[p + 2].data();
            const float* k33 = k3[p + 3].data();

            for (int k = 0; k < maxk; k++) {
                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];

                g00[4] = k01[k];
                g00[5] = k11[k];
                g00[6] = k21[k];
                g00[7] = k31[k];

                g00[8] = k02[k];
                g00[9] = k12[k];
                g00[10] = k22[k];
                g00[11] = k32[k];

                g00[12] = k03[k];
                g00[13] = k13[k];
                g00[14] = k23[k];
                g00[15] = k33[k];

                g00 += 16;
            }
        }
    }
}

void convolution_im2col_sgemm_transform_kernel_pack4to1_sse(const Tensor& kernel_, Tensor& kernel_tf, int inch, int outch, int kernel_w, int kernel_h) {
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = pb-pa-maxk-inch/pa-outch/pb
    Tensor kernel = kernel_.view({outch, inch, maxk});
    kernel_tf = otter::empty({outch / 4 + outch % 4, inch / 4, 4 * 4 * maxk}, otter::ScalarType::Float);
    
    auto kernel_a = kernel.accessor<float, 3>();
    auto kernel_tf_a = kernel_tf.accessor<float, 3>();

    int q = 0;
    for (; q + 3 < outch; q += 4) {
        float* g00 = (float*)kernel_tf_a[q / 4].data();

        for (int p = 0; p + 3 < inch; p += 4) {
            for (int k = 0; k < maxk; k++) {
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        const float* k00 = (const float*)kernel_a[q + j][p + i].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
    }
    for (; q < outch; q++) {
        const auto k0 = kernel_a[q];

        float* g00 = (float*)kernel_tf_a[q / 4 + q % 4].data();

        for (int p = 0; p + 3 < inch; p += 4) {
            for (int k = 0; k < maxk; k++) {
                for (int j = 0; j < 4; j++) {
                    const float* k00 = (const float*)k0[p + j].data();

                    g00[0] = k00[k];

                    g00++;
                }
            }
        }
    }
}

void convolution_im2col_sgemm_transform_kernel_pack1to4_sse(const Tensor& kernel_, Tensor& kernel_tf, int inch, int outch, int kernel_w, int kernel_h) {
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 4b-4a-maxk-inch/4a-outch/4b
    Tensor kernel = kernel_.view({outch, inch, maxk});
    kernel_tf = otter::empty({outch / 4, inch, 4 * maxk}, otter::ScalarType::Float);
    
    auto kernel_a = kernel.accessor<float, 3>();
    auto kernel_tf_a = kernel_tf.accessor<float, 3>();

    int q = 0;
    for (; q + 3 < outch; q += 4) {
        const auto k0 = kernel_a[q];
        const auto k1 = kernel_a[q + 1];
        const auto k2 = kernel_a[q + 2];
        const auto k3 = kernel_a[q + 3];

        float* g00 = kernel_tf_a[q / 4].data();

        for (int p = 0; p < inch; p++) {
            const float* k00 = k0[p].data();
            const float* k10 = k1[p].data();
            const float* k20 = k2[p].data();
            const float* k30 = k3[p].data();

            for (int k = 0; k < maxk; k++) {
                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];

                g00 += 4;
            }
        }
    }
}

Tensor& sgemm_conv2d_pack4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_({output_size[0], output_size[1] / 4, output_size[2], output_size[3]});
    
    int inch = self.size(1);
    
    const int kernel_h = kernel_size[0];
    const int kernel_w = kernel_size[1];
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];
    
    int outw = output.size(3);
    int outh = output.size(2);
    int outch = output.size(1);
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        convolution_im2col_sgemm_transform_kernel_pack4_sse(weight, kernel_tf, inch * 4, outch * 4, kernel_w, kernel_h);
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    
    Tensor im2col = otter::empty({inch, maxk, size}, ScalarType::Float4);
    
    auto input_a = input.accessor<float, 3, 4>();
    auto im2col_a = im2col.accessor<float, 3, 4>();
    // im2col
    {
        const int gap = (w * stride_h - outw * stride_w) * 4;

        otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
            for (const auto p : otter::irange(begin, end)) {
                const auto img = input_a[p];
                float* ptr = (float*)im2col_a[p].data();

                for (int u = 0; u < kernel_h; u++) {
                    for (int v = 0; v < kernel_w; v++) {
                        const float* sptr = (const float*)img[dilation_h * u].data() + dilation_w * v * 4;

                        for (int i = 0; i < outh; i++) {
                            int j = 0;
                            for (; j < outw; j++) {
                                __m128 _val = _mm_load_ps(sptr);
                                _mm_store_ps(ptr, _val);

                                sptr += stride_w * 4;
                                ptr += 4;
                            }

                            sptr += gap;
                        }
                    }
                }
            }
        });
    }
    
    im2col_sgemm_conv2d_pack4_impl_x86(im2col, output, kernel_tf, bias);
    
    return output;
}
    
Tensor sgemm_conv2d_pack4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    Tensor output = otter::empty({}, otter::ScalarType::Float4);
    sgemm_conv2d_pack4_x86_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
    
    return output;
}

Tensor& sgemm_conv2d_pack4to1_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    output.resize_(output_size);
    
    int inch = self.size(1);
    
    const int kernel_h = kernel_size[0];
    const int kernel_w = kernel_size[1];
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];
    
    int outw = output.size(3);
    int outh = output.size(2);
    int outch = output.size(1);
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        convolution_im2col_sgemm_transform_kernel_pack4to1_sse(weight, kernel_tf, inch * 4, outch, kernel_w, kernel_h);
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    
    Tensor im2col = otter::empty({inch, maxk, size}, ScalarType::Float4);
    
    auto input_a = input.accessor<float, 3, 4>();
    auto im2col_a = im2col.accessor<float, 3, 4>();
    
    {
        const int gap = (w * stride_h - outw * stride_w) * 4;

        otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
            for (const auto p : otter::irange(begin, end))
            {
                const auto img = input_a[p];
                float* ptr = (float*)im2col_a[p].data();

                for (int u = 0; u < kernel_h; u++) {
                    for (int v = 0; v < kernel_w; v++) {
                        const float* sptr = (const float*)img[dilation_h * u].data() + dilation_w * v * 4;

                        for (int i = 0; i < outh; i++) {
                            int j = 0;
                            for (; j < outw; j++) {
                                __m128 _val = _mm_load_ps(sptr);
                                _mm_store_ps(ptr, _val);

                                sptr += stride_w * 4;
                                ptr += 4;
                            }

                            sptr += gap;
                        }
                    }
                }
            }
        });
    }
    
    im2col_sgemm_conv2d_pack4to1_impl_x86(im2col, output, kernel_tf, bias);
    
    return output;
}
    
Tensor sgemm_conv2d_pack4to1_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    Tensor output = otter::empty({}, otter::ScalarType::Float);
    sgemm_conv2d_pack4to1_x86_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
    
    return output;
}

Tensor& sgemm_conv2d_pack1to4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding, dilation);
    output.resize_({output_size[0], output_size[1] / 4, output_size[2], output_size[3]});
    
    int inch = self.size(1);
    
    const int kernel_h = kernel_size[0];
    const int kernel_w = kernel_size[1];
    
    int outw = output.size(3);
    int outh = output.size(2);
    int outch = output.size(1);
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        convolution_im2col_sgemm_transform_kernel_pack1to4_sse(weight, kernel_tf, inch, outch * 4, kernel_w, kernel_h);
    
    Tensor im2col = otter::im2col_cpu(self, kernel_size, stride, padding, dilation).view({inch, maxk, size});
    
    im2col_sgemm_conv2d_pack1to4_impl_x86(im2col, output, kernel_tf, bias);
    
    return output;
}
    
Tensor sgemm_conv2d_pack1to4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    Tensor output = otter::empty({}, otter::ScalarType::Float4);
    
    sgemm_conv2d_pack1to4_x86_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
    
    return output;
}
    
Tensor conv2d_1x1s1_sgemm_pack4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
    output.resize_({output_size[0], output_size[1] / 4, output_size[2], output_size[3]});
    
    int inch = self.size(1);
    int outch = output.size(1);
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        convolution_im2col_sgemm_transform_kernel_pack4_sse(weight, kernel_tf, inch * 4, outch * 4, 1, 1);
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    int h = input.size(1);
    const int size = w * h;
    
    Tensor im2col = input.view({-1, 1, size});
    
    im2col_sgemm_conv2d_pack4_impl_x86(im2col, output, kernel_tf, bias);
    
    return output;
}

Tensor conv2d_1x1s1_sgemm_pack4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
               
    auto output = otter::empty({}, otter::ScalarType::Float4);
    
    return conv2d_1x1s1_sgemm_pack4_x86_out(self, weight, weight_o, bias, padding, output);
}

Tensor conv2d_1x1s2_sgemm_pack4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {2, 2}, padding);
    output.resize_({output_size[0], output_size[1] / 4, output_size[2], output_size[3]});
    
    int inch = self.size(1);
    int outch = output.size(1);
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        convolution_im2col_sgemm_transform_kernel_pack4_sse(weight, kernel_tf, inch * 4, outch * 4, 1, 1);
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    int channels = input.size(0);
    
    int outw = output_size[3];
    int outh = output_size[2];
    
    const int tailstep = (w - 2 * outw + w) * 4;
    
    Tensor shrinked = otter::empty({channels, outh, outw}, otter::ScalarType::Float4);
    
    auto input_a = input.accessor<float, 3, 4>();
    auto shrinked_a = shrinked.accessor<float, 3, 4>();
    
    otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            const float* r0 = input_a[p].data();
            float* outptr = shrinked_a[p].data();

            for (int i = 0; i < outh; i++) {
                for (int j = 0; j < outw; j++) {
                    __m128 _v = _mm_load_ps(r0);
                    _mm_store_ps(outptr, _v);

                    r0 += 8;
                    outptr += 4;
                }

                r0 += tailstep;
            }
        }
    });
    
    const int size = outw * outh;
    
    Tensor im2col = shrinked.view({-1, 1, size});
    
    im2col_sgemm_conv2d_pack4_impl_x86(im2col, output, kernel_tf, bias);
    
    return output;
}

Tensor conv2d_1x1s2_sgemm_pack4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, otter::ScalarType::Float4);
    
    return conv2d_1x1s2_sgemm_pack4_x86_out(self, weight, weight_o, bias, padding, output);
}

Tensor conv2d_1x1s1_sgemm_pack1to4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
    output.resize_({output_size[0], output_size[1] / 4, output_size[2], output_size[3]});
    
    int inch = self.size(1);
    int outch = output.size(1);
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        convolution_im2col_sgemm_transform_kernel_pack1to4_sse(weight, kernel_tf, inch, outch * 4, 1, 1);
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    int h = input.size(1);
    const int size = w * h;
    
    Tensor im2col = input.view({-1, 1, size});
    
    im2col_sgemm_conv2d_pack1to4_impl_x86(im2col, output, kernel_tf, bias);
    
    return output;
}

Tensor conv2d_1x1s1_sgemm_pack1to4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, otter::ScalarType::Float4);
    
    return conv2d_1x1s1_sgemm_pack1to4_x86_out(self, weight, weight_o, bias, padding, output);
}

Tensor conv2d_1x1s1_sgemm_pack4to1_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
    output.resize_(output_size);
    
    int inch = self.size(1);
    int outch = output.size(1);
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        convolution_im2col_sgemm_transform_kernel_pack4to1_sse(weight, kernel_tf, inch * 4, outch, 1, 1);
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    int h = input.size(1);
    const int size = w * h;
    
    Tensor im2col = input.view({-1, 1, size});
    
    im2col_sgemm_conv2d_pack4to1_impl_x86(im2col, output, kernel_tf, bias);
    
    return output;
}

Tensor conv2d_1x1s1_sgemm_pack4to1_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, otter::ScalarType::Float);
    
    return conv2d_1x1s1_sgemm_pack4to1_x86_out(self, weight, weight_o, bias, padding, output);
}

#endif  // __SSE2__

}   // end namespace otter
