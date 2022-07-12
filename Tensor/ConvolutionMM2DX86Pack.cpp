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
#include "TensorTransform.hpp"

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

void conv3x3s1_winograd63_transform_input_pack4_sse(const Tensor& bottom_blob, Tensor& bottom_blob_tm) {
    const int w = bottom_blob.size(2);
    const int h = bottom_blob.size(1);
    const int inch = bottom_blob.size(0);

    const int w_tiles = (w - 2) / 6;
    const int h_tiles = (h - 2) / 6;
    const int tiles = w_tiles * h_tiles;

    // const float itm[8][8] = {
    //     {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
    //
    //     {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
    //     {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
    //
    //     {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
    //     {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
    //
    //     {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
    //     {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
    //
    //     {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
    // };

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
    
    auto bottom_blob_a = bottom_blob.accessor<float, 3, 4>();
    auto bottom_blob_tm_a = bottom_blob_tm.accessor<float, 3, 4>();

    otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
        for (const auto q : otter::irange(begin, end)) {
            const auto img0 = bottom_blob_a[q];
            auto img0_tm = bottom_blob_tm_a[q];

    #ifdef _MSC_VER
            __declspec(align(16))
    #else
            __attribute__((aligned(16)))
    #endif
            float tmp[8][8][4];

            __m128 _v5_25 = _mm_set1_ps(5.25f);
            __m128 _vm4_25 = _mm_set1_ps(-4.25f);
            __m128 _vm1_25 = _mm_set1_ps(-1.25f);
            __m128 _v0_25 = _mm_set1_ps(0.25f);
            __m128 _vm2_5 = _mm_set1_ps(-2.5f);
            __m128 _v0_5 = _mm_set1_ps(0.5f);
            __m128 _v2 = _mm_set1_ps(2.f);
            __m128 _v4 = _mm_set1_ps(4.f);

            // tile
            for (int i = 0; i < h_tiles; i++)
            {
                for (int j = 0; j < w_tiles; j++)
                {
                    const float* r0 = img0[i * 6].data() + (j * 6) * 4;

                    for (int m = 0; m < 8; m++)
                    {
                        __m128 _r00 = _mm_load_ps(r0);
                        __m128 _r01 = _mm_load_ps(r0 + 4);
                        __m128 _r02 = _mm_load_ps(r0 + 4 * 2);
                        __m128 _r03 = _mm_load_ps(r0 + 4 * 3);
                        __m128 _r04 = _mm_load_ps(r0 + 4 * 4);
                        __m128 _r05 = _mm_load_ps(r0 + 4 * 5);
                        __m128 _r06 = _mm_load_ps(r0 + 4 * 6);
                        __m128 _r07 = _mm_load_ps(r0 + 4 * 7);

                        __m128 _tmp0m = _mm_comp_fmadd_ps(_v5_25, _mm_sub_ps(_r04, _r02), _mm_sub_ps(_r00, _r06));
                        __m128 _tmp7m = _mm_comp_fmadd_ps(_v5_25, _mm_sub_ps(_r03, _r05), _mm_sub_ps(_r07, _r01));
                        _mm_store_ps(tmp[0][m], _tmp0m);
                        _mm_store_ps(tmp[7][m], _tmp7m);

                        __m128 _tmp12a = _mm_comp_fmadd_ps(_vm4_25, _r04, _mm_add_ps(_r02, _r06));
                        __m128 _tmp12b = _mm_comp_fmadd_ps(_vm4_25, _r03, _mm_add_ps(_r01, _r05));

                        __m128 _tmp1m = _mm_add_ps(_tmp12a, _tmp12b);
                        __m128 _tmp2m = _mm_sub_ps(_tmp12a, _tmp12b);
                        _mm_store_ps(tmp[1][m], _tmp1m);
                        _mm_store_ps(tmp[2][m], _tmp2m);

                        __m128 _tmp34a = _mm_comp_fmadd_ps(_vm1_25, _r04, _mm_comp_fmadd_ps(_v0_25, _r02, _r06));
                        __m128 _tmp34b = _mm_comp_fmadd_ps(_v2, _r05, _mm_comp_fmadd_ps(_vm2_5, _r03, _mm_mul_ps(_r01, _v0_5)));

                        __m128 _tmp3m = _mm_add_ps(_tmp34a, _tmp34b);
                        __m128 _tmp4m = _mm_sub_ps(_tmp34a, _tmp34b);
                        _mm_store_ps(tmp[3][m], _tmp3m);
                        _mm_store_ps(tmp[4][m], _tmp4m);

                        __m128 _tmp56a = _mm_comp_fmadd_ps(_v4, _mm_comp_fmadd_ps(_vm1_25, _r04, _r02), _r06);
                        __m128 _tmp56b = _mm_comp_fmadd_ps(_v0_5, _r05, _mm_comp_fmadd_ps(_vm2_5, _r03, _mm_mul_ps(_r01, _v2)));

                        __m128 _tmp5m = _mm_add_ps(_tmp56a, _tmp56b);
                        __m128 _tmp6m = _mm_sub_ps(_tmp56a, _tmp56b);
                        _mm_store_ps(tmp[5][m], _tmp5m);
                        _mm_store_ps(tmp[6][m], _tmp6m);

                        r0 += w * 4;
                    }

                    float* r0_tm_0 = (float*)img0_tm.data() + (i * w_tiles + j) * 4;
                    float* r0_tm_1 = r0_tm_0 + tiles * 4;
                    float* r0_tm_2 = r0_tm_0 + tiles * 4 * 2;
                    float* r0_tm_3 = r0_tm_0 + tiles * 4 * 3;
                    float* r0_tm_4 = r0_tm_0 + tiles * 4 * 4;
                    float* r0_tm_5 = r0_tm_0 + tiles * 4 * 5;
                    float* r0_tm_6 = r0_tm_0 + tiles * 4 * 6;
                    float* r0_tm_7 = r0_tm_0 + tiles * 4 * 7;

                    for (int m = 0; m < 8; m++)
                    {
                        __m128 _tmp00 = _mm_load_ps(tmp[m][0]);
                        __m128 _tmp01 = _mm_load_ps(tmp[m][1]);
                        __m128 _tmp02 = _mm_load_ps(tmp[m][2]);
                        __m128 _tmp03 = _mm_load_ps(tmp[m][3]);
                        __m128 _tmp04 = _mm_load_ps(tmp[m][4]);
                        __m128 _tmp05 = _mm_load_ps(tmp[m][5]);
                        __m128 _tmp06 = _mm_load_ps(tmp[m][6]);
                        __m128 _tmp07 = _mm_load_ps(tmp[m][7]);

                        __m128 _r0tm0 = _mm_comp_fmadd_ps(_v5_25, _mm_sub_ps(_tmp04, _tmp02), _mm_sub_ps(_tmp00, _tmp06));
                        __m128 _r0tm7 = _mm_comp_fmadd_ps(_v5_25, _mm_sub_ps(_tmp03, _tmp05), _mm_sub_ps(_tmp07, _tmp01));

                        __m128 _tmp12a = _mm_comp_fmadd_ps(_vm4_25, _tmp04, _mm_add_ps(_tmp02, _tmp06));
                        __m128 _tmp12b = _mm_comp_fmadd_ps(_vm4_25, _tmp03, _mm_add_ps(_tmp01, _tmp05));

                        __m128 _r0tm1 = _mm_add_ps(_tmp12a, _tmp12b);
                        __m128 _r0tm2 = _mm_sub_ps(_tmp12a, _tmp12b);

                        __m128 _tmp34a = _mm_comp_fmadd_ps(_vm1_25, _tmp04, _mm_comp_fmadd_ps(_v0_25, _tmp02, _tmp06));
                        __m128 _tmp34b = _mm_comp_fmadd_ps(_v2, _tmp05, _mm_comp_fmadd_ps(_vm2_5, _tmp03, _mm_mul_ps(_tmp01, _v0_5)));

                        __m128 _r0tm3 = _mm_add_ps(_tmp34a, _tmp34b);
                        __m128 _r0tm4 = _mm_sub_ps(_tmp34a, _tmp34b);

                        __m128 _tmp56a = _mm_comp_fmadd_ps(_v4, _mm_comp_fmadd_ps(_vm1_25, _tmp04, _tmp02), _tmp06);
                        __m128 _tmp56b = _mm_comp_fmadd_ps(_v0_5, _tmp05, _mm_comp_fmadd_ps(_vm2_5, _tmp03, _mm_mul_ps(_tmp01, _v2)));

                        __m128 _r0tm5 = _mm_add_ps(_tmp56a, _tmp56b);
                        __m128 _r0tm6 = _mm_sub_ps(_tmp56a, _tmp56b);

                        _mm_store_ps(r0_tm_0, _r0tm0);
                        _mm_store_ps(r0_tm_1, _r0tm1);
                        _mm_store_ps(r0_tm_2, _r0tm2);
                        _mm_store_ps(r0_tm_3, _r0tm3);
                        _mm_store_ps(r0_tm_4, _r0tm4);
                        _mm_store_ps(r0_tm_5, _r0tm5);
                        _mm_store_ps(r0_tm_6, _r0tm6);
                        _mm_store_ps(r0_tm_7, _r0tm7);

                        r0_tm_0 += tiles * 4 * 8;
                        r0_tm_1 += tiles * 4 * 8;
                        r0_tm_2 += tiles * 4 * 8;
                        r0_tm_3 += tiles * 4 * 8;
                        r0_tm_4 += tiles * 4 * 8;
                        r0_tm_5 += tiles * 4 * 8;
                        r0_tm_6 += tiles * 4 * 8;
                        r0_tm_7 += tiles * 4 * 8;
                    }
                }
            }
        }
    });
}

void conv3x3s1_winograd63_transform_kernel_pack4_sse(const Tensor& kernel, Tensor& kernel_tm_pack4, int inch, int outch)
{
    // winograd63 transform kernel
    Tensor kernel_tm = otter::empty({outch, inch, 8 * 8}, otter::ScalarType::Float);

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
    
    const float* kernel_ptr = (const float*)kernel.data_ptr();
    auto kernel_tm_a = kernel_tm.accessor<float, 3>();

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            for (int q = 0; q < inch; q++) {
                const float* kernel0 = (const float*)kernel_ptr + p * inch * 9 + q * 9;
                float* kernel_tm0 = kernel_tm_a[p][q].data();

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

    // interleave
    // src = 64-inch-outch
    // dst = pb-pa-inch/pa-64-outch/pb
    kernel_tm_pack4 = otter::empty({outch / 4, 64, inch / 4}, otter::ScalarType::Float16);
    
    auto kernel_tm_pack4_a = kernel_tm_pack4.accessor<float, 3, 16>();

    for (int q = 0; q + 3 < outch; q += 4)
    {
        auto g0 = kernel_tm_pack4_a[q / 4];

        for (int k = 0; k < 64; k++)
        {
            float* g00 = g0[k].data();

            for (int p = 0; p + 3 < inch; p += 4)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const float* k00 = kernel_tm_a[q + j][p + i].data();
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }
}

void conv3x3s1_winograd43_transform_kernel_pack4_sse(const Tensor& kernel, Tensor& kernel_tm_pack4, int inch, int outch)
{
    // winograd43 transform kernel
    Tensor kernel_tm = otter::empty({outch, inch, 6 * 6}, otter::ScalarType::Float);

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
        for (const auto p : otter::irange(begin, end))
        {
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
    // dst = pb-pa-inch/pa-36-outch/pb
    kernel_tm_pack4 = otter::empty({outch / 4, 36, inch / 4}, otter::ScalarType::Float16);
    auto kernel_tm_pack4_a = kernel_tm_pack4.accessor<float, 3, 16>();

    for (int q = 0; q + 3 < outch; q += 4)
    {
        auto g0 = kernel_tm_pack4_a[q / 4];

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0[k].data();

            for (int p = 0; p + 3 < inch; p += 4)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const float* k00 = kernel_tm_a[q + j][p + i].data();
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_kernel_pack4_sse(const Tensor& kernel, Tensor& kernel_tm_pack4, int inch, int outch)
{
    // winograd23 transform kernel
    Tensor kernel_tm = otter::empty({outch, inch, 4 * 4}, otter::ScalarType::Float);

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
    // dst = pb-pa-inch/pa-16-outch/pb
    kernel_tm_pack4 = otter::empty({outch / 4, 16, inch / 4}, otter::ScalarType::Float16);
    auto kernel_tm_pack4_a = kernel_tm_pack4.accessor<float, 3, 16>();

    for (int q = 0; q + 3 < outch; q += 4)
    {
        auto g0 = kernel_tm_pack4_a[q / 4];

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0[k].data();

            for (int p = 0; p + 3 < inch; p += 4)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const float* k00 = kernel_tm_a[q + j][p + i].data();
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }
}

void conv3x3s1_winograd63_transform_output_pack4_sse(const Tensor& top_blob_tm, Tensor& top_blob, const Tensor& bias) {
    const int outw = top_blob.size(2);
    const int outh = top_blob.size(1);
    const int outch = top_blob.size(0);

    const int w_tiles = outw / 6;
    const int h_tiles = outh / 6;
    const int tiles = w_tiles * h_tiles;

    const float* biasptr = (bias.defined()) ? bias.data_ptr<float>() : nullptr;

    // const float otm[6][8] = {
    //     {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
    //     {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
    //     {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
    //     {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
    //     {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
    //     {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
    // };

    // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
    // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
    // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
    // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
    // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
    // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)
    
    auto top_blob_a = top_blob.accessor<float, 3, 4>();
    auto top_blob_tm_a = top_blob_tm.accessor<float, 3, 4>();

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end))
        {
            const auto out0_tm = top_blob_tm_a[p];
            auto out0 = top_blob_a[p];

            __m128 _bias0 = biasptr ? _mm_loadu_ps(biasptr + p * 4) : _mm_setzero_ps();

    #ifdef _MSC_VER
            __declspec(align(16))
    #else
            __attribute__((aligned(16)))
    #endif
            float tmp[6][8][4];

            __m128 _v32 = _mm_set1_ps(32.f);
            __m128 _v16 = _mm_set1_ps(16.f);
            __m128 _v8 = _mm_set1_ps(8.f);
            __m128 _v4 = _mm_set1_ps(4.f);
            __m128 _v2 = _mm_set1_ps(2.f);

            // tile
            for (int i = 0; i < h_tiles; i++)
            {
                for (int j = 0; j < w_tiles; j++)
                {
                    const float* output0_tm_0 = (const float*)out0_tm.data() + (i * w_tiles + j) * 4;
                    const float* output0_tm_1 = output0_tm_0 + tiles * 4;
                    const float* output0_tm_2 = output0_tm_0 + tiles * 4 * 2;
                    const float* output0_tm_3 = output0_tm_0 + tiles * 4 * 3;
                    const float* output0_tm_4 = output0_tm_0 + tiles * 4 * 4;
                    const float* output0_tm_5 = output0_tm_0 + tiles * 4 * 5;
                    const float* output0_tm_6 = output0_tm_0 + tiles * 4 * 6;
                    const float* output0_tm_7 = output0_tm_0 + tiles * 4 * 7;

                    float* output0 = out0[i * 6].data() + (j * 6) * 4;

                    for (int m = 0; m < 8; m++)
                    {
                        __m128 _out0tm0 = _mm_load_ps(output0_tm_0);
                        __m128 _out0tm1 = _mm_load_ps(output0_tm_1);
                        __m128 _out0tm2 = _mm_load_ps(output0_tm_2);
                        __m128 _out0tm3 = _mm_load_ps(output0_tm_3);
                        __m128 _out0tm4 = _mm_load_ps(output0_tm_4);
                        __m128 _out0tm5 = _mm_load_ps(output0_tm_5);
                        __m128 _out0tm6 = _mm_load_ps(output0_tm_6);
                        __m128 _out0tm7 = _mm_load_ps(output0_tm_7);

                        __m128 _tmp024a = _mm_add_ps(_out0tm1, _out0tm2);
                        __m128 _tmp135a = _mm_sub_ps(_out0tm1, _out0tm2);

                        __m128 _tmp024b = _mm_add_ps(_out0tm3, _out0tm4);
                        __m128 _tmp135b = _mm_sub_ps(_out0tm3, _out0tm4);

                        __m128 _tmp024c = _mm_add_ps(_out0tm5, _out0tm6);
                        __m128 _tmp135c = _mm_sub_ps(_out0tm5, _out0tm6);

                        __m128 _tmp0m = _mm_add_ps(_mm_add_ps(_out0tm0, _tmp024a), _mm_comp_fmadd_ps(_v32, _tmp024c, _tmp024b));
                        __m128 _tmp2m = _mm_comp_fmadd_ps(_v8, _tmp024c, _mm_comp_fmadd_ps(_v4, _tmp024b, _tmp024a));
                        __m128 _tmp4m = _mm_comp_fmadd_ps(_v2, _tmp024c, _mm_comp_fmadd_ps(_v16, _tmp024b, _tmp024a));
                        _mm_store_ps(tmp[0][m], _tmp0m);
                        _mm_store_ps(tmp[2][m], _tmp2m);
                        _mm_store_ps(tmp[4][m], _tmp4m);

                        __m128 _tmp1m = _mm_comp_fmadd_ps(_v16, _tmp135c, _mm_comp_fmadd_ps(_v2, _tmp135b, _tmp135a));
                        __m128 _tmp3m = _mm_comp_fmadd_ps(_v4, _tmp135c, _mm_comp_fmadd_ps(_v8, _tmp135b, _tmp135a));
                        __m128 _tmp5m = _mm_add_ps(_mm_add_ps(_out0tm7, _tmp135a), _mm_comp_fmadd_ps(_v32, _tmp135b, _tmp135c));
                        _mm_store_ps(tmp[1][m], _tmp1m);
                        _mm_store_ps(tmp[3][m], _tmp3m);
                        _mm_store_ps(tmp[5][m], _tmp5m);

                        output0_tm_0 += tiles * 4 * 8;
                        output0_tm_1 += tiles * 4 * 8;
                        output0_tm_2 += tiles * 4 * 8;
                        output0_tm_3 += tiles * 4 * 8;
                        output0_tm_4 += tiles * 4 * 8;
                        output0_tm_5 += tiles * 4 * 8;
                        output0_tm_6 += tiles * 4 * 8;
                        output0_tm_7 += tiles * 4 * 8;
                    }

                    for (int m = 0; m < 6; m++)
                    {
                        __m128 _tmp00 = _mm_load_ps(tmp[m][0]);
                        __m128 _tmp01 = _mm_load_ps(tmp[m][1]);
                        __m128 _tmp02 = _mm_load_ps(tmp[m][2]);
                        __m128 _tmp03 = _mm_load_ps(tmp[m][3]);
                        __m128 _tmp04 = _mm_load_ps(tmp[m][4]);
                        __m128 _tmp05 = _mm_load_ps(tmp[m][5]);
                        __m128 _tmp06 = _mm_load_ps(tmp[m][6]);
                        __m128 _tmp07 = _mm_load_ps(tmp[m][7]);

                        __m128 _tmp024a = _mm_add_ps(_tmp01, _tmp02);
                        __m128 _tmp135a = _mm_sub_ps(_tmp01, _tmp02);

                        __m128 _tmp024b = _mm_add_ps(_tmp03, _tmp04);
                        __m128 _tmp135b = _mm_sub_ps(_tmp03, _tmp04);

                        __m128 _tmp024c = _mm_add_ps(_tmp05, _tmp06);
                        __m128 _tmp135c = _mm_sub_ps(_tmp05, _tmp06);

                        __m128 _out00 = _mm_add_ps(_bias0, _mm_add_ps(_mm_add_ps(_tmp00, _tmp024a), _mm_comp_fmadd_ps(_v32, _tmp024c, _tmp024b)));
                        __m128 _out02 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v8, _tmp024c, _mm_comp_fmadd_ps(_v4, _tmp024b, _tmp024a)));
                        __m128 _out04 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v2, _tmp024c, _mm_comp_fmadd_ps(_v16, _tmp024b, _tmp024a)));
                        _mm_store_ps(output0, _out00);
                        _mm_store_ps(output0 + 4 * 2, _out02);
                        _mm_store_ps(output0 + 4 * 4, _out04);

                        __m128 _out01 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v16, _tmp135c, _mm_comp_fmadd_ps(_v2, _tmp135b, _tmp135a)));
                        __m128 _out03 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v4, _tmp135c, _mm_comp_fmadd_ps(_v8, _tmp135b, _tmp135a)));
                        __m128 _out05 = _mm_add_ps(_bias0, _mm_add_ps(_mm_add_ps(_tmp07, _tmp135a), _mm_comp_fmadd_ps(_v32, _tmp135b, _tmp135c)));
                        _mm_store_ps(output0 + 4, _out01);
                        _mm_store_ps(output0 + 4 * 3, _out03);
                        _mm_store_ps(output0 + 4 * 5, _out05);

                        output0 += outw * 4;
                    }
                }
            }
        }
    });
}

void conv3x3s1_winograd43_transform_input_pack4_sse(const Tensor& bottom_blob, Tensor& bottom_blob_tm) {
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
    
    auto bottom_blob_a = bottom_blob.accessor<float, 3, 4>();
    auto bottom_blob_tm_a = bottom_blob_tm.accessor<float, 3, 4>();

    otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
        for (const auto q : otter::irange(begin, end))
        {
            const auto img0 = bottom_blob_a[q];
            auto img0_tm = bottom_blob_tm_a[q];

    #ifdef _MSC_VER
            __declspec(align(16))
    #else
            __attribute__((aligned(16)))
    #endif
            float tmp[6][6][4];

            __m128 _vm5 = _mm_set1_ps(-5.f);
            __m128 _vm4 = _mm_set1_ps(-4.f);
            __m128 _v4 = _mm_set1_ps(4.f);
            __m128 _vm2 = _mm_set1_ps(-2.f);
            __m128 _v2 = _mm_set1_ps(2.f);

            // tile
            for (int i = 0; i < h_tiles; i++)
            {
                for (int j = 0; j < w_tiles; j++)
                {
                    const float* r0 = img0[i * 4].data() + (j * 4) * 4;

                    for (int m = 0; m < 6; m++)
                    {
                        __m128 _r00 = _mm_load_ps(r0);
                        __m128 _r01 = _mm_load_ps(r0 + 4);
                        __m128 _r02 = _mm_load_ps(r0 + 4 * 2);
                        __m128 _r03 = _mm_load_ps(r0 + 4 * 3);
                        __m128 _r04 = _mm_load_ps(r0 + 4 * 4);
                        __m128 _r05 = _mm_load_ps(r0 + 4 * 5);

                        __m128 _tmp0m = _mm_comp_fmadd_ps(_vm5, _r02, _mm_comp_fmadd_ps(_v4, _r00, _r04));
                        __m128 _tmp1m = _mm_comp_fmadd_ps(_vm4, _mm_add_ps(_r01, _r02), _mm_add_ps(_r04, _r03));
                        __m128 _tmp2m = _mm_comp_fmadd_ps(_v4, _mm_sub_ps(_r01, _r02), _mm_sub_ps(_r04, _r03));
                        __m128 _tmp3m = _mm_comp_fmadd_ps(_vm2, _mm_sub_ps(_r01, _r03), _mm_sub_ps(_r04, _r02));
                        __m128 _tmp4m = _mm_comp_fmadd_ps(_v2, _mm_sub_ps(_r01, _r03), _mm_sub_ps(_r04, _r02));
                        __m128 _tmp5m = _mm_comp_fmadd_ps(_vm5, _r03, _mm_comp_fmadd_ps(_v4, _r01, _r05));

                        _mm_store_ps(tmp[0][m], _tmp0m);
                        _mm_store_ps(tmp[1][m], _tmp1m);
                        _mm_store_ps(tmp[2][m], _tmp2m);
                        _mm_store_ps(tmp[3][m], _tmp3m);
                        _mm_store_ps(tmp[4][m], _tmp4m);
                        _mm_store_ps(tmp[5][m], _tmp5m);

                        r0 += w * 4;
                    }

                    float* r0_tm_0 = (float*)img0_tm.data() + (i * w_tiles + j) * 4;
                    float* r0_tm_1 = r0_tm_0 + tiles * 4;
                    float* r0_tm_2 = r0_tm_0 + tiles * 4 * 2;
                    float* r0_tm_3 = r0_tm_0 + tiles * 4 * 3;
                    float* r0_tm_4 = r0_tm_0 + tiles * 4 * 4;
                    float* r0_tm_5 = r0_tm_0 + tiles * 4 * 5;

                    for (int m = 0; m < 6; m++)
                    {
                        __m128 _tmp00 = _mm_load_ps(tmp[m][0]);
                        __m128 _tmp01 = _mm_load_ps(tmp[m][1]);
                        __m128 _tmp02 = _mm_load_ps(tmp[m][2]);
                        __m128 _tmp03 = _mm_load_ps(tmp[m][3]);
                        __m128 _tmp04 = _mm_load_ps(tmp[m][4]);
                        __m128 _tmp05 = _mm_load_ps(tmp[m][5]);

                        __m128 _r0tm0 = _mm_comp_fmadd_ps(_vm5, _tmp02, _mm_comp_fmadd_ps(_v4, _tmp00, _tmp04));
                        __m128 _r0tm1 = _mm_comp_fmadd_ps(_vm4, _mm_add_ps(_tmp01, _tmp02), _mm_add_ps(_tmp04, _tmp03));
                        __m128 _r0tm2 = _mm_comp_fmadd_ps(_v4, _mm_sub_ps(_tmp01, _tmp02), _mm_sub_ps(_tmp04, _tmp03));
                        __m128 _r0tm3 = _mm_comp_fmadd_ps(_vm2, _mm_sub_ps(_tmp01, _tmp03), _mm_sub_ps(_tmp04, _tmp02));
                        __m128 _r0tm4 = _mm_comp_fmadd_ps(_v2, _mm_sub_ps(_tmp01, _tmp03), _mm_sub_ps(_tmp04, _tmp02));
                        __m128 _r0tm5 = _mm_comp_fmadd_ps(_vm5, _tmp03, _mm_comp_fmadd_ps(_v4, _tmp01, _tmp05));

                        _mm_store_ps(r0_tm_0, _r0tm0);
                        _mm_store_ps(r0_tm_1, _r0tm1);
                        _mm_store_ps(r0_tm_2, _r0tm2);
                        _mm_store_ps(r0_tm_3, _r0tm3);
                        _mm_store_ps(r0_tm_4, _r0tm4);
                        _mm_store_ps(r0_tm_5, _r0tm5);

                        r0_tm_0 += tiles * 4 * 6;
                        r0_tm_1 += tiles * 4 * 6;
                        r0_tm_2 += tiles * 4 * 6;
                        r0_tm_3 += tiles * 4 * 6;
                        r0_tm_4 += tiles * 4 * 6;
                        r0_tm_5 += tiles * 4 * 6;
                    }
                }
            }
        }
    });
}

void conv3x3s1_winograd43_transform_output_pack4_sse(const Tensor& top_blob_tm, Tensor& top_blob, const Tensor& bias) {
    const int outw = top_blob.size(2);
    const int outh = top_blob.size(1);
    const int outch = top_blob.size(0);

    const int w_tiles = outw / 4;
    const int h_tiles = outh / 4;
    const int tiles = w_tiles * h_tiles;

    const float* biasptr = (bias.defined()) ? bias.data_ptr<float>() : nullptr;

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
    
    auto top_blob_a = top_blob.accessor<float, 3, 4>();
    auto top_blob_tm_a = top_blob_tm.accessor<float, 3, 4>();

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            const auto out0_tm = top_blob_tm_a[p];
            auto out0 = top_blob_a[p];

            __m128 _bias0 = biasptr ? _mm_loadu_ps(biasptr + p * 4) : _mm_setzero_ps();

    #ifdef _MSC_VER
            __declspec(align(16))
    #else
            __attribute__((aligned(16)))
    #endif
            float tmp[4][6][4];

            __m128 _v2 = _mm_set1_ps(2.f);
            __m128 _v4 = _mm_set1_ps(4.f);
            __m128 _v8 = _mm_set1_ps(8.f);

            // tile
            for (int i = 0; i < h_tiles; i++)
            {
                for (int j = 0; j < w_tiles; j++)
                {
                    const float* output0_tm_0 = (const float*)out0_tm.data() + (i * w_tiles + j) * 4;
                    const float* output0_tm_1 = output0_tm_0 + tiles * 4;
                    const float* output0_tm_2 = output0_tm_0 + tiles * 4 * 2;
                    const float* output0_tm_3 = output0_tm_0 + tiles * 4 * 3;
                    const float* output0_tm_4 = output0_tm_0 + tiles * 4 * 4;
                    const float* output0_tm_5 = output0_tm_0 + tiles * 4 * 5;

                    float* output0 = out0[i * 4].data() + (j * 4) * 4;

                    for (int m = 0; m < 6; m++)
                    {
                        __m128 _out0tm0 = _mm_load_ps(output0_tm_0);
                        __m128 _out0tm1 = _mm_load_ps(output0_tm_1);
                        __m128 _out0tm2 = _mm_load_ps(output0_tm_2);
                        __m128 _out0tm3 = _mm_load_ps(output0_tm_3);
                        __m128 _out0tm4 = _mm_load_ps(output0_tm_4);
                        __m128 _out0tm5 = _mm_load_ps(output0_tm_5);

                        __m128 _tmp02a = _mm_add_ps(_out0tm1, _out0tm2);
                        __m128 _tmp13a = _mm_sub_ps(_out0tm1, _out0tm2);

                        __m128 _tmp02b = _mm_add_ps(_out0tm3, _out0tm4);
                        __m128 _tmp13b = _mm_sub_ps(_out0tm3, _out0tm4);

                        __m128 _tmp0m = _mm_add_ps(_mm_add_ps(_out0tm0, _tmp02a), _tmp02b);
                        __m128 _tmp1m = _mm_comp_fmadd_ps(_v2, _tmp13b, _tmp13a);
                        __m128 _tmp2m = _mm_comp_fmadd_ps(_v4, _tmp02b, _tmp02a);
                        __m128 _tmp3m = _mm_comp_fmadd_ps(_v8, _tmp13b, _mm_add_ps(_out0tm5, _tmp13a));

                        _mm_store_ps(tmp[0][m], _tmp0m);
                        _mm_store_ps(tmp[1][m], _tmp1m);
                        _mm_store_ps(tmp[2][m], _tmp2m);
                        _mm_store_ps(tmp[3][m], _tmp3m);

                        output0_tm_0 += tiles * 4 * 6;
                        output0_tm_1 += tiles * 4 * 6;
                        output0_tm_2 += tiles * 4 * 6;
                        output0_tm_3 += tiles * 4 * 6;
                        output0_tm_4 += tiles * 4 * 6;
                        output0_tm_5 += tiles * 4 * 6;
                    }

                    for (int m = 0; m < 4; m++)
                    {
                        __m128 _tmp00 = _mm_load_ps(tmp[m][0]);
                        __m128 _tmp01 = _mm_load_ps(tmp[m][1]);
                        __m128 _tmp02 = _mm_load_ps(tmp[m][2]);
                        __m128 _tmp03 = _mm_load_ps(tmp[m][3]);
                        __m128 _tmp04 = _mm_load_ps(tmp[m][4]);
                        __m128 _tmp05 = _mm_load_ps(tmp[m][5]);

                        __m128 _tmp02a = _mm_add_ps(_tmp01, _tmp02);
                        __m128 _tmp13a = _mm_sub_ps(_tmp01, _tmp02);

                        __m128 _tmp02b = _mm_add_ps(_tmp03, _tmp04);
                        __m128 _tmp13b = _mm_sub_ps(_tmp03, _tmp04);

                        __m128 _out00 = _mm_add_ps(_bias0, _mm_add_ps(_mm_add_ps(_tmp00, _tmp02a), _tmp02b));
                        __m128 _out01 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v2, _tmp13b, _tmp13a));
                        __m128 _out02 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v4, _tmp02b, _tmp02a));
                        __m128 _out03 = _mm_add_ps(_bias0, _mm_comp_fmadd_ps(_v8, _tmp13b, _mm_add_ps(_tmp05, _tmp13a)));

                        _mm_store_ps(output0, _out00);
                        _mm_store_ps(output0 + 4, _out01);
                        _mm_store_ps(output0 + 4 * 2, _out02);
                        _mm_store_ps(output0 + 4 * 3, _out03);

                        output0 += outw * 4;
                    }
                }
            }
        }
    });
}

void conv3x3s1_winograd23_transform_input_pack4_sse(const Tensor& bottom_blob, Tensor& bottom_blob_tm) {
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
    
    auto bottom_blob_a = bottom_blob.accessor<float, 3, 4>();
    auto bottom_blob_tm_a = bottom_blob_tm.accessor<float, 3, 4>();

    otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
        for (const auto q : otter::irange(begin, end)) {
            const auto img0 = bottom_blob_a[q];
            auto img0_tm = bottom_blob_tm_a[q];

    #ifdef _MSC_VER
            __declspec(align(16))
    #else
            __attribute__((aligned(16)))
    #endif
            float tmp[4][4][4];

            // tile
            for (int i = 0; i < h_tiles; i++)
            {
                for (int j = 0; j < w_tiles; j++)
                {
                    const float* r0 = img0[i * 2].data() + (j * 2) * 4;

                    for (int m = 0; m < 4; m++)
                    {
                        __m128 _r00 = _mm_load_ps(r0);
                        __m128 _r01 = _mm_load_ps(r0 + 4);
                        __m128 _r02 = _mm_load_ps(r0 + 4 * 2);
                        __m128 _r03 = _mm_load_ps(r0 + 4 * 3);

                        __m128 _tmp0m = _mm_sub_ps(_r00, _r02);
                        __m128 _tmp1m = _mm_add_ps(_r01, _r02);
                        __m128 _tmp2m = _mm_sub_ps(_r02, _r01);
                        __m128 _tmp3m = _mm_sub_ps(_r03, _r01);

                        _mm_store_ps(tmp[0][m], _tmp0m);
                        _mm_store_ps(tmp[1][m], _tmp1m);
                        _mm_store_ps(tmp[2][m], _tmp2m);
                        _mm_store_ps(tmp[3][m], _tmp3m);

                        r0 += w * 4;
                    }

                    float* r0_tm_0 = (float*)img0_tm.data() + (i * w_tiles + j) * 4;
                    float* r0_tm_1 = r0_tm_0 + tiles * 4;
                    float* r0_tm_2 = r0_tm_0 + tiles * 4 * 2;
                    float* r0_tm_3 = r0_tm_0 + tiles * 4 * 3;

                    for (int m = 0; m < 4; m++)
                    {
                        __m128 _tmp00 = _mm_load_ps(tmp[m][0]);
                        __m128 _tmp01 = _mm_load_ps(tmp[m][1]);
                        __m128 _tmp02 = _mm_load_ps(tmp[m][2]);
                        __m128 _tmp03 = _mm_load_ps(tmp[m][3]);

                        __m128 _r0tm0 = _mm_sub_ps(_tmp00, _tmp02);
                        __m128 _r0tm1 = _mm_add_ps(_tmp01, _tmp02);
                        __m128 _r0tm2 = _mm_sub_ps(_tmp02, _tmp01);
                        __m128 _r0tm3 = _mm_sub_ps(_tmp03, _tmp01);

                        _mm_store_ps(r0_tm_0, _r0tm0);
                        _mm_store_ps(r0_tm_1, _r0tm1);
                        _mm_store_ps(r0_tm_2, _r0tm2);
                        _mm_store_ps(r0_tm_3, _r0tm3);

                        r0_tm_0 += tiles * 4 * 4;
                        r0_tm_1 += tiles * 4 * 4;
                        r0_tm_2 += tiles * 4 * 4;
                        r0_tm_3 += tiles * 4 * 4;
                    }
                }
            }
        }
    });
}

void conv3x3s1_winograd23_transform_output_pack4_sse(const Tensor& top_blob_tm, Tensor& top_blob, const Tensor& bias) {
    const int outw = top_blob.size(2);
    const int outh = top_blob.size(1);
    const int outch = top_blob.size(0);

    const int w_tiles = outw / 2;
    const int h_tiles = outh / 2;
    const int tiles = w_tiles * h_tiles;

    const float* biasptr = (bias.defined()) ? bias.data_ptr<float>() : nullptr;

    // const float otm[2][4] = {
    //     {1.0f,  1.0f,  1.0f,  0.0f},
    //     {0.0f,  1.0f, -1.0f,  1.0f}
    // };

    // 0 = r00 + r01 + r02
    // 1 = r01 - r02 + r03
    
    auto top_blob_a = top_blob.accessor<float, 3, 4>();
    auto top_blob_tm_a = top_blob_tm.accessor<float, 3, 4>();

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            const auto out0_tm = top_blob_tm_a[p];
            auto out0 = top_blob_a[p];

            __m128 _bias0 = biasptr ? _mm_loadu_ps(biasptr + p * 4) : _mm_setzero_ps();

    #ifdef _MSC_VER
            __declspec(align(16))
    #else
            __attribute__((aligned(16)))
    #endif
            float tmp[2][4][4];

            // tile
            for (int i = 0; i < h_tiles; i++)
            {
                for (int j = 0; j < w_tiles; j++)
                {
                    const float* output0_tm_0 = (const float*)out0_tm.data() + (i * w_tiles + j) * 4;
                    const float* output0_tm_1 = output0_tm_0 + tiles * 4;
                    const float* output0_tm_2 = output0_tm_0 + tiles * 4 * 2;
                    const float* output0_tm_3 = output0_tm_0 + tiles * 4 * 3;

                    float* output0 = out0[i * 2].data() + (j * 2) * 4;

                    for (int m = 0; m < 4; m++)
                    {
                        __m128 _out0tm0 = _mm_load_ps(output0_tm_0);
                        __m128 _out0tm1 = _mm_load_ps(output0_tm_1);
                        __m128 _out0tm2 = _mm_load_ps(output0_tm_2);
                        __m128 _out0tm3 = _mm_load_ps(output0_tm_3);

                        __m128 _tmp0m = _mm_add_ps(_mm_add_ps(_out0tm0, _out0tm1), _out0tm2);
                        __m128 _tmp1m = _mm_add_ps(_mm_sub_ps(_out0tm1, _out0tm2), _out0tm3);

                        _mm_store_ps(tmp[0][m], _tmp0m);
                        _mm_store_ps(tmp[1][m], _tmp1m);

                        output0_tm_0 += tiles * 4 * 4;
                        output0_tm_1 += tiles * 4 * 4;
                        output0_tm_2 += tiles * 4 * 4;
                        output0_tm_3 += tiles * 4 * 4;
                    }

                    for (int m = 0; m < 2; m++)
                    {
                        __m128 _tmp00 = _mm_load_ps(tmp[m][0]);
                        __m128 _tmp01 = _mm_load_ps(tmp[m][1]);
                        __m128 _tmp02 = _mm_load_ps(tmp[m][2]);
                        __m128 _tmp03 = _mm_load_ps(tmp[m][3]);

                        __m128 _out00 = _mm_add_ps(_bias0, _mm_add_ps(_mm_add_ps(_tmp00, _tmp01), _tmp02));
                        __m128 _out01 = _mm_add_ps(_bias0, _mm_add_ps(_mm_sub_ps(_tmp01, _tmp02), _tmp03));

                        _mm_store_ps(output0, _out00);
                        _mm_store_ps(output0 + 4, _out01);

                        output0 += outw * 4;
                    }
                }
            }
        }
    });
}

void convolution_winograd_dot_pack4_sse(Tensor& bottom_blob_tm, int outch, const Tensor& kernel_tm, Tensor& top_blob_tm) {
    // Tensor bottom_blob_tm(tiles, 16/36/64, inch, 16u, 4, opt.workspace_allocator);

    const int tiles = bottom_blob_tm.size(2);
    const int batch = bottom_blob_tm.size(1);
    const int inch = bottom_blob_tm.size(0);

    // permute
    Tensor bottom_blob_tm2;
    if (tiles >= 12)
        bottom_blob_tm2 = otter::empty({batch, tiles / 12 + (tiles % 12) / 8 + (tiles % 12 % 8) / 4 + (tiles % 12 % 4) / 2 + tiles % 12 % 2, 12 * inch}, otter::ScalarType::Float4);
    else if (tiles >= 8)
        bottom_blob_tm2 = otter::empty({batch, tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2, 8 * inch}, otter::ScalarType::Float4);
    else if (tiles >= 4)
        bottom_blob_tm2 = otter::empty({batch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, 4 * inch}, otter::ScalarType::Float4);
    else if (tiles >= 2)
        bottom_blob_tm2 = otter::empty({batch, tiles / 2 + tiles % 2, 2 * inch}, otter::ScalarType::Float4);
    else // if (tiles >= 1)
        bottom_blob_tm2 = otter::empty({batch, tiles, 1 * inch}, otter::ScalarType::Float4);

    auto bottom_blob_tm_a = bottom_blob_tm.accessor<float, 3, 4>();
    auto bottom_blob_tm2_a = bottom_blob_tm2.accessor<float, 3, 4>();
    
    int bottom_blob_tm_cstep = tiles * batch;
    
    otter::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
        for (const auto r : otter::irange(begin, end)) {
            auto tm2 = bottom_blob_tm2_a[r];

            // tile
            int i = 0;
            for (; i + 11 < tiles; i += 12)
            {
                float* tmpptr = tm2[i / 12].data();

                const float* r0 = bottom_blob_tm_a.data();

                r0 += (r * tiles + i) * 4;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 4x12
                    __m128 _r0 = _mm_load_ps(r0);
                    __m128 _r1 = _mm_load_ps(r0 + 4);
                    __m128 _r2 = _mm_load_ps(r0 + 4 * 2);
                    __m128 _r3 = _mm_load_ps(r0 + 4 * 3);
                    __m128 _r4 = _mm_load_ps(r0 + 4 * 4);
                    __m128 _r5 = _mm_load_ps(r0 + 4 * 5);
                    __m128 _r6 = _mm_load_ps(r0 + 4 * 6);
                    __m128 _r7 = _mm_load_ps(r0 + 4 * 7);
                    __m128 _r8 = _mm_load_ps(r0 + 4 * 8);
                    __m128 _r9 = _mm_load_ps(r0 + 4 * 9);
                    __m128 _ra = _mm_load_ps(r0 + 4 * 10);
                    __m128 _rb = _mm_load_ps(r0 + 4 * 11);

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

                    r0 += bottom_blob_tm_cstep * 4;
                    tmpptr += 48;
                }
            }
            for (; i + 7 < tiles; i += 8)
            {
                float* tmpptr = tm2[i / 12 + (i % 12) / 8].data();

                const float* r0 = bottom_blob_tm_a.data();

                r0 += (r * tiles + i) * 4;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 4x8
                    __m128 _r0 = _mm_load_ps(r0);
                    __m128 _r1 = _mm_load_ps(r0 + 4);
                    __m128 _r2 = _mm_load_ps(r0 + 4 * 2);
                    __m128 _r3 = _mm_load_ps(r0 + 4 * 3);
                    __m128 _r4 = _mm_load_ps(r0 + 4 * 4);
                    __m128 _r5 = _mm_load_ps(r0 + 4 * 5);
                    __m128 _r6 = _mm_load_ps(r0 + 4 * 6);
                    __m128 _r7 = _mm_load_ps(r0 + 4 * 7);

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

                    r0 += bottom_blob_tm_cstep * 4;
                    tmpptr += 32;
                }
            }
            for (; i + 3 < tiles; i += 4)
            {
                float* tmpptr = tm2[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4].data();

                const float* r0 = bottom_blob_tm_a.data();

                r0 += (r * tiles + i) * 4;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 4x4
                    __m128 _r0 = _mm_load_ps(r0);
                    __m128 _r1 = _mm_load_ps(r0 + 4);
                    __m128 _r2 = _mm_load_ps(r0 + 4 * 2);
                    __m128 _r3 = _mm_load_ps(r0 + 4 * 3);

                    _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

                    _mm_store_ps(tmpptr, _r0);
                    _mm_store_ps(tmpptr + 4, _r1);
                    _mm_store_ps(tmpptr + 4 * 2, _r2);
                    _mm_store_ps(tmpptr + 4 * 3, _r3);

                    r0 += bottom_blob_tm_cstep * 4;
                    tmpptr += 16;
                }
            }
            for (; i + 1 < tiles; i += 2)
            {
                float* tmpptr = tm2[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2].data();

                const float* r0 = bottom_blob_tm_a.data();

                r0 += (r * tiles + i) * 4;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 4x2
                    __m128 _r0 = _mm_load_ps(r0);
                    __m128 _r1 = _mm_load_ps(r0 + 4);

                    __m128 _r01_0 = _mm_unpacklo_ps(_r0, _r1);
                    __m128 _r01_1 = _mm_unpackhi_ps(_r0, _r1);

                    _mm_store_ps(tmpptr, _r01_0);
                    _mm_store_ps(tmpptr + 4, _r01_1);

                    r0 += bottom_blob_tm_cstep * 4;
                    tmpptr += 8;
                }
            }
            for (; i < tiles; i++)
            {
                float* tmpptr = tm2[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2].data();

                const float* r0 = bottom_blob_tm_a.data();

                r0 += (r * tiles + i) * 4;

                for (int q = 0; q < inch; q++)
                {
                    __m128 _val = _mm_load_ps(r0);
                    _mm_store_ps(tmpptr, _val);

                    r0 += bottom_blob_tm_cstep * 4;
                    tmpptr += 4;
                }
            }
        }
    });

    bottom_blob_tm.reset();
    // permute end

    top_blob_tm = otter::empty({outch, batch, tiles}, otter::ScalarType::Float4);
    auto top_blob_tm_a = top_blob_tm.accessor<float, 3, 4>();
    auto kernel_tm_a = kernel_tm.accessor<float, 3, 16>();

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end))
        {
            float* output0_tm = top_blob_tm_a[p].data();

            const auto kernel0_tm = kernel_tm_a[p];

            for (int r = 0; r < batch; r++)
            {
                const auto bb2 = bottom_blob_tm2_a[r];

                int i = 0;
                for (; i + 11 < tiles; i += 12)
                {
                    const float* r0 = bb2[i / 12].data();
                    const float* k0 = kernel0_tm[r].data();

                    int nn = inch * 4; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();
                    __m128 _sum4 = _mm_setzero_ps();
                    __m128 _sum5 = _mm_setzero_ps();
                    __m128 _sum6 = _mm_setzero_ps();
                    __m128 _sum7 = _mm_setzero_ps();
                    __m128 _sum8 = _mm_setzero_ps();
                    __m128 _sum9 = _mm_setzero_ps();
                    __m128 _suma = _mm_setzero_ps();
                    __m128 _sumb = _mm_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m128 _w0 = _mm_load_ps(k0);

                        __m128 _val0 = _mm_load1_ps(r0);
                        __m128 _val1 = _mm_load1_ps(r0 + 1);
                        __m128 _val2 = _mm_load1_ps(r0 + 2);
                        __m128 _val3 = _mm_load1_ps(r0 + 3);
                        __m128 _val4 = _mm_load1_ps(r0 + 4);
                        __m128 _val5 = _mm_load1_ps(r0 + 5);
                        __m128 _val6 = _mm_load1_ps(r0 + 6);
                        __m128 _val7 = _mm_load1_ps(r0 + 7);
                        __m128 _val8 = _mm_load1_ps(r0 + 8);
                        __m128 _val9 = _mm_load1_ps(r0 + 9);
                        __m128 _vala = _mm_load1_ps(r0 + 10);
                        __m128 _valb = _mm_load1_ps(r0 + 11);

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

                        r0 += 12;
                        k0 += 4;
                    }

                    _mm_store_ps(output0_tm, _sum0);
                    _mm_store_ps(output0_tm + 4, _sum1);
                    _mm_store_ps(output0_tm + 4 * 2, _sum2);
                    _mm_store_ps(output0_tm + 4 * 3, _sum3);
                    _mm_store_ps(output0_tm + 4 * 4, _sum4);
                    _mm_store_ps(output0_tm + 4 * 5, _sum5);
                    _mm_store_ps(output0_tm + 4 * 6, _sum6);
                    _mm_store_ps(output0_tm + 4 * 7, _sum7);
                    _mm_store_ps(output0_tm + 4 * 8, _sum8);
                    _mm_store_ps(output0_tm + 4 * 9, _sum9);
                    _mm_store_ps(output0_tm + 4 * 10, _suma);
                    _mm_store_ps(output0_tm + 4 * 11, _sumb);

                    output0_tm += 4 * 12;
                }
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2[i / 12 + (i % 12) / 8].data();
                    const float* k0 = kernel0_tm[r].data();

                    int nn = inch * 4; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();
                    __m128 _sum4 = _mm_setzero_ps();
                    __m128 _sum5 = _mm_setzero_ps();
                    __m128 _sum6 = _mm_setzero_ps();
                    __m128 _sum7 = _mm_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m128 _w0 = _mm_load_ps(k0);

                        __m128 _val0 = _mm_load1_ps(r0);
                        __m128 _val1 = _mm_load1_ps(r0 + 1);
                        __m128 _val2 = _mm_load1_ps(r0 + 2);
                        __m128 _val3 = _mm_load1_ps(r0 + 3);
                        __m128 _val4 = _mm_load1_ps(r0 + 4);
                        __m128 _val5 = _mm_load1_ps(r0 + 5);
                        __m128 _val6 = _mm_load1_ps(r0 + 6);
                        __m128 _val7 = _mm_load1_ps(r0 + 7);

                        _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val3, _w0, _sum3);
                        _sum4 = _mm_comp_fmadd_ps(_val4, _w0, _sum4);
                        _sum5 = _mm_comp_fmadd_ps(_val5, _w0, _sum5);
                        _sum6 = _mm_comp_fmadd_ps(_val6, _w0, _sum6);
                        _sum7 = _mm_comp_fmadd_ps(_val7, _w0, _sum7);

                        r0 += 8;
                        k0 += 4;
                    }

                    _mm_store_ps(output0_tm, _sum0);
                    _mm_store_ps(output0_tm + 4, _sum1);
                    _mm_store_ps(output0_tm + 4 * 2, _sum2);
                    _mm_store_ps(output0_tm + 4 * 3, _sum3);
                    _mm_store_ps(output0_tm + 4 * 4, _sum4);
                    _mm_store_ps(output0_tm + 4 * 5, _sum5);
                    _mm_store_ps(output0_tm + 4 * 6, _sum6);
                    _mm_store_ps(output0_tm + 4 * 7, _sum7);

                    output0_tm += 4 * 8;
                }
                for (; i + 3 < tiles; i += 4)
                {
                    const float* r0 = bb2[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4].data();
                    const float* k0 = kernel0_tm[r].data();

                    int nn = inch * 4; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();
                    __m128 _sum2 = _mm_setzero_ps();
                    __m128 _sum3 = _mm_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m128 _w0 = _mm_load_ps(k0);

                        __m128 _val0 = _mm_load1_ps(r0);
                        __m128 _val1 = _mm_load1_ps(r0 + 1);
                        __m128 _val2 = _mm_load1_ps(r0 + 2);
                        __m128 _val3 = _mm_load1_ps(r0 + 3);

                        _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                        _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                        _sum3 = _mm_comp_fmadd_ps(_val3, _w0, _sum3);

                        r0 += 4;
                        k0 += 4;
                    }

                    _mm_store_ps(output0_tm, _sum0);
                    _mm_store_ps(output0_tm + 4, _sum1);
                    _mm_store_ps(output0_tm + 4 * 2, _sum2);
                    _mm_store_ps(output0_tm + 4 * 3, _sum3);

                    output0_tm += 4 * 4;
                }
                for (; i + 1 < tiles; i += 2)
                {
                    const float* r0 = bb2[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2].data();
                    const float* k0 = kernel0_tm[r].data();

                    int nn = inch * 4; // inch always > 0

                    __m128 _sum0 = _mm_setzero_ps();
                    __m128 _sum1 = _mm_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m128 _w0 = _mm_load_ps(k0);

                        __m128 _val0 = _mm_load1_ps(r0);
                        __m128 _val1 = _mm_load1_ps(r0 + 1);

                        _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);

                        r0 += 2;
                        k0 += 4;
                    }

                    _mm_store_ps(output0_tm, _sum0);
                    _mm_store_ps(output0_tm + 4, _sum1);

                    output0_tm += 4 * 2;
                }
                for (; i < tiles; i++)
                {
                    const float* r0 = bb2[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2].data();
                    const float* k0 = kernel0_tm[r].data();

                    int nn = inch * 4; // inch always > 0

                    __m128 _sum = _mm_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m128 _w0 = _mm_load_ps(k0);
                        __m128 _val0 = _mm_load1_ps(r0);
                        _sum = _mm_comp_fmadd_ps(_val0, _w0, _sum);

                        r0 += 1;
                        k0 += 4;
                    }

                    _mm_store_ps(output0_tm, _sum);

                    output0_tm += 4;
                }
            }
        }
    });
}

Tensor conv2d_3x3s1_winograd63_pack4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_shape = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
    output.resize_({output_shape[0], output_shape[1] / 4, output_shape[2], output_shape[3]});
    
    int origin_w = (int)self.size(3) + 2 * (int)padding[1];
    int origin_h = (int)self.size(2) + 2 * (int)padding[0];
    
    int w = origin_w;
    int h = origin_h;
    int inch  = (int)self.size(1);
    
    int outw  = (int)output_shape[3];
    int outh  = (int)output_shape[2];
    int outch = (int)output_shape[1] / 4;
    
    outw = (outw + 5) / 6 * 6;
    outh = (outh + 5) / 6 * 6;

    w = outw + 2;
    h = outh + 2;
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1] + w - origin_w, padding[0], padding[0] + h - origin_h}, 0);
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        otter::conv3x3s1_winograd63_transform_kernel_pack4_sse(weight, kernel_tf, inch * 4, outch * 4);
    
    // BEGIN transform input
    Tensor bottom_blob_tm;
    {
        int w_tiles = outw / 6;
        int h_tiles = outh / 6;
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm = otter::empty({inch, 64, tiles}, otter::ScalarType::Float4);
        conv3x3s1_winograd63_transform_input_pack4_sse(input[0], bottom_blob_tm);
    }
    input.reset();
    // END transform input

    // BEGIN dot
    Tensor top_blob_tm;
    convolution_winograd_dot_pack4_sse(bottom_blob_tm, outch, kernel_tf, top_blob_tm);
    // END dot

    // BEGIN transform output
    Tensor top_blob_bordered;
    if (outw == output_shape[3] && outh == output_shape[2]) {
        top_blob_bordered = output;
    } else {
        top_blob_bordered = otter::empty({1, outch, outh, outw}, otter::ScalarType::Float4);
    }
    {
        Tensor top_blob_bordered_t = top_blob_bordered[0];
        conv3x3s1_winograd63_transform_output_pack4_sse(top_blob_tm, top_blob_bordered_t, bias);
    }
    // END transform output
    
    otter::crop_(top_blob_bordered, {0, top_blob_bordered.size(3) - output_shape[3], 0, top_blob_bordered.size(2) - output_shape[2]}, output);
    
    return output;
}

Tensor conv2d_3x3s1_winograd63_pack4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, otter::ScalarType::Float4);
    
    return conv2d_3x3s1_winograd63_pack4_x86_out(self, weight, weight_o, bias, padding, output);
}

Tensor conv2d_3x3s1_winograd43_pack4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_shape = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
    output.resize_({output_shape[0], output_shape[1] / 4, output_shape[2], output_shape[3]});
    
    int origin_w = (int)self.size(3) + 2 * (int)padding[1];
    int origin_h = (int)self.size(2) + 2 * (int)padding[0];
    
    int w = origin_w;
    int h = origin_h;
    int inch  = (int)self.size(1);
    
    int outw  = (int)output_shape[3];
    int outh  = (int)output_shape[2];
    int outch = (int)output_shape[1] / 4;
    
    outw = (outw + 3) / 4 * 4;
    outh = (outh + 3) / 4 * 4;

    w = outw + 2;
    h = outh + 2;
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1] + w - origin_w, padding[0], padding[0] + h - origin_h}, 0);
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        otter::conv3x3s1_winograd43_transform_kernel_pack4_sse(weight, kernel_tf, inch * 4, outch * 4);
    
    // BEGIN transform input
    Tensor bottom_blob_tm;
    {
        int w_tiles = outw / 4;
        int h_tiles = outh / 4;
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm = otter::empty({inch, 36, tiles}, otter::ScalarType::Float4);
        conv3x3s1_winograd43_transform_input_pack4_sse(input[0], bottom_blob_tm);
    }
    input.reset();
    // END transform input

    // BEGIN dot
    Tensor top_blob_tm;
    convolution_winograd_dot_pack4_sse(bottom_blob_tm, outch, kernel_tf, top_blob_tm);
    // END dot

    // BEGIN transform output
    Tensor top_blob_bordered;
    if (outw == output_shape[3] && outh == output_shape[2]) {
        top_blob_bordered = output;
    } else {
        top_blob_bordered = otter::empty({1, outch, outh, outw}, otter::ScalarType::Float4);
    }
    {
        Tensor top_blob_bordered_t = top_blob_bordered[0];
        conv3x3s1_winograd43_transform_output_pack4_sse(top_blob_tm, top_blob_bordered_t, bias);
    }
    // END transform output
    
    otter::crop_(top_blob_bordered, {0, top_blob_bordered.size(3) - output_shape[3], 0, top_blob_bordered.size(2) - output_shape[2]}, output);
    
    return output;
}

Tensor conv2d_3x3s1_winograd43_pack4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, otter::ScalarType::Float4);
    
    return conv2d_3x3s1_winograd43_pack4_x86_out(self, weight, weight_o, bias, padding, output);
}

Tensor conv2d_3x3s1_winograd23_pack4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_shape = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
    output.resize_({output_shape[0], output_shape[1] / 4, output_shape[2], output_shape[3]});
    
    int origin_w = (int)self.size(3) + 2 * (int)padding[1];
    int origin_h = (int)self.size(2) + 2 * (int)padding[0];
    
    int w = origin_w;
    int h = origin_h;
    int inch  = (int)self.size(1);
    
    int outw  = (int)output_shape[3];
    int outh  = (int)output_shape[2];
    int outch = (int)output_shape[1] / 4;
    
    outw = (outw + 1) / 2 * 2;
    outh = (outh + 1) / 2 * 2;

    w = outw + 2;
    h = outh + 2;
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1] + w - origin_w, padding[0], padding[0] + h - origin_h}, 0);
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        otter::conv3x3s1_winograd23_transform_kernel_pack4_sse(weight, kernel_tf, inch * 4, outch * 4);
    
    // BEGIN transform input
    Tensor bottom_blob_tm;
    {
        int w_tiles = outw / 2;
        int h_tiles = outh / 2;
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm = otter::empty({inch, 16, tiles}, otter::ScalarType::Float4);
        conv3x3s1_winograd23_transform_input_pack4_sse(input[0], bottom_blob_tm);
    }
    input.reset();
    // END transform input

    // BEGIN dot
    Tensor top_blob_tm;
    convolution_winograd_dot_pack4_sse(bottom_blob_tm, outch, kernel_tf, top_blob_tm);
    // END dot

    // BEGIN transform output
    Tensor top_blob_bordered;
    if (outw == output_shape[3] && outh == output_shape[2]) {
        top_blob_bordered = output;
    } else {
        top_blob_bordered = otter::empty({1, outch, outh, outw}, otter::ScalarType::Float4);
    }
    {
        Tensor top_blob_bordered_t = top_blob_bordered[0];
        conv3x3s1_winograd23_transform_output_pack4_sse(top_blob_tm, top_blob_bordered_t, bias);
    }
    // END transform output
    
    otter::crop_(top_blob_bordered, {0, top_blob_bordered.size(3) - output_shape[3], 0, top_blob_bordered.size(2) - output_shape[2]}, output);
    
    return output;
}

Tensor conv2d_3x3s1_winograd23_pack4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, otter::ScalarType::Float4);
    
    return conv2d_3x3s1_winograd23_pack4_x86_out(self, weight, weight_o, bias, padding, output);
}

#if __AVX__

void im2col_sgemm_pack1to8_avx(const Tensor& bottom_im2col, Tensor& top_blob, const Tensor& kernel, const Tensor& _bias) {
    // Tensor bottom_im2col(size, maxk, inch, 4u, 1, opt.workspace_allocator);

    const int size = bottom_im2col.size(2);
    const int maxk = bottom_im2col.size(1);
    const int inch = bottom_im2col.size(0);

    const int outch = top_blob.size(1);

    const float* bias = (_bias.defined()) ? _bias.data_ptr<float>() : nullptr;

    // permute
    Tensor tmp;
    if (size >= 8)
        tmp = otter::empty({size / 8 + (size % 8) / 4 + size % 4, inch, 8 * maxk}, otter::ScalarType::Float);
    else if (size >= 4)
        tmp = otter::empty({size / 4 + size % 4, inch, 4 * maxk}, otter::ScalarType::Float);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Float);
    
    auto tmp_a = tmp.accessor<float, 3>();
    auto bottom_im2col_a = bottom_im2col.accessor<float, 3>();
    auto kernel_a = kernel.accessor<float, 3>();
    auto top_blob_a = top_blob.accessor<float, 4, 8>()[0];
    
    {
        int nn_size = size >> 3;
        int remain_size_start = 0;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end))
            {
                int i = remain_size_start + ii * 8;

                float* tmpptr = tmp_a[i / 8].data();

                for (int q = 0; q < inch; q++)
                {
                    const float* img0 = (const float*)bottom_im2col_a[q].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        __m256 _r0 = _mm256_loadu_ps(img0);
                        _mm256_store_ps(tmpptr, _r0);

                        img0 += size;
                        tmpptr += 8;
                    }
                }
            }
        });

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end))
            {
                int i = remain_size_start + ii * 4;

                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();

                for (int q = 0; q < inch; q++)
                {
                    const float* img0 = (const float*)bottom_im2col_a[q].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
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
            for (const auto i : otter::irange(begin, end))
            {
                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();

                for (int q = 0; q < inch; q++)
                {
                    const float* img0 = (const float*)bottom_im2col_a[q].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        tmpptr[0] = img0[0];
                        img0 += size;
                        tmpptr += 1;
                    }
                }
            }
        });
    }

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end))
        {
            float* outptr0 = top_blob_a[p].data();

            const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + p * 8 : zeros;

            int i = 0;
            for (; i + 7 < size; i += 8)
            {
                const float* tmpptr = tmp_a[i / 8].data();
                const float* kptr0 = kernel_a[p].data();

                int nn = inch * maxk; // inch always > 0

                __m256 _sum0 = _mm256_loadu_ps(biasptr);
                __m256 _sum1 = _sum0;
                __m256 _sum2 = _sum0;
                __m256 _sum3 = _sum0;
                __m256 _sum4 = _sum0;
                __m256 _sum5 = _sum0;
                __m256 _sum6 = _sum0;
                __m256 _sum7 = _sum0;

                for (int j = 0; j < nn; j++)
                {
                    __m256 _w0 = _mm256_load_ps(kptr0);

                    __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                    __m256 _val1 = _mm256_broadcast_ss(tmpptr + 1);
                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);
                    __m256 _val2 = _mm256_broadcast_ss(tmpptr + 2);
                    __m256 _val3 = _mm256_broadcast_ss(tmpptr + 3);
                    _sum2 = _mm256_comp_fmadd_ps(_val2, _w0, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val3, _w0, _sum3);
                    __m256 _val4 = _mm256_broadcast_ss(tmpptr + 4);
                    __m256 _val5 = _mm256_broadcast_ss(tmpptr + 5);
                    _sum4 = _mm256_comp_fmadd_ps(_val4, _w0, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_val5, _w0, _sum5);
                    __m256 _val6 = _mm256_broadcast_ss(tmpptr + 6);
                    __m256 _val7 = _mm256_broadcast_ss(tmpptr + 7);
                    _sum6 = _mm256_comp_fmadd_ps(_val6, _w0, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_val7, _w0, _sum7);

                    tmpptr += 8;
                    kptr0 += 8;
                }

                _mm256_store_ps(outptr0, _sum0);
                _mm256_store_ps(outptr0 + 8, _sum1);
                _mm256_store_ps(outptr0 + 8 * 2, _sum2);
                _mm256_store_ps(outptr0 + 8 * 3, _sum3);
                _mm256_store_ps(outptr0 + 8 * 4, _sum4);
                _mm256_store_ps(outptr0 + 8 * 5, _sum5);
                _mm256_store_ps(outptr0 + 8 * 6, _sum6);
                _mm256_store_ps(outptr0 + 8 * 7, _sum7);
                outptr0 += 64;
            }
            for (; i + 3 < size; i += 4)
            {
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();
                const float* kptr0 = kernel_a[p].data();

                int nn = inch * maxk; // inch always > 0

                __m256 _sum0 = _mm256_loadu_ps(biasptr);
                __m256 _sum1 = _sum0;
                __m256 _sum2 = _sum0;
                __m256 _sum3 = _sum0;

                for (int j = 0; j < nn; j++)
                {
                    __m256 _w0 = _mm256_load_ps(kptr0);

                    __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                    __m256 _val1 = _mm256_broadcast_ss(tmpptr + 1);
                    __m256 _val2 = _mm256_broadcast_ss(tmpptr + 2);
                    __m256 _val3 = _mm256_broadcast_ss(tmpptr + 3);

                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);
                    _sum2 = _mm256_comp_fmadd_ps(_val2, _w0, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val3, _w0, _sum3);

                    tmpptr += 4;
                    kptr0 += 8;
                }

                _mm256_store_ps(outptr0, _sum0);
                _mm256_store_ps(outptr0 + 8, _sum1);
                _mm256_store_ps(outptr0 + 16, _sum2);
                _mm256_store_ps(outptr0 + 24, _sum3);
                outptr0 += 32;
            }
            for (; i < size; i++)
            {
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();
                const float* kptr0 = kernel_a[p].data();

                int nn = inch * maxk; // inch always > 0

                __m256 _sum = _mm256_loadu_ps(biasptr);

                for (int j = 0; j < nn; j++)
                {
                    __m256 _w0 = _mm256_load_ps(kptr0);
                    __m256 _val = _mm256_broadcast_ss(tmpptr);
                    _sum = _mm256_comp_fmadd_ps(_w0, _val, _sum);

                    tmpptr += 1;
                    kptr0 += 8;
                }

                _mm256_store_ps(outptr0, _sum);
                outptr0 += 8;
            }
        }
    });
}

void im2col_sgemm_pack4to8_avx(const Tensor& bottom_im2col, Tensor& top_blob, const Tensor& kernel, const Tensor& _bias) {
    // Tensor bottom_im2col(size, maxk, inch, 16u, 4, opt.workspace_allocator);

    const int size = bottom_im2col.size(2);
    const int maxk = bottom_im2col.size(1);
    const int inch = bottom_im2col.size(0);

    const int outch = top_blob.size(1);

    const float* bias = (_bias.defined()) ? _bias.data_ptr<float>() : nullptr;

    // permute
    Tensor tmp;
    if (size >= 8)
        tmp = otter::empty({size / 8 + size % 8, inch, 8 * maxk}, otter::ScalarType::Float4);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Float4);
    
    auto tmp_a = tmp.accessor<float, 3, 4>();
    auto bottom_im2col_a = bottom_im2col.accessor<float, 3, 4>();
    auto kernel_a = kernel.accessor<float, 3>();
    auto top_blob_a = top_blob.accessor<float, 4, 8>()[0];
    
    {
        int nn_size = size >> 3;
        int remain_size_start = 0;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end))
            {
                int i = remain_size_start + ii * 8;

                float* tmpptr = tmp_a[i / 8].data();

                for (int q = 0; q < inch; q++)
                {
                    const float* img0 = (const float*)bottom_im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++)
                    {
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

        otter::parallel_for(remain_size_start, size, 0, [&](int64_t begin, int64_t end) {
            for (const auto i : otter::irange(begin, end))
            {
                float* tmpptr = tmp_a[i / 8 + i % 8].data();

                for (int q = 0; q < inch; q++)
                {
                    const float* img0 = (const float*)bottom_im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++)
                    {
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
        for (const auto p : otter::irange(begin, end))
        {
            float* outptr0 = top_blob_a[p].data();

            const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + p * 8 : zeros;

            int i = 0;
            for (; i + 7 < size; i += 8)
            {
                float* tmpptr = tmp_a[i / 8].data();
                const float* kptr = kernel_a[p].data();

                int nn = inch * maxk * 4; // inch always > 0

                __m256 _sum0 = _mm256_loadu_ps(biasptr);
                __m256 _sum1 = _sum0;
                __m256 _sum2 = _sum0;
                __m256 _sum3 = _sum0;
                __m256 _sum4 = _sum0;
                __m256 _sum5 = _sum0;
                __m256 _sum6 = _sum0;
                __m256 _sum7 = _sum0;

                for (int j = 0; j < nn; j++)
                {
                    __m256 _w0 = _mm256_load_ps(kptr);

                    __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                    __m256 _val1 = _mm256_broadcast_ss(tmpptr + 1);
                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);
                    __m256 _val2 = _mm256_broadcast_ss(tmpptr + 2);
                    __m256 _val3 = _mm256_broadcast_ss(tmpptr + 3);
                    _sum2 = _mm256_comp_fmadd_ps(_val2, _w0, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val3, _w0, _sum3);
                    __m256 _val4 = _mm256_broadcast_ss(tmpptr + 4);
                    __m256 _val5 = _mm256_broadcast_ss(tmpptr + 5);
                    _sum4 = _mm256_comp_fmadd_ps(_val4, _w0, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_val5, _w0, _sum5);
                    __m256 _val6 = _mm256_broadcast_ss(tmpptr + 6);
                    __m256 _val7 = _mm256_broadcast_ss(tmpptr + 7);
                    _sum6 = _mm256_comp_fmadd_ps(_val6, _w0, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_val7, _w0, _sum7);

                    kptr += 8;
                    tmpptr += 8;
                }

                _mm256_store_ps(outptr0, _sum0);
                _mm256_store_ps(outptr0 + 8, _sum1);
                _mm256_store_ps(outptr0 + 8 * 2, _sum2);
                _mm256_store_ps(outptr0 + 8 * 3, _sum3);
                _mm256_store_ps(outptr0 + 8 * 4, _sum4);
                _mm256_store_ps(outptr0 + 8 * 5, _sum5);
                _mm256_store_ps(outptr0 + 8 * 6, _sum6);
                _mm256_store_ps(outptr0 + 8 * 7, _sum7);

                outptr0 += 8 * 8;
            }
            for (; i < size; i++)
            {
                float* tmpptr = tmp_a[i / 8 + i % 8].data();
                const float* kptr = kernel_a[p].data();

                int nn = inch * maxk * 4; // inch always > 0

                __m256 _sum0 = _mm256_loadu_ps(biasptr);

                for (int j = 0; j < nn; j++)
                {
                    __m256 _w0 = _mm256_load_ps(kptr);
                    __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);

                    kptr += 8;
                    tmpptr += 1;
                }

                _mm256_store_ps(outptr0, _sum0);
                outptr0 += 8;
            }
        }
    });
}

void im2col_sgemm_pack8_avx(const Tensor& bottom_im2col, Tensor& top_blob, const Tensor& kernel, const Tensor& _bias)
{
    // Tensor bottom_im2col(size, maxk, inch, 32u, 8, opt.workspace_allocator);

    const int size = bottom_im2col.size(2);
    const int maxk = bottom_im2col.size(1);
    const int inch = bottom_im2col.size(0);

    const int outch = top_blob.size(1);

    const float* bias = (_bias.defined()) ? _bias.data_ptr<float>() : nullptr;

    // permute
    Tensor tmp;
    if (size >= 12)
        tmp = otter::empty({size / 12 + (size % 12) / 8 + (size % 12 % 8) / 4 + (size % 12 % 4) / 2 + size % 12 % 2, inch, 12 * maxk}, otter::ScalarType::Float8);
    else if (size >= 8)
        tmp = otter::empty({size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, inch, 8 * maxk}, otter::ScalarType::Float8);
    else if (size >= 4)
        tmp = otter::empty({size / 4 + (size % 4) / 2 + size % 2, inch, 4 * maxk}, otter::ScalarType::Float8);
    else if (size >= 2)
        tmp = otter::empty({size / 2 + size % 2, inch, 2 * maxk}, otter::ScalarType::Float8);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Float8);
    
    auto tmp_a = tmp.accessor<float, 3, 8>();
    auto bottom_im2col_a = bottom_im2col.accessor<float, 3, 8>();
    auto kernel_a = kernel.accessor<float, 3>();
    auto top_blob_a = top_blob.accessor<float, 4, 8>()[0];
    
    {
        int nn_size = size / 12;
        int remain_size_start = 0;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 12;

                float* tmpptr = tmp_a[i / 12].data();

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)bottom_im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++) {
                        // transpose 8x12
                        __m256 _r0 = _mm256_load_ps(img0);
                        __m256 _r1 = _mm256_load_ps(img0 + 8);
                        __m256 _r2 = _mm256_load_ps(img0 + 8 * 2);
                        __m256 _r3 = _mm256_load_ps(img0 + 8 * 3);
                        __m256 _r4 = _mm256_load_ps(img0 + 8 * 4);
                        __m256 _r5 = _mm256_load_ps(img0 + 8 * 5);
                        __m256 _r6 = _mm256_load_ps(img0 + 8 * 6);
                        __m256 _r7 = _mm256_load_ps(img0 + 8 * 7);
                        __m256 _r8 = _mm256_load_ps(img0 + 8 * 8);
                        __m256 _r9 = _mm256_load_ps(img0 + 8 * 9);
                        __m256 _ra = _mm256_load_ps(img0 + 8 * 10);
                        __m256 _rb = _mm256_load_ps(img0 + 8 * 11);

                        __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                        __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                        __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
                        __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
                        __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
                        __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);
                        __m256 _tmp6 = _mm256_unpacklo_ps(_r6, _r7);
                        __m256 _tmp7 = _mm256_unpackhi_ps(_r6, _r7);
                        __m256 _tmp8 = _mm256_unpacklo_ps(_r8, _r9);
                        __m256 _tmp9 = _mm256_unpackhi_ps(_r8, _r9);
                        __m256 _tmpa = _mm256_unpacklo_ps(_ra, _rb);
                        __m256 _tmpb = _mm256_unpackhi_ps(_ra, _rb);
                        __m256 _tmpc = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmpd = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                        __m256 _tmpe = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmpf = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
                        __m256 _tmpg = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmph = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
                        __m256 _tmpi = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmpj = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
                        __m256 _tmpk = _mm256_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmpl = _mm256_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(3, 2, 3, 2));
                        __m256 _tmpm = _mm256_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmpn = _mm256_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(3, 2, 3, 2));
                        _r0 = _mm256_permute2f128_ps(_tmpc, _tmpg, _MM_SHUFFLE(0, 2, 0, 0));
                        _r1 = _mm256_permute2f128_ps(_tmpk, _tmpd, _MM_SHUFFLE(0, 2, 0, 0));
                        _r2 = _mm256_permute2f128_ps(_tmph, _tmpl, _MM_SHUFFLE(0, 2, 0, 0));
                        _r3 = _mm256_permute2f128_ps(_tmpe, _tmpi, _MM_SHUFFLE(0, 2, 0, 0));
                        _r4 = _mm256_permute2f128_ps(_tmpm, _tmpf, _MM_SHUFFLE(0, 2, 0, 0));
                        _r5 = _mm256_permute2f128_ps(_tmpj, _tmpn, _MM_SHUFFLE(0, 2, 0, 0));
                        _r6 = _mm256_permute2f128_ps(_tmpc, _tmpg, _MM_SHUFFLE(0, 3, 0, 1));
                        _r7 = _mm256_permute2f128_ps(_tmpk, _tmpd, _MM_SHUFFLE(0, 3, 0, 1));
                        _r8 = _mm256_permute2f128_ps(_tmph, _tmpl, _MM_SHUFFLE(0, 3, 0, 1));
                        _r9 = _mm256_permute2f128_ps(_tmpe, _tmpi, _MM_SHUFFLE(0, 3, 0, 1));
                        _ra = _mm256_permute2f128_ps(_tmpm, _tmpf, _MM_SHUFFLE(0, 3, 0, 1));
                        _rb = _mm256_permute2f128_ps(_tmpj, _tmpn, _MM_SHUFFLE(0, 3, 0, 1));

                        _mm256_store_ps(tmpptr, _r0);
                        _mm256_store_ps(tmpptr + 8, _r1);
                        _mm256_store_ps(tmpptr + 8 * 2, _r2);
                        _mm256_store_ps(tmpptr + 8 * 3, _r3);
                        _mm256_store_ps(tmpptr + 8 * 4, _r4);
                        _mm256_store_ps(tmpptr + 8 * 5, _r5);
                        _mm256_store_ps(tmpptr + 8 * 6, _r6);
                        _mm256_store_ps(tmpptr + 8 * 7, _r7);
                        _mm256_store_ps(tmpptr + 8 * 8, _r8);
                        _mm256_store_ps(tmpptr + 8 * 9, _r9);
                        _mm256_store_ps(tmpptr + 8 * 10, _ra);
                        _mm256_store_ps(tmpptr + 8 * 11, _rb);

                        img0 += size * 8;
                        tmpptr += 96;
                    }
                }
            }
        });

        remain_size_start += nn_size * 12;
        nn_size = (size - remain_size_start) >> 3;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 8;

                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8].data();

                for (int q = 0; q < inch; q++)
                {
                    const float* img0 = (const float*)bottom_im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++)
                    {
                        // transpose 8x8
                        __m256 _r0 = _mm256_load_ps(img0);
                        __m256 _r1 = _mm256_load_ps(img0 + 8);
                        __m256 _r2 = _mm256_load_ps(img0 + 8 * 2);
                        __m256 _r3 = _mm256_load_ps(img0 + 8 * 3);
                        __m256 _r4 = _mm256_load_ps(img0 + 8 * 4);
                        __m256 _r5 = _mm256_load_ps(img0 + 8 * 5);
                        __m256 _r6 = _mm256_load_ps(img0 + 8 * 6);
                        __m256 _r7 = _mm256_load_ps(img0 + 8 * 7);

                        __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                        __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                        __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
                        __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
                        __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
                        __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);
                        __m256 _tmp6 = _mm256_unpacklo_ps(_r6, _r7);
                        __m256 _tmp7 = _mm256_unpackhi_ps(_r6, _r7);
                        __m256 _tmp8 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmp9 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                        __m256 _tmpa = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmpb = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
                        __m256 _tmpc = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmpd = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
                        __m256 _tmpe = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmpf = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
                        _r0 = _mm256_permute2f128_ps(_tmp8, _tmpc, _MM_SHUFFLE(0, 2, 0, 0));
                        _r1 = _mm256_permute2f128_ps(_tmp9, _tmpd, _MM_SHUFFLE(0, 2, 0, 0));
                        _r2 = _mm256_permute2f128_ps(_tmpa, _tmpe, _MM_SHUFFLE(0, 2, 0, 0));
                        _r3 = _mm256_permute2f128_ps(_tmpb, _tmpf, _MM_SHUFFLE(0, 2, 0, 0));
                        _r4 = _mm256_permute2f128_ps(_tmp8, _tmpc, _MM_SHUFFLE(0, 3, 0, 1));
                        _r5 = _mm256_permute2f128_ps(_tmp9, _tmpd, _MM_SHUFFLE(0, 3, 0, 1));
                        _r6 = _mm256_permute2f128_ps(_tmpa, _tmpe, _MM_SHUFFLE(0, 3, 0, 1));
                        _r7 = _mm256_permute2f128_ps(_tmpb, _tmpf, _MM_SHUFFLE(0, 3, 0, 1));

                        _mm256_store_ps(tmpptr, _r0);
                        _mm256_store_ps(tmpptr + 8, _r1);
                        _mm256_store_ps(tmpptr + 8 * 2, _r2);
                        _mm256_store_ps(tmpptr + 8 * 3, _r3);
                        _mm256_store_ps(tmpptr + 8 * 4, _r4);
                        _mm256_store_ps(tmpptr + 8 * 5, _r5);
                        _mm256_store_ps(tmpptr + 8 * 6, _r6);
                        _mm256_store_ps(tmpptr + 8 * 7, _r7);

                        img0 += size * 8;
                        tmpptr += 64;
                    }
                }
            }
        });

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 4;

                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4].data();

                for (int q = 0; q < inch; q++)
                {
                    const float* img0 = (const float*)bottom_im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++)
                    {
                        // transpose 8x4
                        __m256 _r0 = _mm256_load_ps(img0);
                        __m256 _r1 = _mm256_load_ps(img0 + 8);
                        __m256 _r2 = _mm256_load_ps(img0 + 8 * 2);
                        __m256 _r3 = _mm256_load_ps(img0 + 8 * 3);

                        __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                        __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                        __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
                        __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
                        __m256 _tmp4 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmp5 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                        __m256 _tmp6 = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmp7 = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
                        _r0 = _mm256_permute2f128_ps(_tmp4, _tmp5, _MM_SHUFFLE(0, 2, 0, 0));
                        _r1 = _mm256_permute2f128_ps(_tmp6, _tmp7, _MM_SHUFFLE(0, 2, 0, 0));
                        _r2 = _mm256_permute2f128_ps(_tmp4, _tmp5, _MM_SHUFFLE(0, 3, 0, 1));
                        _r3 = _mm256_permute2f128_ps(_tmp6, _tmp7, _MM_SHUFFLE(0, 3, 0, 1));

                        _mm256_store_ps(tmpptr, _r0);
                        _mm256_store_ps(tmpptr + 8, _r1);
                        _mm256_store_ps(tmpptr + 8 * 2, _r2);
                        _mm256_store_ps(tmpptr + 8 * 3, _r3);

                        img0 += size * 8;
                        tmpptr += 32;
                    }
                }
            }
        });

        remain_size_start += nn_size << 2;
        nn_size = (size - remain_size_start) >> 1;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 2;

                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2].data();

                for (int q = 0; q < inch; q++)
                {
                    const float* img0 = (const float*)bottom_im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++)
                    {
                        // transpose 8x2
                        __m256 _r0 = _mm256_load_ps(img0);
                        __m256 _r1 = _mm256_load_ps(img0 + 8);

                        __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                        __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                        _r0 = _mm256_permute2f128_ps(_tmp0, _tmp1, _MM_SHUFFLE(0, 2, 0, 0));
                        _r1 = _mm256_permute2f128_ps(_tmp0, _tmp1, _MM_SHUFFLE(0, 3, 0, 1));

                        _mm256_store_ps(tmpptr, _r0);
                        _mm256_store_ps(tmpptr + 8, _r1);

                        img0 += size * 8;
                        tmpptr += 16;
                    }
                }
            }
        });

        remain_size_start += nn_size << 1;

        otter::parallel_for(remain_size_start, size, 0, [&](int64_t begin, int64_t end) {
            for (const auto i : otter::irange(begin, end))
            {
                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2].data();

                for (int q = 0; q < inch; q++)
                {
                    const float* img0 = (const float*)bottom_im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++)
                    {
                        __m256 _val = _mm256_load_ps(img0);
                        _mm256_store_ps(tmpptr, _val);

                        img0 += size * 8;
                        tmpptr += 8;
                    }
                }
            }
        });
    }

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            float* outptr0 = top_blob_a[p].data();

            const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + p * 8 : zeros;

            int i = 0;
            for (; i + 11 < size; i += 12)
            {
                const float* tmpptr = tmp_a[i / 12].data();
                const float* kptr0 = kernel_a[p].data();

                int nn = inch * maxk * 8; // inch always > 0

                __m256 _sum0 = _mm256_loadu_ps(biasptr);
                __m256 _sum1 = _sum0;
                __m256 _sum2 = _sum0;
                __m256 _sum3 = _sum0;
                __m256 _sum4 = _sum0;
                __m256 _sum5 = _sum0;
                __m256 _sum6 = _sum0;
                __m256 _sum7 = _sum0;
                __m256 _sum8 = _sum0;
                __m256 _sum9 = _sum0;
                __m256 _suma = _sum0;
                __m256 _sumb = _sum0;

                for (int j = 0; j < nn; j++)
                {
                    __m256 _w0 = _mm256_load_ps(kptr0);

                    __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                    __m256 _val1 = _mm256_broadcast_ss(tmpptr + 1);
                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);
                    __m256 _val2 = _mm256_broadcast_ss(tmpptr + 2);
                    __m256 _val3 = _mm256_broadcast_ss(tmpptr + 3);
                    _sum2 = _mm256_comp_fmadd_ps(_val2, _w0, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val3, _w0, _sum3);
                    __m256 _val4 = _mm256_broadcast_ss(tmpptr + 4);
                    __m256 _val5 = _mm256_broadcast_ss(tmpptr + 5);
                    _sum4 = _mm256_comp_fmadd_ps(_val4, _w0, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_val5, _w0, _sum5);
                    __m256 _val6 = _mm256_broadcast_ss(tmpptr + 6);
                    __m256 _val7 = _mm256_broadcast_ss(tmpptr + 7);
                    _sum6 = _mm256_comp_fmadd_ps(_val6, _w0, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_val7, _w0, _sum7);
                    __m256 _val8 = _mm256_broadcast_ss(tmpptr + 8);
                    __m256 _val9 = _mm256_broadcast_ss(tmpptr + 9);
                    _sum8 = _mm256_comp_fmadd_ps(_val8, _w0, _sum8);
                    _sum9 = _mm256_comp_fmadd_ps(_val9, _w0, _sum9);
                    __m256 _vala = _mm256_broadcast_ss(tmpptr + 10);
                    __m256 _valb = _mm256_broadcast_ss(tmpptr + 11);
                    _suma = _mm256_comp_fmadd_ps(_vala, _w0, _suma);
                    _sumb = _mm256_comp_fmadd_ps(_valb, _w0, _sumb);

                    tmpptr += 12;
                    kptr0 += 8;
                }

                _mm256_store_ps(outptr0, _sum0);
                _mm256_store_ps(outptr0 + 8, _sum1);
                _mm256_store_ps(outptr0 + 8 * 2, _sum2);
                _mm256_store_ps(outptr0 + 8 * 3, _sum3);
                _mm256_store_ps(outptr0 + 8 * 4, _sum4);
                _mm256_store_ps(outptr0 + 8 * 5, _sum5);
                _mm256_store_ps(outptr0 + 8 * 6, _sum6);
                _mm256_store_ps(outptr0 + 8 * 7, _sum7);
                _mm256_store_ps(outptr0 + 8 * 8, _sum8);
                _mm256_store_ps(outptr0 + 8 * 9, _sum9);
                _mm256_store_ps(outptr0 + 8 * 10, _suma);
                _mm256_store_ps(outptr0 + 8 * 11, _sumb);

                outptr0 += 8 * 12;
            }
            for (; i + 7 < size; i += 8)
            {
                const float* tmpptr = tmp_a[i / 12 + (i % 12) / 8].data();
                const float* kptr0 = kernel_a[p].data();

                int nn = inch * maxk * 8; // inch always > 0

                __m256 _sum0 = _mm256_loadu_ps(biasptr);
                __m256 _sum1 = _sum0;
                __m256 _sum2 = _sum0;
                __m256 _sum3 = _sum0;
                __m256 _sum4 = _sum0;
                __m256 _sum5 = _sum0;
                __m256 _sum6 = _sum0;
                __m256 _sum7 = _sum0;

                for (int j = 0; j < nn; j++)
                {
                    __m256 _w0 = _mm256_load_ps(kptr0);

                    __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                    __m256 _val1 = _mm256_broadcast_ss(tmpptr + 1);
                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);
                    __m256 _val2 = _mm256_broadcast_ss(tmpptr + 2);
                    __m256 _val3 = _mm256_broadcast_ss(tmpptr + 3);
                    _sum2 = _mm256_comp_fmadd_ps(_val2, _w0, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val3, _w0, _sum3);
                    __m256 _val4 = _mm256_broadcast_ss(tmpptr + 4);
                    __m256 _val5 = _mm256_broadcast_ss(tmpptr + 5);
                    _sum4 = _mm256_comp_fmadd_ps(_val4, _w0, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_val5, _w0, _sum5);
                    __m256 _val6 = _mm256_broadcast_ss(tmpptr + 6);
                    __m256 _val7 = _mm256_broadcast_ss(tmpptr + 7);
                    _sum6 = _mm256_comp_fmadd_ps(_val6, _w0, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_val7, _w0, _sum7);

                    tmpptr += 8;
                    kptr0 += 8;
                }

                _mm256_store_ps(outptr0, _sum0);
                _mm256_store_ps(outptr0 + 8, _sum1);
                _mm256_store_ps(outptr0 + 8 * 2, _sum2);
                _mm256_store_ps(outptr0 + 8 * 3, _sum3);
                _mm256_store_ps(outptr0 + 8 * 4, _sum4);
                _mm256_store_ps(outptr0 + 8 * 5, _sum5);
                _mm256_store_ps(outptr0 + 8 * 6, _sum6);
                _mm256_store_ps(outptr0 + 8 * 7, _sum7);

                outptr0 += 8 * 8;
            }
            for (; i + 3 < size; i += 4)
            {
                const float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4].data();
                const float* kptr0 = kernel_a[p].data();

                int nn = inch * maxk * 8; // inch always > 0

                __m256 _sum0 = _mm256_loadu_ps(biasptr);
                __m256 _sum1 = _sum0;
                __m256 _sum2 = _sum0;
                __m256 _sum3 = _sum0;

                for (int j = 0; j < nn; j++)
                {
                    __m256 _w0 = _mm256_load_ps(kptr0);

                    __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                    __m256 _val1 = _mm256_broadcast_ss(tmpptr + 1);
                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);
                    __m256 _val2 = _mm256_broadcast_ss(tmpptr + 2);
                    __m256 _val3 = _mm256_broadcast_ss(tmpptr + 3);
                    _sum2 = _mm256_comp_fmadd_ps(_val2, _w0, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val3, _w0, _sum3);

                    tmpptr += 4;
                    kptr0 += 8;
                }

                _mm256_store_ps(outptr0, _sum0);
                _mm256_store_ps(outptr0 + 8, _sum1);
                _mm256_store_ps(outptr0 + 8 * 2, _sum2);
                _mm256_store_ps(outptr0 + 8 * 3, _sum3);

                outptr0 += 8 * 4;
            }
            for (; i + 1 < size; i += 2)
            {
                const float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2].data();
                const float* kptr0 = kernel_a[p].data();

                int nn = inch * maxk * 8; // inch always > 0

                __m256 _sum0 = _mm256_loadu_ps(biasptr);
                __m256 _sum1 = _sum0;

                for (int j = 0; j < nn; j++)
                {
                    __m256 _w0 = _mm256_load_ps(kptr0);

                    __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                    __m256 _val1 = _mm256_broadcast_ss(tmpptr + 1);
                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);

                    tmpptr += 2;
                    kptr0 += 8;
                }

                _mm256_store_ps(outptr0, _sum0);
                _mm256_store_ps(outptr0 + 8, _sum1);

                outptr0 += 8 * 2;
            }
            for (; i < size; i++)
            {
                const float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2].data();
                const float* kptr0 = kernel_a[p].data();

                int nn = inch * maxk * 8; // inch always > 0

                __m256 _sum = _mm256_loadu_ps(biasptr);

                for (int j = 0; j < nn; j++)
                {
                    __m256 _w0 = _mm256_load_ps(kptr0);
                    __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                    _sum = _mm256_comp_fmadd_ps(_val0, _w0, _sum);

                    tmpptr += 1;
                    kptr0 += 8;
                }

                _mm256_store_ps(outptr0, _sum);

                outptr0 += 8;
            }
        }
    });
}

void im2col_sgemm_pack8to1_avx(const Tensor& bottom_im2col, Tensor& top_blob, const Tensor& kernel, const Tensor& _bias)
{
    // Tensor bottom_im2col(size, maxk, inch, 4u * 8, 8, opt.workspace_allocator);

    const int size = bottom_im2col.size(2);
    const int maxk = bottom_im2col.size(1);
    const int inch = bottom_im2col.size(0);

    const int outch = top_blob.size(1);

    const float* bias = (_bias.defined()) ? _bias.data_ptr<float>() : nullptr;

    Tensor tmp;
    if (size >= 8)
        tmp = otter::empty({size / 8 + size % 8, inch, 8 * maxk}, otter::ScalarType::Float8);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Float8);
    
    auto tmp_a = tmp.accessor<float, 3, 8>();
    auto bottom_im2col_a = bottom_im2col.accessor<float, 3, 8>();
    auto kernel_a = kernel.accessor<float, 3>();
    auto top_blob_a = top_blob.accessor<float, 4>()[0];
    
    {
        int remain_size_start = 0;
        int nn_size = size >> 3;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end))
            {
                int i = ii * 8;

                float* tmpptr = tmp_a[i / 8].data();

                for (int q = 0; q < inch; q++)
                {
                    const float* img0 = (const float*)bottom_im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++)
                    {
                        // transpose 8x8
                        __m256 _r0 = _mm256_load_ps(img0);
                        __m256 _r1 = _mm256_load_ps(img0 + 8);
                        __m256 _r2 = _mm256_load_ps(img0 + 8 * 2);
                        __m256 _r3 = _mm256_load_ps(img0 + 8 * 3);
                        __m256 _r4 = _mm256_load_ps(img0 + 8 * 4);
                        __m256 _r5 = _mm256_load_ps(img0 + 8 * 5);
                        __m256 _r6 = _mm256_load_ps(img0 + 8 * 6);
                        __m256 _r7 = _mm256_load_ps(img0 + 8 * 7);

                        __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                        __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                        __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
                        __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
                        __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
                        __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);
                        __m256 _tmp6 = _mm256_unpacklo_ps(_r6, _r7);
                        __m256 _tmp7 = _mm256_unpackhi_ps(_r6, _r7);
                        __m256 _tmp8 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmp9 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                        __m256 _tmpa = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmpb = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
                        __m256 _tmpc = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmpd = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
                        __m256 _tmpe = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmpf = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
                        _r0 = _mm256_permute2f128_ps(_tmp8, _tmpc, _MM_SHUFFLE(0, 2, 0, 0));
                        _r1 = _mm256_permute2f128_ps(_tmp9, _tmpd, _MM_SHUFFLE(0, 2, 0, 0));
                        _r2 = _mm256_permute2f128_ps(_tmpa, _tmpe, _MM_SHUFFLE(0, 2, 0, 0));
                        _r3 = _mm256_permute2f128_ps(_tmpb, _tmpf, _MM_SHUFFLE(0, 2, 0, 0));
                        _r4 = _mm256_permute2f128_ps(_tmp8, _tmpc, _MM_SHUFFLE(0, 3, 0, 1));
                        _r5 = _mm256_permute2f128_ps(_tmp9, _tmpd, _MM_SHUFFLE(0, 3, 0, 1));
                        _r6 = _mm256_permute2f128_ps(_tmpa, _tmpe, _MM_SHUFFLE(0, 3, 0, 1));
                        _r7 = _mm256_permute2f128_ps(_tmpb, _tmpf, _MM_SHUFFLE(0, 3, 0, 1));

                        _mm256_store_ps(tmpptr, _r0);
                        _mm256_store_ps(tmpptr + 8, _r1);
                        _mm256_store_ps(tmpptr + 8 * 2, _r2);
                        _mm256_store_ps(tmpptr + 8 * 3, _r3);
                        _mm256_store_ps(tmpptr + 8 * 4, _r4);
                        _mm256_store_ps(tmpptr + 8 * 5, _r5);
                        _mm256_store_ps(tmpptr + 8 * 6, _r6);
                        _mm256_store_ps(tmpptr + 8 * 7, _r7);

                        img0 += size * 8;
                        tmpptr += 64;
                    }
                }
            }
        });

        remain_size_start += nn_size << 3;

        otter::parallel_for(remain_size_start, size, 0, [&](int64_t begin, int64_t end) {
            for (const auto i : otter::irange(begin, end))
            {
                float* tmpptr = tmp_a[i / 8 + i % 8].data();

                for (int q = 0; q < inch; q++)
                {
                    const float* img0 = (const float*)bottom_im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++)
                    {
                        __m256 _val = _mm256_load_ps(img0);
                        _mm256_store_ps(tmpptr, _val);

                        img0 += size * 8;
                        tmpptr += 8;
                    }
                }
            }
        });
    }

    int nn_outch = outch / 8;
    int remain_outch_start = nn_outch * 8;

    otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto pp : otter::irange(begin, end))
        {
            int p = pp * 8;

            float* outptr0 = top_blob_a[p].data();
            float* outptr1 = top_blob_a[p + 1].data();
            float* outptr2 = top_blob_a[p + 2].data();
            float* outptr3 = top_blob_a[p + 3].data();
            float* outptr4 = top_blob_a[p + 4].data();
            float* outptr5 = top_blob_a[p + 5].data();
            float* outptr6 = top_blob_a[p + 6].data();
            float* outptr7 = top_blob_a[p + 7].data();

            const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + p : zeros;

            int i = 0;
            for (; i + 7 < size; i += 8)
            {
                const float* tmpptr = tmp_a[i / 8].data();
                const float* kptr0 = kernel_a[p / 8].data();

                int nn = inch * maxk * 8; // inch always > 0

                __m256 _sum0 = _mm256_broadcast_ss(biasptr);
                __m256 _sum1 = _mm256_broadcast_ss(biasptr + 1);
                __m256 _sum2 = _mm256_broadcast_ss(biasptr + 2);
                __m256 _sum3 = _mm256_broadcast_ss(biasptr + 3);
                __m256 _sum4 = _mm256_broadcast_ss(biasptr + 4);
                __m256 _sum5 = _mm256_broadcast_ss(biasptr + 5);
                __m256 _sum6 = _mm256_broadcast_ss(biasptr + 6);
                __m256 _sum7 = _mm256_broadcast_ss(biasptr + 7);

                for (int j = 0; j < nn; j++)
                {
                    __m256 _val0 = _mm256_load_ps(tmpptr);

                    __m256 _w0 = _mm256_broadcast_ss(kptr0);
                    __m256 _w1 = _mm256_broadcast_ss(kptr0 + 1);
                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val0, _w1, _sum1);
                    __m256 _w2 = _mm256_broadcast_ss(kptr0 + 2);
                    __m256 _w3 = _mm256_broadcast_ss(kptr0 + 3);
                    _sum2 = _mm256_comp_fmadd_ps(_val0, _w2, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val0, _w3, _sum3);
                    __m256 _w4 = _mm256_broadcast_ss(kptr0 + 4);
                    __m256 _w5 = _mm256_broadcast_ss(kptr0 + 5);
                    _sum4 = _mm256_comp_fmadd_ps(_val0, _w4, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_val0, _w5, _sum5);
                    __m256 _w6 = _mm256_broadcast_ss(kptr0 + 6);
                    __m256 _w7 = _mm256_broadcast_ss(kptr0 + 7);
                    _sum6 = _mm256_comp_fmadd_ps(_val0, _w6, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_val0, _w7, _sum7);

                    tmpptr += 8;
                    kptr0 += 8;
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
            for (; i < size; i++)
            {
                const float* tmpptr = tmp_a[i / 8 + i % 8].data();
                const float* kptr0 = kernel_a[p / 8].data();

                int nn = inch * maxk * 8; // inch always > 0

                __m256 _sum = _mm256_loadu_ps(biasptr);

                for (int j = 0; j < nn; j++)
                {
                    __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                    __m256 _w0 = _mm256_load_ps(kptr0);
                    _sum = _mm256_comp_fmadd_ps(_val0, _w0, _sum);

                    tmpptr += 1;
                    kptr0 += 8;
                }

                float sum[8];
                _mm256_storeu_ps(sum, _sum);

                outptr0[0] = sum[0];
                outptr1[0] = sum[1];
                outptr2[0] = sum[2];
                outptr3[0] = sum[3];
                outptr4[0] = sum[4];
                outptr5[0] = sum[5];
                outptr6[0] = sum[6];
                outptr7[0] = sum[7];

                outptr0 += 1;
                outptr1 += 1;
                outptr2 += 1;
                outptr3 += 1;
                outptr4 += 1;
                outptr5 += 1;
                outptr6 += 1;
                outptr7 += 1;
            }
        }
    });

    otter::parallel_for(remain_outch_start, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end))
        {
            float* outptr0 = top_blob_a[p].data();

            const float bias0 = bias ? bias[p] : 0.f;

            int i = 0;
            for (; i + 7 < size; i += 8)
            {
                const float* tmpptr = tmp_a[i / 8].data();
                const float* kptr0 = kernel_a[p / 8 + p % 8].data();

                int nn = inch * maxk * 8; // inch always > 0

                __m256 _sum0 = _mm256_set1_ps(bias0);

                for (int j = 0; j < nn; j++)
                {
                    __m256 _val0 = _mm256_load_ps(tmpptr);
                    __m256 _w0 = _mm256_broadcast_ss(kptr0);
                    _sum0 = _mm256_comp_fmadd_ps(_w0, _val0, _sum0);

                    tmpptr += 8;
                    kptr0 += 1;
                }

                _mm256_storeu_ps(outptr0, _sum0);
                outptr0 += 8;
            }
            for (; i < size; i++)
            {
                const float* tmpptr = tmp_a[i / 8 + i % 8].data();
                const float* kptr0 = kernel_a[p / 8 + p % 8].data();

                int nn = inch * maxk; // inch always > 0

                float sum0 = bias0;

                __m256 _sum0 = _mm256_setzero_ps();

                for (int j = 0; j < nn; j++)
                {
                    __m256 _val0 = _mm256_load_ps(tmpptr);
                    __m256 _w0 = _mm256_load_ps(kptr0);
                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);

                    tmpptr += 8;
                    kptr0 += 8;
                }

                sum0 += _mm256_reduce_add_ps(_sum0);

                outptr0[0] = sum0;
                outptr0 += 1;
            }
        }
    });
}

void im2col_sgemm_pack8to4_avx(const Tensor& bottom_im2col, Tensor& top_blob, const Tensor& kernel, const Tensor& _bias)
{
    // Tensor bottom_im2col(size, maxk, inch, 32u, 8, opt.workspace_allocator);

    const int size = bottom_im2col.size(2);
    const int maxk = bottom_im2col.size(1);
    const int inch = bottom_im2col.size(0);

    const int outch = top_blob.size(1);

    const float* bias = (_bias.defined()) ? _bias.data_ptr<float>() : nullptr;

    // permute
    Tensor tmp;
    if (size >= 8)
        tmp = otter::empty({size / 8 + (size % 8) / 4 + size % 4, inch, 8 * maxk}, otter::ScalarType::Float8);
    else if (size >= 4)
        tmp = otter::empty({size / 4 + size % 4, inch, 4 * maxk}, otter::ScalarType::Float8);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Float8);
    
    auto tmp_a = tmp.accessor<float, 3, 8>();
    auto bottom_im2col_a = bottom_im2col.accessor<float, 3, 8>();
    auto kernel_a = kernel.accessor<float, 3>();
    auto top_blob_a = top_blob.accessor<float, 4, 4>()[0];
    
    {
        int nn_size = size / 8;
        int remain_size_start = 0;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end))
            {
                int i = remain_size_start + ii * 8;

                float* tmpptr = tmp_a[i / 8].data();

                for (int q = 0; q < inch; q++)
                {
                    const float* img0 = (const float*)bottom_im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++)
                    {
                        // transpose 8x8
                        __m256 _r0 = _mm256_load_ps(img0);
                        __m256 _r1 = _mm256_load_ps(img0 + 8);
                        __m256 _r2 = _mm256_load_ps(img0 + 8 * 2);
                        __m256 _r3 = _mm256_load_ps(img0 + 8 * 3);
                        __m256 _r4 = _mm256_load_ps(img0 + 8 * 4);
                        __m256 _r5 = _mm256_load_ps(img0 + 8 * 5);
                        __m256 _r6 = _mm256_load_ps(img0 + 8 * 6);
                        __m256 _r7 = _mm256_load_ps(img0 + 8 * 7);

                        __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                        __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                        __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
                        __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
                        __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
                        __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);
                        __m256 _tmp6 = _mm256_unpacklo_ps(_r6, _r7);
                        __m256 _tmp7 = _mm256_unpackhi_ps(_r6, _r7);
                        __m256 _tmp8 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmp9 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                        __m256 _tmpa = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmpb = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
                        __m256 _tmpc = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmpd = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
                        __m256 _tmpe = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmpf = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
                        _r0 = _mm256_permute2f128_ps(_tmp8, _tmpc, _MM_SHUFFLE(0, 2, 0, 0));
                        _r1 = _mm256_permute2f128_ps(_tmp9, _tmpd, _MM_SHUFFLE(0, 2, 0, 0));
                        _r2 = _mm256_permute2f128_ps(_tmpa, _tmpe, _MM_SHUFFLE(0, 2, 0, 0));
                        _r3 = _mm256_permute2f128_ps(_tmpb, _tmpf, _MM_SHUFFLE(0, 2, 0, 0));
                        _r4 = _mm256_permute2f128_ps(_tmp8, _tmpc, _MM_SHUFFLE(0, 3, 0, 1));
                        _r5 = _mm256_permute2f128_ps(_tmp9, _tmpd, _MM_SHUFFLE(0, 3, 0, 1));
                        _r6 = _mm256_permute2f128_ps(_tmpa, _tmpe, _MM_SHUFFLE(0, 3, 0, 1));
                        _r7 = _mm256_permute2f128_ps(_tmpb, _tmpf, _MM_SHUFFLE(0, 3, 0, 1));

                        _mm256_store_ps(tmpptr, _r0);
                        _mm256_store_ps(tmpptr + 8, _r1);
                        _mm256_store_ps(tmpptr + 8 * 2, _r2);
                        _mm256_store_ps(tmpptr + 8 * 3, _r3);
                        _mm256_store_ps(tmpptr + 8 * 4, _r4);
                        _mm256_store_ps(tmpptr + 8 * 5, _r5);
                        _mm256_store_ps(tmpptr + 8 * 6, _r6);
                        _mm256_store_ps(tmpptr + 8 * 7, _r7);

                        img0 += size * 8;
                        tmpptr += 64;
                    }
                }
            }
        });

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end))
            {
                int i = remain_size_start + ii * 4;

                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();

                for (int q = 0; q < inch; q++)
                {
                    const float* img0 = (const float*)bottom_im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++)
                    {
                        // transpose 8x4
                        __m256 _r0 = _mm256_load_ps(img0);
                        __m256 _r1 = _mm256_load_ps(img0 + 8);
                        __m256 _r2 = _mm256_load_ps(img0 + 8 * 2);
                        __m256 _r3 = _mm256_load_ps(img0 + 8 * 3);

                        __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                        __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                        __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
                        __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
                        __m256 _tmp4 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmp5 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                        __m256 _tmp6 = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                        __m256 _tmp7 = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
                        _r0 = _mm256_permute2f128_ps(_tmp4, _tmp5, _MM_SHUFFLE(0, 2, 0, 0));
                        _r1 = _mm256_permute2f128_ps(_tmp6, _tmp7, _MM_SHUFFLE(0, 2, 0, 0));
                        _r2 = _mm256_permute2f128_ps(_tmp4, _tmp5, _MM_SHUFFLE(0, 3, 0, 1));
                        _r3 = _mm256_permute2f128_ps(_tmp6, _tmp7, _MM_SHUFFLE(0, 3, 0, 1));

                        _mm256_store_ps(tmpptr, _r0);
                        _mm256_store_ps(tmpptr + 8, _r1);
                        _mm256_store_ps(tmpptr + 8 * 2, _r2);
                        _mm256_store_ps(tmpptr + 8 * 3, _r3);

                        img0 += size * 8;
                        tmpptr += 32;
                    }
                }
            }
        });

        remain_size_start += nn_size << 2;

        otter::parallel_for(remain_size_start, size, 0, [&](int64_t begin, int64_t end) {
            for (const auto i : otter::irange(begin, end))
            {
                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();

                for (int q = 0; q < inch; q++)
                {
                    const float* img0 = (const float*)bottom_im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++)
                    {
                        __m256 _val = _mm256_load_ps(img0);
                        _mm256_store_ps(tmpptr, _val);

                        img0 += size * 8;
                        tmpptr += 8;
                    }
                }
            }
        });
    }

    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = outch >> 1;

    otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto pp : otter::irange(begin, end))
        {
            int p = pp * 2;

            float* outptr0 = top_blob_a[p].data();
            float* outptr1 = top_blob_a[p + 1].data();

            const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + p * 4 : zeros;

            int i = 0;
            for (; i + 7 < size; i += 8)
            {
                float* tmpptr = tmp_a[i / 8].data();
                const float* kptr = kernel_a[p / 2].data();

                int nn = inch * maxk * 8; // inch always > 0

                __m256 _sum0 = _mm256_loadu_ps(biasptr);
                __m256 _sum1 = _sum0;
                __m256 _sum2 = _sum0;
                __m256 _sum3 = _sum0;
                __m256 _sum4 = _sum0;
                __m256 _sum5 = _sum0;
                __m256 _sum6 = _sum0;
                __m256 _sum7 = _sum0;

                for (int j = 0; j < nn; j++)
                {
                    __m256 _w0 = _mm256_load_ps(kptr);

                    __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                    __m256 _val1 = _mm256_broadcast_ss(tmpptr + 1);
                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);
                    __m256 _val2 = _mm256_broadcast_ss(tmpptr + 2);
                    __m256 _val3 = _mm256_broadcast_ss(tmpptr + 3);
                    _sum2 = _mm256_comp_fmadd_ps(_val2, _w0, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val3, _w0, _sum3);
                    __m256 _val4 = _mm256_broadcast_ss(tmpptr + 4);
                    __m256 _val5 = _mm256_broadcast_ss(tmpptr + 5);
                    _sum4 = _mm256_comp_fmadd_ps(_val4, _w0, _sum4);
                    _sum5 = _mm256_comp_fmadd_ps(_val5, _w0, _sum5);
                    __m256 _val6 = _mm256_broadcast_ss(tmpptr + 6);
                    __m256 _val7 = _mm256_broadcast_ss(tmpptr + 7);
                    _sum6 = _mm256_comp_fmadd_ps(_val6, _w0, _sum6);
                    _sum7 = _mm256_comp_fmadd_ps(_val7, _w0, _sum7);

                    tmpptr += 8;
                    kptr += 8;
                }

                _mm_store_ps(outptr0, _mm256_extractf128_ps(_sum0, 0));
                _mm_store_ps(outptr0 + 4, _mm256_extractf128_ps(_sum1, 0));
                _mm_store_ps(outptr0 + 8, _mm256_extractf128_ps(_sum2, 0));
                _mm_store_ps(outptr0 + 12, _mm256_extractf128_ps(_sum3, 0));
                _mm_store_ps(outptr0 + 16, _mm256_extractf128_ps(_sum4, 0));
                _mm_store_ps(outptr0 + 20, _mm256_extractf128_ps(_sum5, 0));
                _mm_store_ps(outptr0 + 24, _mm256_extractf128_ps(_sum6, 0));
                _mm_store_ps(outptr0 + 28, _mm256_extractf128_ps(_sum7, 0));
                _mm_store_ps(outptr1, _mm256_extractf128_ps(_sum0, 1));
                _mm_store_ps(outptr1 + 4, _mm256_extractf128_ps(_sum1, 1));
                _mm_store_ps(outptr1 + 8, _mm256_extractf128_ps(_sum2, 1));
                _mm_store_ps(outptr1 + 12, _mm256_extractf128_ps(_sum3, 1));
                _mm_store_ps(outptr1 + 16, _mm256_extractf128_ps(_sum4, 1));
                _mm_store_ps(outptr1 + 20, _mm256_extractf128_ps(_sum5, 1));
                _mm_store_ps(outptr1 + 24, _mm256_extractf128_ps(_sum6, 1));
                _mm_store_ps(outptr1 + 28, _mm256_extractf128_ps(_sum7, 1));

                outptr0 += 32;
                outptr1 += 32;
            }
            for (; i + 3 < size; i += 4)
            {
                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();
                const float* kptr = kernel_a[p / 2].data();

                int nn = inch * maxk * 8; // inch always > 0

                __m256 _sum0 = _mm256_loadu_ps(biasptr);
                __m256 _sum1 = _sum0;
                __m256 _sum2 = _sum0;
                __m256 _sum3 = _sum0;

                for (int j = 0; j < nn; j++)
                {
                    __m256 _w0 = _mm256_load_ps(kptr);

                    __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                    __m256 _val1 = _mm256_broadcast_ss(tmpptr + 1);
                    _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);
                    __m256 _val2 = _mm256_broadcast_ss(tmpptr + 2);
                    __m256 _val3 = _mm256_broadcast_ss(tmpptr + 3);
                    _sum2 = _mm256_comp_fmadd_ps(_val2, _w0, _sum2);
                    _sum3 = _mm256_comp_fmadd_ps(_val3, _w0, _sum3);

                    tmpptr += 4;
                    kptr += 8;
                }

                _mm_store_ps(outptr0, _mm256_extractf128_ps(_sum0, 0));
                _mm_store_ps(outptr0 + 4, _mm256_extractf128_ps(_sum1, 0));
                _mm_store_ps(outptr0 + 8, _mm256_extractf128_ps(_sum2, 0));
                _mm_store_ps(outptr0 + 12, _mm256_extractf128_ps(_sum3, 0));
                _mm_store_ps(outptr1, _mm256_extractf128_ps(_sum0, 1));
                _mm_store_ps(outptr1 + 4, _mm256_extractf128_ps(_sum1, 1));
                _mm_store_ps(outptr1 + 8, _mm256_extractf128_ps(_sum2, 1));
                _mm_store_ps(outptr1 + 12, _mm256_extractf128_ps(_sum3, 1));

                outptr0 += 16;
                outptr1 += 16;
            }
            for (; i < size; i++)
            {
                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();
                const float* kptr = kernel_a[p / 2].data();

                int nn = inch * maxk * 8; // inch always > 0

                __m256 _sum = _mm256_loadu_ps(biasptr);

                for (int j = 0; j < nn; j++)
                {
                    __m256 _w0 = _mm256_load_ps(kptr);
                    __m256 _val0 = _mm256_broadcast_ss(tmpptr);
                    _sum = _mm256_comp_fmadd_ps(_val0, _w0, _sum);

                    tmpptr += 1;
                    kptr += 8;
                }

                _mm_store_ps(outptr0, _mm256_extractf128_ps(_sum, 0));
                _mm_store_ps(outptr1, _mm256_extractf128_ps(_sum, 1));

                outptr0 += 4;
                outptr1 += 4;
            }
        }
    });

    remain_outch_start += nn_outch << 1;

    otter::parallel_for(remain_outch_start, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end))
        {
            float* outptr0 = top_blob_a[p].data();

            const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + p * 4 : zeros;

            int i = 0;
            for (; i + 7 < size; i += 8)
            {
                float* tmpptr = tmp_a[i / 8].data();
                const float* kptr = kernel_a[p / 2 + p % 2].data();

                int nn = inch * maxk * 8; // inch always > 0

                __m128 _sum0 = _mm_loadu_ps(biasptr);
                __m128 _sum1 = _sum0;
                __m128 _sum2 = _sum0;
                __m128 _sum3 = _sum0;
                __m128 _sum4 = _sum0;
                __m128 _sum5 = _sum0;
                __m128 _sum6 = _sum0;
                __m128 _sum7 = _sum0;

                for (int j = 0; j < nn; j++)
                {
                    __m128 _w0 = _mm_load_ps(kptr);

                    __m128 _val0 = _mm_load1_ps(tmpptr);
                    __m128 _val1 = _mm_load1_ps(tmpptr + 1);
                    _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                    __m128 _val2 = _mm_load1_ps(tmpptr + 2);
                    __m128 _val3 = _mm_load1_ps(tmpptr + 3);
                    _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val3, _w0, _sum3);
                    __m128 _val4 = _mm_load1_ps(tmpptr + 4);
                    __m128 _val5 = _mm_load1_ps(tmpptr + 5);
                    _sum4 = _mm_comp_fmadd_ps(_val4, _w0, _sum4);
                    _sum5 = _mm_comp_fmadd_ps(_val5, _w0, _sum5);
                    __m128 _val6 = _mm_load1_ps(tmpptr + 6);
                    __m128 _val7 = _mm_load1_ps(tmpptr + 7);
                    _sum6 = _mm_comp_fmadd_ps(_val6, _w0, _sum6);
                    _sum7 = _mm_comp_fmadd_ps(_val7, _w0, _sum7);

                    tmpptr += 8;
                    kptr += 4;
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
            for (; i + 3 < size; i += 4)
            {
                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();
                const float* kptr = kernel_a[p / 2 + p % 2].data();

                int nn = inch * maxk * 8; // inch always > 0

                __m128 _sum0 = _mm_loadu_ps(biasptr);
                __m128 _sum1 = _sum0;
                __m128 _sum2 = _sum0;
                __m128 _sum3 = _sum0;

                for (int j = 0; j < nn; j++)
                {
                    __m128 _w0 = _mm_load_ps(kptr);

                    __m128 _val0 = _mm_load1_ps(tmpptr);
                    __m128 _val1 = _mm_load1_ps(tmpptr + 1);
                    _sum0 = _mm_comp_fmadd_ps(_val0, _w0, _sum0);
                    _sum1 = _mm_comp_fmadd_ps(_val1, _w0, _sum1);
                    __m128 _val2 = _mm_load1_ps(tmpptr + 2);
                    __m128 _val3 = _mm_load1_ps(tmpptr + 3);
                    _sum2 = _mm_comp_fmadd_ps(_val2, _w0, _sum2);
                    _sum3 = _mm_comp_fmadd_ps(_val3, _w0, _sum3);

                    tmpptr += 4;
                    kptr += 4;
                }

                _mm_store_ps(outptr0, _sum0);
                _mm_store_ps(outptr0 + 4, _sum1);
                _mm_store_ps(outptr0 + 8, _sum2);
                _mm_store_ps(outptr0 + 12, _sum3);

                outptr0 += 16;
            }
            for (; i < size; i++)
            {
                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();
                const float* kptr = kernel_a[p / 2 + p % 2].data();

                int nn = inch * maxk * 8; // inch always > 0

                __m128 _sum = _mm_loadu_ps(biasptr);

                for (int j = 0; j < nn; j++)
                {
                    __m128 _w0 = _mm_load_ps(kptr);
                    __m128 _val0 = _mm_load1_ps(tmpptr);
                    _sum = _mm_comp_fmadd_ps(_val0, _w0, _sum);

                    tmpptr += 1;
                    kptr += 4;
                }

                _mm_store_ps(outptr0, _sum);

                outptr0 += 4;
            }
        }
    });
}

void convolution_im2col_sgemm_transform_kernel_pack1to8_avx(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h) {
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8b-4a-maxk-inch/4a-outch/8b
    Tensor kernel = _kernel.view({outch, inch, maxk});
    kernel_tm = otter::empty({outch / 8, inch, 8 * maxk}, otter::ScalarType::Float);
    
    auto kernel_a = kernel.accessor<float, 3>();
    auto kernel_tm_a = kernel_tm.accessor<float, 3>();

    int q = 0;
    for (; q + 7 < outch; q += 8)
    {
        const auto k0 = kernel_a[q + 0];
        const auto k1 = kernel_a[q + 1];
        const auto k2 = kernel_a[q + 2];
        const auto k3 = kernel_a[q + 3];
        const auto k4 = kernel_a[q + 4];
        const auto k5 = kernel_a[q + 5];
        const auto k6 = kernel_a[q + 6];
        const auto k7 = kernel_a[q + 7];

        float* g00 = kernel_tm_a[q / 8].data();

        for (int p = 0; p < inch; p++)
        {
            const float* k00 = k0[p].data();
            const float* k10 = k1[p].data();
            const float* k20 = k2[p].data();
            const float* k30 = k3[p].data();
            const float* k40 = k4[p].data();
            const float* k50 = k5[p].data();
            const float* k60 = k6[p].data();
            const float* k70 = k7[p].data();

            for (int k = 0; k < maxk; k++)
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
}

void convolution_im2col_sgemm_transform_kernel_pack4to8_avx(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8b-4a-maxk-inch/4a-outch/8b
    Tensor kernel = _kernel.view({outch, inch, maxk});
    kernel_tm = otter::empty({outch / 8, inch / 4, 32 * maxk}, otter::ScalarType::Float);
    
    auto kernel_a = kernel.accessor<float, 3>();
    auto kernel_tm_a = kernel_tm.accessor<float, 3>();

    for (int q = 0; q + 7 < outch; q += 8)
    {
        float* g00 = kernel_tm_a[q / 8].data();

        for (int p = 0; p + 3 < inch; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const float* k00 = kernel_a[q + j][p + i].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
    }
}

void convolution_im2col_sgemm_transform_kernel_pack8_avx(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8b-8a-maxk-inch/8a-outch/8b
    Tensor kernel = _kernel.view({outch, inch, maxk});
    kernel_tm = otter::empty({outch / 8, inch / 8, 64 * maxk}, otter::ScalarType::Float);
    
    auto kernel_a = kernel.accessor<float, 3>();
    auto kernel_tm_a = kernel_tm.accessor<float, 3>();

    for (int q = 0; q + 7 < outch; q += 8)
    {
        float* g00 = kernel_tm_a[q / 8].data();

        for (int p = 0; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const float* k00 = kernel_a[q + j][p + i].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
    }
}

void convolution_im2col_sgemm_transform_kernel_pack8to1_avx(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = pb-pa-maxk-inch/pa-outch/pb
    Tensor kernel = _kernel.view({outch, inch, maxk});
    kernel_tm = otter::empty({outch / 8 + outch % 8, inch / 8, 8 * 8 * maxk}, otter::ScalarType::Float);
    
    auto kernel_a = kernel.accessor<float, 3>();
    auto kernel_tm_a = kernel_tm.accessor<float, 3>();

    int q = 0;
    for (; q + 7 < outch; q += 8)
    {
        float* g00 = kernel_tm_a[q / 8].data();

        for (int p = 0; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const float* k00 = kernel_a[q + j][p + i].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
    }
    for (; q < outch; q++)
    {
        const auto k0 = kernel_a[q];

        float* g00 = kernel_tm_a[q / 8 + q % 8].data();

        for (int p = 0; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 8; j++)
                {
                    const float* k00 = k0[p + j].data();

                    g00[0] = k00[k];

                    g00++;
                }
            }
        }
    }
}

void convolution_im2col_sgemm_transform_kernel_pack8to4_avx(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8b-8a-maxk-inch/8a-outch/8b
    Tensor kernel = _kernel.view({outch, inch, maxk});
    kernel_tm = otter::empty({outch / 8 + (outch % 8) / 4, inch / 8, 64 * maxk}, otter::ScalarType::Float);
    
    auto kernel_a = kernel.accessor<float, 3>();
    auto kernel_tm_a = kernel_tm.accessor<float, 3>();

    int q = 0;
    for (; q + 7 < outch; q += 8)
    {
        float* g00 = kernel_tm_a[q / 8].data();

        for (int p = 0; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const float* k00 = kernel_a[q + j][p + i].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
    }
    for (; q + 3 < outch; q += 4)
    {
        float* g00 = kernel_tm_a[q / 8 + (q % 8) / 4].data();

        for (int p = 0; p + 7 < inch; p += 8)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const float* k00 = kernel_a[q + j][p + i].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
    }
}

Tensor& sgemm_conv2d_pack1to8_x86_out(
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
    output.resize_({output_size[0], output_size[1] / 8, output_size[2], output_size[3]});
    
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
        convolution_im2col_sgemm_transform_kernel_pack1to8_avx(weight, kernel_tf, inch, outch * 8, kernel_w, kernel_h);
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    Tensor im2col = otter::im2col_cpu(self, kernel_size, stride, padding, dilation).view({inch, maxk, size});
    
    im2col_sgemm_pack1to8_avx(im2col, output, kernel_tf, bias);
    
    return output;
}
    
Tensor sgemm_conv2d_pack1to8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    Tensor output = otter::empty({}, otter::ScalarType::Float8);
    sgemm_conv2d_pack1to8_x86_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
    
    return output;
}

Tensor& sgemm_conv2d_pack4to8_x86_out(
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
    output.resize_({output_size[0], output_size[1] / 8, output_size[2], output_size[3]});
    
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
        convolution_im2col_sgemm_transform_kernel_pack4to8_avx(weight, kernel_tf, inch * 4, outch * 8, kernel_w, kernel_h);
    
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
                float* ptr = im2col_a[p].data();

                for (int u = 0; u < kernel_h; u++)
                {
                    for (int v = 0; v < kernel_w; v++)
                    {
                        const float* sptr = img[dilation_h * u].data() + dilation_w * v * 4;

                        for (int i = 0; i < outh; i++)
                        {
                            int j = 0;
                            for (; j < outw; j++)
                            {
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
    
    im2col_sgemm_pack4to8_avx(im2col, output, kernel_tf, bias);
    
    return output;
}
    
Tensor sgemm_conv2d_pack4to8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    Tensor output = otter::empty({}, otter::ScalarType::Float8);
    sgemm_conv2d_pack4to8_x86_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
    
    return output;
}

Tensor& sgemm_conv2d_pack8_x86_out(
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
    output.resize_({output_size[0], output_size[1] / 8, output_size[2], output_size[3]});
    
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
        convolution_im2col_sgemm_transform_kernel_pack8_avx(weight, kernel_tf, inch * 8, outch * 8, kernel_w, kernel_h);
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    
    Tensor im2col = otter::empty({inch, maxk, size}, ScalarType::Float8);
    
    auto input_a = input.accessor<float, 3, 8>();
    auto im2col_a = im2col.accessor<float, 3, 8>();
    // im2col
    {
        const int gap = (w * stride_h - outw * stride_w) * 8;

        otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
            for (const auto p : otter::irange(begin, end)) {
                const auto img = input_a[p];
                float* ptr = im2col_a[p].data();

                for (int u = 0; u < kernel_h; u++)
                {
                    for (int v = 0; v < kernel_w; v++)
                    {
                        const float* sptr = img[dilation_h * u].data() + dilation_w * v * 8;

                        for (int i = 0; i < outh; i++)
                        {
                            int j = 0;
                            for (; j < outw; j++)
                            {
                                __m256 _v = _mm256_load_ps(sptr);
                                _mm256_store_ps(ptr, _v);

                                sptr += stride_w * 8;
                                ptr += 8;
                            }

                            sptr += gap;
                        }
                    }
                }
            }
        });
    }
    
//    std::cout << im2col << std::endl;
//    std::cout << otter::im2col_cpu(self.packing(1), kernel_size, stride, padding, dilation).view({inch * 8, maxk, size}).packing(8) << std::endl;
    
    im2col_sgemm_pack8_avx(im2col, output, kernel_tf, bias);
    
    return output;
}
    
Tensor sgemm_conv2d_pack8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    Tensor output = otter::empty({}, otter::ScalarType::Float8);
    sgemm_conv2d_pack8_x86_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
    
    return output;
}

Tensor& sgemm_conv2d_pack8to1_x86_out(
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
        convolution_im2col_sgemm_transform_kernel_pack8to1_avx(weight, kernel_tf, inch * 8, outch, kernel_w, kernel_h);
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    
    Tensor im2col = otter::empty({inch, maxk, size}, ScalarType::Float8);
    
    auto input_a = input.accessor<float, 3, 8>();
    auto im2col_a = im2col.accessor<float, 3, 8>();
    // im2col
    {
        const int gap = (w * stride_h - outw * stride_w) * 8;

        otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
            for (const auto p : otter::irange(begin, end))
            {
                const auto img = input_a[p];
                float* ptr = im2col_a[p].data();

                for (int u = 0; u < kernel_h; u++)
                {
                    for (int v = 0; v < kernel_w; v++)
                    {
                        const float* sptr = img[dilation_h * u].data() + dilation_w * v * 8;

                        for (int i = 0; i < outh; i++)
                        {
                            int j = 0;
                            for (; j < outw; j++)
                            {
                                __m256 _val = _mm256_load_ps(sptr);
                                _mm256_store_ps(ptr, _val);

                                sptr += stride_w * 8;
                                ptr += 8;
                            }

                            sptr += gap;
                        }
                    }
                }
            }
        });
    }
    
    im2col_sgemm_pack8to1_avx(im2col, output, kernel_tf, bias);
    
    return output;
}
    
Tensor sgemm_conv2d_pack8to1_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    Tensor output = otter::empty({}, otter::ScalarType::Float);
    sgemm_conv2d_pack8to1_x86_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
    
    return output;
}

Tensor& sgemm_conv2d_pack8to4_x86_out(
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
        convolution_im2col_sgemm_transform_kernel_pack8to4_avx(weight, kernel_tf, inch * 8, outch * 4, kernel_w, kernel_h);
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    
    Tensor im2col = otter::empty({inch, maxk, size}, ScalarType::Float8);
    
    auto input_a = input.accessor<float, 3, 8>();
    auto im2col_a = im2col.accessor<float, 3, 8>();
    // im2col
    {
        const int gap = (w * stride_h - outw * stride_w) * 8;

        otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
            for (const auto p : otter::irange(begin, end))
            {
                const auto img = input_a[p];
                float* ptr = im2col_a[p].data();

                for (int u = 0; u < kernel_h; u++)
                {
                    for (int v = 0; v < kernel_w; v++)
                    {
                        const float* sptr = img[dilation_h * u].data() + dilation_w * v * 8;

                        for (int i = 0; i < outh; i++)
                        {
                            int j = 0;
                            for (; j < outw; j++)
                            {
                                __m256 _v = _mm256_load_ps(sptr);
                                _mm256_store_ps(ptr, _v);

                                sptr += stride_w * 8;
                                ptr += 8;
                            }

                            sptr += gap;
                        }
                    }
                }
            }
        });
    }
    
    im2col_sgemm_pack8to4_avx(im2col, output, kernel_tf, bias);
    
    return output;
}
    
Tensor sgemm_conv2d_pack8to4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    Tensor output = otter::empty({}, otter::ScalarType::Float4);
    sgemm_conv2d_pack8to4_x86_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
    
    return output;
}

Tensor conv2d_1x1s1_sgemm_pack1to8_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
    output.resize_({output_size[0], output_size[1] / 8, output_size[2], output_size[3]});
    
    int inch = self.size(1);
    int outch = output.size(1);
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        convolution_im2col_sgemm_transform_kernel_pack1to8_avx(weight, kernel_tf, inch, outch * 8, 1, 1);
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    int h = input.size(1);
    const int size = w * h;
    
    Tensor im2col = input.view({-1, 1, size});
    
    im2col_sgemm_pack1to8_avx(im2col, output, kernel_tf, bias);
    
    return output;
}

Tensor conv2d_1x1s1_sgemm_pack1to8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
               
    auto output = otter::empty({}, otter::ScalarType::Float8);
    
    return conv2d_1x1s1_sgemm_pack1to8_x86_out(self, weight, weight_o, bias, padding, output);
}

Tensor conv2d_1x1s1_sgemm_pack4to8_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
    output.resize_({output_size[0], output_size[1] / 8, output_size[2], output_size[3]});
    
    int inch = self.size(1);
    int outch = output.size(1);
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        convolution_im2col_sgemm_transform_kernel_pack4to8_avx(weight, kernel_tf, inch * 4, outch * 8, 1, 1);
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    int h = input.size(1);
    const int size = w * h;
    
    Tensor im2col = input.view({-1, 1, size});
    
    im2col_sgemm_pack4to8_avx(im2col, output, kernel_tf, bias);
    
    return output;
}

Tensor conv2d_1x1s1_sgemm_pack4to8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
               
    auto output = otter::empty({}, otter::ScalarType::Float8);
    
    return conv2d_1x1s1_sgemm_pack4to8_x86_out(self, weight, weight_o, bias, padding, output);
}

Tensor conv2d_1x1s1_sgemm_pack8_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
    output.resize_({output_size[0], output_size[1] / 8, output_size[2], output_size[3]});
    
    int inch = self.size(1);
    int outch = output.size(1);
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        convolution_im2col_sgemm_transform_kernel_pack8_avx(weight, kernel_tf, inch * 8, outch * 8, 1, 1);
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    int h = input.size(1);
    const int size = w * h;
    
    Tensor im2col = input.view({-1, 1, size});
    
    im2col_sgemm_pack8_avx(im2col, output, kernel_tf, bias);
    
    return output;
}

Tensor conv2d_1x1s1_sgemm_pack8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
               
    auto output = otter::empty({}, otter::ScalarType::Float8);
    
    return conv2d_1x1s1_sgemm_pack8_x86_out(self, weight, weight_o, bias, padding, output);
}

Tensor conv2d_1x1s2_sgemm_pack8_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {2, 2}, padding);
    output.resize_({output_size[0], output_size[1] / 8, output_size[2], output_size[3]});
    
    int inch = self.size(1);
    int outch = output.size(1);
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        convolution_im2col_sgemm_transform_kernel_pack8_avx(weight, kernel_tf, inch * 8, outch * 8, 1, 1);
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    int channels = input.size(0);
    
    int outw = output_size[3];
    int outh = output_size[2];
    
    const int tailstep = (w - 2 * outw + w) * 8;
    
    Tensor shrinked = otter::empty({channels, outh, outw}, otter::ScalarType::Float8);
    
    auto input_a = input.accessor<float, 3, 8>();
    auto shrinked_a = shrinked.accessor<float, 3, 8>();
    
    otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            const float* r0 = input_a[p].data();
            float* outptr = shrinked_a[p].data();

            for (int i = 0; i < outh; i++) {
                int j = 0;
                for (; j < outw; j++) {
                    __m256 _v = _mm256_loadu_ps(r0);
                    _mm256_storeu_ps(outptr, _v);

                    r0 += 16;
                    outptr += 8;
                }

                r0 += tailstep;
            }
        }
    });
    
    const int size = outw * outh;
    
    Tensor im2col = shrinked.view({-1, 1, size});
    
    im2col_sgemm_pack8_avx(im2col, output, kernel_tf, bias);
    
    return output;
}

Tensor conv2d_1x1s2_sgemm_pack8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
               
    auto output = otter::empty({}, otter::ScalarType::Float8);
    
    return conv2d_1x1s2_sgemm_pack8_x86_out(self, weight, weight_o, bias, padding, output);
}

Tensor conv2d_1x1s1_sgemm_pack8to1_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
    output.resize_({output_size[0], output_size[1], output_size[2], output_size[3]});
    
    int inch = self.size(1);
    int outch = output.size(1);
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        convolution_im2col_sgemm_transform_kernel_pack8to1_avx(weight, kernel_tf, inch * 8, outch, 1, 1);
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    int h = input.size(1);
    const int size = w * h;
    
    Tensor im2col = input.view({-1, 1, size});
    
    im2col_sgemm_pack8to1_avx(im2col, output, kernel_tf, bias);
    
    return output;
}

Tensor conv2d_1x1s1_sgemm_pack8to1_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
               
    auto output = otter::empty({}, otter::ScalarType::Float);
    
    return conv2d_1x1s1_sgemm_pack8to1_x86_out(self, weight, weight_o, bias, padding, output);
}

Tensor conv2d_1x1s1_sgemm_pack8to4_x86_out(
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
        convolution_im2col_sgemm_transform_kernel_pack8to4_avx(weight, kernel_tf, inch * 8, outch * 4, 1, 1);
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    int h = input.size(1);
    const int size = w * h;
    
    Tensor im2col = input.view({-1, 1, size});
    
    im2col_sgemm_pack8to4_avx(im2col, output, kernel_tf, bias);
    
    return output;
}

Tensor conv2d_1x1s1_sgemm_pack8to4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
               
    auto output = otter::empty({}, otter::ScalarType::Float4);
    
    return conv2d_1x1s1_sgemm_pack8to4_x86_out(self, weight, weight_o, bias, padding, output);
}

static void convolution_winograd_dot_pack8_avx(Tensor& bottom_blob_tm, int outch, const Tensor& kernel_tm, Tensor& top_blob_tm) {
    // Tensor bottom_blob_tm(tiles, 16/36/64, inch, 32u, 4, opt.workspace_allocator);

    const int tiles = bottom_blob_tm.size(2);
    const int batch = bottom_blob_tm.size(1);
    const int inch = bottom_blob_tm.size(0);

    // permute
    Tensor bottom_blob_tm2;
    if (tiles >= 12)
        bottom_blob_tm2 = otter::empty({batch, tiles / 12 + (tiles % 12) / 8 + (tiles % 12 % 8) / 4 + (tiles % 12 % 4) / 2 + tiles % 12 % 2, 12 * inch}, otter::ScalarType::Float8);
    else if (tiles >= 8)
        bottom_blob_tm2 = otter::empty({batch, tiles / 8 + (tiles % 8) / 4 + (tiles % 4) / 2 + tiles % 2, 8 * inch}, otter::ScalarType::Float8);
    else if (tiles >= 4)
        bottom_blob_tm2 = otter::empty({batch, tiles / 4 + (tiles % 4) / 2 + tiles % 2, 4 * inch}, otter::ScalarType::Float8);
    else if (tiles >= 2)
        bottom_blob_tm2 = otter::empty({batch, tiles / 2 + tiles % 2, 2 * inch}, otter::ScalarType::Float8);
    else // if (tiles >= 1)
        bottom_blob_tm2 = otter::empty({batch, tiles, 1 * inch}, otter::ScalarType::Float8);
    
    auto bottom_blob_tm_a = bottom_blob_tm.accessor<float, 3, 8>();
    auto bottom_blob_tm2_a = bottom_blob_tm2.accessor<float, 3, 8>();
    
    int bottom_blob_tm_cstep = tiles * batch;

    otter::parallel_for(0, batch, 0, [&](int64_t begin, int64_t end) {
        for (const auto r : otter::irange(begin, end)) {
            auto tm2 = bottom_blob_tm2_a[r];

            // tile
            int i = 0;

            for (; i + 11 < tiles; i += 12)
            {
                float* tmpptr = tm2[i / 12].data();

                const float* r0 = bottom_blob_tm_a.data();

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 8x12
                    __m256 _r0 = _mm256_load_ps(r0);
                    __m256 _r1 = _mm256_load_ps(r0 + 8);
                    __m256 _r2 = _mm256_load_ps(r0 + 8 * 2);
                    __m256 _r3 = _mm256_load_ps(r0 + 8 * 3);
                    __m256 _r4 = _mm256_load_ps(r0 + 8 * 4);
                    __m256 _r5 = _mm256_load_ps(r0 + 8 * 5);
                    __m256 _r6 = _mm256_load_ps(r0 + 8 * 6);
                    __m256 _r7 = _mm256_load_ps(r0 + 8 * 7);
                    __m256 _r8 = _mm256_load_ps(r0 + 8 * 8);
                    __m256 _r9 = _mm256_load_ps(r0 + 8 * 9);
                    __m256 _ra = _mm256_load_ps(r0 + 8 * 10);
                    __m256 _rb = _mm256_load_ps(r0 + 8 * 11);

                    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
                    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
                    __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
                    __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);
                    __m256 _tmp6 = _mm256_unpacklo_ps(_r6, _r7);
                    __m256 _tmp7 = _mm256_unpackhi_ps(_r6, _r7);
                    __m256 _tmp8 = _mm256_unpacklo_ps(_r8, _r9);
                    __m256 _tmp9 = _mm256_unpackhi_ps(_r8, _r9);
                    __m256 _tmpa = _mm256_unpacklo_ps(_ra, _rb);
                    __m256 _tmpb = _mm256_unpackhi_ps(_ra, _rb);
                    __m256 _tmpc = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpd = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpe = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpf = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpg = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmph = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpi = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpj = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpk = _mm256_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpl = _mm256_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpm = _mm256_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpn = _mm256_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(3, 2, 3, 2));
                    _r0 = _mm256_permute2f128_ps(_tmpc, _tmpg, _MM_SHUFFLE(0, 2, 0, 0));
                    _r1 = _mm256_permute2f128_ps(_tmpk, _tmpd, _MM_SHUFFLE(0, 2, 0, 0));
                    _r2 = _mm256_permute2f128_ps(_tmph, _tmpl, _MM_SHUFFLE(0, 2, 0, 0));
                    _r3 = _mm256_permute2f128_ps(_tmpe, _tmpi, _MM_SHUFFLE(0, 2, 0, 0));
                    _r4 = _mm256_permute2f128_ps(_tmpm, _tmpf, _MM_SHUFFLE(0, 2, 0, 0));
                    _r5 = _mm256_permute2f128_ps(_tmpj, _tmpn, _MM_SHUFFLE(0, 2, 0, 0));
                    _r6 = _mm256_permute2f128_ps(_tmpc, _tmpg, _MM_SHUFFLE(0, 3, 0, 1));
                    _r7 = _mm256_permute2f128_ps(_tmpk, _tmpd, _MM_SHUFFLE(0, 3, 0, 1));
                    _r8 = _mm256_permute2f128_ps(_tmph, _tmpl, _MM_SHUFFLE(0, 3, 0, 1));
                    _r9 = _mm256_permute2f128_ps(_tmpe, _tmpi, _MM_SHUFFLE(0, 3, 0, 1));
                    _ra = _mm256_permute2f128_ps(_tmpm, _tmpf, _MM_SHUFFLE(0, 3, 0, 1));
                    _rb = _mm256_permute2f128_ps(_tmpj, _tmpn, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_store_ps(tmpptr, _r0);
                    _mm256_store_ps(tmpptr + 8, _r1);
                    _mm256_store_ps(tmpptr + 8 * 2, _r2);
                    _mm256_store_ps(tmpptr + 8 * 3, _r3);
                    _mm256_store_ps(tmpptr + 8 * 4, _r4);
                    _mm256_store_ps(tmpptr + 8 * 5, _r5);
                    _mm256_store_ps(tmpptr + 8 * 6, _r6);
                    _mm256_store_ps(tmpptr + 8 * 7, _r7);
                    _mm256_store_ps(tmpptr + 8 * 8, _r8);
                    _mm256_store_ps(tmpptr + 8 * 9, _r9);
                    _mm256_store_ps(tmpptr + 8 * 10, _ra);
                    _mm256_store_ps(tmpptr + 8 * 11, _rb);

                    tmpptr += 96;
                    r0 += bottom_blob_tm_cstep * 8;
                }
            }
            for (; i + 7 < tiles; i += 8)
            {
                float* tmpptr = tm2[i / 12 + (i % 12) / 8].data();

                const float* r0 = bottom_blob_tm_a.data();

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 8x8
                    __m256 _r0 = _mm256_load_ps(r0);
                    __m256 _r1 = _mm256_load_ps(r0 + 8);
                    __m256 _r2 = _mm256_load_ps(r0 + 8 * 2);
                    __m256 _r3 = _mm256_load_ps(r0 + 8 * 3);
                    __m256 _r4 = _mm256_load_ps(r0 + 8 * 4);
                    __m256 _r5 = _mm256_load_ps(r0 + 8 * 5);
                    __m256 _r6 = _mm256_load_ps(r0 + 8 * 6);
                    __m256 _r7 = _mm256_load_ps(r0 + 8 * 7);

                    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
                    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
                    __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
                    __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);
                    __m256 _tmp6 = _mm256_unpacklo_ps(_r6, _r7);
                    __m256 _tmp7 = _mm256_unpackhi_ps(_r6, _r7);
                    __m256 _tmp8 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmp9 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpa = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpb = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpc = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpd = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmpe = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmpf = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
                    _r0 = _mm256_permute2f128_ps(_tmp8, _tmpc, _MM_SHUFFLE(0, 2, 0, 0));
                    _r1 = _mm256_permute2f128_ps(_tmp9, _tmpd, _MM_SHUFFLE(0, 2, 0, 0));
                    _r2 = _mm256_permute2f128_ps(_tmpa, _tmpe, _MM_SHUFFLE(0, 2, 0, 0));
                    _r3 = _mm256_permute2f128_ps(_tmpb, _tmpf, _MM_SHUFFLE(0, 2, 0, 0));
                    _r4 = _mm256_permute2f128_ps(_tmp8, _tmpc, _MM_SHUFFLE(0, 3, 0, 1));
                    _r5 = _mm256_permute2f128_ps(_tmp9, _tmpd, _MM_SHUFFLE(0, 3, 0, 1));
                    _r6 = _mm256_permute2f128_ps(_tmpa, _tmpe, _MM_SHUFFLE(0, 3, 0, 1));
                    _r7 = _mm256_permute2f128_ps(_tmpb, _tmpf, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_store_ps(tmpptr, _r0);
                    _mm256_store_ps(tmpptr + 8, _r1);
                    _mm256_store_ps(tmpptr + 8 * 2, _r2);
                    _mm256_store_ps(tmpptr + 8 * 3, _r3);
                    _mm256_store_ps(tmpptr + 8 * 4, _r4);
                    _mm256_store_ps(tmpptr + 8 * 5, _r5);
                    _mm256_store_ps(tmpptr + 8 * 6, _r6);
                    _mm256_store_ps(tmpptr + 8 * 7, _r7);

                    tmpptr += 64;
                    r0 += bottom_blob_tm_cstep * 8;
                }
            }
            for (; i + 3 < tiles; i += 4)
            {
                float* tmpptr = tm2[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4].data();

                const float* r0 = bottom_blob_tm_a.data();

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 8x4
                    __m256 _r0 = _mm256_load_ps(r0);
                    __m256 _r1 = _mm256_load_ps(r0 + 8);
                    __m256 _r2 = _mm256_load_ps(r0 + 8 * 2);
                    __m256 _r3 = _mm256_load_ps(r0 + 8 * 3);

                    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
                    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
                    __m256 _tmp4 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmp5 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
                    __m256 _tmp6 = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
                    __m256 _tmp7 = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
                    _r0 = _mm256_permute2f128_ps(_tmp4, _tmp5, _MM_SHUFFLE(0, 2, 0, 0));
                    _r1 = _mm256_permute2f128_ps(_tmp6, _tmp7, _MM_SHUFFLE(0, 2, 0, 0));
                    _r2 = _mm256_permute2f128_ps(_tmp4, _tmp5, _MM_SHUFFLE(0, 3, 0, 1));
                    _r3 = _mm256_permute2f128_ps(_tmp6, _tmp7, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_store_ps(tmpptr, _r0);
                    _mm256_store_ps(tmpptr + 8, _r1);
                    _mm256_store_ps(tmpptr + 8 * 2, _r2);
                    _mm256_store_ps(tmpptr + 8 * 3, _r3);

                    tmpptr += 32;
                    r0 += bottom_blob_tm_cstep * 8;
                }
            }
            for (; i + 1 < tiles; i += 2)
            {
                float* tmpptr = tm2[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2].data();

                const float* r0 = bottom_blob_tm_a.data();

                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    // transpose 8x2
                    __m256 _r0 = _mm256_load_ps(r0);
                    __m256 _r1 = _mm256_load_ps(r0 + 8);

                    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
                    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
                    _r0 = _mm256_permute2f128_ps(_tmp0, _tmp1, _MM_SHUFFLE(0, 2, 0, 0));
                    _r1 = _mm256_permute2f128_ps(_tmp0, _tmp1, _MM_SHUFFLE(0, 3, 0, 1));

                    _mm256_store_ps(tmpptr, _r0);
                    _mm256_store_ps(tmpptr + 8, _r1);

                    tmpptr += 16;
                    r0 += bottom_blob_tm_cstep * 8;
                }
            }

            for (; i < tiles; i++)
            {
                float* tmpptr = tm2[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2].data();

                const float* r0 = bottom_blob_tm_a.data();
                r0 += (r * tiles + i) * 8;

                for (int q = 0; q < inch; q++)
                {
                    __m256 _val = _mm256_load_ps(r0);
                    _mm256_store_ps(tmpptr, _val);

                    tmpptr += 8;
                    r0 += bottom_blob_tm_cstep * 8;
                }
            }
        }
    });

    bottom_blob_tm.reset();
    // permute end

    top_blob_tm = otter::empty({outch, batch, tiles}, otter::ScalarType::Float8);
    auto top_blob_tm_a = top_blob_tm.accessor<float, 3, 8>();
    auto kernel_tm_a = kernel_tm.accessor<float, 3, 64>();

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            float* output0_tm = top_blob_tm_a[p].data();

            const auto kernel0_tm = kernel_tm_a[p];

            for (int r = 0; r < batch; r++)
            {
                const auto bb2 = bottom_blob_tm2_a[r];

                int i = 0;
                for (; i + 11 < tiles; i += 12)
                {
                    const float* r0 = bb2[i / 12].data();
                    const float* k0 = kernel0_tm[r].data();

                    int nn = inch * 8; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    __m256 _sum4 = _mm256_setzero_ps();
                    __m256 _sum5 = _mm256_setzero_ps();
                    __m256 _sum6 = _mm256_setzero_ps();
                    __m256 _sum7 = _mm256_setzero_ps();
                    __m256 _sum8 = _mm256_setzero_ps();
                    __m256 _sum9 = _mm256_setzero_ps();
                    __m256 _suma = _mm256_setzero_ps();
                    __m256 _sumb = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _w0 = _mm256_load_ps(k0);

                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        __m256 _val1 = _mm256_broadcast_ss(r0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);
                        __m256 _val2 = _mm256_broadcast_ss(r0 + 2);
                        __m256 _val3 = _mm256_broadcast_ss(r0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val2, _w0, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val3, _w0, _sum3);
                        __m256 _val4 = _mm256_broadcast_ss(r0 + 4);
                        __m256 _val5 = _mm256_broadcast_ss(r0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val4, _w0, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val5, _w0, _sum5);
                        __m256 _val6 = _mm256_broadcast_ss(r0 + 6);
                        __m256 _val7 = _mm256_broadcast_ss(r0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val6, _w0, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val7, _w0, _sum7);
                        __m256 _val8 = _mm256_broadcast_ss(r0 + 8);
                        __m256 _val9 = _mm256_broadcast_ss(r0 + 9);
                        _sum8 = _mm256_comp_fmadd_ps(_val8, _w0, _sum8);
                        _sum9 = _mm256_comp_fmadd_ps(_val9, _w0, _sum9);
                        __m256 _vala = _mm256_broadcast_ss(r0 + 10);
                        __m256 _valb = _mm256_broadcast_ss(r0 + 11);
                        _suma = _mm256_comp_fmadd_ps(_vala, _w0, _suma);
                        _sumb = _mm256_comp_fmadd_ps(_valb, _w0, _sumb);

                        r0 += 12;
                        k0 += 8;
                    }

                    _mm256_store_ps(output0_tm, _sum0);
                    _mm256_store_ps(output0_tm + 8, _sum1);
                    _mm256_store_ps(output0_tm + 8 * 2, _sum2);
                    _mm256_store_ps(output0_tm + 8 * 3, _sum3);
                    _mm256_store_ps(output0_tm + 8 * 4, _sum4);
                    _mm256_store_ps(output0_tm + 8 * 5, _sum5);
                    _mm256_store_ps(output0_tm + 8 * 6, _sum6);
                    _mm256_store_ps(output0_tm + 8 * 7, _sum7);
                    _mm256_store_ps(output0_tm + 8 * 8, _sum8);
                    _mm256_store_ps(output0_tm + 8 * 9, _sum9);
                    _mm256_store_ps(output0_tm + 8 * 10, _suma);
                    _mm256_store_ps(output0_tm + 8 * 11, _sumb);

                    output0_tm += 8 * 12;
                }
                for (; i + 7 < tiles; i += 8)
                {
                    const float* r0 = bb2[i / 12 + (i % 12) / 8].data();
                    const float* k0 = kernel0_tm[r].data();

                    int nn = inch * 8; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();
                    __m256 _sum4 = _mm256_setzero_ps();
                    __m256 _sum5 = _mm256_setzero_ps();
                    __m256 _sum6 = _mm256_setzero_ps();
                    __m256 _sum7 = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _w0 = _mm256_load_ps(k0);

                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        __m256 _val1 = _mm256_broadcast_ss(r0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);
                        __m256 _val2 = _mm256_broadcast_ss(r0 + 2);
                        __m256 _val3 = _mm256_broadcast_ss(r0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val2, _w0, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val3, _w0, _sum3);
                        __m256 _val4 = _mm256_broadcast_ss(r0 + 4);
                        __m256 _val5 = _mm256_broadcast_ss(r0 + 5);
                        _sum4 = _mm256_comp_fmadd_ps(_val4, _w0, _sum4);
                        _sum5 = _mm256_comp_fmadd_ps(_val5, _w0, _sum5);
                        __m256 _val6 = _mm256_broadcast_ss(r0 + 6);
                        __m256 _val7 = _mm256_broadcast_ss(r0 + 7);
                        _sum6 = _mm256_comp_fmadd_ps(_val6, _w0, _sum6);
                        _sum7 = _mm256_comp_fmadd_ps(_val7, _w0, _sum7);

                        r0 += 8;
                        k0 += 8;
                    }

                    _mm256_store_ps(output0_tm, _sum0);
                    _mm256_store_ps(output0_tm + 8, _sum1);
                    _mm256_store_ps(output0_tm + 8 * 2, _sum2);
                    _mm256_store_ps(output0_tm + 8 * 3, _sum3);
                    _mm256_store_ps(output0_tm + 8 * 4, _sum4);
                    _mm256_store_ps(output0_tm + 8 * 5, _sum5);
                    _mm256_store_ps(output0_tm + 8 * 6, _sum6);
                    _mm256_store_ps(output0_tm + 8 * 7, _sum7);

                    output0_tm += 8 * 8;
                }
                for (; i + 3 < tiles; i += 4)
                {
                    const float* r0 = bb2[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4].data();
                    const float* k0 = kernel0_tm[r].data();

                    int nn = inch * 8; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();
                    __m256 _sum2 = _mm256_setzero_ps();
                    __m256 _sum3 = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _w0 = _mm256_load_ps(k0);

                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        __m256 _val1 = _mm256_broadcast_ss(r0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);
                        __m256 _val2 = _mm256_broadcast_ss(r0 + 2);
                        __m256 _val3 = _mm256_broadcast_ss(r0 + 3);
                        _sum2 = _mm256_comp_fmadd_ps(_val2, _w0, _sum2);
                        _sum3 = _mm256_comp_fmadd_ps(_val3, _w0, _sum3);

                        r0 += 4;
                        k0 += 8;
                    }

                    _mm256_store_ps(output0_tm, _sum0);
                    _mm256_store_ps(output0_tm + 8, _sum1);
                    _mm256_store_ps(output0_tm + 8 * 2, _sum2);
                    _mm256_store_ps(output0_tm + 8 * 3, _sum3);

                    output0_tm += 8 * 4;
                }
                for (; i + 1 < tiles; i += 2)
                {
                    const float* r0 = bb2[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2].data();
                    const float* k0 = kernel0_tm[r].data();

                    int nn = inch * 8; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();
                    __m256 _sum1 = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _w0 = _mm256_load_ps(k0);

                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        __m256 _val1 = _mm256_broadcast_ss(r0 + 1);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);
                        _sum1 = _mm256_comp_fmadd_ps(_val1, _w0, _sum1);

                        r0 += 2;
                        k0 += 8;
                    }

                    _mm256_store_ps(output0_tm, _sum0);
                    _mm256_store_ps(output0_tm + 8, _sum1);

                    output0_tm += 8 * 2;
                }

                for (; i < tiles; i++)
                {
                    const float* r0 = bb2[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2].data();
                    const float* k0 = kernel0_tm[r].data();

                    int nn = inch * 8; // inch always > 0

                    __m256 _sum0 = _mm256_setzero_ps();

                    for (int j = 0; j < nn; j++)
                    {
                        __m256 _w0 = _mm256_load_ps(k0);
                        __m256 _val0 = _mm256_broadcast_ss(r0);
                        _sum0 = _mm256_comp_fmadd_ps(_val0, _w0, _sum0);

                        r0 += 1;
                        k0 += 8;
                    }

                    _mm256_store_ps(output0_tm, _sum0);

                    output0_tm += 8;
                }
            }
        }
    });
}

void conv3x3s1_winograd63_transform_input_pack8_avx(const Tensor& bottom_blob, Tensor& bottom_blob_tm) {
    const int w = bottom_blob.size(2);
    const int h = bottom_blob.size(1);
    const int inch = bottom_blob.size(0);

    const int w_tiles = (w - 2) / 6;
    const int h_tiles = (h - 2) / 6;
    const int tiles = w_tiles * h_tiles;

    // const float itm[8][8] = {
    //     {1.0f,  0.0f, -5.25f,  0.00f,  5.25f,  0.00f, -1.0f, 0.0f},
    //
    //     {0.0f,  1.0f,  1.00f, -4.25f, -4.25f,  1.00f,  1.0f, 0.0f},
    //     {0.0f, -1.0f,  1.00f,  4.25f, -4.25f, -1.00f,  1.0f, 0.0f},
    //
    //     {0.0f,  0.5f,  0.25f, -2.50f, -1.25f,  2.00f,  1.0f, 0.0f},
    //     {0.0f, -0.5f,  0.25f,  2.50f, -1.25f, -2.00f,  1.0f, 0.0f},
    //
    //     {0.0f,  2.0f,  4.00f, -2.50f, -5.00f,  0.50f,  1.0f, 0.0f},
    //     {0.0f, -2.0f,  4.00f,  2.50f, -5.00f, -0.50f,  1.0f, 0.0f},
    //
    //     {0.0f, -1.0f,  0.00f,  5.25f,  0.00f, -5.25f,  0.0f, 1.0f}
    // };

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
    
    auto bottom_blob_a = bottom_blob.accessor<float, 3, 8>();
    auto bottom_blob_tm_a = bottom_blob_tm.accessor<float, 3, 8>();

    otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
        for (const auto q : otter::irange(begin, end)) {
            const auto img0 = bottom_blob_a[q];
            auto img0_tm = bottom_blob_tm_a[q];

    #ifdef _MSC_VER
            __declspec(align(32))
    #else
            __attribute__((aligned(32)))
    #endif
            float tmp[8][8][8];

            // tile
            for (int i = 0; i < h_tiles; i++)
            {
                for (int j = 0; j < w_tiles; j++)
                {
                    const float* r0 = img0[i * 6].data() + (j * 6) * 8;

                    for (int m = 0; m < 8; m++)
                    {
                        __m256 _r00 = _mm256_load_ps(r0);
                        __m256 _r01 = _mm256_load_ps(r0 + 8);
                        __m256 _r02 = _mm256_load_ps(r0 + 16);
                        __m256 _r03 = _mm256_load_ps(r0 + 24);
                        __m256 _r04 = _mm256_load_ps(r0 + 32);
                        __m256 _r05 = _mm256_load_ps(r0 + 40);
                        __m256 _r06 = _mm256_load_ps(r0 + 48);
                        __m256 _r07 = _mm256_load_ps(r0 + 56);

                        __m256 _tmp0m = _mm256_comp_fmadd_ps(_mm256_set1_ps(5.25f), _mm256_sub_ps(_r04, _r02), _mm256_sub_ps(_r00, _r06));
                        __m256 _tmp7m = _mm256_comp_fmadd_ps(_mm256_set1_ps(5.25f), _mm256_sub_ps(_r03, _r05), _mm256_sub_ps(_r07, _r01));

                        __m256 _tmp12a = _mm256_comp_fmadd_ps(_mm256_set1_ps(-4.25f), _r04, _mm256_add_ps(_r02, _r06));
                        __m256 _tmp12b = _mm256_comp_fmadd_ps(_mm256_set1_ps(-4.25f), _r03, _mm256_add_ps(_r01, _r05));

                        __m256 _tmp1m = _mm256_add_ps(_tmp12a, _tmp12b);
                        __m256 _tmp2m = _mm256_sub_ps(_tmp12a, _tmp12b);

                        __m256 _tmp34a = _mm256_comp_fmadd_ps(_mm256_set1_ps(-1.25f), _r04, _mm256_comp_fmadd_ps(_mm256_set1_ps(0.25f), _r02, _r06));
                        __m256 _tmp34b = _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _r05, _mm256_comp_fmadd_ps(_mm256_set1_ps(-2.5f), _r03, _mm256_mul_ps(_r01, _mm256_set1_ps(0.5f))));

                        __m256 _tmp3m = _mm256_add_ps(_tmp34a, _tmp34b);
                        __m256 _tmp4m = _mm256_sub_ps(_tmp34a, _tmp34b);

                        __m256 _tmp56a = _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _mm256_comp_fmadd_ps(_mm256_set1_ps(-1.25f), _r04, _r02), _r06);
                        __m256 _tmp56b = _mm256_comp_fmadd_ps(_mm256_set1_ps(0.5f), _r05, _mm256_comp_fmadd_ps(_mm256_set1_ps(-2.5f), _r03, _mm256_mul_ps(_r01, _mm256_set1_ps(2.f))));

                        __m256 _tmp5m = _mm256_add_ps(_tmp56a, _tmp56b);
                        __m256 _tmp6m = _mm256_sub_ps(_tmp56a, _tmp56b);

    #if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                        // old gcc breaks stack variable alignement
                        // ref https://gcc.gnu.org/bugzilla/show_bug.cgi?id=16660
                        _mm256_storeu_ps(tmp[0][m], _tmp0m);
                        _mm256_storeu_ps(tmp[1][m], _tmp1m);
                        _mm256_storeu_ps(tmp[2][m], _tmp2m);
                        _mm256_storeu_ps(tmp[3][m], _tmp3m);
                        _mm256_storeu_ps(tmp[4][m], _tmp4m);
                        _mm256_storeu_ps(tmp[5][m], _tmp5m);
                        _mm256_storeu_ps(tmp[6][m], _tmp6m);
                        _mm256_storeu_ps(tmp[7][m], _tmp7m);
    #else
                        _mm256_store_ps(tmp[0][m], _tmp0m);
                        _mm256_store_ps(tmp[1][m], _tmp1m);
                        _mm256_store_ps(tmp[2][m], _tmp2m);
                        _mm256_store_ps(tmp[3][m], _tmp3m);
                        _mm256_store_ps(tmp[4][m], _tmp4m);
                        _mm256_store_ps(tmp[5][m], _tmp5m);
                        _mm256_store_ps(tmp[6][m], _tmp6m);
                        _mm256_store_ps(tmp[7][m], _tmp7m);
    #endif

                        r0 += w * 8;
                    }

                    float* r0_tm_0 = (float*)img0_tm.data() + (i * w_tiles + j) * 8;
                    float* r0_tm_1 = r0_tm_0 + tiles * 8;
                    float* r0_tm_2 = r0_tm_0 + tiles * 16;
                    float* r0_tm_3 = r0_tm_0 + tiles * 24;
                    float* r0_tm_4 = r0_tm_0 + tiles * 32;
                    float* r0_tm_5 = r0_tm_0 + tiles * 40;
                    float* r0_tm_6 = r0_tm_0 + tiles * 48;
                    float* r0_tm_7 = r0_tm_0 + tiles * 56;

                    for (int m = 0; m < 8; m++)
                    {
    #if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                        __m256 _tmp00 = _mm256_loadu_ps(tmp[m][0]);
                        __m256 _tmp01 = _mm256_loadu_ps(tmp[m][1]);
                        __m256 _tmp02 = _mm256_loadu_ps(tmp[m][2]);
                        __m256 _tmp03 = _mm256_loadu_ps(tmp[m][3]);
                        __m256 _tmp04 = _mm256_loadu_ps(tmp[m][4]);
                        __m256 _tmp05 = _mm256_loadu_ps(tmp[m][5]);
                        __m256 _tmp06 = _mm256_loadu_ps(tmp[m][6]);
                        __m256 _tmp07 = _mm256_loadu_ps(tmp[m][7]);
    #else
                        __m256 _tmp00 = _mm256_load_ps(tmp[m][0]);
                        __m256 _tmp01 = _mm256_load_ps(tmp[m][1]);
                        __m256 _tmp02 = _mm256_load_ps(tmp[m][2]);
                        __m256 _tmp03 = _mm256_load_ps(tmp[m][3]);
                        __m256 _tmp04 = _mm256_load_ps(tmp[m][4]);
                        __m256 _tmp05 = _mm256_load_ps(tmp[m][5]);
                        __m256 _tmp06 = _mm256_load_ps(tmp[m][6]);
                        __m256 _tmp07 = _mm256_load_ps(tmp[m][7]);
    #endif

                        __m256 _r0tm0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(5.25f), _mm256_sub_ps(_tmp04, _tmp02), _mm256_sub_ps(_tmp00, _tmp06));
                        __m256 _r0tm7 = _mm256_comp_fmadd_ps(_mm256_set1_ps(5.25f), _mm256_sub_ps(_tmp03, _tmp05), _mm256_sub_ps(_tmp07, _tmp01));

                        __m256 _tmp12a = _mm256_comp_fmadd_ps(_mm256_set1_ps(-4.25f), _tmp04, _mm256_add_ps(_tmp02, _tmp06));
                        __m256 _tmp12b = _mm256_comp_fmadd_ps(_mm256_set1_ps(-4.25f), _tmp03, _mm256_add_ps(_tmp01, _tmp05));

                        __m256 _r0tm1 = _mm256_add_ps(_tmp12a, _tmp12b);
                        __m256 _r0tm2 = _mm256_sub_ps(_tmp12a, _tmp12b);

                        __m256 _tmp34a = _mm256_comp_fmadd_ps(_mm256_set1_ps(-1.25f), _tmp04, _mm256_comp_fmadd_ps(_mm256_set1_ps(0.25f), _tmp02, _tmp06));
                        __m256 _tmp34b = _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _tmp05, _mm256_comp_fmadd_ps(_mm256_set1_ps(-2.5f), _tmp03, _mm256_mul_ps(_tmp01, _mm256_set1_ps(0.5f))));

                        __m256 _r0tm3 = _mm256_add_ps(_tmp34a, _tmp34b);
                        __m256 _r0tm4 = _mm256_sub_ps(_tmp34a, _tmp34b);

                        __m256 _tmp56a = _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _mm256_comp_fmadd_ps(_mm256_set1_ps(-1.25f), _tmp04, _tmp02), _tmp06);
                        __m256 _tmp56b = _mm256_comp_fmadd_ps(_mm256_set1_ps(0.5f), _tmp05, _mm256_comp_fmadd_ps(_mm256_set1_ps(-2.5f), _tmp03, _mm256_mul_ps(_tmp01, _mm256_set1_ps(2.f))));

                        __m256 _r0tm5 = _mm256_add_ps(_tmp56a, _tmp56b);
                        __m256 _r0tm6 = _mm256_sub_ps(_tmp56a, _tmp56b);

                        _mm256_store_ps(r0_tm_0, _r0tm0);
                        _mm256_store_ps(r0_tm_1, _r0tm1);
                        _mm256_store_ps(r0_tm_2, _r0tm2);
                        _mm256_store_ps(r0_tm_3, _r0tm3);
                        _mm256_store_ps(r0_tm_4, _r0tm4);
                        _mm256_store_ps(r0_tm_5, _r0tm5);
                        _mm256_store_ps(r0_tm_6, _r0tm6);
                        _mm256_store_ps(r0_tm_7, _r0tm7);

                        r0_tm_0 += tiles * 64;
                        r0_tm_1 += tiles * 64;
                        r0_tm_2 += tiles * 64;
                        r0_tm_3 += tiles * 64;
                        r0_tm_4 += tiles * 64;
                        r0_tm_5 += tiles * 64;
                        r0_tm_6 += tiles * 64;
                        r0_tm_7 += tiles * 64;
                    }
                }
            }
        }
    });
}

void conv3x3s1_winograd63_transform_output_pack8_avx(const Tensor& top_blob_tm, Tensor& top_blob, const Tensor& bias) {
    const int outw = top_blob.size(2);
    const int outh = top_blob.size(1);
    const int outch = top_blob.size(0);

    const int w_tiles = outw / 6;
    const int h_tiles = outh / 6;
    const int tiles = w_tiles * h_tiles;

    const float* biasptr = (bias.defined()) ? bias.data_ptr<float>() : nullptr;

    // const float otm[6][8] = {
    //     {1.0f,  1.0f,   1.0f,   1.0f,   1.0f,  32.0f, 32.0f, 0.0f},
    //     {0.0f,  1.0f,  -1.0f,   2.0f,  -2.0f,  16.0f,-16.0f, 0.0f},
    //     {0.0f,  1.0f,   1.0f,   4.0f,   4.0f,   8.0f,  8.0f, 0.0f},
    //     {0.0f,  1.0f,  -1.0f,   8.0f,  -8.0f,   4.0f, -4.0f, 0.0f},
    //     {0.0f,  1.0f,   1.0f,  16.0f,  16.0f,   2.0f,  2.0f, 0.0f},
    //     {0.0f,  1.0f,  -1.0f,  32.0f, -32.0f,   1.0f, -1.0f, 1.0f}
    // };

    // 0 = r0 + (r1 + r2) + (r3 + r4)     + (r5 + r6) * 32
    // 1 =      (r1 - r2) + (r3 - r4) * 2 + (r5 - r6) * 16
    // 2 =      (r1 + r2) + (r3 + r4) * 4 + (r5 + r6) * 8
    // 3 =      (r1 - r2) + (r3 - r4) * 8 + (r5 - r6) * 4
    // 4 =      (r1 + r2) + (r3 + r4) * 16+ (r5 + r6) * 2
    // 5 = r7 + (r1 - r2) + (r3 - r4) * 32+ (r5 - r6)
    
    auto top_blob_a = top_blob.accessor<float, 3, 8>();
    auto top_blob_tm_a = top_blob_tm.accessor<float, 3, 8>();

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end))
        {
            const auto out0_tm = top_blob_tm_a[p];
            auto out0 = top_blob_a[p];

            __m256 _bias0 = biasptr ? _mm256_loadu_ps(biasptr + p * 8) : _mm256_setzero_ps();

    #ifdef _MSC_VER
            __declspec(align(32))
    #else
            __attribute__((aligned(32)))
    #endif
            float tmp[6][8][8];

            // tile
            for (int i = 0; i < h_tiles; i++)
            {
                for (int j = 0; j < w_tiles; j++)
                {
                    const float* output0_tm_0 = (const float*)out0_tm.data() + (i * w_tiles + j) * 8;
                    const float* output0_tm_1 = output0_tm_0 + tiles * 8;
                    const float* output0_tm_2 = output0_tm_0 + tiles * 16;
                    const float* output0_tm_3 = output0_tm_0 + tiles * 24;
                    const float* output0_tm_4 = output0_tm_0 + tiles * 32;
                    const float* output0_tm_5 = output0_tm_0 + tiles * 40;
                    const float* output0_tm_6 = output0_tm_0 + tiles * 48;
                    const float* output0_tm_7 = output0_tm_0 + tiles * 56;

                    float* output0 = out0[i * 6].data() + (j * 6) * 8;

                    for (int m = 0; m < 8; m++)
                    {
                        __m256 _out0tm0 = _mm256_load_ps(output0_tm_0);
                        __m256 _out0tm1 = _mm256_load_ps(output0_tm_1);
                        __m256 _out0tm2 = _mm256_load_ps(output0_tm_2);
                        __m256 _out0tm3 = _mm256_load_ps(output0_tm_3);
                        __m256 _out0tm4 = _mm256_load_ps(output0_tm_4);
                        __m256 _out0tm5 = _mm256_load_ps(output0_tm_5);
                        __m256 _out0tm6 = _mm256_load_ps(output0_tm_6);
                        __m256 _out0tm7 = _mm256_load_ps(output0_tm_7);

                        __m256 _tmp024a = _mm256_add_ps(_out0tm1, _out0tm2);
                        __m256 _tmp135a = _mm256_sub_ps(_out0tm1, _out0tm2);

                        __m256 _tmp024b = _mm256_add_ps(_out0tm3, _out0tm4);
                        __m256 _tmp135b = _mm256_sub_ps(_out0tm3, _out0tm4);

                        __m256 _tmp024c = _mm256_add_ps(_out0tm5, _out0tm6);
                        __m256 _tmp135c = _mm256_sub_ps(_out0tm5, _out0tm6);

                        __m256 _tmp0m = _mm256_add_ps(_mm256_add_ps(_out0tm0, _tmp024a), _mm256_comp_fmadd_ps(_mm256_set1_ps(32.f), _tmp024c, _tmp024b));
                        __m256 _tmp2m = _mm256_comp_fmadd_ps(_mm256_set1_ps(8.f), _tmp024c, _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _tmp024b, _tmp024a));
                        __m256 _tmp4m = _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _tmp024c, _mm256_comp_fmadd_ps(_mm256_set1_ps(16.f), _tmp024b, _tmp024a));

                        __m256 _tmp1m = _mm256_comp_fmadd_ps(_mm256_set1_ps(16.f), _tmp135c, _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _tmp135b, _tmp135a));
                        __m256 _tmp3m = _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _tmp135c, _mm256_comp_fmadd_ps(_mm256_set1_ps(8.f), _tmp135b, _tmp135a));
                        __m256 _tmp5m = _mm256_add_ps(_mm256_add_ps(_out0tm7, _tmp135a), _mm256_comp_fmadd_ps(_mm256_set1_ps(32.f), _tmp135b, _tmp135c));

    #if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                        _mm256_storeu_ps(tmp[0][m], _tmp0m);
                        _mm256_storeu_ps(tmp[1][m], _tmp1m);
                        _mm256_storeu_ps(tmp[2][m], _tmp2m);
                        _mm256_storeu_ps(tmp[3][m], _tmp3m);
                        _mm256_storeu_ps(tmp[4][m], _tmp4m);
                        _mm256_storeu_ps(tmp[5][m], _tmp5m);
    #else
                        _mm256_store_ps(tmp[0][m], _tmp0m);
                        _mm256_store_ps(tmp[1][m], _tmp1m);
                        _mm256_store_ps(tmp[2][m], _tmp2m);
                        _mm256_store_ps(tmp[3][m], _tmp3m);
                        _mm256_store_ps(tmp[4][m], _tmp4m);
                        _mm256_store_ps(tmp[5][m], _tmp5m);
    #endif

                        output0_tm_0 += tiles * 64;
                        output0_tm_1 += tiles * 64;
                        output0_tm_2 += tiles * 64;
                        output0_tm_3 += tiles * 64;
                        output0_tm_4 += tiles * 64;
                        output0_tm_5 += tiles * 64;
                        output0_tm_6 += tiles * 64;
                        output0_tm_7 += tiles * 64;
                    }

                    for (int m = 0; m < 6; m++)
                    {
    #if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                        __m256 _tmp00 = _mm256_loadu_ps(tmp[m][0]);
                        __m256 _tmp01 = _mm256_loadu_ps(tmp[m][1]);
                        __m256 _tmp02 = _mm256_loadu_ps(tmp[m][2]);
                        __m256 _tmp03 = _mm256_loadu_ps(tmp[m][3]);
                        __m256 _tmp04 = _mm256_loadu_ps(tmp[m][4]);
                        __m256 _tmp05 = _mm256_loadu_ps(tmp[m][5]);
                        __m256 _tmp06 = _mm256_loadu_ps(tmp[m][6]);
                        __m256 _tmp07 = _mm256_loadu_ps(tmp[m][7]);
    #else
                        __m256 _tmp00 = _mm256_load_ps(tmp[m][0]);
                        __m256 _tmp01 = _mm256_load_ps(tmp[m][1]);
                        __m256 _tmp02 = _mm256_load_ps(tmp[m][2]);
                        __m256 _tmp03 = _mm256_load_ps(tmp[m][3]);
                        __m256 _tmp04 = _mm256_load_ps(tmp[m][4]);
                        __m256 _tmp05 = _mm256_load_ps(tmp[m][5]);
                        __m256 _tmp06 = _mm256_load_ps(tmp[m][6]);
                        __m256 _tmp07 = _mm256_load_ps(tmp[m][7]);
    #endif

                        __m256 _tmp024a = _mm256_add_ps(_tmp01, _tmp02);
                        __m256 _tmp135a = _mm256_sub_ps(_tmp01, _tmp02);

                        __m256 _tmp024b = _mm256_add_ps(_tmp03, _tmp04);
                        __m256 _tmp135b = _mm256_sub_ps(_tmp03, _tmp04);

                        __m256 _tmp024c = _mm256_add_ps(_tmp05, _tmp06);
                        __m256 _tmp135c = _mm256_sub_ps(_tmp05, _tmp06);

                        __m256 _out00 = _mm256_add_ps(_bias0, _mm256_add_ps(_mm256_add_ps(_tmp00, _tmp024a), _mm256_comp_fmadd_ps(_mm256_set1_ps(32.f), _tmp024c, _tmp024b)));
                        __m256 _out02 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_mm256_set1_ps(8.f), _tmp024c, _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _tmp024b, _tmp024a)));
                        __m256 _out04 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _tmp024c, _mm256_comp_fmadd_ps(_mm256_set1_ps(16.f), _tmp024b, _tmp024a)));

                        __m256 _out01 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_mm256_set1_ps(16.f), _tmp135c, _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _tmp135b, _tmp135a)));
                        __m256 _out03 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _tmp135c, _mm256_comp_fmadd_ps(_mm256_set1_ps(8.f), _tmp135b, _tmp135a)));
                        __m256 _out05 = _mm256_add_ps(_bias0, _mm256_add_ps(_mm256_add_ps(_tmp07, _tmp135a), _mm256_comp_fmadd_ps(_mm256_set1_ps(32.f), _tmp135b, _tmp135c)));

                        _mm256_store_ps(output0, _out00);
                        _mm256_store_ps(output0 + 8, _out01);
                        _mm256_store_ps(output0 + 16, _out02);
                        _mm256_store_ps(output0 + 24, _out03);
                        _mm256_store_ps(output0 + 32, _out04);
                        _mm256_store_ps(output0 + 40, _out05);

                        output0 += outw * 8;
                    }
                }
            }
        }
    });
}

void conv3x3s1_winograd43_transform_input_pack8_avx(const Tensor& bottom_blob, Tensor& bottom_blob_tm) {
    const int w = bottom_blob.size(2);
    const int h = bottom_blob.size(1);
    const int inch = bottom_blob.size(0);

    const int w_tiles = (w - 2) / 4;
    const int h_tiles = (h - 2) / 4;
    const int tiles = w_tiles * h_tiles;

    // const float itm[4][4] = {
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
    
    auto bottom_blob_a = bottom_blob.accessor<float, 3, 8>();
    auto bottom_blob_tm_a = bottom_blob_tm.accessor<float, 3, 8>();

    otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
        for (const auto q : otter::irange(begin, end))
        {
            const auto img0 = bottom_blob_a[q];
            auto img0_tm = bottom_blob_tm_a[q];

    #ifdef _MSC_VER
            __declspec(align(32))
    #else
            __attribute__((aligned(32)))
    #endif
            float tmp[6][6][8];

            // tile
            for (int i = 0; i < h_tiles; i++)
            {
                for (int j = 0; j < w_tiles; j++)
                {
                    const float* r0 = img0[i * 4].data() + (j * 4) * 8;

                    for (int m = 0; m < 6; m++)
                    {
                        __m256 _r00 = _mm256_load_ps(r0);
                        __m256 _r01 = _mm256_load_ps(r0 + 8);
                        __m256 _r02 = _mm256_load_ps(r0 + 8 * 2);
                        __m256 _r03 = _mm256_load_ps(r0 + 8 * 3);
                        __m256 _r04 = _mm256_load_ps(r0 + 8 * 4);
                        __m256 _r05 = _mm256_load_ps(r0 + 8 * 5);

                        __m256 _tmp0m = _mm256_comp_fmadd_ps(_mm256_set1_ps(-5.f), _r02, _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _r00, _r04));
                        __m256 _tmp1m = _mm256_comp_fmadd_ps(_mm256_set1_ps(-4.f), _mm256_add_ps(_r01, _r02), _mm256_add_ps(_r04, _r03));
                        __m256 _tmp2m = _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _mm256_sub_ps(_r01, _r02), _mm256_sub_ps(_r04, _r03));
                        __m256 _tmp3m = _mm256_comp_fmadd_ps(_mm256_set1_ps(-2.f), _mm256_sub_ps(_r01, _r03), _mm256_sub_ps(_r04, _r02));
                        __m256 _tmp4m = _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _mm256_sub_ps(_r01, _r03), _mm256_sub_ps(_r04, _r02));
                        __m256 _tmp5m = _mm256_comp_fmadd_ps(_mm256_set1_ps(-5.f), _r03, _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _r01, _r05));

    #if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                        _mm256_storeu_ps(tmp[0][m], _tmp0m);
                        _mm256_storeu_ps(tmp[1][m], _tmp1m);
                        _mm256_storeu_ps(tmp[2][m], _tmp2m);
                        _mm256_storeu_ps(tmp[3][m], _tmp3m);
                        _mm256_storeu_ps(tmp[4][m], _tmp4m);
                        _mm256_storeu_ps(tmp[5][m], _tmp5m);
    #else
                        _mm256_store_ps(tmp[0][m], _tmp0m);
                        _mm256_store_ps(tmp[1][m], _tmp1m);
                        _mm256_store_ps(tmp[2][m], _tmp2m);
                        _mm256_store_ps(tmp[3][m], _tmp3m);
                        _mm256_store_ps(tmp[4][m], _tmp4m);
                        _mm256_store_ps(tmp[5][m], _tmp5m);
    #endif

                        r0 += w * 8;
                    }

                    float* r0_tm_0 = (float*)img0_tm.data() + (i * w_tiles + j) * 8;
                    float* r0_tm_1 = r0_tm_0 + tiles * 8;
                    float* r0_tm_2 = r0_tm_0 + tiles * 8 * 2;
                    float* r0_tm_3 = r0_tm_0 + tiles * 8 * 3;
                    float* r0_tm_4 = r0_tm_0 + tiles * 8 * 4;
                    float* r0_tm_5 = r0_tm_0 + tiles * 8 * 5;

                    for (int m = 0; m < 6; m++)
                    {
    #if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                        __m256 _tmp00 = _mm256_loadu_ps(tmp[m][0]);
                        __m256 _tmp01 = _mm256_loadu_ps(tmp[m][1]);
                        __m256 _tmp02 = _mm256_loadu_ps(tmp[m][2]);
                        __m256 _tmp03 = _mm256_loadu_ps(tmp[m][3]);
                        __m256 _tmp04 = _mm256_loadu_ps(tmp[m][4]);
                        __m256 _tmp05 = _mm256_loadu_ps(tmp[m][5]);
    #else
                        __m256 _tmp00 = _mm256_load_ps(tmp[m][0]);
                        __m256 _tmp01 = _mm256_load_ps(tmp[m][1]);
                        __m256 _tmp02 = _mm256_load_ps(tmp[m][2]);
                        __m256 _tmp03 = _mm256_load_ps(tmp[m][3]);
                        __m256 _tmp04 = _mm256_load_ps(tmp[m][4]);
                        __m256 _tmp05 = _mm256_load_ps(tmp[m][5]);
    #endif

                        __m256 _r0tm0 = _mm256_comp_fmadd_ps(_mm256_set1_ps(-5.f), _tmp02, _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _tmp00, _tmp04));
                        __m256 _r0tm1 = _mm256_comp_fmadd_ps(_mm256_set1_ps(-4.f), _mm256_add_ps(_tmp01, _tmp02), _mm256_add_ps(_tmp04, _tmp03));
                        __m256 _r0tm2 = _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _mm256_sub_ps(_tmp01, _tmp02), _mm256_sub_ps(_tmp04, _tmp03));
                        __m256 _r0tm3 = _mm256_comp_fmadd_ps(_mm256_set1_ps(-2.f), _mm256_sub_ps(_tmp01, _tmp03), _mm256_sub_ps(_tmp04, _tmp02));
                        __m256 _r0tm4 = _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _mm256_sub_ps(_tmp01, _tmp03), _mm256_sub_ps(_tmp04, _tmp02));
                        __m256 _r0tm5 = _mm256_comp_fmadd_ps(_mm256_set1_ps(-5.f), _tmp03, _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _tmp01, _tmp05));

                        _mm256_store_ps(r0_tm_0, _r0tm0);
                        _mm256_store_ps(r0_tm_1, _r0tm1);
                        _mm256_store_ps(r0_tm_2, _r0tm2);
                        _mm256_store_ps(r0_tm_3, _r0tm3);
                        _mm256_store_ps(r0_tm_4, _r0tm4);
                        _mm256_store_ps(r0_tm_5, _r0tm5);

                        r0_tm_0 += tiles * 8 * 6;
                        r0_tm_1 += tiles * 8 * 6;
                        r0_tm_2 += tiles * 8 * 6;
                        r0_tm_3 += tiles * 8 * 6;
                        r0_tm_4 += tiles * 8 * 6;
                        r0_tm_5 += tiles * 8 * 6;
                    }
                }
            }
        }
    });
}

void conv3x3s1_winograd43_transform_output_pack8_avx(const Tensor& top_blob_tm, Tensor& top_blob, const Tensor& bias) {
    const int outw = top_blob.size(2);
    const int outh = top_blob.size(1);
    const int outch = top_blob.size(0);

    const int w_tiles = outw / 4;
    const int h_tiles = outh / 4;
    const int tiles = w_tiles * h_tiles;

    const float* biasptr = (bias.defined()) ? bias.data_ptr<float>() : nullptr;

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
    
    auto top_blob_a = top_blob.accessor<float, 3, 8>();
    auto top_blob_tm_a = top_blob_tm.accessor<float, 3, 8>();

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end))
        {
            const auto out0_tm = top_blob_tm_a[p];
            auto out0 = top_blob_a[p];

            __m256 _bias0 = biasptr ? _mm256_loadu_ps(biasptr + p * 8) : _mm256_setzero_ps();

    #ifdef _MSC_VER
            __declspec(align(32))
    #else
            __attribute__((aligned(32)))
    #endif
            float tmp[4][6][8];

            // tile
            for (int i = 0; i < h_tiles; i++)
            {
                for (int j = 0; j < w_tiles; j++)
                {
                    const float* output0_tm_0 = (const float*)out0_tm.data() + (i * w_tiles + j) * 8;
                    const float* output0_tm_1 = output0_tm_0 + tiles * 8;
                    const float* output0_tm_2 = output0_tm_0 + tiles * 8 * 2;
                    const float* output0_tm_3 = output0_tm_0 + tiles * 8 * 3;
                    const float* output0_tm_4 = output0_tm_0 + tiles * 8 * 4;
                    const float* output0_tm_5 = output0_tm_0 + tiles * 8 * 5;

                    float* output0 = out0[i * 4].data() + (j * 4) * 8;

                    for (int m = 0; m < 6; m++)
                    {
                        __m256 _out0tm0 = _mm256_load_ps(output0_tm_0);
                        __m256 _out0tm1 = _mm256_load_ps(output0_tm_1);
                        __m256 _out0tm2 = _mm256_load_ps(output0_tm_2);
                        __m256 _out0tm3 = _mm256_load_ps(output0_tm_3);
                        __m256 _out0tm4 = _mm256_load_ps(output0_tm_4);
                        __m256 _out0tm5 = _mm256_load_ps(output0_tm_5);

                        __m256 _tmp02a = _mm256_add_ps(_out0tm1, _out0tm2);
                        __m256 _tmp13a = _mm256_sub_ps(_out0tm1, _out0tm2);

                        __m256 _tmp02b = _mm256_add_ps(_out0tm3, _out0tm4);
                        __m256 _tmp13b = _mm256_sub_ps(_out0tm3, _out0tm4);

                        __m256 _tmp0m = _mm256_add_ps(_mm256_add_ps(_out0tm0, _tmp02a), _tmp02b);
                        __m256 _tmp1m = _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _tmp13b, _tmp13a);
                        __m256 _tmp2m = _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _tmp02b, _tmp02a);
                        __m256 _tmp3m = _mm256_comp_fmadd_ps(_mm256_set1_ps(8.f), _tmp13b, _mm256_add_ps(_out0tm5, _tmp13a));

    #if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                        _mm256_storeu_ps(tmp[0][m], _tmp0m);
                        _mm256_storeu_ps(tmp[1][m], _tmp1m);
                        _mm256_storeu_ps(tmp[2][m], _tmp2m);
                        _mm256_storeu_ps(tmp[3][m], _tmp3m);
    #else
                        _mm256_store_ps(tmp[0][m], _tmp0m);
                        _mm256_store_ps(tmp[1][m], _tmp1m);
                        _mm256_store_ps(tmp[2][m], _tmp2m);
                        _mm256_store_ps(tmp[3][m], _tmp3m);
    #endif

                        output0_tm_0 += tiles * 8 * 6;
                        output0_tm_1 += tiles * 8 * 6;
                        output0_tm_2 += tiles * 8 * 6;
                        output0_tm_3 += tiles * 8 * 6;
                        output0_tm_4 += tiles * 8 * 6;
                        output0_tm_5 += tiles * 8 * 6;
                    }

                    for (int m = 0; m < 4; m++)
                    {
    #if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                        __m256 _tmp00 = _mm256_loadu_ps(tmp[m][0]);
                        __m256 _tmp01 = _mm256_loadu_ps(tmp[m][1]);
                        __m256 _tmp02 = _mm256_loadu_ps(tmp[m][2]);
                        __m256 _tmp03 = _mm256_loadu_ps(tmp[m][3]);
                        __m256 _tmp04 = _mm256_loadu_ps(tmp[m][4]);
                        __m256 _tmp05 = _mm256_loadu_ps(tmp[m][5]);
    #else
                        __m256 _tmp00 = _mm256_load_ps(tmp[m][0]);
                        __m256 _tmp01 = _mm256_load_ps(tmp[m][1]);
                        __m256 _tmp02 = _mm256_load_ps(tmp[m][2]);
                        __m256 _tmp03 = _mm256_load_ps(tmp[m][3]);
                        __m256 _tmp04 = _mm256_load_ps(tmp[m][4]);
                        __m256 _tmp05 = _mm256_load_ps(tmp[m][5]);
    #endif

                        __m256 _tmp02a = _mm256_add_ps(_tmp01, _tmp02);
                        __m256 _tmp13a = _mm256_sub_ps(_tmp01, _tmp02);

                        __m256 _tmp02b = _mm256_add_ps(_tmp03, _tmp04);
                        __m256 _tmp13b = _mm256_sub_ps(_tmp03, _tmp04);

                        __m256 _out00 = _mm256_add_ps(_bias0, _mm256_add_ps(_mm256_add_ps(_tmp00, _tmp02a), _tmp02b));
                        __m256 _out01 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_mm256_set1_ps(2.f), _tmp13b, _tmp13a));
                        __m256 _out02 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_mm256_set1_ps(4.f), _tmp02b, _tmp02a));
                        __m256 _out03 = _mm256_add_ps(_bias0, _mm256_comp_fmadd_ps(_mm256_set1_ps(8.f), _tmp13b, _mm256_add_ps(_tmp05, _tmp13a)));

                        _mm256_store_ps(output0, _out00);
                        _mm256_store_ps(output0 + 8, _out01);
                        _mm256_store_ps(output0 + 8 * 2, _out02);
                        _mm256_store_ps(output0 + 8 * 3, _out03);

                        output0 += outw * 8;
                    }
                }
            }
        }
    });
}

void conv3x3s1_winograd23_transform_input_pack8_avx(const Tensor& bottom_blob, Tensor& bottom_blob_tm) {
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
    
    auto bottom_blob_a = bottom_blob.accessor<float, 3, 8>();
    auto bottom_blob_tm_a = bottom_blob_tm.accessor<float, 3, 8>();

    otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
        for (const auto q : otter::irange(begin, end))
        {
            const auto img0 = bottom_blob_a[q];
            auto img0_tm = bottom_blob_tm_a[q];

    #ifdef _MSC_VER
            __declspec(align(32))
    #else
            __attribute__((aligned(32)))
    #endif
            float tmp[4][4][8];

            // tile
            for (int i = 0; i < h_tiles; i++)
            {
                for (int j = 0; j < w_tiles; j++)
                {
                    const float* r0 = img0[i * 2].data() + (j * 2) * 8;

                    for (int m = 0; m < 4; m++)
                    {
                        __m256 _r00 = _mm256_load_ps(r0);
                        __m256 _r01 = _mm256_load_ps(r0 + 8);
                        __m256 _r02 = _mm256_load_ps(r0 + 8 * 2);
                        __m256 _r03 = _mm256_load_ps(r0 + 8 * 3);

                        __m256 _tmp0m = _mm256_sub_ps(_r00, _r02);
                        __m256 _tmp1m = _mm256_add_ps(_r01, _r02);
                        __m256 _tmp2m = _mm256_sub_ps(_r02, _r01);
                        __m256 _tmp3m = _mm256_sub_ps(_r03, _r01);

    #if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                        _mm256_storeu_ps(tmp[0][m], _tmp0m);
                        _mm256_storeu_ps(tmp[1][m], _tmp1m);
                        _mm256_storeu_ps(tmp[2][m], _tmp2m);
                        _mm256_storeu_ps(tmp[3][m], _tmp3m);
    #else
                        _mm256_store_ps(tmp[0][m], _tmp0m);
                        _mm256_store_ps(tmp[1][m], _tmp1m);
                        _mm256_store_ps(tmp[2][m], _tmp2m);
                        _mm256_store_ps(tmp[3][m], _tmp3m);
    #endif

                        r0 += w * 8;
                    }

                    float* r0_tm_0 = (float*)img0_tm.data() + (i * w_tiles + j) * 8;
                    float* r0_tm_1 = r0_tm_0 + tiles * 8;
                    float* r0_tm_2 = r0_tm_0 + tiles * 8 * 2;
                    float* r0_tm_3 = r0_tm_0 + tiles * 8 * 3;

                    for (int m = 0; m < 4; m++)
                    {
    #if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                        __m256 _tmp00 = _mm256_loadu_ps(tmp[m][0]);
                        __m256 _tmp01 = _mm256_loadu_ps(tmp[m][1]);
                        __m256 _tmp02 = _mm256_loadu_ps(tmp[m][2]);
                        __m256 _tmp03 = _mm256_loadu_ps(tmp[m][3]);
    #else
                        __m256 _tmp00 = _mm256_load_ps(tmp[m][0]);
                        __m256 _tmp01 = _mm256_load_ps(tmp[m][1]);
                        __m256 _tmp02 = _mm256_load_ps(tmp[m][2]);
                        __m256 _tmp03 = _mm256_load_ps(tmp[m][3]);
    #endif

                        __m256 _r0tm0 = _mm256_sub_ps(_tmp00, _tmp02);
                        __m256 _r0tm1 = _mm256_add_ps(_tmp01, _tmp02);
                        __m256 _r0tm2 = _mm256_sub_ps(_tmp02, _tmp01);
                        __m256 _r0tm3 = _mm256_sub_ps(_tmp03, _tmp01);

                        _mm256_store_ps(r0_tm_0, _r0tm0);
                        _mm256_store_ps(r0_tm_1, _r0tm1);
                        _mm256_store_ps(r0_tm_2, _r0tm2);
                        _mm256_store_ps(r0_tm_3, _r0tm3);

                        r0_tm_0 += tiles * 8 * 4;
                        r0_tm_1 += tiles * 8 * 4;
                        r0_tm_2 += tiles * 8 * 4;
                        r0_tm_3 += tiles * 8 * 4;
                    }
                }
            }
        }
    });
}

void conv3x3s1_winograd23_transform_output_pack8_avx(const Tensor& top_blob_tm, Tensor& top_blob, const Tensor& bias) {
    const int outw = top_blob.size(2);
    const int outh = top_blob.size(1);
    const int outch = top_blob.size(0);

    const int w_tiles = outw / 2;
    const int h_tiles = outh / 2;
    const int tiles = w_tiles * h_tiles;

    const float* biasptr = (bias.defined()) ? bias.data_ptr<float>() : nullptr;

    // const float otm[2][4] = {
    //     {1.0f,  1.0f,  1.0f,  0.0f},
    //     {0.0f,  1.0f, -1.0f,  1.0f}
    // };

    // 0 = r00 + r01 + r02
    // 1 = r01 - r02 + r03
    
    auto top_blob_a = top_blob.accessor<float, 3, 8>();
    auto top_blob_tm_a = top_blob_tm.accessor<float, 3, 8>();

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end))
        {
            const auto out0_tm = top_blob_tm_a[p];
            auto out0 = top_blob_a[p];

            __m256 _bias0 = biasptr ? _mm256_loadu_ps(biasptr + p * 8) : _mm256_setzero_ps();

    #ifdef _MSC_VER
            __declspec(align(32))
    #else
            __attribute__((aligned(32)))
    #endif
            float tmp[2][4][8];

            // tile
            for (int i = 0; i < h_tiles; i++)
            {
                for (int j = 0; j < w_tiles; j++)
                {
                    const float* output0_tm_0 = (const float*)out0_tm.data() + (i * w_tiles + j) * 8;
                    const float* output0_tm_1 = output0_tm_0 + tiles * 8;
                    const float* output0_tm_2 = output0_tm_0 + tiles * 8 * 2;
                    const float* output0_tm_3 = output0_tm_0 + tiles * 8 * 3;

                    float* output0 = out0[i * 2].data() + (j * 2) * 8;

                    for (int m = 0; m < 4; m++)
                    {
                        __m256 _out0tm0 = _mm256_load_ps(output0_tm_0);
                        __m256 _out0tm1 = _mm256_load_ps(output0_tm_1);
                        __m256 _out0tm2 = _mm256_load_ps(output0_tm_2);
                        __m256 _out0tm3 = _mm256_load_ps(output0_tm_3);

                        __m256 _tmp0m = _mm256_add_ps(_mm256_add_ps(_out0tm0, _out0tm1), _out0tm2);
                        __m256 _tmp1m = _mm256_add_ps(_mm256_sub_ps(_out0tm1, _out0tm2), _out0tm3);

    #if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                        _mm256_storeu_ps(tmp[0][m], _tmp0m);
                        _mm256_storeu_ps(tmp[1][m], _tmp1m);
    #else
                        _mm256_store_ps(tmp[0][m], _tmp0m);
                        _mm256_store_ps(tmp[1][m], _tmp1m);
    #endif

                        output0_tm_0 += tiles * 8 * 4;
                        output0_tm_1 += tiles * 8 * 4;
                        output0_tm_2 += tiles * 8 * 4;
                        output0_tm_3 += tiles * 8 * 4;
                    }

                    for (int m = 0; m < 2; m++)
                    {
    #if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 6)
                        __m256 _tmp00 = _mm256_loadu_ps(tmp[m][0]);
                        __m256 _tmp01 = _mm256_loadu_ps(tmp[m][1]);
                        __m256 _tmp02 = _mm256_loadu_ps(tmp[m][2]);
                        __m256 _tmp03 = _mm256_loadu_ps(tmp[m][3]);
    #else
                        __m256 _tmp00 = _mm256_load_ps(tmp[m][0]);
                        __m256 _tmp01 = _mm256_load_ps(tmp[m][1]);
                        __m256 _tmp02 = _mm256_load_ps(tmp[m][2]);
                        __m256 _tmp03 = _mm256_load_ps(tmp[m][3]);
    #endif

                        __m256 _out00 = _mm256_add_ps(_bias0, _mm256_add_ps(_mm256_add_ps(_tmp00, _tmp01), _tmp02));
                        __m256 _out01 = _mm256_add_ps(_bias0, _mm256_add_ps(_mm256_sub_ps(_tmp01, _tmp02), _tmp03));

                        _mm256_store_ps(output0, _out00);
                        _mm256_store_ps(output0 + 8, _out01);

                        output0 += outw * 8;
                    }
                }
            }
        }
    });
}

void conv3x3s1_winograd63_transform_kernel_pack8_avx(const Tensor& kernel, Tensor& kernel_tm_pack8, int inch, int outch)
{
    // winograd63 transform kernel
    Tensor kernel_tm = otter::empty({outch, inch, 8 * 8}, otter::ScalarType::Float);

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
    
    const float* kernel_ptr = kernel.data_ptr<float>();
    auto kernel_tm_a = kernel_tm.accessor<float, 3>();

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            for (int q = 0; q < inch; q++)
            {
                const float* kernel0 = (const float*)kernel_ptr + p * inch * 9 + q * 9;
                float* kernel_tm0 = kernel_tm_a[p][q].data();

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

    // interleave
    // src = 64-inch-outch
    // dst = 8b-8a-inch/8a-64-outch/8b
    kernel_tm_pack8 = otter::empty({outch / 8, 64, inch / 8}, otter::ScalarType::Float64);
    auto kernel_tm_pack8_a = kernel_tm_pack8.accessor<float, 3, 64>();
    for (int q = 0; q + 7 < outch; q += 8)
    {
        auto g0 = kernel_tm_pack8_a[q / 8];

        for (int k = 0; k < 64; k++)
        {
            float* g00 = g0[k].data();

            for (int p = 0; p + 7 < inch; p += 8)
            {
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const float* k00 = kernel_tm_a[q + j][p + i].data();
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }
}

void conv3x3s1_winograd43_transform_kernel_pack8_avx(const Tensor& kernel, Tensor& kernel_tm_pack8, int inch, int outch)
{
    // winograd43 transform kernel
    Tensor kernel_tm = otter::empty({outch, inch, 6 * 6}, otter::ScalarType::Float);

    const float ktm[6][3] = {
        {1.0f / 4, 0.0f, 0.0f},
        {-1.0f / 6, -1.0f / 6, -1.0f / 6},
        {-1.0f / 6, 1.0f / 6, -1.0f / 6},
        {1.0f / 24, 1.0f / 12, 1.0f / 6},
        {1.0f / 24, -1.0f / 12, 1.0f / 6},
        {0.0f, 0.0f, 1.0f}
    };
    
    const float* kernel_ptr = kernel.data_ptr<float>();
    auto kernel_tm_a = kernel_tm.accessor<float, 3>();

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end))
        {
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
    // dst = 8b-8a-inch/8a-36-outch/8b
    kernel_tm_pack8 = otter::empty({outch / 8, 36, inch / 8}, otter::ScalarType::Float64);
    auto kernel_tm_pack8_a = kernel_tm_pack8.accessor<float, 3, 64>();
    for (int q = 0; q + 7 < outch; q += 8)
    {
        auto g0 = kernel_tm_pack8_a[q / 8];

        for (int k = 0; k < 36; k++)
        {
            float* g00 = g0[k].data();

            for (int p = 0; p + 7 < inch; p += 8)
            {
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const float* k00 = kernel_tm_a[q + j][p + i].data();
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }
}

void conv3x3s1_winograd23_transform_kernel_pack8_avx(const Tensor& kernel, Tensor& kernel_tm_pack8, int inch, int outch)
{
    // winograd23 transform kernel
    Tensor kernel_tm = otter::empty({outch, inch, 4 * 4}, otter::ScalarType::Float);

    const float ktm[4][3] = {
        {1.0f, 0.0f, 0.0f},
        {1.0f / 2, 1.0f / 2, 1.0f / 2},
        {1.0f / 2, -1.0f / 2, 1.0f / 2},
        {0.0f, 0.0f, 1.0f}
    };
    
    const float* kernel_ptr = kernel.data_ptr<float>();
    auto kernel_tm_a = kernel_tm.accessor<float, 3>();

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end))
        {
            for (int q = 0; q < inch; q++)
            {
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
    // dst = pb-pa-inch/pa-16-outch/pb
    kernel_tm_pack8 = otter::empty({outch / 8, 16, inch / 8}, otter::ScalarType::Float64);
    auto kernel_tm_pack8_a = kernel_tm_pack8.accessor<float, 3, 64>();
    for (int q = 0; q + 7 < outch; q += 8)
    {
        auto g0 = kernel_tm_pack8_a[q / 8];

        for (int k = 0; k < 16; k++)
        {
            float* g00 = g0[k].data();

            for (int p = 0; p + 7 < inch; p += 8)
            {
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        const float* k00 = kernel_tm_a[q + j][p + i].data();
                        g00[0] = k00[k];
                        g00++;
                    }
                }
            }
        }
    }
}

Tensor conv2d_3x3s1_winograd63_pack8_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_shape = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
    output.resize_({output_shape[0], output_shape[1] / 8, output_shape[2], output_shape[3]});
    
    int origin_w = (int)self.size(3) + 2 * (int)padding[1];
    int origin_h = (int)self.size(2) + 2 * (int)padding[0];
    
    int w = origin_w;
    int h = origin_h;
    int inch  = (int)self.size(1);
    
    int outw  = (int)output_shape[3];
    int outh  = (int)output_shape[2];
    int outch = (int)output_shape[1] / 8;
    
    outw = (outw + 5) / 6 * 6;
    outh = (outh + 5) / 6 * 6;

    w = outw + 2;
    h = outh + 2;
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1] + w - origin_w, padding[0], padding[0] + h - origin_h}, 0);
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        otter::conv3x3s1_winograd63_transform_kernel_pack8_avx(weight, kernel_tf, inch * 8, outch * 8);
    
    // BEGIN transform input
    Tensor bottom_blob_tm;
    {
        int w_tiles = outw / 6;
        int h_tiles = outh / 6;
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm = otter::empty({inch, 64, tiles}, otter::ScalarType::Float8);
        conv3x3s1_winograd63_transform_input_pack8_avx(input[0], bottom_blob_tm);
    }
    input.reset();
    // END transform input

    // BEGIN dot
    Tensor top_blob_tm;
    convolution_winograd_dot_pack8_avx(bottom_blob_tm, outch, kernel_tf, top_blob_tm);
    // END dot

    // BEGIN transform output
    Tensor top_blob_bordered;
    if (outw == output_shape[3] && outh == output_shape[2]) {
        top_blob_bordered = output;
    } else {
        top_blob_bordered = otter::empty({1, outch, outh, outw}, otter::ScalarType::Float8);
    }
    {
        Tensor top_blob_bordered_t = top_blob_bordered[0];
        conv3x3s1_winograd63_transform_output_pack8_avx(top_blob_tm, top_blob_bordered_t, bias);
    }
    // END transform output
    
    otter::crop_(top_blob_bordered, {0, top_blob_bordered.size(3) - output_shape[3], 0, top_blob_bordered.size(2) - output_shape[2]}, output);
    
    return output;
}

Tensor conv2d_3x3s1_winograd63_pack8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, otter::ScalarType::Float8);
    
    return conv2d_3x3s1_winograd63_pack8_x86_out(self, weight, weight_o, bias, padding, output);
}

Tensor conv2d_3x3s1_winograd43_pack8_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_shape = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
    output.resize_({output_shape[0], output_shape[1] / 8, output_shape[2], output_shape[3]});
    
    int origin_w = (int)self.size(3) + 2 * (int)padding[1];
    int origin_h = (int)self.size(2) + 2 * (int)padding[0];
    
    int w = origin_w;
    int h = origin_h;
    int inch  = (int)self.size(1);
    
    int outw  = (int)output_shape[3];
    int outh  = (int)output_shape[2];
    int outch = (int)output_shape[1] / 8;
    
    outw = (outw + 3) / 4 * 4;
    outh = (outh + 3) / 4 * 4;

    w = outw + 2;
    h = outh + 2;
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1] + w - origin_w, padding[0], padding[0] + h - origin_h}, 0);
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        otter::conv3x3s1_winograd43_transform_kernel_pack8_avx(weight, kernel_tf, inch * 8, outch * 8);
    
    // BEGIN transform input
    Tensor bottom_blob_tm;
    {
        int w_tiles = outw / 4;
        int h_tiles = outh / 4;
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm = otter::empty({inch, 36, tiles}, otter::ScalarType::Float8);
        conv3x3s1_winograd43_transform_input_pack8_avx(input[0], bottom_blob_tm);
    }
    input.reset();
    // END transform input

    // BEGIN dot
    Tensor top_blob_tm;
    convolution_winograd_dot_pack8_avx(bottom_blob_tm, outch, kernel_tf, top_blob_tm);
    // END dot

    // BEGIN transform output
    Tensor top_blob_bordered;
    if (outw == output_shape[3] && outh == output_shape[2]) {
        top_blob_bordered = output;
    } else {
        top_blob_bordered = otter::empty({1, outch, outh, outw}, otter::ScalarType::Float8);
    }
    {
        Tensor top_blob_bordered_t = top_blob_bordered[0];
        conv3x3s1_winograd43_transform_output_pack8_avx(top_blob_tm, top_blob_bordered_t, bias);
    }
    // END transform output
    
    otter::crop_(top_blob_bordered, {0, top_blob_bordered.size(3) - output_shape[3], 0, top_blob_bordered.size(2) - output_shape[2]}, output);
    
    return output;
}

Tensor conv2d_3x3s1_winograd43_pack8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, otter::ScalarType::Float8);
    
    return conv2d_3x3s1_winograd43_pack8_x86_out(self, weight, weight_o, bias, padding, output);
}

Tensor conv2d_3x3s1_winograd23_pack8_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_shape = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
    output.resize_({output_shape[0], output_shape[1] / 8, output_shape[2], output_shape[3]});
    
    int origin_w = (int)self.size(3) + 2 * (int)padding[1];
    int origin_h = (int)self.size(2) + 2 * (int)padding[0];
    
    int w = origin_w;
    int h = origin_h;
    int inch  = (int)self.size(1);
    
    int outw  = (int)output_shape[3];
    int outh  = (int)output_shape[2];
    int outch = (int)output_shape[1] / 8;
    
    outw = (outw + 1) / 2 * 2;
    outh = (outh + 1) / 2 * 2;

    w = outw + 2;
    h = outh + 2;
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1] + w - origin_w, padding[0], padding[0] + h - origin_h}, 0);
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        otter::conv3x3s1_winograd23_transform_kernel_pack8_avx(weight, kernel_tf, inch * 8, outch * 8);
    
    // BEGIN transform input
    Tensor bottom_blob_tm;
    {
        int w_tiles = outw / 2;
        int h_tiles = outh / 2;
        int tiles = w_tiles * h_tiles;

        bottom_blob_tm = otter::empty({inch, 16, tiles}, otter::ScalarType::Float8);
        conv3x3s1_winograd23_transform_input_pack8_avx(input[0], bottom_blob_tm);
    }
    input.reset();
    // END transform input

    // BEGIN dot
    Tensor top_blob_tm;
    convolution_winograd_dot_pack8_avx(bottom_blob_tm, outch, kernel_tf, top_blob_tm);
    // END dot

    // BEGIN transform output
    Tensor top_blob_bordered;
    if (outw == output_shape[3] && outh == output_shape[2]) {
        top_blob_bordered = output;
    } else {
        top_blob_bordered = otter::empty({1, outch, outh, outw}, otter::ScalarType::Float8);
    }
    {
        Tensor top_blob_bordered_t = top_blob_bordered[0];
        conv3x3s1_winograd23_transform_output_pack8_avx(top_blob_tm, top_blob_bordered_t, bias);
    }
    // END transform output
    
    otter::crop_(top_blob_bordered, {0, top_blob_bordered.size(3) - output_shape[3], 0, top_blob_bordered.size(2) - output_shape[2]}, output);
    
    return output;
}

Tensor conv2d_3x3s1_winograd23_pack8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, otter::ScalarType::Float8);
    
    return conv2d_3x3s1_winograd23_pack8_x86_out(self, weight, weight_o, bias, padding, output);
}

#endif  // __AVX__
#endif  // __SSE2__

}   // end namespace otter
