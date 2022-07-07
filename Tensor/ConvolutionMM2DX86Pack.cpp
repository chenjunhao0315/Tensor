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

//    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(0, outch)) {
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
//    });

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

void conv3x3s1_winograd63_transform_output_pack4_sse(const Tensor& top_blob_tm, Tensor& top_blob, const Tensor& bias) {
    const int outw = top_blob.size(2);
    const int outh = top_blob.size(1);
    const int outch = top_blob.size(0);

    const int w_tiles = outw / 6;
    const int h_tiles = outh / 6;
    const int tiles = w_tiles * h_tiles;

    const float* biasptr = bias.data_ptr<float>();

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
        for (int p = 0; p < outch; p++)
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
    
}

void conv3x3s1_winograd43_transform_output_pack4_sse(const Tensor& top_blob_tm, Tensor& top_blob, const Tensor& bias) {
    
}

void conv3x3s1_winograd23_transform_input_pack4_sse(const Tensor& bottom_blob, Tensor& bottom_blob_tm) {
    
}

void conv3x3s1_winograd23_transform_output_pack4_sse(const Tensor& top_blob_tm, Tensor& top_blob, const Tensor& bias) {
    
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

#endif  // __SSE2__

}   // end namespace otter
