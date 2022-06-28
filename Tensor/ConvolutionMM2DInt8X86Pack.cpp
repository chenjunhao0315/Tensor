//
//  ConvolutionMM2DInt8X86Pack.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/25.
//

#include "ConvolutionMM2DInt8X86Pack.hpp"
#include "TensorFactory.hpp"
#include "Parallel.hpp"
#include "im2col.hpp"
#include "ConvolutionUtils.hpp"
#include "Padding.hpp"
#include "VecIntrinsic.hpp"

namespace otter {

#if __SSE2__

void convolution_im2col_sgemm_transform_kernel_pack1to4_int8_x86(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h) {
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 4a-4b-maxk-inch/4a-outch/4b
    Tensor kernel = _kernel.view({outch, inch, maxk});
    if (inch >= 4)
        kernel_tm = otter::empty({outch / 4, inch / 4 + inch % 4, 16 * maxk}, otter::ScalarType::Byte);
    else
        kernel_tm = otter::empty({outch / 4, inch, 4 * maxk}, otter::ScalarType::Byte);

    auto kernel_ra = kernel.raw_accessor<signed char, 3>();
    auto kernel_tm_ra = kernel_tm.raw_accessor<signed char, 3>();
    
    for (int q = 0; q + 3 < outch; q += 4) {
        signed char* g00 = kernel_tm_ra[q / 4].data();

        int p = 0;
        for (; p + 3 < inch; p += 4) {
            for (int k = 0; k < maxk; k++) {
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        const signed char* k00 = kernel_ra[q + i][p + j].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
        for (; p < inch; p++) {
            for (int k = 0; k < maxk; k++) {
                for (int i = 0; i < 4; i++) {
                    const signed char* k00 = kernel_ra[q + i][p].data();

                    g00[0] = k00[k];

                    g00++;
                }
            }
        }
    }
}

void convolution_im2col_sgemm_transform_kernel_pack8to1_int8_x86(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h) {
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8a-4b-maxk-inch/8a-outch/4b
    Tensor kernel = _kernel.view({outch, inch, maxk});
    if (outch >= 4)
        kernel_tm = otter::empty({outch / 4 + outch % 4, inch / 8, 32 * maxk}, otter::ScalarType::Byte);
    else
        kernel_tm = otter::empty({outch, inch / 8, 8 * maxk}, otter::ScalarType::Byte);

    auto kernel_ra = kernel.raw_accessor<signed char, 3>();
    auto kernel_tm_ra = kernel_tm.raw_accessor<signed char, 3>();
    
    int q = 0;
    for (; q + 3 < outch; q += 4) {
        signed char* g00 = kernel_tm_ra[q / 4].data();

        for (int p = 0; p + 7 < inch; p += 8) {
            for (int k = 0; k < maxk; k++) {
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 8; j++) {
                        const signed char* k00 = kernel_ra[q + i][p + j].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
    }
    // TODO unroll 2
    for (; q < outch; q++) {
        signed char* g00 = kernel_tm_ra[q / 4 + q % 4].data();

        for (int p = 0; p + 7 < inch; p += 8) {
            for (int k = 0; k < maxk; k++) {
                for (int j = 0; j < 8; j++) {
                    const signed char* k00 = kernel_ra[q][p + j].data();

                    g00[0] = k00[k];

                    g00++;
                }
            }
        }
    }
}

void convolution_im2col_sgemm_transform_kernel_pack8to4_int8_x86(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h) {
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8a-4b-maxk-inch/8a-outch/4b
    Tensor kernel = _kernel.view({outch, inch, maxk});
    kernel_tm = otter::empty({outch / 4, inch / 8, 32 * maxk}, otter::ScalarType::Byte);

    auto kernel_ra = kernel.raw_accessor<signed char, 3>();
    auto kernel_tm_ra = kernel_tm.raw_accessor<signed char, 3>();
    
    for (int q = 0; q + 3 < outch; q += 4) {
        signed char* g00 = kernel_tm_ra[q / 4].data();

        for (int p = 0; p + 7 < inch; p += 8) {
            for (int k = 0; k < maxk; k++) {
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 8; j++) {
                        const signed char* k00 = kernel_ra[q + i][p + j].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
    }
}

void im2col_sgemm_conv2d_int8_pack1to4_impl_x86(
    const Tensor& im2col,
    const Tensor& kernel_tf_,
    Tensor& output) {
    
    const int size = im2col.size(2);
    const int maxk = im2col.size(1);
    const int inch = im2col.size(0);

    const int outch = output.size(1);
    
    auto output_a = output.accessor<int, 4, 4>()[0];
    auto im2col_a = im2col.accessor<unsigned char, 3>();
    auto kernel_a = kernel_tf_.accessor<unsigned char, 3>();

    // permute
    Tensor tmp;
    if (inch >= 4)
    {
#if __AVX2__
        if (size >= 4)
            tmp = otter::empty({size / 4 + (size % 4) / 2 + size % 2, inch / 4 + inch % 4, 4 * maxk}, otter::ScalarType::Byte4);
        else if (size >= 2)
            tmp = otter::empty({size / 2 + size % 2, inch / 4 + inch % 4, 2 * maxk}, otter::ScalarType::Byte4);
        else
            tmp = otter::empty({size, inch / 4 + inch % 4, maxk}, otter::ScalarType::Byte4);
#else
        if (size >= 2)
            tmp = otter::empty({size / 2 + size % 2, inch / 4 + inch % 4, 2 * maxk}, otter::ScalarType::Byte4);
        else
            tmp = otter::empty({size, inch / 4 + inch % 4, maxk}, otter::ScalarType::Byte4);
#endif
    }
    else
    {
#if __AVX2__
        if (size >= 4)
            tmp = otter::empty({size / 4 + (size % 4) / 2 + size % 2, inch, 4 * maxk}, otter::ScalarType::Byte);
        else if (size >= 2)
            tmp = otter::empty({size / 2 + size % 2, inch, 2 * maxk}, otter::ScalarType::Byte);
        else
            tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Byte);
#else
        if (size >= 2)
            tmp = otter::empty({size / 2 + size % 2, inch, 2 * maxk}, otter::ScalarType::Byte);
        else
            tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Byte);
#endif
    }
    auto tmp_ra = tmp.raw_accessor<signed char, 3>();
    {
#if __AVX2__
        int remain_size_start = 0;
        int nn_size = size >> 2;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 4;

                signed char* tmpptr = tmp_ra[i / 4].data();

                int q = 0;
                for (; q + 3 < inch; q += 4)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q + 0].data() + i;
                    const signed char* img1 = (const signed char*)im2col_a[q + 1].data() + i;
                    const signed char* img2 = (const signed char*)im2col_a[q + 2].data() + i;
                    const signed char* img3 = (const signed char*)im2col_a[q + 3].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        tmpptr[0] = img0[0];
                        tmpptr[1] = img1[0];
                        tmpptr[2] = img2[0];
                        tmpptr[3] = img3[0];
                        tmpptr[4] = img0[1];
                        tmpptr[5] = img1[1];
                        tmpptr[6] = img2[1];
                        tmpptr[7] = img3[1];
                        tmpptr[8] = img0[2];
                        tmpptr[9] = img1[2];
                        tmpptr[10] = img2[2];
                        tmpptr[11] = img3[2];
                        tmpptr[12] = img0[3];
                        tmpptr[13] = img1[3];
                        tmpptr[14] = img2[3];
                        tmpptr[15] = img3[3];
                        tmpptr += 16;

                        img0 += size;
                        img1 += size;
                        img2 += size;
                        img3 += size;
                    }
                }
                for (; q < inch; q++)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        tmpptr[0] = img0[0];
                        tmpptr[1] = img0[1];
                        tmpptr[2] = img0[2];
                        tmpptr[3] = img0[3];

                        tmpptr += 4;

                        img0 += size;
                    }
                }
            }
        });

        remain_size_start += nn_size << 2;
        nn_size = (size - remain_size_start) >> 1;
#else
        int remain_size_start = 0;
        int nn_size = (size - remain_size_start) >> 1;
#endif

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 2;

    #if __AVX2__
                signed char* tmpptr = tmp_ra[i / 4 + (i % 4) / 2].data();
    #else
                signed char* tmpptr = tmp_ra[i / 2].data();
    #endif

                int q = 0;
                for (; q + 3 < inch; q += 4)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q + 0].data() + i;
                    const signed char* img1 = (const signed char*)im2col_a[q + 1].data() + i;
                    const signed char* img2 = (const signed char*)im2col_a[q + 2].data() + i;
                    const signed char* img3 = (const signed char*)im2col_a[q + 3].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        tmpptr[0] = img0[0];
                        tmpptr[1] = img1[0];
                        tmpptr[2] = img2[0];
                        tmpptr[3] = img3[0];
                        tmpptr[4] = img0[1];
                        tmpptr[5] = img1[1];
                        tmpptr[6] = img2[1];
                        tmpptr[7] = img3[1];
                        tmpptr += 8;

                        img0 += size;
                        img1 += size;
                        img2 += size;
                        img3 += size;
                    }
                }
                for (; q < inch; q++)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        tmpptr[0] = img0[0];
                        tmpptr[1] = img0[1];

                        tmpptr += 2;

                        img0 += size;
                    }
                }
            }
        });

        remain_size_start += nn_size << 1;

        otter::parallel_for(remain_size_start, size, 0, [&](int64_t begin, int64_t end) {
            for (const auto i : otter::irange(begin, end))
            {
    #if __AVX2__
                signed char* tmpptr = tmp_ra[i / 4 + (i % 4) / 2 + i % 2].data();
    #else
                signed char* tmpptr = tmp_ra[i / 2 + i % 2].data();
    #endif

                int q = 0;
                for (; q + 3 < inch; q += 4)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q + 0].data() + i;
                    const signed char* img1 = (const signed char*)im2col_a[q + 1].data() + i;
                    const signed char* img2 = (const signed char*)im2col_a[q + 2].data() + i;
                    const signed char* img3 = (const signed char*)im2col_a[q + 3].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        tmpptr[0] = img0[0];
                        tmpptr[1] = img1[0];
                        tmpptr[2] = img2[0];
                        tmpptr[3] = img3[0];
                        tmpptr += 4;

                        img0 += size;
                        img1 += size;
                        img2 += size;
                        img3 += size;
                    }
                }
                for (; q < inch; q++)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        tmpptr[0] = img0[0];

                        tmpptr += 1;

                        img0 += size;
                    }
                }
            }
        });
    }

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            int* outptr0 = output_a[p].data();

            int i = 0;
    #if __AVX2__
            for (; i + 3 < size; i += 4)
            {
                const signed char* tmpptr = (const signed char*)tmp_ra[i / 4].data();
                const signed char* kptr0 = (const signed char*)kernel_a[p].data();

                int nn4 = (inch / 4) * maxk;
                int nn1 = (inch % 4) * maxk;

                __m256i _sum00_12 = _mm256_setzero_si256();
                __m256i _sum20_32 = _mm256_setzero_si256();

                if (nn4 > 0)
                {
    #if __AVXVNNI__ || __AVX512VNNI__
                    __m256i _sum10_02 = _mm256_setzero_si256();
                    __m256i _sum30_22 = _mm256_setzero_si256();
    #else
                    __m256i _sum10_02 = _mm256_setzero_si256();
                    __m256i _sum01_13 = _mm256_setzero_si256();
                    __m256i _sum11_03 = _mm256_setzero_si256();
                    __m256i _sum30_22 = _mm256_setzero_si256();
                    __m256i _sum21_33 = _mm256_setzero_si256();
                    __m256i _sum31_23 = _mm256_setzero_si256();
    #endif

                    int j = 0;
                    for (; j < nn4; j++)
                    {
                        __m128i _val0123 = _mm_loadu_si128((const __m128i*)tmpptr);
                        __m256i _val0123_16 = _mm256_cvtepi8_epi16(_val0123);

                        __m256i _val01_16 = _mm256_permute4x64_epi64(_val0123_16, _MM_SHUFFLE(1, 1, 0, 0));
                        __m256i _val23_16 = _mm256_permute4x64_epi64(_val0123_16, _MM_SHUFFLE(3, 3, 2, 2));

                        __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                        __m256i _w01_16 = _mm256_cvtepi8_epi16(_w01);

                        __m256i _val10_16 = _mm256_permute4x64_epi64(_val01_16, 78);
                        __m256i _val32_16 = _mm256_permute4x64_epi64(_val23_16, 78);

    #if __AVXVNNI__ || __AVX512VNNI__
                        _sum00_12 = _mm256_dpwssd_epi32(_sum00_12, _val01_16, _w01_16);
                        _sum10_02 = _mm256_dpwssd_epi32(_sum10_02, _val10_16, _w01_16);
                        _sum20_32 = _mm256_dpwssd_epi32(_sum20_32, _val23_16, _w01_16);
                        _sum30_22 = _mm256_dpwssd_epi32(_sum30_22, _val32_16, _w01_16);
    #else
                        __m256i _sl00_11 = _mm256_mullo_epi16(_val01_16, _w01_16);
                        __m256i _sh00_11 = _mm256_mulhi_epi16(_val01_16, _w01_16);
                        __m256i _sl10_01 = _mm256_mullo_epi16(_val10_16, _w01_16);
                        __m256i _sh10_01 = _mm256_mulhi_epi16(_val10_16, _w01_16);
                        __m256i _sl20_31 = _mm256_mullo_epi16(_val23_16, _w01_16);
                        __m256i _sh20_31 = _mm256_mulhi_epi16(_val23_16, _w01_16);
                        __m256i _sl30_21 = _mm256_mullo_epi16(_val32_16, _w01_16);
                        __m256i _sh30_21 = _mm256_mulhi_epi16(_val32_16, _w01_16);

                        _sum00_12 = _mm256_add_epi32(_sum00_12, _mm256_unpacklo_epi16(_sl00_11, _sh00_11));
                        _sum10_02 = _mm256_add_epi32(_sum10_02, _mm256_unpacklo_epi16(_sl10_01, _sh10_01));
                        _sum01_13 = _mm256_add_epi32(_sum01_13, _mm256_unpackhi_epi16(_sl00_11, _sh00_11));
                        _sum11_03 = _mm256_add_epi32(_sum11_03, _mm256_unpackhi_epi16(_sl10_01, _sh10_01));
                        _sum20_32 = _mm256_add_epi32(_sum20_32, _mm256_unpacklo_epi16(_sl20_31, _sh20_31));
                        _sum30_22 = _mm256_add_epi32(_sum30_22, _mm256_unpacklo_epi16(_sl30_21, _sh30_21));
                        _sum21_33 = _mm256_add_epi32(_sum21_33, _mm256_unpackhi_epi16(_sl20_31, _sh20_31));
                        _sum31_23 = _mm256_add_epi32(_sum31_23, _mm256_unpackhi_epi16(_sl30_21, _sh30_21));
    #endif

                        tmpptr += 16;
                        kptr0 += 16;
                    }

    #if __AVXVNNI__ || __AVX512VNNI__
                    _sum00_12 = _mm256_hadd_epi32(_sum00_12, _sum10_02);
                    _sum20_32 = _mm256_hadd_epi32(_sum20_32, _sum30_22);

                    _sum00_12 = _mm256_permute4x64_epi64(_sum00_12, _MM_SHUFFLE(2, 1, 3, 0));
                    _sum20_32 = _mm256_permute4x64_epi64(_sum20_32, _MM_SHUFFLE(2, 1, 3, 0));
    #else
                    // transpose 4x8
                    {
                        __m256i _tmp0, _tmp1, _tmp2, _tmp3;
                        _tmp0 = _mm256_unpacklo_epi32(_sum00_12, _sum10_02);
                        _tmp1 = _mm256_unpacklo_epi32(_sum01_13, _sum11_03);
                        _tmp2 = _mm256_unpackhi_epi32(_sum00_12, _sum10_02);
                        _tmp3 = _mm256_unpackhi_epi32(_sum01_13, _sum11_03);
                        _sum00_12 = _mm256_unpacklo_epi64(_tmp0, _tmp1);
                        _sum10_02 = _mm256_unpackhi_epi64(_tmp0, _tmp1);
                        _sum01_13 = _mm256_unpacklo_epi64(_tmp2, _tmp3);
                        _sum11_03 = _mm256_unpackhi_epi64(_tmp2, _tmp3);
                    }
                    {
                        __m256i _tmp0, _tmp1, _tmp2, _tmp3;
                        _tmp0 = _mm256_unpacklo_epi32(_sum20_32, _sum30_22);
                        _tmp1 = _mm256_unpacklo_epi32(_sum21_33, _sum31_23);
                        _tmp2 = _mm256_unpackhi_epi32(_sum20_32, _sum30_22);
                        _tmp3 = _mm256_unpackhi_epi32(_sum21_33, _sum31_23);
                        _sum20_32 = _mm256_unpacklo_epi64(_tmp0, _tmp1);
                        _sum30_22 = _mm256_unpackhi_epi64(_tmp0, _tmp1);
                        _sum21_33 = _mm256_unpacklo_epi64(_tmp2, _tmp3);
                        _sum31_23 = _mm256_unpackhi_epi64(_tmp2, _tmp3);
                    }

                    _sum00_12 = _mm256_add_epi32(_sum00_12, _sum10_02);
                    _sum01_13 = _mm256_add_epi32(_sum01_13, _sum11_03);
                    _sum00_12 = _mm256_add_epi32(_sum00_12, _sum01_13);

                    _sum20_32 = _mm256_add_epi32(_sum20_32, _sum30_22);
                    _sum21_33 = _mm256_add_epi32(_sum21_33, _sum31_23);
                    _sum20_32 = _mm256_add_epi32(_sum20_32, _sum21_33);

                    __m256i _perm_mask = _mm256_set_epi32(6, 4, 3, 1, 7, 5, 2, 0);
                    _sum00_12 = _mm256_permutevar8x32_epi32(_sum00_12, _perm_mask);
                    _sum20_32 = _mm256_permutevar8x32_epi32(_sum20_32, _perm_mask);
    #endif
                }

                __m128i _sum00 = _mm256_extracti128_si256(_sum00_12, 0);
                __m128i _sum10 = _mm256_extracti128_si256(_sum00_12, 1);
                __m128i _sum20 = _mm256_extracti128_si256(_sum20_32, 0);
                __m128i _sum30 = _mm256_extracti128_si256(_sum20_32, 1);

                int j = 0;
                for (; j < nn1; j++)
                {
                    __m128i _val01 = _mm_set_epi16(tmpptr[1], tmpptr[1], tmpptr[1], tmpptr[1], tmpptr[0], tmpptr[0], tmpptr[0], tmpptr[0]);
                    __m128i _val23 = _mm_set_epi16(tmpptr[3], tmpptr[3], tmpptr[3], tmpptr[3], tmpptr[2], tmpptr[2], tmpptr[2], tmpptr[2]);

                    __m128i _w0123 = _mm_set_epi16(kptr0[3], kptr0[2], kptr0[1], kptr0[0], kptr0[3], kptr0[2], kptr0[1], kptr0[0]);

                    __m128i _sl00 = _mm_mullo_epi16(_val01, _w0123);
                    __m128i _sh00 = _mm_mulhi_epi16(_val01, _w0123);
                    __m128i _sl10 = _mm_mullo_epi16(_val23, _w0123);
                    __m128i _sh10 = _mm_mulhi_epi16(_val23, _w0123);

                    _sum00 = _mm_add_epi32(_sum00, _mm_unpacklo_epi16(_sl00, _sh00));
                    _sum10 = _mm_add_epi32(_sum10, _mm_unpackhi_epi16(_sl00, _sh00));
                    _sum20 = _mm_add_epi32(_sum20, _mm_unpacklo_epi16(_sl10, _sh10));
                    _sum30 = _mm_add_epi32(_sum30, _mm_unpackhi_epi16(_sl10, _sh10));

                    tmpptr += 4;
                    kptr0 += 4;
                }

                _mm_storeu_si128((__m128i*)outptr0, _sum00);
                _mm_storeu_si128((__m128i*)(outptr0 + 4), _sum10);
                _mm_storeu_si128((__m128i*)(outptr0 + 8), _sum20);
                _mm_storeu_si128((__m128i*)(outptr0 + 12), _sum30);
                outptr0 += 16;
            }
    #endif
            for (; i + 1 < size; i += 2)
            {
    #if __AVX2__
                const signed char* tmpptr = tmp_ra[i / 4 + (i % 4) / 2].data();
    #else
                const signed char* tmpptr = tmp_ra[i / 2].data();
    #endif
                const signed char* kptr0 = (const signed char*)kernel_a[p].data();

                int nn4 = (inch / 4) * maxk;
                int nn1 = (inch % 4) * maxk;

    #if __AVX2__
                __m256i _sum00_12 = _mm256_setzero_si256();
    #else
                __m128i _sum00 = _mm_setzero_si128();
                __m128i _sum10 = _mm_setzero_si128();
    #endif

                if (nn4 > 0)
                {
    #if __AVX2__
    #if __AVXVNNI__ || __AVX512VNNI__
                    __m256i _sum10_02 = _mm256_setzero_si256();
    #else
                    __m256i _sum10_02 = _mm256_setzero_si256();
                    __m256i _sum01_13 = _mm256_setzero_si256();
                    __m256i _sum11_03 = _mm256_setzero_si256();
    #endif
    #else
    #if __XOP__
                    __m128i _sum01 = _mm_setzero_si128();
                    __m128i _sum11 = _mm_setzero_si128();
    #else
                    __m128i _sum01 = _mm_setzero_si128();
                    __m128i _sum02 = _mm_setzero_si128();
                    __m128i _sum03 = _mm_setzero_si128();
                    __m128i _sum11 = _mm_setzero_si128();
                    __m128i _sum12 = _mm_setzero_si128();
                    __m128i _sum13 = _mm_setzero_si128();
    #endif
    #endif

                    int j = 0;
                    for (; j < nn4; j++)
                    {
    #if __AVX2__
                        __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                        __m256i _val01_16 = _mm256_cvtepi8_epi16(_val01);

                        _val01_16 = _mm256_permute4x64_epi64(_val01_16, _MM_SHUFFLE(1, 1, 0, 0));

                        __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                        __m256i _w01_16 = _mm256_cvtepi8_epi16(_w01);

                        __m256i _val10_16 = _mm256_permute4x64_epi64(_val01_16, 78);

    #if __AVXVNNI__ || __AVX512VNNI__
                        _sum00_12 = _mm256_dpwssd_epi32(_sum00_12, _val01_16, _w01_16);
                        _sum10_02 = _mm256_dpwssd_epi32(_sum10_02, _val10_16, _w01_16);
    #else
                        __m256i _sl00_11 = _mm256_mullo_epi16(_val01_16, _w01_16);
                        __m256i _sh00_11 = _mm256_mulhi_epi16(_val01_16, _w01_16);
                        __m256i _sl10_01 = _mm256_mullo_epi16(_val10_16, _w01_16);
                        __m256i _sh10_01 = _mm256_mulhi_epi16(_val10_16, _w01_16);

                        _sum00_12 = _mm256_add_epi32(_sum00_12, _mm256_unpacklo_epi16(_sl00_11, _sh00_11));
                        _sum10_02 = _mm256_add_epi32(_sum10_02, _mm256_unpacklo_epi16(_sl10_01, _sh10_01));
                        _sum01_13 = _mm256_add_epi32(_sum01_13, _mm256_unpackhi_epi16(_sl00_11, _sh00_11));
                        _sum11_03 = _mm256_add_epi32(_sum11_03, _mm256_unpackhi_epi16(_sl10_01, _sh10_01));
    #endif
    #else
                        __m128i _val01 = _mm_loadl_epi64((const __m128i*)tmpptr);
    #if __SSE4_1__
                        _val01 = _mm_cvtepi8_epi16(_val01);
    #else
                        __m128i _extval01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val01);
                        _val01 = _mm_unpacklo_epi8(_val01, _extval01);
    #endif

                        __m128i _val0 = _mm_shuffle_epi32(_val01, _MM_SHUFFLE(1, 0, 1, 0));
                        __m128i _val1 = _mm_shuffle_epi32(_val01, _MM_SHUFFLE(3, 2, 3, 2));

                        __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                        __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                        __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                        __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);

    #if __XOP__
                        _sum00 = _mm_maddd_epi16(_val0, _w0, _sum00);
                        _sum01 = _mm_maddd_epi16(_val0, _w1, _sum01);
                        _sum10 = _mm_maddd_epi16(_val1, _w0, _sum10);
                        _sum11 = _mm_maddd_epi16(_val1, _w1, _sum11);
    #else
                        __m128i _sl00 = _mm_mullo_epi16(_val0, _w0);
                        __m128i _sh00 = _mm_mulhi_epi16(_val0, _w0);
                        __m128i _sl01 = _mm_mullo_epi16(_val0, _w1);
                        __m128i _sh01 = _mm_mulhi_epi16(_val0, _w1);
                        __m128i _sl10 = _mm_mullo_epi16(_val1, _w0);
                        __m128i _sh10 = _mm_mulhi_epi16(_val1, _w0);
                        __m128i _sl11 = _mm_mullo_epi16(_val1, _w1);
                        __m128i _sh11 = _mm_mulhi_epi16(_val1, _w1);

                        _sum00 = _mm_add_epi32(_sum00, _mm_unpacklo_epi16(_sl00, _sh00));
                        _sum01 = _mm_add_epi32(_sum01, _mm_unpackhi_epi16(_sl00, _sh00));
                        _sum02 = _mm_add_epi32(_sum02, _mm_unpacklo_epi16(_sl01, _sh01));
                        _sum03 = _mm_add_epi32(_sum03, _mm_unpackhi_epi16(_sl01, _sh01));
                        _sum10 = _mm_add_epi32(_sum10, _mm_unpacklo_epi16(_sl10, _sh10));
                        _sum11 = _mm_add_epi32(_sum11, _mm_unpackhi_epi16(_sl10, _sh10));
                        _sum12 = _mm_add_epi32(_sum12, _mm_unpacklo_epi16(_sl11, _sh11));
                        _sum13 = _mm_add_epi32(_sum13, _mm_unpackhi_epi16(_sl11, _sh11));
    #endif
    #endif

                        tmpptr += 8;
                        kptr0 += 16;
                    }

    #if __AVX2__
    #if __AVXVNNI__ || __AVX512VNNI__
                    _sum00_12 = _mm256_hadd_epi32(_sum00_12, _sum10_02);

                    _sum00_12 = _mm256_permute4x64_epi64(_sum00_12, _MM_SHUFFLE(2, 1, 3, 0));
    #else
                    // transpose 4x8
                    {
                        __m256i _tmp0, _tmp1, _tmp2, _tmp3;
                        _tmp0 = _mm256_unpacklo_epi32(_sum00_12, _sum10_02);
                        _tmp1 = _mm256_unpacklo_epi32(_sum01_13, _sum11_03);
                        _tmp2 = _mm256_unpackhi_epi32(_sum00_12, _sum10_02);
                        _tmp3 = _mm256_unpackhi_epi32(_sum01_13, _sum11_03);
                        _sum00_12 = _mm256_unpacklo_epi64(_tmp0, _tmp1);
                        _sum10_02 = _mm256_unpackhi_epi64(_tmp0, _tmp1);
                        _sum01_13 = _mm256_unpacklo_epi64(_tmp2, _tmp3);
                        _sum11_03 = _mm256_unpackhi_epi64(_tmp2, _tmp3);
                    }

                    _sum00_12 = _mm256_add_epi32(_sum00_12, _sum10_02);
                    _sum01_13 = _mm256_add_epi32(_sum01_13, _sum11_03);
                    _sum00_12 = _mm256_add_epi32(_sum00_12, _sum01_13);

                    __m256i _perm_mask = _mm256_set_epi32(6, 4, 3, 1, 7, 5, 2, 0);
                    _sum00_12 = _mm256_permutevar8x32_epi32(_sum00_12, _perm_mask);
    #endif
    #else
    #if __XOP__
                    _sum00 = _mm_hadd_epi32(_sum00, _sum01);
                    _sum10 = _mm_hadd_epi32(_sum10, _sum11);
    #else
                    // transpose 4x4
                    {
                        __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                        _tmp0 = _mm_unpacklo_epi32(_sum00, _sum01);
                        _tmp1 = _mm_unpacklo_epi32(_sum02, _sum03);
                        _tmp2 = _mm_unpackhi_epi32(_sum00, _sum01);
                        _tmp3 = _mm_unpackhi_epi32(_sum02, _sum03);
                        _sum00 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                        _sum01 = _mm_unpackhi_epi64(_tmp0, _tmp1);
                        _sum02 = _mm_unpacklo_epi64(_tmp2, _tmp3);
                        _sum03 = _mm_unpackhi_epi64(_tmp2, _tmp3);
                    }
                    {
                        __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                        _tmp0 = _mm_unpacklo_epi32(_sum10, _sum11);
                        _tmp1 = _mm_unpacklo_epi32(_sum12, _sum13);
                        _tmp2 = _mm_unpackhi_epi32(_sum10, _sum11);
                        _tmp3 = _mm_unpackhi_epi32(_sum12, _sum13);
                        _sum10 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                        _sum11 = _mm_unpackhi_epi64(_tmp0, _tmp1);
                        _sum12 = _mm_unpacklo_epi64(_tmp2, _tmp3);
                        _sum13 = _mm_unpackhi_epi64(_tmp2, _tmp3);
                    }

                    _sum00 = _mm_add_epi32(_sum00, _sum01);
                    _sum02 = _mm_add_epi32(_sum02, _sum03);
                    _sum10 = _mm_add_epi32(_sum10, _sum11);
                    _sum12 = _mm_add_epi32(_sum12, _sum13);

                    _sum00 = _mm_add_epi32(_sum00, _sum02);
                    _sum10 = _mm_add_epi32(_sum10, _sum12);
    #endif
    #endif
                }

    #if __AVX2__
                __m128i _sum00 = _mm256_extracti128_si256(_sum00_12, 0);
                __m128i _sum10 = _mm256_extracti128_si256(_sum00_12, 1);
    #endif

                int j = 0;
                for (; j < nn1; j++)
                {
                    __m128i _val = _mm_set_epi16(tmpptr[1], tmpptr[1], tmpptr[1], tmpptr[1], tmpptr[0], tmpptr[0], tmpptr[0], tmpptr[0]);

                    __m128i _w0123 = _mm_loadl_epi64((const __m128i*)kptr0);
    #if __SSE4_1__
                    _w0123 = _mm_cvtepi8_epi16(_w0123);
    #else
                    __m128i _extw0123 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w0123);
                    _w0123 = _mm_unpacklo_epi8(_w0123, _extw0123);
    #endif

                    _w0123 = _mm_shuffle_epi32(_w0123, _MM_SHUFFLE(1, 0, 1, 0));

                    __m128i _sl00 = _mm_mullo_epi16(_val, _w0123);
                    __m128i _sh00 = _mm_mulhi_epi16(_val, _w0123);

                    _sum00 = _mm_add_epi32(_sum00, _mm_unpacklo_epi16(_sl00, _sh00));
                    _sum10 = _mm_add_epi32(_sum10, _mm_unpackhi_epi16(_sl00, _sh00));

                    tmpptr += 2;
                    kptr0 += 4;
                }

                _mm_storeu_si128((__m128i*)outptr0, _sum00);
                _mm_storeu_si128((__m128i*)(outptr0 + 4), _sum10);
                outptr0 += 8;
            }
            for (; i < size; i++)
            {
    #if __AVX2__
                const signed char* tmpptr = tmp_ra[i / 4 + (i % 4) / 2 + i % 2].data();
    #else
                const signed char* tmpptr = tmp_ra[i / 2 + i % 2].data();
    #endif
                const signed char* kptr0 = (const signed char*)kernel_a[p].data();

                int nn4 = (inch / 4) * maxk;
                int nn1 = (inch % 4) * maxk;

                __m128i _sum0 = _mm_setzero_si128();

                if (nn4 > 0)
                {
                    __m128i _sum1 = _mm_setzero_si128();
                    __m128i _sum2 = _mm_setzero_si128();
                    __m128i _sum3 = _mm_setzero_si128();

                    int j = 0;
                    for (; j < nn4; j++)
                    {
                        __m128i _val01 = _mm_loadl_epi64((const __m128i*)tmpptr);
    #if __SSE4_1__
                        __m128i _val0 = _mm_cvtepi8_epi16(_val01);
    #else
                        __m128i _extval01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val01);
                        __m128i _val0 = _mm_unpacklo_epi8(_val01, _extval01);
    #endif

                        _val0 = _mm_shuffle_epi32(_val0, _MM_SHUFFLE(1, 0, 1, 0));

                        __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                        __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                        __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                        __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);

                        __m128i _sl00 = _mm_mullo_epi16(_val0, _w0);
                        __m128i _sh00 = _mm_mulhi_epi16(_val0, _w0);
                        __m128i _sl01 = _mm_mullo_epi16(_val0, _w1);
                        __m128i _sh01 = _mm_mulhi_epi16(_val0, _w1);

                        _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl00, _sh00));
                        _sum1 = _mm_add_epi32(_sum1, _mm_unpackhi_epi16(_sl00, _sh00));
                        _sum2 = _mm_add_epi32(_sum2, _mm_unpacklo_epi16(_sl01, _sh01));
                        _sum3 = _mm_add_epi32(_sum3, _mm_unpackhi_epi16(_sl01, _sh01));

                        tmpptr += 4;
                        kptr0 += 16;
                    }

                    // transpose 4x4
                    {
                        __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                        _tmp0 = _mm_unpacklo_epi32(_sum0, _sum1);
                        _tmp1 = _mm_unpacklo_epi32(_sum2, _sum3);
                        _tmp2 = _mm_unpackhi_epi32(_sum0, _sum1);
                        _tmp3 = _mm_unpackhi_epi32(_sum2, _sum3);
                        _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                        _sum1 = _mm_unpackhi_epi64(_tmp0, _tmp1);
                        _sum2 = _mm_unpacklo_epi64(_tmp2, _tmp3);
                        _sum3 = _mm_unpackhi_epi64(_tmp2, _tmp3);
                    }

                    _sum0 = _mm_add_epi32(_sum0, _sum1);
                    _sum2 = _mm_add_epi32(_sum2, _sum3);
                    _sum0 = _mm_add_epi32(_sum0, _sum2);
                }

                int j = 0;
                for (; j < nn1; j++)
                {
                    __m128i _val = _mm_set1_epi16(tmpptr[0]);

                    __m128i _w0123 = _mm_loadl_epi64((const __m128i*)kptr0);
    #if __SSE4_1__
                    _w0123 = _mm_cvtepi8_epi16(_w0123);
    #else
                    __m128i _extw0123 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w0123);
                    _w0123 = _mm_unpacklo_epi8(_w0123, _extw0123);
    #endif

                    __m128i _sl00 = _mm_mullo_epi16(_val, _w0123);
                    __m128i _sh00 = _mm_mulhi_epi16(_val, _w0123);

                    _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl00, _sh00));

                    tmpptr += 1;
                    kptr0 += 4;
                }

                _mm_storeu_si128((__m128i*)outptr0, _sum0);
                outptr0 += 4;
            }
        }
    });
}

void im2col_sgemm_conv2d_int8_pack8to1_impl_x86(
    const Tensor& im2col,
    const Tensor& kernel_tf_,
    Tensor& output) {
    
    const int size = im2col.size(2);
    const int maxk = im2col.size(1);
    const int inch = im2col.size(0);

    const int outch = output.size(1);
    
    auto output_a = output.accessor<int, 4>()[0];
    auto im2col_ra = im2col.raw_accessor<int64_t, 3>();
    auto kernel_a = kernel_tf_.accessor<signed char, 3>();
    
    // permute
    Tensor tmp;
#if __AVX2__
    if (size >= 4)
        tmp = otter::empty({size / 4 + (size % 4) / 2 + size % 2, inch, 4 * maxk}, otter::ScalarType::Byte8);
    else if (size >= 2)
        tmp = otter::empty({size / 2 + size % 2, inch, 2 * maxk}, otter::ScalarType::Byte8);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Byte8);
#else
    if (size >= 2)
        tmp = otter::empty({size / 2 + size % 2, inch, 2 * maxk}, otter::ScalarType::Byte8);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Byte8);
#endif
    
    auto tmp_a = tmp.accessor<signed char, 3, 8>();
    auto tmp_ra = tmp.raw_accessor<int64_t, 3>();
    
    {
#if __AVX2__
        int remain_size_start = 0;
        int nn_size = size >> 2;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 4;

                int64_t* tmpptr = tmp_ra[i / 4].data();

                for (int q = 0; q < inch; q++)
                {
                    const int64_t* img0 = (const int64_t*)im2col_ra[q].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        __m256i _v = _mm256_loadu_si256((const __m256i*)img0);
                        _mm256_storeu_si256((__m256i*)tmpptr, _v);
                        tmpptr += 4;
                        img0 += size;
                    }
                }
            }
        });

        remain_size_start += nn_size << 2;
        nn_size = (size - remain_size_start) >> 1;
#else
        int remain_size_start = 0;
        int nn_size = (size - remain_size_start) >> 1;
#endif

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 2;

    #if __AVX2__
                int64_t* tmpptr = tmp_ra[i / 4 + (i % 4) / 2].data();
    #else
                int64_t* tmpptr = tmp_ra[i / 2].data();
    #endif

                for (int q = 0; q < inch; q++) {
                    const int64_t* img0 = (const int64_t*)im2col_ra[q].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        __m128i _v = _mm_loadu_si128((const __m128i*)img0);
                        _mm_storeu_si128((__m128i*)tmpptr, _v);
                        tmpptr += 2;
                        img0 += size;
                    }
                }
            }
        });

        remain_size_start += nn_size << 1;

        otter::parallel_for(remain_size_start, size, 0, [&](int64_t begin, int64_t end) {
            for (const auto i : otter::irange(begin, end)) {
    #if __AVX2__
                int64_t* tmpptr = tmp_ra[i / 4 + (i % 4) / 2 + i % 2].data();
    #else
                int64_t* tmpptr = tmp_ra[i / 2 + i % 2].data();
    #endif

                for (int q = 0; q < inch; q++)
                {
                    const int64_t* img0 = (const int64_t*)im2col_ra[q].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        tmpptr[0] = img0[0];
                        tmpptr += 1;
                        img0 += size;
                    }
                }
            }
        });
    }

    int nn_outch = 0;
    int remain_outch_start = 0;

    nn_outch = outch >> 2;

    otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto pp : otter::irange(begin, end)) {
            int p = pp * 4;

            int* outptr0 = output_a[p + 0].data();
            int* outptr1 = output_a[p + 1].data();
            int* outptr2 = output_a[p + 2].data();
            int* outptr3 = output_a[p + 3].data();

            int i = 0;
    #if __AVX2__
            for (; i + 3 < size; i += 4)
            {
                const signed char* tmpptr = tmp_a[i / 4].data();
                const signed char* kptr0 = kernel_a[p / 4].data();

                int nn = inch * maxk; // inch always > 0

                __m256i _sum00_11 = _mm256_setzero_si256();
                __m256i _sum10_01 = _mm256_setzero_si256();
                __m256i _sum02_13 = _mm256_setzero_si256();
                __m256i _sum12_03 = _mm256_setzero_si256();

                __m256i _sum04_15 = _mm256_setzero_si256();
                __m256i _sum14_05 = _mm256_setzero_si256();
                __m256i _sum06_17 = _mm256_setzero_si256();
                __m256i _sum16_07 = _mm256_setzero_si256();

                int j = 0;
                for (; j < nn; j++)
                {
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m256i _val01_16 = _mm256_cvtepi8_epi16(_val01);

                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _w23 = _mm_loadu_si128((const __m128i*)(kptr0 + 16));
                    __m256i _w01_16 = _mm256_cvtepi8_epi16(_w01);
                    __m256i _w23_16 = _mm256_cvtepi8_epi16(_w23);

                    __m256i _val10_16 = _mm256_permute4x64_epi64(_val01_16, 78);

    #if __AVXVNNI__ || __AVX512VNNI__
                    _sum00_11 = _mm256_dpwssd_epi32(_sum00_11, _val01_16, _w01_16);
                    _sum10_01 = _mm256_dpwssd_epi32(_sum10_01, _val10_16, _w01_16);
                    _sum02_13 = _mm256_dpwssd_epi32(_sum02_13, _val01_16, _w23_16);
                    _sum12_03 = _mm256_dpwssd_epi32(_sum12_03, _val10_16, _w23_16);
    #else
                    __m256i _sl00_11 = _mm256_mullo_epi16(_val01_16, _w01_16);
                    __m256i _sh00_11 = _mm256_mulhi_epi16(_val01_16, _w01_16);
                    __m256i _sl10_01 = _mm256_mullo_epi16(_val10_16, _w01_16);
                    __m256i _sh10_01 = _mm256_mulhi_epi16(_val10_16, _w01_16);
                    __m256i _sl02_13 = _mm256_mullo_epi16(_val01_16, _w23_16);
                    __m256i _sh02_13 = _mm256_mulhi_epi16(_val01_16, _w23_16);
                    __m256i _sl12_03 = _mm256_mullo_epi16(_val10_16, _w23_16);
                    __m256i _sh12_03 = _mm256_mulhi_epi16(_val10_16, _w23_16);

                    _sum00_11 = _mm256_add_epi32(_sum00_11, _mm256_unpacklo_epi16(_sl00_11, _sh00_11));
                    _sum10_01 = _mm256_add_epi32(_sum10_01, _mm256_unpacklo_epi16(_sl10_01, _sh10_01));
                    _sum02_13 = _mm256_add_epi32(_sum02_13, _mm256_unpacklo_epi16(_sl02_13, _sh02_13));
                    _sum12_03 = _mm256_add_epi32(_sum12_03, _mm256_unpacklo_epi16(_sl12_03, _sh12_03));
                    _sum00_11 = _mm256_add_epi32(_sum00_11, _mm256_unpackhi_epi16(_sl00_11, _sh00_11));
                    _sum10_01 = _mm256_add_epi32(_sum10_01, _mm256_unpackhi_epi16(_sl10_01, _sh10_01));
                    _sum02_13 = _mm256_add_epi32(_sum02_13, _mm256_unpackhi_epi16(_sl02_13, _sh02_13));
                    _sum12_03 = _mm256_add_epi32(_sum12_03, _mm256_unpackhi_epi16(_sl12_03, _sh12_03));
    #endif

                    __m128i _val23 = _mm_loadu_si128((const __m128i*)(tmpptr + 16));
                    __m256i _val23_16 = _mm256_cvtepi8_epi16(_val23);
                    __m256i _val32_16 = _mm256_permute4x64_epi64(_val23_16, 78);

    #if __AVXVNNI__ || __AVX512VNNI__
                    _sum04_15 = _mm256_dpwssd_epi32(_sum04_15, _val23_16, _w01_16);
                    _sum14_05 = _mm256_dpwssd_epi32(_sum14_05, _val32_16, _w01_16);
                    _sum06_17 = _mm256_dpwssd_epi32(_sum06_17, _val23_16, _w23_16);
                    _sum16_07 = _mm256_dpwssd_epi32(_sum16_07, _val32_16, _w23_16);
    #else
                    __m256i _sl04_15 = _mm256_mullo_epi16(_val23_16, _w01_16);
                    __m256i _sh04_15 = _mm256_mulhi_epi16(_val23_16, _w01_16);
                    __m256i _sl14_05 = _mm256_mullo_epi16(_val32_16, _w01_16);
                    __m256i _sh14_05 = _mm256_mulhi_epi16(_val32_16, _w01_16);
                    __m256i _sl06_17 = _mm256_mullo_epi16(_val23_16, _w23_16);
                    __m256i _sh06_17 = _mm256_mulhi_epi16(_val23_16, _w23_16);
                    __m256i _sl16_07 = _mm256_mullo_epi16(_val32_16, _w23_16);
                    __m256i _sh16_07 = _mm256_mulhi_epi16(_val32_16, _w23_16);

                    _sum04_15 = _mm256_add_epi32(_sum04_15, _mm256_unpacklo_epi16(_sl04_15, _sh04_15));
                    _sum14_05 = _mm256_add_epi32(_sum14_05, _mm256_unpacklo_epi16(_sl14_05, _sh14_05));
                    _sum06_17 = _mm256_add_epi32(_sum06_17, _mm256_unpacklo_epi16(_sl06_17, _sh06_17));
                    _sum16_07 = _mm256_add_epi32(_sum16_07, _mm256_unpacklo_epi16(_sl16_07, _sh16_07));
                    _sum04_15 = _mm256_add_epi32(_sum04_15, _mm256_unpackhi_epi16(_sl04_15, _sh04_15));
                    _sum14_05 = _mm256_add_epi32(_sum14_05, _mm256_unpackhi_epi16(_sl14_05, _sh14_05));
                    _sum06_17 = _mm256_add_epi32(_sum06_17, _mm256_unpackhi_epi16(_sl06_17, _sh06_17));
                    _sum16_07 = _mm256_add_epi32(_sum16_07, _mm256_unpackhi_epi16(_sl16_07, _sh16_07));
    #endif

                    tmpptr += 32;
                    kptr0 += 32;
                }

                // transpose 4x8
                {
                    __m256i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = _mm256_unpacklo_epi32(_sum00_11, _sum10_01);
                    _tmp1 = _mm256_unpacklo_epi32(_sum02_13, _sum12_03);
                    _tmp2 = _mm256_unpackhi_epi32(_sum00_11, _sum10_01);
                    _tmp3 = _mm256_unpackhi_epi32(_sum02_13, _sum12_03);
                    _sum00_11 = _mm256_unpacklo_epi64(_tmp0, _tmp1);
                    _sum10_01 = _mm256_unpackhi_epi64(_tmp0, _tmp1);
                    _sum02_13 = _mm256_unpacklo_epi64(_tmp2, _tmp3);
                    _sum12_03 = _mm256_unpackhi_epi64(_tmp2, _tmp3);
                }
                {
                    __m256i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = _mm256_unpacklo_epi32(_sum04_15, _sum14_05);
                    _tmp1 = _mm256_unpacklo_epi32(_sum06_17, _sum16_07);
                    _tmp2 = _mm256_unpackhi_epi32(_sum04_15, _sum14_05);
                    _tmp3 = _mm256_unpackhi_epi32(_sum06_17, _sum16_07);
                    _sum04_15 = _mm256_unpacklo_epi64(_tmp0, _tmp1);
                    _sum14_05 = _mm256_unpackhi_epi64(_tmp0, _tmp1);
                    _sum06_17 = _mm256_unpacklo_epi64(_tmp2, _tmp3);
                    _sum16_07 = _mm256_unpackhi_epi64(_tmp2, _tmp3);
                }

                _sum00_11 = _mm256_add_epi32(_sum00_11, _sum10_01);
                _sum02_13 = _mm256_add_epi32(_sum02_13, _sum12_03);
                _sum00_11 = _mm256_add_epi32(_sum00_11, _sum02_13);

                _sum04_15 = _mm256_add_epi32(_sum04_15, _sum14_05);
                _sum06_17 = _mm256_add_epi32(_sum06_17, _sum16_07);
                _sum04_15 = _mm256_add_epi32(_sum04_15, _sum06_17);

                __m256i _perm_mask = _mm256_set_epi32(6, 3, 4, 1, 7, 2, 5, 0);
                _sum00_11 = _mm256_permutevar8x32_epi32(_sum00_11, _perm_mask);
                _sum04_15 = _mm256_permutevar8x32_epi32(_sum04_15, _perm_mask);

                int sum[16];
                _mm256_storeu_si256((__m256i*)sum, _sum00_11);
                _mm256_storeu_si256((__m256i*)(sum + 8), _sum04_15);

                outptr0[0] = sum[0];
                outptr1[0] = sum[1];
                outptr2[0] = sum[2];
                outptr3[0] = sum[3];
                outptr0[1] = sum[4];
                outptr1[1] = sum[5];
                outptr2[1] = sum[6];
                outptr3[1] = sum[7];
                outptr0[2] = sum[8];
                outptr1[2] = sum[9];
                outptr2[2] = sum[10];
                outptr3[2] = sum[11];
                outptr0[3] = sum[12];
                outptr1[3] = sum[13];
                outptr2[3] = sum[14];
                outptr3[3] = sum[15];
                outptr0 += 4;
                outptr1 += 4;
                outptr2 += 4;
                outptr3 += 4;
            }
    #endif
            for (; i + 1 < size; i += 2)
            {
    #if __AVX2__
                const signed char* tmpptr = tmp_a[i / 4 + (i % 4) / 2].data();
    #else
                const signed char* tmpptr = tmp_a[i / 2].data();
    #endif
                const signed char* kptr0 = kernel_a[p / 4].data();

                int nn = inch * maxk; // inch always > 0

    #if __AVX2__
                __m256i _sum00_11 = _mm256_setzero_si256();
                __m256i _sum10_01 = _mm256_setzero_si256();
                __m256i _sum02_13 = _mm256_setzero_si256();
                __m256i _sum12_03 = _mm256_setzero_si256();
    #else
                __m128i _sum00 = _mm_setzero_si128();
                __m128i _sum01 = _mm_setzero_si128();
                __m128i _sum02 = _mm_setzero_si128();
                __m128i _sum03 = _mm_setzero_si128();
                __m128i _sum10 = _mm_setzero_si128();
                __m128i _sum11 = _mm_setzero_si128();
                __m128i _sum12 = _mm_setzero_si128();
                __m128i _sum13 = _mm_setzero_si128();
    #endif

                int j = 0;
                for (; j < nn; j++)
                {
    #if __AVX2__
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m256i _val01_16 = _mm256_cvtepi8_epi16(_val01);

                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _w23 = _mm_loadu_si128((const __m128i*)(kptr0 + 16));
                    __m256i _w01_16 = _mm256_cvtepi8_epi16(_w01);
                    __m256i _w23_16 = _mm256_cvtepi8_epi16(_w23);

                    __m256i _val10_16 = _mm256_permute4x64_epi64(_val01_16, 78);

    #if __AVXVNNI__ || __AVX512VNNI__
                    _sum00_11 = _mm256_dpwssd_epi32(_sum00_11, _val01_16, _w01_16);
                    _sum10_01 = _mm256_dpwssd_epi32(_sum10_01, _val10_16, _w01_16);
                    _sum02_13 = _mm256_dpwssd_epi32(_sum02_13, _val01_16, _w23_16);
                    _sum12_03 = _mm256_dpwssd_epi32(_sum12_03, _val10_16, _w23_16);
    #else
                    __m256i _sl00_11 = _mm256_mullo_epi16(_val01_16, _w01_16);
                    __m256i _sh00_11 = _mm256_mulhi_epi16(_val01_16, _w01_16);
                    __m256i _sl10_01 = _mm256_mullo_epi16(_val10_16, _w01_16);
                    __m256i _sh10_01 = _mm256_mulhi_epi16(_val10_16, _w01_16);
                    __m256i _sl02_13 = _mm256_mullo_epi16(_val01_16, _w23_16);
                    __m256i _sh02_13 = _mm256_mulhi_epi16(_val01_16, _w23_16);
                    __m256i _sl12_03 = _mm256_mullo_epi16(_val10_16, _w23_16);
                    __m256i _sh12_03 = _mm256_mulhi_epi16(_val10_16, _w23_16);

                    _sum00_11 = _mm256_add_epi32(_sum00_11, _mm256_unpacklo_epi16(_sl00_11, _sh00_11));
                    _sum10_01 = _mm256_add_epi32(_sum10_01, _mm256_unpacklo_epi16(_sl10_01, _sh10_01));
                    _sum02_13 = _mm256_add_epi32(_sum02_13, _mm256_unpacklo_epi16(_sl02_13, _sh02_13));
                    _sum12_03 = _mm256_add_epi32(_sum12_03, _mm256_unpacklo_epi16(_sl12_03, _sh12_03));
                    _sum00_11 = _mm256_add_epi32(_sum00_11, _mm256_unpackhi_epi16(_sl00_11, _sh00_11));
                    _sum10_01 = _mm256_add_epi32(_sum10_01, _mm256_unpackhi_epi16(_sl10_01, _sh10_01));
                    _sum02_13 = _mm256_add_epi32(_sum02_13, _mm256_unpackhi_epi16(_sl02_13, _sh02_13));
                    _sum12_03 = _mm256_add_epi32(_sum12_03, _mm256_unpackhi_epi16(_sl12_03, _sh12_03));
    #endif
    #else
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m128i _extval01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val01);
                    __m128i _val0 = _mm_unpacklo_epi8(_val01, _extval01);
                    __m128i _val1 = _mm_unpackhi_epi8(_val01, _extval01);

                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _w23 = _mm_loadu_si128((const __m128i*)(kptr0 + 16));
                    __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                    __m128i _extw23 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w23);
                    __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                    __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);
                    __m128i _w2 = _mm_unpacklo_epi8(_w23, _extw23);
                    __m128i _w3 = _mm_unpackhi_epi8(_w23, _extw23);

    #if __XOP__
                    _sum00 = _mm_maddd_epi16(_val0, _w0, _sum00);
                    _sum01 = _mm_maddd_epi16(_val0, _w1, _sum01);
                    _sum02 = _mm_maddd_epi16(_val0, _w2, _sum02);
                    _sum03 = _mm_maddd_epi16(_val0, _w3, _sum03);
                    _sum10 = _mm_maddd_epi16(_val1, _w0, _sum10);
                    _sum11 = _mm_maddd_epi16(_val1, _w1, _sum11);
                    _sum12 = _mm_maddd_epi16(_val1, _w2, _sum12);
                    _sum13 = _mm_maddd_epi16(_val1, _w3, _sum13);
    #else
                    __m128i _sl00 = _mm_mullo_epi16(_val0, _w0);
                    __m128i _sh00 = _mm_mulhi_epi16(_val0, _w0);
                    __m128i _sl01 = _mm_mullo_epi16(_val0, _w1);
                    __m128i _sh01 = _mm_mulhi_epi16(_val0, _w1);
                    __m128i _sl02 = _mm_mullo_epi16(_val0, _w2);
                    __m128i _sh02 = _mm_mulhi_epi16(_val0, _w2);
                    __m128i _sl03 = _mm_mullo_epi16(_val0, _w3);
                    __m128i _sh03 = _mm_mulhi_epi16(_val0, _w3);
                    __m128i _sl10 = _mm_mullo_epi16(_val1, _w0);
                    __m128i _sh10 = _mm_mulhi_epi16(_val1, _w0);
                    __m128i _sl11 = _mm_mullo_epi16(_val1, _w1);
                    __m128i _sh11 = _mm_mulhi_epi16(_val1, _w1);
                    __m128i _sl12 = _mm_mullo_epi16(_val1, _w2);
                    __m128i _sh12 = _mm_mulhi_epi16(_val1, _w2);
                    __m128i _sl13 = _mm_mullo_epi16(_val1, _w3);
                    __m128i _sh13 = _mm_mulhi_epi16(_val1, _w3);

                    _sum00 = _mm_add_epi32(_sum00, _mm_unpacklo_epi16(_sl00, _sh00));
                    _sum01 = _mm_add_epi32(_sum01, _mm_unpacklo_epi16(_sl01, _sh01));
                    _sum02 = _mm_add_epi32(_sum02, _mm_unpacklo_epi16(_sl02, _sh02));
                    _sum03 = _mm_add_epi32(_sum03, _mm_unpacklo_epi16(_sl03, _sh03));
                    _sum00 = _mm_add_epi32(_sum00, _mm_unpackhi_epi16(_sl00, _sh00));
                    _sum01 = _mm_add_epi32(_sum01, _mm_unpackhi_epi16(_sl01, _sh01));
                    _sum02 = _mm_add_epi32(_sum02, _mm_unpackhi_epi16(_sl02, _sh02));
                    _sum03 = _mm_add_epi32(_sum03, _mm_unpackhi_epi16(_sl03, _sh03));
                    _sum10 = _mm_add_epi32(_sum10, _mm_unpacklo_epi16(_sl10, _sh10));
                    _sum11 = _mm_add_epi32(_sum11, _mm_unpacklo_epi16(_sl11, _sh11));
                    _sum12 = _mm_add_epi32(_sum12, _mm_unpacklo_epi16(_sl12, _sh12));
                    _sum13 = _mm_add_epi32(_sum13, _mm_unpacklo_epi16(_sl13, _sh13));
                    _sum10 = _mm_add_epi32(_sum10, _mm_unpackhi_epi16(_sl10, _sh10));
                    _sum11 = _mm_add_epi32(_sum11, _mm_unpackhi_epi16(_sl11, _sh11));
                    _sum12 = _mm_add_epi32(_sum12, _mm_unpackhi_epi16(_sl12, _sh12));
                    _sum13 = _mm_add_epi32(_sum13, _mm_unpackhi_epi16(_sl13, _sh13));
    #endif
    #endif

                    tmpptr += 16;
                    kptr0 += 32;
                }

    #if __AVX2__
                // transpose 4x8
                {
                    __m256i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = _mm256_unpacklo_epi32(_sum00_11, _sum10_01);
                    _tmp1 = _mm256_unpacklo_epi32(_sum02_13, _sum12_03);
                    _tmp2 = _mm256_unpackhi_epi32(_sum00_11, _sum10_01);
                    _tmp3 = _mm256_unpackhi_epi32(_sum02_13, _sum12_03);
                    _sum00_11 = _mm256_unpacklo_epi64(_tmp0, _tmp1);
                    _sum10_01 = _mm256_unpackhi_epi64(_tmp0, _tmp1);
                    _sum02_13 = _mm256_unpacklo_epi64(_tmp2, _tmp3);
                    _sum12_03 = _mm256_unpackhi_epi64(_tmp2, _tmp3);
                }

                _sum00_11 = _mm256_add_epi32(_sum00_11, _sum10_01);
                _sum02_13 = _mm256_add_epi32(_sum02_13, _sum12_03);
                _sum00_11 = _mm256_add_epi32(_sum00_11, _sum02_13);

                __m256i _perm_mask = _mm256_set_epi32(6, 3, 4, 1, 7, 2, 5, 0);
                _sum00_11 = _mm256_permutevar8x32_epi32(_sum00_11, _perm_mask);

                int sum[8];
                _mm256_storeu_si256((__m256i*)sum, _sum00_11);
    #else
                // transpose 4x4
                {
                    __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = _mm_unpacklo_epi32(_sum00, _sum01);
                    _tmp1 = _mm_unpacklo_epi32(_sum02, _sum03);
                    _tmp2 = _mm_unpackhi_epi32(_sum00, _sum01);
                    _tmp3 = _mm_unpackhi_epi32(_sum02, _sum03);
                    _sum00 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                    _sum01 = _mm_unpackhi_epi64(_tmp0, _tmp1);
                    _sum02 = _mm_unpacklo_epi64(_tmp2, _tmp3);
                    _sum03 = _mm_unpackhi_epi64(_tmp2, _tmp3);
                }
                {
                    __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = _mm_unpacklo_epi32(_sum10, _sum11);
                    _tmp1 = _mm_unpacklo_epi32(_sum12, _sum13);
                    _tmp2 = _mm_unpackhi_epi32(_sum10, _sum11);
                    _tmp3 = _mm_unpackhi_epi32(_sum12, _sum13);
                    _sum10 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                    _sum11 = _mm_unpackhi_epi64(_tmp0, _tmp1);
                    _sum12 = _mm_unpacklo_epi64(_tmp2, _tmp3);
                    _sum13 = _mm_unpackhi_epi64(_tmp2, _tmp3);
                }

                _sum00 = _mm_add_epi32(_sum00, _sum01);
                _sum02 = _mm_add_epi32(_sum02, _sum03);
                _sum10 = _mm_add_epi32(_sum10, _sum11);
                _sum12 = _mm_add_epi32(_sum12, _sum13);

                _sum00 = _mm_add_epi32(_sum00, _sum02);
                _sum10 = _mm_add_epi32(_sum10, _sum12);

                int sum[8];
                _mm_storeu_si128((__m128i*)sum, _sum00);
                _mm_storeu_si128((__m128i*)(sum + 4), _sum10);
    #endif

                outptr0[0] = sum[0];
                outptr1[0] = sum[1];
                outptr2[0] = sum[2];
                outptr3[0] = sum[3];
                outptr0[1] = sum[4];
                outptr1[1] = sum[5];
                outptr2[1] = sum[6];
                outptr3[1] = sum[7];
                outptr0 += 2;
                outptr1 += 2;
                outptr2 += 2;
                outptr3 += 2;
            }
            for (; i < size; i++)
            {
    #if __AVX2__
                const signed char* tmpptr = tmp_a[i / 4 + (i % 4) / 2 + i % 2].data();
    #else
                const signed char* tmpptr = tmp_a[i / 2 + i % 2].data();
    #endif
                const signed char* kptr0 = kernel_a[p / 4].data();

                int nn = inch * maxk; // inch always > 0

    #if __AVX2__
                __m256i _sum0_1 = _mm256_setzero_si256();
                __m256i _sum2_3 = _mm256_setzero_si256();
    #else
                __m128i _sum0 = _mm_setzero_si128();
                __m128i _sum1 = _mm_setzero_si128();
                __m128i _sum2 = _mm_setzero_si128();
                __m128i _sum3 = _mm_setzero_si128();
    #endif

                int j = 0;
                for (; j < nn; j++)
                {
    #if __AVX2__
                    __m128i _val = _mm_loadl_epi64((const __m128i*)tmpptr);
                    _val = _mm_cvtepi8_epi16(_val);

                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _w23 = _mm_loadu_si128((const __m128i*)(kptr0 + 16));
                    __m256i _w01_16 = _mm256_cvtepi8_epi16(_w01);
                    __m256i _w23_16 = _mm256_cvtepi8_epi16(_w23);

                    __m256i _valval = _mm256_inserti128_si256(_mm256_castsi128_si256(_val), _val, 1);

    #if __AVXVNNI__ || __AVX512VNNI__
                    _sum0_1 = _mm256_dpwssd_epi32(_sum0_1, _valval, _w01_16);
                    _sum2_3 = _mm256_dpwssd_epi32(_sum2_3, _valval, _w23_16);
    #else
                    __m256i _sl0_1 = _mm256_mullo_epi16(_valval, _w01_16);
                    __m256i _sh0_1 = _mm256_mulhi_epi16(_valval, _w01_16);
                    __m256i _sl2_3 = _mm256_mullo_epi16(_valval, _w23_16);
                    __m256i _sh2_3 = _mm256_mulhi_epi16(_valval, _w23_16);

                    _sum0_1 = _mm256_add_epi32(_sum0_1, _mm256_unpacklo_epi16(_sl0_1, _sh0_1));
                    _sum2_3 = _mm256_add_epi32(_sum2_3, _mm256_unpacklo_epi16(_sl2_3, _sh2_3));
                    _sum0_1 = _mm256_add_epi32(_sum0_1, _mm256_unpackhi_epi16(_sl0_1, _sh0_1));
                    _sum2_3 = _mm256_add_epi32(_sum2_3, _mm256_unpackhi_epi16(_sl2_3, _sh2_3));
    #endif
    #else
                    __m128i _val = _mm_loadl_epi64((const __m128i*)tmpptr);
    #if __SSE4_1__
                    _val = _mm_cvtepi8_epi16(_val);
    #else
                    _val = _mm_unpacklo_epi8(_val, _mm_cmpgt_epi8(_mm_setzero_si128(), _val));
    #endif

                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _w23 = _mm_loadu_si128((const __m128i*)(kptr0 + 16));
                    __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                    __m128i _extw23 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w23);
                    __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                    __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);
                    __m128i _w2 = _mm_unpacklo_epi8(_w23, _extw23);
                    __m128i _w3 = _mm_unpackhi_epi8(_w23, _extw23);

    #if __XOP__
                    _sum0 = _mm_maddd_epi16(_val, _w0, _sum0);
                    _sum1 = _mm_maddd_epi16(_val, _w1, _sum1);
                    _sum2 = _mm_maddd_epi16(_val, _w2, _sum2);
                    _sum3 = _mm_maddd_epi16(_val, _w3, _sum3);
    #else
                    __m128i _sl0 = _mm_mullo_epi16(_val, _w0);
                    __m128i _sh0 = _mm_mulhi_epi16(_val, _w0);
                    __m128i _sl1 = _mm_mullo_epi16(_val, _w1);
                    __m128i _sh1 = _mm_mulhi_epi16(_val, _w1);
                    __m128i _sl2 = _mm_mullo_epi16(_val, _w2);
                    __m128i _sh2 = _mm_mulhi_epi16(_val, _w2);
                    __m128i _sl3 = _mm_mullo_epi16(_val, _w3);
                    __m128i _sh3 = _mm_mulhi_epi16(_val, _w3);

                    _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl0, _sh0));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpacklo_epi16(_sl1, _sh1));
                    _sum2 = _mm_add_epi32(_sum2, _mm_unpacklo_epi16(_sl2, _sh2));
                    _sum3 = _mm_add_epi32(_sum3, _mm_unpacklo_epi16(_sl3, _sh3));
                    _sum0 = _mm_add_epi32(_sum0, _mm_unpackhi_epi16(_sl0, _sh0));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpackhi_epi16(_sl1, _sh1));
                    _sum2 = _mm_add_epi32(_sum2, _mm_unpackhi_epi16(_sl2, _sh2));
                    _sum3 = _mm_add_epi32(_sum3, _mm_unpackhi_epi16(_sl3, _sh3));
    #endif
    #endif

                    tmpptr += 8;
                    kptr0 += 32;
                }

    #if __AVX2__
                __m128i _sum0 = _mm256_extracti128_si256(_sum0_1, 0);
                __m128i _sum1 = _mm256_extracti128_si256(_sum0_1, 1);
                __m128i _sum2 = _mm256_extracti128_si256(_sum2_3, 0);
                __m128i _sum3 = _mm256_extracti128_si256(_sum2_3, 1);
    #endif

                // transpose 4x4
                {
                    __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = _mm_unpacklo_epi32(_sum0, _sum1);
                    _tmp1 = _mm_unpacklo_epi32(_sum2, _sum3);
                    _tmp2 = _mm_unpackhi_epi32(_sum0, _sum1);
                    _tmp3 = _mm_unpackhi_epi32(_sum2, _sum3);
                    _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                    _sum1 = _mm_unpackhi_epi64(_tmp0, _tmp1);
                    _sum2 = _mm_unpacklo_epi64(_tmp2, _tmp3);
                    _sum3 = _mm_unpackhi_epi64(_tmp2, _tmp3);
                }

                _sum0 = _mm_add_epi32(_sum0, _sum1);
                _sum2 = _mm_add_epi32(_sum2, _sum3);

                _sum0 = _mm_add_epi32(_sum0, _sum2);

                int sum[4];
                _mm_storeu_si128((__m128i*)sum, _sum0);

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

    remain_outch_start += nn_outch << 2;

    otter::parallel_for(remain_outch_start, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            int* outptr0 = output_a[p].data();

            int i = 0;
    #if __AVX2__
            for (; i + 3 < size; i += 4)
            {
                const signed char* tmpptr = tmp_a[i / 4].data();
                const signed char* kptr0 = kernel_a[p / 4 + p % 4].data();

                int nn = inch * maxk; // inch always > 0

                __m256i _sum0_2 = _mm256_setzero_si256();
                __m256i _sum1_3 = _mm256_setzero_si256();
                __m256i _sum4_6 = _mm256_setzero_si256();
                __m256i _sum5_7 = _mm256_setzero_si256();

                int j = 0;
                for (; j < nn; j++)
                {
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m128i _val23 = _mm_loadu_si128((const __m128i*)(tmpptr + 16));
                    __m256i _val01_16 = _mm256_cvtepi8_epi16(_val01);
                    __m256i _val23_16 = _mm256_cvtepi8_epi16(_val23);

                    __m128i _w01 = _mm_loadl_epi64((const __m128i*)kptr0);
                    __m256i _w01_16 = _mm256_cvtepi8_epi16(_w01);
                    _w01_16 = _mm256_permute4x64_epi64(_w01_16, _MM_SHUFFLE(1, 0, 1, 0));

                    __m256i _sl00_10 = _mm256_mullo_epi16(_val01_16, _w01_16);
                    __m256i _sh00_10 = _mm256_mulhi_epi16(_val01_16, _w01_16);
                    __m256i _sl20_30 = _mm256_mullo_epi16(_val23_16, _w01_16);
                    __m256i _sh20_30 = _mm256_mulhi_epi16(_val23_16, _w01_16);

                    _sum0_2 = _mm256_add_epi32(_sum0_2, _mm256_unpacklo_epi16(_sl00_10, _sh00_10));
                    _sum1_3 = _mm256_add_epi32(_sum1_3, _mm256_unpackhi_epi16(_sl00_10, _sh00_10));
                    _sum4_6 = _mm256_add_epi32(_sum4_6, _mm256_unpacklo_epi16(_sl20_30, _sh20_30));
                    _sum5_7 = _mm256_add_epi32(_sum5_7, _mm256_unpackhi_epi16(_sl20_30, _sh20_30));

                    tmpptr += 32;
                    kptr0 += 8;
                }

                _sum0_2 = _mm256_add_epi32(_sum0_2, _sum1_3);
                _sum4_6 = _mm256_add_epi32(_sum4_6, _sum5_7);
                __m128i _sum0 = _mm256_extracti128_si256(_sum0_2, 0);
                __m128i _sum2 = _mm256_extracti128_si256(_sum0_2, 1);
                __m128i _sum4 = _mm256_extracti128_si256(_sum4_6, 0);
                __m128i _sum6 = _mm256_extracti128_si256(_sum4_6, 1);

                outptr0[0] = _mm_reduce_add_epi32(_sum0);
                outptr0[1] = _mm_reduce_add_epi32(_sum2);
                outptr0[2] = _mm_reduce_add_epi32(_sum4);
                outptr0[3] = _mm_reduce_add_epi32(_sum6);
                outptr0 += 4;
            }
    #endif
            for (; i + 1 < size; i += 2)
            {
    #if __AVX2__
                const signed char* tmpptr = tmp_a[i / 4 + (i % 4) / 2].data();
    #else
                const signed char* tmpptr = tmp_a[i / 2].data();
    #endif
                const signed char* kptr0 = kernel_a[p / 4 + p % 4].data();

                int nn = inch * maxk; // inch always > 0

    #if __AVX2__
                __m256i _sum0_2 = _mm256_setzero_si256();
                __m256i _sum1_3 = _mm256_setzero_si256();
    #else
                __m128i _sum0 = _mm_setzero_si128();
                __m128i _sum1 = _mm_setzero_si128();
                __m128i _sum2 = _mm_setzero_si128();
                __m128i _sum3 = _mm_setzero_si128();
    #endif

                int j = 0;
                for (; j < nn; j++)
                {
    #if __AVX2__
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m256i _val01_16 = _mm256_cvtepi8_epi16(_val01);

                    __m128i _w01 = _mm_loadl_epi64((const __m128i*)kptr0);
                    __m256i _w01_16 = _mm256_cvtepi8_epi16(_w01);
                    _w01_16 = _mm256_permute4x64_epi64(_w01_16, _MM_SHUFFLE(1, 0, 1, 0));

                    __m256i _sl00_10 = _mm256_mullo_epi16(_val01_16, _w01_16);
                    __m256i _sh00_10 = _mm256_mulhi_epi16(_val01_16, _w01_16);

                    _sum0_2 = _mm256_add_epi32(_sum0_2, _mm256_unpacklo_epi16(_sl00_10, _sh00_10));
                    _sum1_3 = _mm256_add_epi32(_sum1_3, _mm256_unpackhi_epi16(_sl00_10, _sh00_10));
    #else
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m128i _extval01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val01);
                    __m128i _val0 = _mm_unpacklo_epi8(_val01, _extval01);
                    __m128i _val1 = _mm_unpackhi_epi8(_val01, _extval01);

                    __m128i _w01 = _mm_loadl_epi64((const __m128i*)kptr0);
    #if __SSE4_1__
                    __m128i _w0 = _mm_cvtepi8_epi16(_w01);
    #else
                    __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                    __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
    #endif

                    __m128i _sl00 = _mm_mullo_epi16(_val0, _w0);
                    __m128i _sh00 = _mm_mulhi_epi16(_val0, _w0);
                    __m128i _sl10 = _mm_mullo_epi16(_val1, _w0);
                    __m128i _sh10 = _mm_mulhi_epi16(_val1, _w0);

                    _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl00, _sh00));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpackhi_epi16(_sl00, _sh00));
                    _sum2 = _mm_add_epi32(_sum2, _mm_unpacklo_epi16(_sl10, _sh10));
                    _sum3 = _mm_add_epi32(_sum3, _mm_unpackhi_epi16(_sl10, _sh10));
    #endif

                    tmpptr += 16;
                    kptr0 += 8;
                }

    #if __AVX2__
                _sum0_2 = _mm256_add_epi32(_sum0_2, _sum1_3);
                __m128i _sum0 = _mm256_extracti128_si256(_sum0_2, 0);
                __m128i _sum2 = _mm256_extracti128_si256(_sum0_2, 1);
    #else
                _sum0 = _mm_add_epi32(_sum0, _sum1);
                _sum2 = _mm_add_epi32(_sum2, _sum3);
    #endif

                outptr0[0] = _mm_reduce_add_epi32(_sum0);
                outptr0[1] = _mm_reduce_add_epi32(_sum2);
                outptr0 += 2;
            }
            for (; i < size; i++)
            {
    #if __AVX2__
                const signed char* tmpptr = tmp_a[i / 4 + (i % 4) / 2 + i % 2].data();
    #else
                const signed char* tmpptr = tmp_a[i / 2 + i % 2].data();
    #endif
                const signed char* kptr0 = kernel_a[p / 4 + p % 4].data();

                int nn = inch * maxk; // inch always > 0

                __m128i _sum0 = _mm_setzero_si128();
                __m128i _sum1 = _mm_setzero_si128();

                int j = 0;
                for (; j < nn; j++)
                {
                    __m128i _val01 = _mm_loadl_epi64((const __m128i*)tmpptr);
    #if __SSE4_1__
                    __m128i _val0 = _mm_cvtepi8_epi16(_val01);
    #else
                    __m128i _extval01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val01);
                    __m128i _val0 = _mm_unpacklo_epi8(_val01, _extval01);
    #endif

                    __m128i _w01 = _mm_loadl_epi64((const __m128i*)kptr0);
    #if __SSE4_1__
                    __m128i _w0 = _mm_cvtepi8_epi16(_w01);
    #else
                    __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                    __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
    #endif

                    __m128i _sl00 = _mm_mullo_epi16(_val0, _w0);
                    __m128i _sh00 = _mm_mulhi_epi16(_val0, _w0);

                    _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl00, _sh00));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpackhi_epi16(_sl00, _sh00));

                    tmpptr += 8;
                    kptr0 += 8;
                }

                _sum0 = _mm_add_epi32(_sum0, _sum1);

                outptr0[0] = _mm_reduce_add_epi32(_sum0);
                outptr0 += 1;
            }
        }
    });
}

void im2col_sgemm_conv2d_int8_pack8to4_impl_x86(
    const Tensor& im2col,
    const Tensor& kernel_tf_,
    Tensor& output) {
    
    const int size = im2col.size(2);
    const int maxk = im2col.size(1);
    const int inch = im2col.size(0);

    const int outch = output.size(1);
    
    auto output_a = output.accessor<int, 4, 4>()[0];
    auto im2col_ra = im2col.raw_accessor<int64_t, 3>();
    auto kernel_a = kernel_tf_.accessor<signed char, 3>();
    
    // permute
    Tensor tmp;
#if __AVX2__
    if (size >= 4)
        tmp = otter::empty({size / 4 + (size % 4) / 2 + size % 2, inch, 4 * maxk}, otter::ScalarType::Byte8);
    else if (size >= 2)
        tmp = otter::empty({size / 2 + size % 2, inch, 2 * maxk}, otter::ScalarType::Byte8);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Byte8);
#else
    if (size >= 2)
        tmp = otter::empty({size / 2 + size % 2, inch, 2 * maxk}, otter::ScalarType::Byte8);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Byte8);
#endif
    
    auto tmp_a = tmp.accessor<signed char, 3, 8>();
    auto tmp_ra = tmp.raw_accessor<int64_t, 3>();
    
    {
#if __AVX2__
        int remain_size_start = 0;
        int nn_size = size >> 2;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 4;

                int64_t* tmpptr = tmp_ra[i / 4].data();

                for (int q = 0; q < inch; q++) {
                    const int64_t* img0 = (const int64_t*)im2col_ra[q].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        __m256i _v = _mm256_loadu_si256((const __m256i*)img0);
                        _mm256_storeu_si256((__m256i*)tmpptr, _v);
                        tmpptr += 4;
                        img0 += size;
                    }
                }
            }
        });

        remain_size_start += nn_size << 2;
        nn_size = (size - remain_size_start) >> 1;
#else
        int remain_size_start = 0;
        int nn_size = size >> 1;
#endif

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 2;

    #if __AVX2__
                int64_t* tmpptr = tmp_ra[i / 4 + (i % 4) / 2].data();
    #else
                int64_t* tmpptr = tmp_ra[i / 2].data();
    #endif

                for (int q = 0; q < inch; q++)
                {
                    const int64_t* img0 = (const int64_t*)im2col_ra[q].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        __m128i _v = _mm_loadu_si128((const __m128i*)img0);
                        _mm_storeu_si128((__m128i*)tmpptr, _v);
                        tmpptr += 2;
                        img0 += size;
                    }
                }
            }
        });

        remain_size_start += nn_size << 1;

        otter::parallel_for(remain_size_start, size, 0, [&](int64_t begin, int64_t end) {
            for (const auto i : otter::irange(begin, end)) {
    #if __AVX2__
                int64_t* tmpptr = tmp_ra[i / 4 + (i % 4) / 2 + i % 2].data();
    #else
                int64_t* tmpptr = tmp_ra[i / 2 + i % 2].data();
    #endif

                for (int q = 0; q < inch; q++)
                {
                    const int64_t* img0 = (const int64_t*)im2col_ra[q].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        tmpptr[0] = img0[0];
                        tmpptr += 1;
                        img0 += size;
                    }
                }
            }
        });
    }

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            int* outptr0 = output_a[p].data();

            int i = 0;
    #if __AVX2__
            for (; i + 3 < size; i += 4)
            {
                const signed char* tmpptr = tmp_a[i / 4].data();
                const signed char* kptr0 = kernel_a[p].data();

                int nn = inch * maxk; // inch always > 0

                __m256i _sum00_11 = _mm256_setzero_si256();
                __m256i _sum10_01 = _mm256_setzero_si256();
                __m256i _sum02_13 = _mm256_setzero_si256();
                __m256i _sum12_03 = _mm256_setzero_si256();

                __m256i _sum04_15 = _mm256_setzero_si256();
                __m256i _sum14_05 = _mm256_setzero_si256();
                __m256i _sum06_17 = _mm256_setzero_si256();
                __m256i _sum16_07 = _mm256_setzero_si256();

                int j = 0;
                for (; j < nn; j++)
                {
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m256i _val01_16 = _mm256_cvtepi8_epi16(_val01);

                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _w23 = _mm_loadu_si128((const __m128i*)(kptr0 + 16));
                    __m256i _w01_16 = _mm256_cvtepi8_epi16(_w01);
                    __m256i _w23_16 = _mm256_cvtepi8_epi16(_w23);

                    __m256i _val10_16 = _mm256_permute4x64_epi64(_val01_16, 78);

    #if __AVXVNNI__ || __AVX512VNNI__
                    _sum00_11 = _mm256_dpwssd_epi32(_sum00_11, _val01_16, _w01_16);
                    _sum10_01 = _mm256_dpwssd_epi32(_sum10_01, _val10_16, _w01_16);
                    _sum02_13 = _mm256_dpwssd_epi32(_sum02_13, _val01_16, _w23_16);
                    _sum12_03 = _mm256_dpwssd_epi32(_sum12_03, _val10_16, _w23_16);
    #else
                    __m256i _sl00_11 = _mm256_mullo_epi16(_val01_16, _w01_16);
                    __m256i _sh00_11 = _mm256_mulhi_epi16(_val01_16, _w01_16);
                    __m256i _sl10_01 = _mm256_mullo_epi16(_val10_16, _w01_16);
                    __m256i _sh10_01 = _mm256_mulhi_epi16(_val10_16, _w01_16);
                    __m256i _sl02_13 = _mm256_mullo_epi16(_val01_16, _w23_16);
                    __m256i _sh02_13 = _mm256_mulhi_epi16(_val01_16, _w23_16);
                    __m256i _sl12_03 = _mm256_mullo_epi16(_val10_16, _w23_16);
                    __m256i _sh12_03 = _mm256_mulhi_epi16(_val10_16, _w23_16);

                    _sum00_11 = _mm256_add_epi32(_sum00_11, _mm256_unpacklo_epi16(_sl00_11, _sh00_11));
                    _sum10_01 = _mm256_add_epi32(_sum10_01, _mm256_unpacklo_epi16(_sl10_01, _sh10_01));
                    _sum02_13 = _mm256_add_epi32(_sum02_13, _mm256_unpacklo_epi16(_sl02_13, _sh02_13));
                    _sum12_03 = _mm256_add_epi32(_sum12_03, _mm256_unpacklo_epi16(_sl12_03, _sh12_03));
                    _sum00_11 = _mm256_add_epi32(_sum00_11, _mm256_unpackhi_epi16(_sl00_11, _sh00_11));
                    _sum10_01 = _mm256_add_epi32(_sum10_01, _mm256_unpackhi_epi16(_sl10_01, _sh10_01));
                    _sum02_13 = _mm256_add_epi32(_sum02_13, _mm256_unpackhi_epi16(_sl02_13, _sh02_13));
                    _sum12_03 = _mm256_add_epi32(_sum12_03, _mm256_unpackhi_epi16(_sl12_03, _sh12_03));
    #endif

                    __m128i _val23 = _mm_loadu_si128((const __m128i*)(tmpptr + 16));
                    __m256i _val23_16 = _mm256_cvtepi8_epi16(_val23);
                    __m256i _val32_16 = _mm256_permute4x64_epi64(_val23_16, 78);

    #if __AVXVNNI__ || __AVX512VNNI__
                    _sum04_15 = _mm256_dpwssd_epi32(_sum04_15, _val23_16, _w01_16);
                    _sum14_05 = _mm256_dpwssd_epi32(_sum14_05, _val32_16, _w01_16);
                    _sum06_17 = _mm256_dpwssd_epi32(_sum06_17, _val23_16, _w23_16);
                    _sum16_07 = _mm256_dpwssd_epi32(_sum16_07, _val32_16, _w23_16);
    #else
                    __m256i _sl04_15 = _mm256_mullo_epi16(_val23_16, _w01_16);
                    __m256i _sh04_15 = _mm256_mulhi_epi16(_val23_16, _w01_16);
                    __m256i _sl14_05 = _mm256_mullo_epi16(_val32_16, _w01_16);
                    __m256i _sh14_05 = _mm256_mulhi_epi16(_val32_16, _w01_16);
                    __m256i _sl06_17 = _mm256_mullo_epi16(_val23_16, _w23_16);
                    __m256i _sh06_17 = _mm256_mulhi_epi16(_val23_16, _w23_16);
                    __m256i _sl16_07 = _mm256_mullo_epi16(_val32_16, _w23_16);
                    __m256i _sh16_07 = _mm256_mulhi_epi16(_val32_16, _w23_16);

                    _sum04_15 = _mm256_add_epi32(_sum04_15, _mm256_unpacklo_epi16(_sl04_15, _sh04_15));
                    _sum14_05 = _mm256_add_epi32(_sum14_05, _mm256_unpacklo_epi16(_sl14_05, _sh14_05));
                    _sum06_17 = _mm256_add_epi32(_sum06_17, _mm256_unpacklo_epi16(_sl06_17, _sh06_17));
                    _sum16_07 = _mm256_add_epi32(_sum16_07, _mm256_unpacklo_epi16(_sl16_07, _sh16_07));
                    _sum04_15 = _mm256_add_epi32(_sum04_15, _mm256_unpackhi_epi16(_sl04_15, _sh04_15));
                    _sum14_05 = _mm256_add_epi32(_sum14_05, _mm256_unpackhi_epi16(_sl14_05, _sh14_05));
                    _sum06_17 = _mm256_add_epi32(_sum06_17, _mm256_unpackhi_epi16(_sl06_17, _sh06_17));
                    _sum16_07 = _mm256_add_epi32(_sum16_07, _mm256_unpackhi_epi16(_sl16_07, _sh16_07));
    #endif

                    tmpptr += 32;
                    kptr0 += 32;
                }

                // transpose 4x8
                {
                    __m256i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = _mm256_unpacklo_epi32(_sum00_11, _sum10_01);
                    _tmp1 = _mm256_unpacklo_epi32(_sum02_13, _sum12_03);
                    _tmp2 = _mm256_unpackhi_epi32(_sum00_11, _sum10_01);
                    _tmp3 = _mm256_unpackhi_epi32(_sum02_13, _sum12_03);
                    _sum00_11 = _mm256_unpacklo_epi64(_tmp0, _tmp1);
                    _sum10_01 = _mm256_unpackhi_epi64(_tmp0, _tmp1);
                    _sum02_13 = _mm256_unpacklo_epi64(_tmp2, _tmp3);
                    _sum12_03 = _mm256_unpackhi_epi64(_tmp2, _tmp3);
                }
                {
                    __m256i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = _mm256_unpacklo_epi32(_sum04_15, _sum14_05);
                    _tmp1 = _mm256_unpacklo_epi32(_sum06_17, _sum16_07);
                    _tmp2 = _mm256_unpackhi_epi32(_sum04_15, _sum14_05);
                    _tmp3 = _mm256_unpackhi_epi32(_sum06_17, _sum16_07);
                    _sum04_15 = _mm256_unpacklo_epi64(_tmp0, _tmp1);
                    _sum14_05 = _mm256_unpackhi_epi64(_tmp0, _tmp1);
                    _sum06_17 = _mm256_unpacklo_epi64(_tmp2, _tmp3);
                    _sum16_07 = _mm256_unpackhi_epi64(_tmp2, _tmp3);
                }

                _sum00_11 = _mm256_add_epi32(_sum00_11, _sum10_01);
                _sum02_13 = _mm256_add_epi32(_sum02_13, _sum12_03);
                _sum00_11 = _mm256_add_epi32(_sum00_11, _sum02_13);

                _sum04_15 = _mm256_add_epi32(_sum04_15, _sum14_05);
                _sum06_17 = _mm256_add_epi32(_sum06_17, _sum16_07);
                _sum04_15 = _mm256_add_epi32(_sum04_15, _sum06_17);

                __m256i _perm_mask = _mm256_set_epi32(6, 3, 4, 1, 7, 2, 5, 0);
                _sum00_11 = _mm256_permutevar8x32_epi32(_sum00_11, _perm_mask);
                _sum04_15 = _mm256_permutevar8x32_epi32(_sum04_15, _perm_mask);

                _mm256_storeu_si256((__m256i*)outptr0, _sum00_11);
                _mm256_storeu_si256((__m256i*)(outptr0 + 8), _sum04_15);
                outptr0 += 16;
            }
    #endif
            for (; i + 1 < size; i += 2)
            {
    #if __AVX2__
                const signed char* tmpptr = tmp_a[i / 4 + (i % 4) / 2].data();
    #else
                const signed char* tmpptr = tmp_a[i / 2].data();
    #endif
                const signed char* kptr0 = kernel_a[p].data();

                int nn = inch * maxk; // inch always > 0

    #if __AVX2__
                __m256i _sum00_11 = _mm256_setzero_si256();
                __m256i _sum10_01 = _mm256_setzero_si256();
                __m256i _sum02_13 = _mm256_setzero_si256();
                __m256i _sum12_03 = _mm256_setzero_si256();
    #else
                __m128i _sum00 = _mm_setzero_si128();
                __m128i _sum01 = _mm_setzero_si128();
                __m128i _sum02 = _mm_setzero_si128();
                __m128i _sum03 = _mm_setzero_si128();
                __m128i _sum10 = _mm_setzero_si128();
                __m128i _sum11 = _mm_setzero_si128();
                __m128i _sum12 = _mm_setzero_si128();
                __m128i _sum13 = _mm_setzero_si128();
    #endif

                int j = 0;
                for (; j < nn; j++)
                {
    #if __AVX2__
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m256i _val01_16 = _mm256_cvtepi8_epi16(_val01);

                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _w23 = _mm_loadu_si128((const __m128i*)(kptr0 + 16));
                    __m256i _w01_16 = _mm256_cvtepi8_epi16(_w01);
                    __m256i _w23_16 = _mm256_cvtepi8_epi16(_w23);

                    __m256i _val10_16 = _mm256_permute4x64_epi64(_val01_16, 78);

    #if __AVXVNNI__ || __AVX512VNNI__
                    _sum00_11 = _mm256_dpwssd_epi32(_sum00_11, _val01_16, _w01_16);
                    _sum10_01 = _mm256_dpwssd_epi32(_sum10_01, _val10_16, _w01_16);
                    _sum02_13 = _mm256_dpwssd_epi32(_sum02_13, _val01_16, _w23_16);
                    _sum12_03 = _mm256_dpwssd_epi32(_sum12_03, _val10_16, _w23_16);
    #else
                    __m256i _sl00_11 = _mm256_mullo_epi16(_val01_16, _w01_16);
                    __m256i _sh00_11 = _mm256_mulhi_epi16(_val01_16, _w01_16);
                    __m256i _sl10_01 = _mm256_mullo_epi16(_val10_16, _w01_16);
                    __m256i _sh10_01 = _mm256_mulhi_epi16(_val10_16, _w01_16);
                    __m256i _sl02_13 = _mm256_mullo_epi16(_val01_16, _w23_16);
                    __m256i _sh02_13 = _mm256_mulhi_epi16(_val01_16, _w23_16);
                    __m256i _sl12_03 = _mm256_mullo_epi16(_val10_16, _w23_16);
                    __m256i _sh12_03 = _mm256_mulhi_epi16(_val10_16, _w23_16);

                    _sum00_11 = _mm256_add_epi32(_sum00_11, _mm256_unpacklo_epi16(_sl00_11, _sh00_11));
                    _sum10_01 = _mm256_add_epi32(_sum10_01, _mm256_unpacklo_epi16(_sl10_01, _sh10_01));
                    _sum02_13 = _mm256_add_epi32(_sum02_13, _mm256_unpacklo_epi16(_sl02_13, _sh02_13));
                    _sum12_03 = _mm256_add_epi32(_sum12_03, _mm256_unpacklo_epi16(_sl12_03, _sh12_03));
                    _sum00_11 = _mm256_add_epi32(_sum00_11, _mm256_unpackhi_epi16(_sl00_11, _sh00_11));
                    _sum10_01 = _mm256_add_epi32(_sum10_01, _mm256_unpackhi_epi16(_sl10_01, _sh10_01));
                    _sum02_13 = _mm256_add_epi32(_sum02_13, _mm256_unpackhi_epi16(_sl02_13, _sh02_13));
                    _sum12_03 = _mm256_add_epi32(_sum12_03, _mm256_unpackhi_epi16(_sl12_03, _sh12_03));
    #endif
    #else
                    __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                    __m128i _extval01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val01);
                    __m128i _val0 = _mm_unpacklo_epi8(_val01, _extval01);
                    __m128i _val1 = _mm_unpackhi_epi8(_val01, _extval01);

                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _w23 = _mm_loadu_si128((const __m128i*)(kptr0 + 16));
                    __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                    __m128i _extw23 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w23);
                    __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                    __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);
                    __m128i _w2 = _mm_unpacklo_epi8(_w23, _extw23);
                    __m128i _w3 = _mm_unpackhi_epi8(_w23, _extw23);

    #if __XOP__
                    _sum00 = _mm_maddd_epi16(_val0, _w0, _sum00);
                    _sum01 = _mm_maddd_epi16(_val0, _w1, _sum01);
                    _sum02 = _mm_maddd_epi16(_val0, _w2, _sum02);
                    _sum03 = _mm_maddd_epi16(_val0, _w3, _sum03);
                    _sum10 = _mm_maddd_epi16(_val1, _w0, _sum10);
                    _sum11 = _mm_maddd_epi16(_val1, _w1, _sum11);
                    _sum12 = _mm_maddd_epi16(_val1, _w2, _sum12);
                    _sum13 = _mm_maddd_epi16(_val1, _w3, _sum13);
    #else
                    __m128i _sl00 = _mm_mullo_epi16(_val0, _w0);
                    __m128i _sh00 = _mm_mulhi_epi16(_val0, _w0);
                    __m128i _sl01 = _mm_mullo_epi16(_val0, _w1);
                    __m128i _sh01 = _mm_mulhi_epi16(_val0, _w1);
                    __m128i _sl02 = _mm_mullo_epi16(_val0, _w2);
                    __m128i _sh02 = _mm_mulhi_epi16(_val0, _w2);
                    __m128i _sl03 = _mm_mullo_epi16(_val0, _w3);
                    __m128i _sh03 = _mm_mulhi_epi16(_val0, _w3);
                    __m128i _sl10 = _mm_mullo_epi16(_val1, _w0);
                    __m128i _sh10 = _mm_mulhi_epi16(_val1, _w0);
                    __m128i _sl11 = _mm_mullo_epi16(_val1, _w1);
                    __m128i _sh11 = _mm_mulhi_epi16(_val1, _w1);
                    __m128i _sl12 = _mm_mullo_epi16(_val1, _w2);
                    __m128i _sh12 = _mm_mulhi_epi16(_val1, _w2);
                    __m128i _sl13 = _mm_mullo_epi16(_val1, _w3);
                    __m128i _sh13 = _mm_mulhi_epi16(_val1, _w3);

                    _sum00 = _mm_add_epi32(_sum00, _mm_unpacklo_epi16(_sl00, _sh00));
                    _sum01 = _mm_add_epi32(_sum01, _mm_unpacklo_epi16(_sl01, _sh01));
                    _sum02 = _mm_add_epi32(_sum02, _mm_unpacklo_epi16(_sl02, _sh02));
                    _sum03 = _mm_add_epi32(_sum03, _mm_unpacklo_epi16(_sl03, _sh03));
                    _sum00 = _mm_add_epi32(_sum00, _mm_unpackhi_epi16(_sl00, _sh00));
                    _sum01 = _mm_add_epi32(_sum01, _mm_unpackhi_epi16(_sl01, _sh01));
                    _sum02 = _mm_add_epi32(_sum02, _mm_unpackhi_epi16(_sl02, _sh02));
                    _sum03 = _mm_add_epi32(_sum03, _mm_unpackhi_epi16(_sl03, _sh03));
                    _sum10 = _mm_add_epi32(_sum10, _mm_unpacklo_epi16(_sl10, _sh10));
                    _sum11 = _mm_add_epi32(_sum11, _mm_unpacklo_epi16(_sl11, _sh11));
                    _sum12 = _mm_add_epi32(_sum12, _mm_unpacklo_epi16(_sl12, _sh12));
                    _sum13 = _mm_add_epi32(_sum13, _mm_unpacklo_epi16(_sl13, _sh13));
                    _sum10 = _mm_add_epi32(_sum10, _mm_unpackhi_epi16(_sl10, _sh10));
                    _sum11 = _mm_add_epi32(_sum11, _mm_unpackhi_epi16(_sl11, _sh11));
                    _sum12 = _mm_add_epi32(_sum12, _mm_unpackhi_epi16(_sl12, _sh12));
                    _sum13 = _mm_add_epi32(_sum13, _mm_unpackhi_epi16(_sl13, _sh13));
    #endif
    #endif

                    tmpptr += 16;
                    kptr0 += 32;
                }

    #if __AVX2__
                // transpose 4x8
                {
                    __m256i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = _mm256_unpacklo_epi32(_sum00_11, _sum10_01);
                    _tmp1 = _mm256_unpacklo_epi32(_sum02_13, _sum12_03);
                    _tmp2 = _mm256_unpackhi_epi32(_sum00_11, _sum10_01);
                    _tmp3 = _mm256_unpackhi_epi32(_sum02_13, _sum12_03);
                    _sum00_11 = _mm256_unpacklo_epi64(_tmp0, _tmp1);
                    _sum10_01 = _mm256_unpackhi_epi64(_tmp0, _tmp1);
                    _sum02_13 = _mm256_unpacklo_epi64(_tmp2, _tmp3);
                    _sum12_03 = _mm256_unpackhi_epi64(_tmp2, _tmp3);
                }

                _sum00_11 = _mm256_add_epi32(_sum00_11, _sum10_01);
                _sum02_13 = _mm256_add_epi32(_sum02_13, _sum12_03);
                _sum00_11 = _mm256_add_epi32(_sum00_11, _sum02_13);

                __m256i _perm_mask = _mm256_set_epi32(6, 3, 4, 1, 7, 2, 5, 0);
                _sum00_11 = _mm256_permutevar8x32_epi32(_sum00_11, _perm_mask);

                _mm256_storeu_si256((__m256i*)outptr0, _sum00_11);
    #else
                // transpose 4x4
                {
                    __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = _mm_unpacklo_epi32(_sum00, _sum01);
                    _tmp1 = _mm_unpacklo_epi32(_sum02, _sum03);
                    _tmp2 = _mm_unpackhi_epi32(_sum00, _sum01);
                    _tmp3 = _mm_unpackhi_epi32(_sum02, _sum03);
                    _sum00 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                    _sum01 = _mm_unpackhi_epi64(_tmp0, _tmp1);
                    _sum02 = _mm_unpacklo_epi64(_tmp2, _tmp3);
                    _sum03 = _mm_unpackhi_epi64(_tmp2, _tmp3);
                }
                {
                    __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = _mm_unpacklo_epi32(_sum10, _sum11);
                    _tmp1 = _mm_unpacklo_epi32(_sum12, _sum13);
                    _tmp2 = _mm_unpackhi_epi32(_sum10, _sum11);
                    _tmp3 = _mm_unpackhi_epi32(_sum12, _sum13);
                    _sum10 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                    _sum11 = _mm_unpackhi_epi64(_tmp0, _tmp1);
                    _sum12 = _mm_unpacklo_epi64(_tmp2, _tmp3);
                    _sum13 = _mm_unpackhi_epi64(_tmp2, _tmp3);
                }

                _sum00 = _mm_add_epi32(_sum00, _sum01);
                _sum02 = _mm_add_epi32(_sum02, _sum03);
                _sum10 = _mm_add_epi32(_sum10, _sum11);
                _sum12 = _mm_add_epi32(_sum12, _sum13);

                _sum00 = _mm_add_epi32(_sum00, _sum02);
                _sum10 = _mm_add_epi32(_sum10, _sum12);

                _mm_storeu_si128((__m128i*)outptr0, _sum00);
                _mm_storeu_si128((__m128i*)(outptr0 + 4), _sum10);
    #endif
                outptr0 += 8;
            }
            for (; i < size; i++)
            {
    #if __AVX2__
                const signed char* tmpptr = tmp_a[i / 4 + (i % 4) / 2 + i % 2].data();
    #else
                const signed char* tmpptr = tmp_a[i / 2 + i % 2].data();
    #endif
                const signed char* kptr0 = kernel_a[p].data();

                int nn = inch * maxk; // inch always > 0

    #if __AVX2__
                __m256i _sum0_1 = _mm256_setzero_si256();
                __m256i _sum2_3 = _mm256_setzero_si256();
    #else
                __m128i _sum0 = _mm_setzero_si128();
                __m128i _sum1 = _mm_setzero_si128();
                __m128i _sum2 = _mm_setzero_si128();
                __m128i _sum3 = _mm_setzero_si128();
    #endif

                int j = 0;
                for (; j < nn; j++)
                {
    #if __AVX2__
                    __m128i _val = _mm_loadl_epi64((const __m128i*)tmpptr);
                    _val = _mm_cvtepi8_epi16(_val);

                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _w23 = _mm_loadu_si128((const __m128i*)(kptr0 + 16));
                    __m256i _w01_16 = _mm256_cvtepi8_epi16(_w01);
                    __m256i _w23_16 = _mm256_cvtepi8_epi16(_w23);

                    __m256i _valval = _mm256_inserti128_si256(_mm256_castsi128_si256(_val), _val, 1);

    #if __AVXVNNI__ || __AVX512VNNI__
                    _sum0_1 = _mm256_dpwssd_epi32(_sum0_1, _valval, _w01_16);
                    _sum2_3 = _mm256_dpwssd_epi32(_sum2_3, _valval, _w23_16);
    #else
                    __m256i _sl0_1 = _mm256_mullo_epi16(_valval, _w01_16);
                    __m256i _sh0_1 = _mm256_mulhi_epi16(_valval, _w01_16);
                    __m256i _sl2_3 = _mm256_mullo_epi16(_valval, _w23_16);
                    __m256i _sh2_3 = _mm256_mulhi_epi16(_valval, _w23_16);

                    _sum0_1 = _mm256_add_epi32(_sum0_1, _mm256_unpacklo_epi16(_sl0_1, _sh0_1));
                    _sum2_3 = _mm256_add_epi32(_sum2_3, _mm256_unpacklo_epi16(_sl2_3, _sh2_3));
                    _sum0_1 = _mm256_add_epi32(_sum0_1, _mm256_unpackhi_epi16(_sl0_1, _sh0_1));
                    _sum2_3 = _mm256_add_epi32(_sum2_3, _mm256_unpackhi_epi16(_sl2_3, _sh2_3));
    #endif
    #else
                    __m128i _val = _mm_loadl_epi64((const __m128i*)tmpptr);
    #if __SSE4_1__
                    _val = _mm_cvtepi8_epi16(_val);
    #else
                    _val = _mm_unpacklo_epi8(_val, _mm_cmpgt_epi8(_mm_setzero_si128(), _val));
    #endif

                    __m128i _w01 = _mm_loadu_si128((const __m128i*)kptr0);
                    __m128i _w23 = _mm_loadu_si128((const __m128i*)(kptr0 + 16));
                    __m128i _extw01 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w01);
                    __m128i _extw23 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w23);
                    __m128i _w0 = _mm_unpacklo_epi8(_w01, _extw01);
                    __m128i _w1 = _mm_unpackhi_epi8(_w01, _extw01);
                    __m128i _w2 = _mm_unpacklo_epi8(_w23, _extw23);
                    __m128i _w3 = _mm_unpackhi_epi8(_w23, _extw23);

    #if __XOP__
                    _sum0 = _mm_maddd_epi16(_val, _w0, _sum0);
                    _sum1 = _mm_maddd_epi16(_val, _w1, _sum1);
                    _sum2 = _mm_maddd_epi16(_val, _w2, _sum2);
                    _sum3 = _mm_maddd_epi16(_val, _w3, _sum3);
    #else
                    __m128i _sl0 = _mm_mullo_epi16(_val, _w0);
                    __m128i _sh0 = _mm_mulhi_epi16(_val, _w0);
                    __m128i _sl1 = _mm_mullo_epi16(_val, _w1);
                    __m128i _sh1 = _mm_mulhi_epi16(_val, _w1);
                    __m128i _sl2 = _mm_mullo_epi16(_val, _w2);
                    __m128i _sh2 = _mm_mulhi_epi16(_val, _w2);
                    __m128i _sl3 = _mm_mullo_epi16(_val, _w3);
                    __m128i _sh3 = _mm_mulhi_epi16(_val, _w3);

                    _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl0, _sh0));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpacklo_epi16(_sl1, _sh1));
                    _sum2 = _mm_add_epi32(_sum2, _mm_unpacklo_epi16(_sl2, _sh2));
                    _sum3 = _mm_add_epi32(_sum3, _mm_unpacklo_epi16(_sl3, _sh3));
                    _sum0 = _mm_add_epi32(_sum0, _mm_unpackhi_epi16(_sl0, _sh0));
                    _sum1 = _mm_add_epi32(_sum1, _mm_unpackhi_epi16(_sl1, _sh1));
                    _sum2 = _mm_add_epi32(_sum2, _mm_unpackhi_epi16(_sl2, _sh2));
                    _sum3 = _mm_add_epi32(_sum3, _mm_unpackhi_epi16(_sl3, _sh3));
    #endif
    #endif

                    tmpptr += 8;
                    kptr0 += 32;
                }

    #if __AVX2__
                __m128i _sum0 = _mm256_extracti128_si256(_sum0_1, 0);
                __m128i _sum1 = _mm256_extracti128_si256(_sum0_1, 1);
                __m128i _sum2 = _mm256_extracti128_si256(_sum2_3, 0);
                __m128i _sum3 = _mm256_extracti128_si256(_sum2_3, 1);
    #endif

                // transpose 4x4
                {
                    __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = _mm_unpacklo_epi32(_sum0, _sum1);
                    _tmp1 = _mm_unpacklo_epi32(_sum2, _sum3);
                    _tmp2 = _mm_unpackhi_epi32(_sum0, _sum1);
                    _tmp3 = _mm_unpackhi_epi32(_sum2, _sum3);
                    _sum0 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                    _sum1 = _mm_unpackhi_epi64(_tmp0, _tmp1);
                    _sum2 = _mm_unpacklo_epi64(_tmp2, _tmp3);
                    _sum3 = _mm_unpackhi_epi64(_tmp2, _tmp3);
                }

                _sum0 = _mm_add_epi32(_sum0, _sum1);
                _sum2 = _mm_add_epi32(_sum2, _sum3);

                _sum0 = _mm_add_epi32(_sum0, _sum2);

                _mm_storeu_si128((__m128i*)outptr0, _sum0);
                outptr0 += 4;
            }
        }
    });
}

Tensor& sgemm_conv2d_int8_pack1to4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
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
        convolution_im2col_sgemm_transform_kernel_pack1to4_int8_x86(weight, kernel_tf, inch, outch * 4, kernel_w, kernel_h);
    
    Tensor im2col = otter::im2col_cpu(self, kernel_size, stride, padding, dilation).view({inch, maxk, size});
    
    im2col_sgemm_conv2d_int8_pack1to4_impl_x86(im2col, kernel_tf, output);
    
    return output;
}
    
Tensor sgemm_conv2d_int8_pack1to4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, otter::ScalarType::Int4);
    
    return sgemm_conv2d_int8_pack1to4_x86_out(self, weight, weight_o, kernel_size, stride, padding, dilation, output);
}

Tensor& sgemm_conv2d_int8_pack8to1_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding, dilation);
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
        convolution_im2col_sgemm_transform_kernel_pack8to1_int8_x86(weight, kernel_tf, inch * 8, outch, kernel_w, kernel_h);
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    
    Tensor im2col = otter::empty({inch, maxk, size}, otter::ScalarType::Byte8);
    
    auto im2col_ra = im2col.raw_accessor<int64_t, 3>();
    auto input_ra = input.raw_accessor<int64_t, 3>();
    
    {
        const int gap = w * stride_h - outw * stride_w;

        otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
            for (const auto p : otter::irange(begin, end)) {
                const auto img = input_ra[p];
                int64_t* ptr = im2col_ra[p].data();

                for (int u = 0; u < kernel_h; u++) {
                    for (int v = 0; v < kernel_w; v++) {
                        const int64_t* sptr = img[dilation_h * u].data() + dilation_w * v;

                        for (int i = 0; i < outh; i++) {
                            int j = 0;
                            for (; j < outw; j++) {
                                ptr[0] = sptr[0];

                                sptr += stride_w;
                                ptr += 1;
                            }

                            sptr += gap;
                        }
                    }
                }
            }
        });
    }
    
    im2col_sgemm_conv2d_int8_pack8to1_impl_x86(im2col, kernel_tf, output);
    
    return output;
}
    
Tensor sgemm_conv2d_int8_pack8to1_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, otter::ScalarType::Int);
    
    return sgemm_conv2d_int8_pack8to1_x86_out(self, weight, weight_o, kernel_size, stride, padding, dilation, output);
}

Tensor& sgemm_conv2d_int8_pack8to4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
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
        convolution_im2col_sgemm_transform_kernel_pack8to4_int8_x86(weight, kernel_tf, inch * 8, outch * 4, kernel_w, kernel_h);
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    
    Tensor im2col = otter::empty({inch, maxk, size}, otter::ScalarType::Byte8);
    
    auto im2col_ra = im2col.raw_accessor<int64_t, 3>();
    auto input_ra = input.raw_accessor<int64_t, 3>();
    
    {
        const int gap = w * stride_h - outw * stride_w;

        otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
            for (const auto p : otter::irange(begin, end)) {
                const auto img = input_ra[p];
                int64_t* ptr = im2col_ra[p].data();

                for (int u = 0; u < kernel_h; u++) {
                    for (int v = 0; v < kernel_w; v++) {
                        const int64_t* sptr = img[dilation_h * u].data() + dilation_w * v;

                        for (int i = 0; i < outh; i++) {
                            int j = 0;
                            for (; j < outw; j++) {
                                ptr[0] = sptr[0];

                                sptr += stride_w;
                                ptr += 1;
                            }

                            sptr += gap;
                        }
                    }
                }
            }
        });
    }
    
    im2col_sgemm_conv2d_int8_pack8to4_impl_x86(im2col, kernel_tf, output);
    
    return output;
}
    
Tensor sgemm_conv2d_int8_pack8to4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, otter::ScalarType::Int4);
    
    return sgemm_conv2d_int8_pack8to4_x86_out(self, weight, weight_o, kernel_size, stride, padding, dilation, output);
}

Tensor& sgemm_conv2d_1x1s1_int8_pack1to4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
    output.resize_({output_size[0], output_size[1] / 4, output_size[2], output_size[3]});
    
    const int64_t inch  = self.size(1);
    const int64_t outch = output.size(1);
    
    Tensor im2col = self.view({self.size(0), self.size(1), -1});
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        convolution_im2col_sgemm_transform_kernel_pack8to4_int8_x86(weight, kernel_tf, inch, outch * 4, 1, 1);
    
    im2col_sgemm_conv2d_int8_pack1to4_impl_x86(im2col, kernel_tf, output);
    
    return output;
}
    
Tensor sgemm_conv2d_1x1s1_int8_pack1to4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, otter::ScalarType::Int4);
    
    return sgemm_conv2d_1x1s1_int8_pack1to4_x86_out(self, weight, weight_o, padding, output);
}

Tensor& sgemm_conv2d_1x1s1_int8_pack8to1_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
    output.resize_(output_size);
    
    const int64_t inch  = self.size(1);
    const int64_t outch = output.size(1);
    
    Tensor im2col = self.view({self.size(0), self.size(1), -1});
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        convolution_im2col_sgemm_transform_kernel_pack8to4_int8_x86(weight, kernel_tf, inch * 8, outch, 1, 1);
    
    im2col_sgemm_conv2d_int8_pack8to1_impl_x86(im2col, kernel_tf, output);
    
    return output;
}
    
Tensor sgemm_conv2d_1x1s1_int8_pack8to1_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, otter::ScalarType::Int);
    
    return sgemm_conv2d_1x1s1_int8_pack8to1_x86_out(self, weight, weight_o, padding, output);
}

Tensor& sgemm_conv2d_1x1s1_int8_pack8to4_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
    output.resize_({output_size[0], output_size[1] / 4, output_size[2], output_size[3]});
    
    const int64_t inch  = self.size(1);
    const int64_t outch = output.size(1);
    
    Tensor im2col = self.view({self.size(0), self.size(1), -1});
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        convolution_im2col_sgemm_transform_kernel_pack8to4_int8_x86(weight, kernel_tf, inch * 8, outch * 4, 1, 1);
    
    im2col_sgemm_conv2d_int8_pack8to4_impl_x86(im2col, kernel_tf, output);
    
    return output;
}
    
Tensor sgemm_conv2d_1x1s1_int8_pack8to4_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, otter::ScalarType::Int4);
    
    return sgemm_conv2d_1x1s1_int8_pack8to4_x86_out(self, weight, weight_o, padding, output);
}


#endif  // __SSE2__

}   // end namespace otter
