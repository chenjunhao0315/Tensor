//
//  ConvolutionMM2DInt8X86.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/5.
//

// https://github.com/Tencent/ncnn/blob/master/src/layer/x86/convolution_sgemm_int8.h

#include "ConvolutionMM2DInt8X86.hpp"

#include "Tensor.hpp"
#include "TensorFactory.hpp"
#include "Padding.hpp"
#include "im2col.hpp"
#include "Parallel.hpp"
#include "VecIntrinsic.hpp"
#include "Quantize.hpp"

namespace otter {

#ifdef __SSE2__
void convolution_im2col_sgemm_transform_kernel_int8_sse(const Tensor& kernel_, Tensor& kernel_tf, int64_t input_channels, int64_t output_channels, int64_t kernel_width, int64_t kernel_height) {
    const int maxk = kernel_width * kernel_height;

#if __SSE2__
    // interleave
    // src = maxk-inch-outch
    // dst = 4a-4b-maxk-inch/4a-outch/4b
    Tensor kernel = kernel_.view({output_channels, input_channels, maxk});
    if (output_channels >= 4) {
        if (input_channels >= 4)
            kernel_tf = otter::empty({output_channels / 4 + output_channels % 4, input_channels / 4 + input_channels % 4, 16 * maxk}, otter::ScalarType::Byte);
        else
            kernel_tf = otter::empty({output_channels / 4 + output_channels % 4, input_channels, 4 * maxk}, otter::ScalarType::Byte);
    } else {
        if (input_channels >= 4)
            kernel_tf = otter::empty({output_channels, input_channels / 4 + input_channels % 4, 4 * maxk}, otter::ScalarType::Byte);
        else
            kernel_tf = otter::empty({output_channels, input_channels, 1 * maxk}, otter::ScalarType::Byte);
    }

    auto kernel_a = kernel.accessor<unsigned char, 3>();
    auto kernel_tf_a = kernel_tf.accessor<unsigned char, 3>();
    
    int q = 0;
    for (; q + 3 < output_channels; q += 4)
    {
        signed char* g00 = (signed char*)kernel_tf_a[q / 4].data();

        int p = 0;
        for (; p + 3 < input_channels; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        const signed char* k00 = (const signed char*)kernel_a[q + i][p + j].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
        for (; p < input_channels; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int i = 0; i < 4; i++)
                {
                    const signed char* k00 = (const signed char*)kernel_a[q + i][p].data();

                    g00[0] = k00[k];

                    g00++;
                }
            }
        }
    }
    // TODO unroll 2
    for (; q < output_channels; q++)
    {
        signed char* g00 = (signed char*)kernel_tf_a[q / 4 + q % 4].data();

        int p = 0;
        for (; p + 3 < input_channels; p += 4)
        {
            for (int k = 0; k < maxk; k++)
            {
                for (int j = 0; j < 4; j++)
                {
                    const signed char* k00 = (const signed char*)kernel_a[q][p + j].data();

                    g00[0] = k00[k];

                    g00++;
                }
            }
        }
        for (; p < input_channels; p++)
        {
            for (int k = 0; k < maxk; k++)
            {
                const signed char* k00 = (const signed char*)kernel_a[q][p].data();

                g00[0] = k00[k];

                g00++;
            }
        }
    }
#else  // __SSE2__
    kernel_tf = kernel_.view({output_channels, input_channels, maxk});
#endif // __SSE2__
}
#else
void convolution_im2col_sgemm_transform_kernel_int8_sse(const Tensor& /*_kernel*/, Tensor& /*kernel_tf*/, int64_t /*input_channels*/, int64_t /*out_chnnels*/, int64_t /*kernel_width*/, int64_t /*kernel_height*/) {}
#endif

#if __SSE2__
void im2col_sgemm_conv2d_int8_impl_x86(
    const Tensor& im2col_,
    const Tensor& kernel_tf,
    int64_t input_channels,
    int64_t output_channels,
    Tensor& output) {
    
    Tensor im2col = im2col_.view({input_channels, -1, im2col_.size(2)});
    
    const int64_t size = im2col.size(2); // im2col width = output_height * output_width
    const int64_t maxk = im2col.size(1); // im2col height = kernel_height * kernel_width
    const int64_t inch = input_channels;
    
    const int64_t outch = output_channels;
    
    Tensor tmp;
#if __SSE2__
    if (inch >= 4)
    {
#if __AVX2__
        if (size >= 4)
            tmp = otter::empty({size / 4 + (size % 4) / 2 + size % 2, inch / 4 + inch % 4, 4 * maxk}, otter::ScalarType::Int);
        else if (size >= 2)
            tmp = otter::empty({size / 2 + size % 2, inch / 4 + inch % 4, 2 * maxk}, otter::ScalarType::Int);
        else
            tmp = otter::empty({size, inch / 4 + inch % 4, maxk}, otter::ScalarType::Int);
#else
        if (size >= 2)
            tmp = otter::empty({size / 2 + size % 2, inch / 4 + inch % 4, 2 * maxk}, otter::ScalarType::Int);
        else
            tmp = otter::empty({size, inch / 4 + inch % 4, maxk}, otter::ScalarType::Int);
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
    auto output_a = output.accessor<int, 4>()[0];
    auto im2col_a = im2col.accessor<unsigned char, 3>();
    auto kernel_tf_a = kernel_tf.accessor<unsigned char, 3>();
    
    {
#if __AVX2__
        int remain_size_start = 0;
        int nn_size = size >> 2;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 4;

                signed char* tmpptr = (signed char*)tmp_ra[i / 4].data();

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
            for (const auto ii : otter::irange(begin, end))
            {
                int i = remain_size_start + ii * 2;

    #if __AVX2__
                signed char* tmpptr = (signed char*)tmp_ra[i / 4 + (i % 4) / 2].data();
    #else
                signed char* tmpptr = (signed char*)tmp_ra[i / 2].data();
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
                signed char* tmpptr = (signed char*)tmp_ra[i / 4 + (i % 4) / 2 + i % 2].data();
    #else
                signed char* tmpptr = (signed char*)tmp_ra[i / 2 + i % 2].data();
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
#else // __SSE2__
    tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Byte);
    auto tmp_ra = tmp.raw_accessor<signed char, 3>();
    {
        otter::parallel_for(0, size, 0, [&](int64_t begin, int64_t end) {
            for (const auto i : otter::irange(begin, end)) {
                signed char* tmpptr = (signed char*)tmp_ra[i].data();

                int q = 0;
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
#endif // __SSE2__

    int nn_outch = 0;
    int remain_outch_start = 0;

#if __SSE2__
    nn_outch = outch >> 2;

    otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto pp : otter::irange(begin, end))
        {
            int p = pp * 4;

            int* outptr0 = output_a[p + 0].data();
            int* outptr1 = output_a[p + 1].data();
            int* outptr2 = output_a[p + 2].data();
            int* outptr3 = output_a[p + 3].data();

            int i = 0;
    #if __AVX2__
            for (; i + 3 < size; i += 4)
            {
                const signed char* tmpptr = (const signed char*)tmp_ra[i / 4].data();
                const signed char* kptr0 = (const signed char*)kernel_tf_a[p / 4].data();

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
                    __m128i _val0123 = _mm_loadl_epi64((const __m128i*)tmpptr);
    #if __SSE4_1__
                    _val0123 = _mm_cvtepi8_epi16(_val0123);
    #else
                    __m128i _extval0123 = _mm_cmpgt_epi8(_mm_setzero_si128(), _val0123);
                    _val0123 = _mm_unpacklo_epi8(_val0123, _extval0123);
    #endif

                    __m128i _val01 = _mm_shufflelo_epi16(_val0123, _MM_SHUFFLE(1, 1, 0, 0));

                    _val01 = _mm_shuffle_epi32(_val01, _MM_SHUFFLE(1, 1, 0, 0));

                    __m128i _val23 = _mm_shufflelo_epi16(_val0123, _MM_SHUFFLE(3, 3, 2, 2));

                    _val23 = _mm_shuffle_epi32(_val23, _MM_SHUFFLE(1, 1, 0, 0));

                    __m128i _w0123 = _mm_loadl_epi64((const __m128i*)kptr0);
    #if __SSE4_1__
                    _w0123 = _mm_cvtepi8_epi16(_w0123);
    #else
                    __m128i _extw0123 = _mm_cmpgt_epi8(_mm_setzero_si128(), _w0123);
                    _w0123 = _mm_unpacklo_epi8(_w0123, _extw0123);
    #endif

                    _w0123 = _mm_shuffle_epi32(_w0123, _MM_SHUFFLE(1, 0, 1, 0));

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

                // transpose 4x4
                {
                    __m128i _tmp0, _tmp1, _tmp2, _tmp3;
                    _tmp0 = _mm_unpacklo_epi32(_sum00, _sum10);
                    _tmp1 = _mm_unpacklo_epi32(_sum20, _sum30);
                    _tmp2 = _mm_unpackhi_epi32(_sum00, _sum10);
                    _tmp3 = _mm_unpackhi_epi32(_sum20, _sum30);
                    _sum00 = _mm_unpacklo_epi64(_tmp0, _tmp1);
                    _sum10 = _mm_unpackhi_epi64(_tmp0, _tmp1);
                    _sum20 = _mm_unpacklo_epi64(_tmp2, _tmp3);
                    _sum30 = _mm_unpackhi_epi64(_tmp2, _tmp3);
                }

                _mm_storeu_si128((__m128i*)outptr0, _sum00);
                _mm_storeu_si128((__m128i*)outptr1, _sum10);
                _mm_storeu_si128((__m128i*)outptr2, _sum20);
                _mm_storeu_si128((__m128i*)outptr3, _sum30);
                outptr0 += 4;
                outptr1 += 4;
                outptr2 += 4;
                outptr3 += 4;
            }
    #endif
            for (; i + 1 < size; i += 2)
            {
    #if __AVX2__
                const signed char* tmpptr = (const signed char*)tmp_ra[i / 4 + (i % 4) / 2].data();
    #else
                const signed char* tmpptr = (const signed char*)tmp_ra[i / 2].data();
    #endif
                const signed char* kptr0 = (const signed char*)kernel_tf_a[p / 4].data();

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

                    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=99754
                    // gcc incorrectly put 32bit to tail with _mm_loadu_si32  :(
                    // 0 1 2 3 x x x x x x x x x x x x
                    // x x x x x x x x x x x x 0 1 2 3
                    // __m128i _w0123 = _mm_loadu_si32(kptr0);
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

                int sum[8];
                _mm_storeu_si128((__m128i*)sum, _sum00);
                _mm_storeu_si128((__m128i*)(sum + 4), _sum10);

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
                const signed char* tmpptr = (const signed char*)tmp_ra[i / 4 + (i % 4) / 2 + i % 2].data();
    #else
                const signed char* tmpptr = (const signed char*)tmp_ra[i / 2 + i % 2].data();
    #endif
                const signed char* kptr0 = (const signed char*)kernel_tf_a[p / 4].data();

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
#endif // __SSE2__

    otter::parallel_for(remain_outch_start, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end))
        {
            int* outptr0 = output_a[p].data();

            int i = 0;
    #if __SSE2__
    #if __AVX2__
            for (; i + 3 < size; i += 4)
            {
                const signed char* tmpptr = (const signed char*)tmp_ra[i / 4].data();
                const signed char* kptr0 = (const signed char*)kernel_tf_a[p / 4 + p % 4].data();

                int nn4 = (inch / 4) * maxk;
                int nn1 = (inch % 4) * maxk;

                int sum0 = 0;
                int sum1 = 0;
                int sum2 = 0;
                int sum3 = 0;

                if (nn4 > 0)
                {
                    __m256i _sum0_2 = _mm256_setzero_si256();
                    __m256i _sum1_3 = _mm256_setzero_si256();

                    int j = 0;
                    for (; j < nn4; j++)
                    {
                        __m128i _val01 = _mm_loadu_si128((const __m128i*)tmpptr);
                        __m256i _val01_16 = _mm256_cvtepi8_epi16(_val01);

                        __m128i _w0123 = _mm_loadl_epi64((const __m128i*)kptr0);
                        __m128i _w = _mm_cvtepi8_epi16(_w0123);
                        _w = _mm_unpacklo_epi64(_w, _w);
                        __m256i _ww = _mm256_inserti128_si256(_mm256_castsi128_si256(_w), _w, 1);

                        __m256i _sl0_1 = _mm256_mullo_epi16(_val01_16, _ww);
                        __m256i _sh0_1 = _mm256_mulhi_epi16(_val01_16, _ww);

                        _sum0_2 = _mm256_add_epi32(_sum0_2, _mm256_unpacklo_epi16(_sl0_1, _sh0_1));
                        _sum1_3 = _mm256_add_epi32(_sum1_3, _mm256_unpackhi_epi16(_sl0_1, _sh0_1));

                        tmpptr += 16;
                        kptr0 += 4;
                    }

                    __m128i _sum0 = _mm256_extracti128_si256(_sum0_2, 0);
                    __m128i _sum1 = _mm256_extracti128_si256(_sum1_3, 0);
                    __m128i _sum2 = _mm256_extracti128_si256(_sum0_2, 1);
                    __m128i _sum3 = _mm256_extracti128_si256(_sum1_3, 1);

                    sum0 = _mm_reduce_add_epi32(_sum0);
                    sum1 = _mm_reduce_add_epi32(_sum1);
                    sum2 = _mm_reduce_add_epi32(_sum2);
                    sum3 = _mm_reduce_add_epi32(_sum3);
                }

                int j = 0;
                for (; j < nn1; j++)
                {
                    signed char val0 = tmpptr[0];
                    signed char val1 = tmpptr[1];
                    signed char val2 = tmpptr[2];
                    signed char val3 = tmpptr[3];
                    signed char w = kptr0[0];

                    sum0 += val0 * w;
                    sum1 += val1 * w;
                    sum2 += val2 * w;
                    sum3 += val3 * w;

                    tmpptr += 4;
                    kptr0 += 1;
                }

                outptr0[0] = sum0;
                outptr0[1] = sum1;
                outptr0[2] = sum2;
                outptr0[3] = sum3;
                outptr0 += 4;
            }
    #endif
            for (; i + 1 < size; i += 2)
            {
    #if __AVX2__
                const signed char* tmpptr = (const signed char*)tmp_ra[i / 4 + (i % 4) / 2].data();
    #else
                const signed char* tmpptr = (const signed char*)tmp_ra[i / 2].data();
    #endif
                const signed char* kptr0 = (const signed char*)kernel_tf_a[p / 4 + p % 4].data();

                int nn4 = (inch / 4) * maxk;
                int nn1 = (inch % 4) * maxk;

                int sum0 = 0;
                int sum1 = 0;

                if (nn4 > 0)
                {
                    __m128i _sum0 = _mm_setzero_si128();
                    __m128i _sum1 = _mm_setzero_si128();

                    int j = 0;
                    for (; j < nn4; j++)
                    {
                        __m128i _val = _mm_loadl_epi64((const __m128i*)tmpptr);
                        __m128i _extval = _mm_cmpgt_epi8(_mm_setzero_si128(), _val);
                        __m128i _val01 = _mm_unpacklo_epi8(_val, _extval);

                        __m128i _w0123 = _mm_loadl_epi64((const __m128i*)kptr0);
    #if __SSE4_1__
                        __m128i _w = _mm_cvtepi8_epi16(_w0123);
    #else
                        __m128i _extw = _mm_cmpgt_epi8(_mm_setzero_si128(), _w0123);
                        __m128i _w = _mm_unpacklo_epi8(_w0123, _extw);
    #endif
                        _w = _mm_shuffle_epi32(_w, _MM_SHUFFLE(1, 0, 1, 0));

                        __m128i _sl01 = _mm_mullo_epi16(_val01, _w);
                        __m128i _sh01 = _mm_mulhi_epi16(_val01, _w);

                        _sum0 = _mm_add_epi32(_sum0, _mm_unpacklo_epi16(_sl01, _sh01));
                        _sum1 = _mm_add_epi32(_sum1, _mm_unpackhi_epi16(_sl01, _sh01));

                        tmpptr += 8;
                        kptr0 += 4;
                    }

                    sum0 = _mm_reduce_add_epi32(_sum0);
                    sum1 = _mm_reduce_add_epi32(_sum1);
                }

                int j = 0;
                for (; j < nn1; j++)
                {
                    signed char val0 = tmpptr[0];
                    signed char val1 = tmpptr[1];
                    signed char w = kptr0[0];

                    sum0 += val0 * w;
                    sum1 += val1 * w;

                    tmpptr += 2;
                    kptr0 += 1;
                }

                outptr0[0] = sum0;
                outptr0[1] = sum1;
                outptr0 += 2;
            }
            for (; i < size; i++)
            {
    #if __AVX2__
                const signed char* tmpptr = (const signed char*)tmp_ra[i / 4 + (i % 4) / 2 + i % 2].data();
    #else
                const signed char* tmpptr = (const signed char*)tmp_ra[i / 2 + i % 2].data();
    #endif
                const signed char* kptr0 = (const signed char*)kernel_tf_a[p / 4 + p % 4].data();

                int nn4 = (inch / 4) * maxk;
                int nn1 = (inch % 4) * maxk;

                int sum = 0;

                if (nn4 > 0)
                {
                    __m128i _sum = _mm_setzero_si128();

                    int j = 0;
                    for (; j < nn4; j++)
                    {
                        __m128i _val0123 = _mm_loadl_epi64((const __m128i*)tmpptr);
    #if __SSE4_1__
                        __m128i _val = _mm_cvtepi8_epi16(_val0123);
    #else
                        __m128i _extval = _mm_cmpgt_epi8(_mm_setzero_si128(), _val0123);
                        __m128i _val = _mm_unpacklo_epi8(_val0123, _extval);
    #endif

                        __m128i _w0123 = _mm_loadl_epi64((const __m128i*)kptr0);
    #if __SSE4_1__
                        __m128i _w = _mm_cvtepi8_epi16(_w0123);
    #else
                        __m128i _extw = _mm_cmpgt_epi8(_mm_setzero_si128(), _w0123);
                        __m128i _w = _mm_unpacklo_epi8(_w0123, _extw);
    #endif

                        __m128i _sl = _mm_mullo_epi16(_val, _w);
                        __m128i _sh = _mm_mulhi_epi16(_val, _w);

                        _sum = _mm_add_epi32(_sum, _mm_unpacklo_epi16(_sl, _sh));

                        tmpptr += 4;
                        kptr0 += 4;
                    }

                    sum = _mm_reduce_add_epi32(_sum);
                }

                int j = 0;
                for (; j < nn1; j++)
                {
                    signed char val = tmpptr[0];
                    signed char w = kptr0[0];

                    sum += val * w;

                    tmpptr += 1;
                    kptr0 += 1;
                }

                outptr0[0] = sum;
                outptr0 += 1;
            }
    #else  // __SSE2__
            for (; i < size; i++)
            {
                const signed char* tmpptr = (const signed char*)tmp_a[i].data();
                const signed char* kptr0 = (const signed char*)kernel_tf_a[p].data();

                int nn1 = inch * maxk;

                int sum = 0;
                int j = 0;
                for (; j < nn1; j++)
                {
                    signed char val = tmpptr[0];
                    signed char w = kptr0[0];

                    sum += val * w;

                    tmpptr += 1;
                    kptr0 += 1;
                }

                outptr0[0] = sum;
                outptr0 += 1;
            }
    #endif // __SSE2__
        }
    });
}
#else
void im2col_sgemm_conv2d_int8_impl_x86(
    const Tensor& /*im2col_*/,
    const Tensor& /*kernel_tf_*/,
    int64_t /*input_channels*/,
    int64_t /*output_channels*/,
    Tensor& /*output*/) {}
#endif

Tensor& sgemm_conv2d_int8_x86_out(
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
    
    const int64_t kernel_height = kernel_size[0];
    const int64_t kernel_width  = kernel_size[1];
    
    const int64_t input_channels  = self.size(1);
    const int64_t output_channels = weight.size(0);
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        otter::convolution_im2col_sgemm_transform_kernel_int8_sse(weight, kernel_tf, input_channels, output_channels, kernel_width, kernel_height);
    
    Tensor im2col = otter::im2col_cpu(self, kernel_size, stride, padding, dilation);
    
    im2col_sgemm_conv2d_int8_impl_x86(im2col, kernel_tf, input_channels, output_channels, output);
    
    return output;
}
    
Tensor sgemm_conv2d_int8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, otter::ScalarType::Int);
    
    return sgemm_conv2d_int8_x86_out(self, weight, weight_o, kernel_size, stride, padding, dilation, output);
}

Tensor& sgemm_conv2d_1x1s1_int8_x86_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    IntArrayRef padding,
    Tensor& output) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), {1, 1}, padding);
    output.resize_(output_size);
    
    const int64_t input_channels  = self.size(1);
    const int64_t output_channels = weight.size(0);
    
    Tensor im2col = self.view({self.size(0), self.size(1), -1});
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        otter::convolution_im2col_sgemm_transform_kernel_int8_sse(weight, kernel_tf, input_channels, output_channels, 1, 1);
    im2col_sgemm_conv2d_int8_impl_x86(im2col, kernel_tf, input_channels, output_channels, output);
    
    return output;
}

Tensor sgemm_conv2d_1x1s1_int8_x86(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, otter::ScalarType::Int);
    
    return sgemm_conv2d_1x1s1_int8_x86_out(self, weight, weight_o, padding, output);
}

}   // end namesapce otter
