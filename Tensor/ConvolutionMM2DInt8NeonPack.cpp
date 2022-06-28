//
//  ConvolutionMM2DInt8NeonPack.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/28.
//

#include "ConvolutionMM2DInt8NeonPack.hpp"
#include "TensorFactory.hpp"
#include "Parallel.hpp"
#include "im2col.hpp"
#include "ConvolutionUtils.hpp"
#include "Padding.hpp"
#include "VecIntrinsic.hpp"

namespace otter {

#if __ARM_NEON__

void convolution_im2col_sgemm_transform_kernel_pack1to4_int8_neon(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h) {
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8a-4b-maxk-inch/8a-outch/4b
    // dst = 4a-4b-2-maxk-inch/8a-outch/4b (arm82)
    Tensor kernel = _kernel.view({outch, inch, maxk});
    if (inch >= 8)
        kernel_tm = otter::empty({outch / 4, inch / 8 + (inch % 8) / 4 + inch % 4, 32 * maxk}, otter::ScalarType::Byte);
    else if (inch >= 4)
        kernel_tm = otter::empty({outch / 4, inch / 4 + inch % 4, 16 * maxk}, otter::ScalarType::Byte);
    else
        kernel_tm = otter::empty({outch / 4, inch, 4 * maxk}, otter::ScalarType::Byte);
    
    auto kernel_a = kernel.accessor<signed char, 3>();
    auto kernel_tm_a = kernel_tm.accessor<signed char, 3>();

    for (int q = 0; q + 3 < outch; q += 4) {
        signed char* g00 = kernel_tm_a[q / 4].data();

        int p = 0;
        for (; p + 7 < inch; p += 8) {
            for (int k = 0; k < maxk; k++) {
#if __ARM_FEATURE_DOTPROD
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        const signed char* k00 = kernel_a[q + i][p + j].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
                for (int i = 0; i < 4; i++) {
                    for (int j = 4; j < 8; j++) {
                        const signed char* k00 = kernel_a[q + i][p + j].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
#else
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 8; j++) {
                        const signed char* k00 = kernel_a[q + i][p + j].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
#endif
            }
        }
        for (; p + 3 < inch; p += 4) {
            for (int k = 0; k < maxk; k++) {
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++)  {
                        const signed char* k00 = kernel_a[q + i][p + j].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
        for (; p < inch; p++) {
            for (int k = 0; k < maxk; k++) {
                for (int i = 0; i < 4; i++) {
                    const signed char* k00 = kernel_a[q + i][p].data();

                    g00[0] = k00[k];

                    g00++;
                }
            }
        }
    }
}

void convolution_im2col_sgemm_transform_kernel_pack8to1_int8_neon(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h) {
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8a-4b-maxk-inch/8a-outch/4b
    // dst = 4a-4b-2-maxk-inch/8a-outch/4b (arm82)
    Tensor kernel = _kernel.view({outch, inch, maxk});
    if (outch >= 4)
        kernel_tm = otter::empty({outch / 4 + outch % 4, inch / 8, 32 * maxk}, otter::ScalarType::Byte);
    else
        kernel_tm = otter::empty({outch, inch / 8, 8 * maxk}, otter::ScalarType::Byte);
    
    auto kernel_a = kernel.accessor<signed char, 3>();
    auto kernel_tm_a = kernel_tm.accessor<signed char, 3>();

    int q = 0;
    for (; q + 3 < outch; q += 4) {
        signed char* g00 = kernel_tm_a[q / 4].data();

        for (int p = 0; p + 7 < inch; p += 8) {
            for (int k = 0; k < maxk; k++) {
#if __ARM_FEATURE_DOTPROD
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        const signed char* k00 = kernel_a[q + i][p + j].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
                for (int i = 0; i < 4; i++) {
                    for (int j = 4; j < 8; j++) {
                        const signed char* k00 = kernel_a[q + i][p + j].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
#else
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 8; j++) {
                        const signed char* k00 = kernel_a[q + i][p + j].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
#endif
            }
        }
    }
    // TODO unroll 2
    for (; q < outch; q++) {
        signed char* g00 = kernel_tm_a[q / 4 + q % 4].data();

        for (int p = 0; p + 7 < inch; p += 8) {
            for (int k = 0; k < maxk; k++) {
                for (int j = 0; j < 8; j++) {
                    const signed char* k00 = kernel_a[q][p + j].data();

                    g00[0] = k00[k];

                    g00++;
                }
            }
        }
    }
}

void convolution_im2col_sgemm_transform_kernel_pack8to4_int8_neon(const Tensor& _kernel, Tensor& kernel_tm, int inch, int outch, int kernel_w, int kernel_h) {
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 8a-4b-maxk-inch/8a-outch/4b
    // dst = 4a-4b-2-maxk-inch/8a-outch/4b (arm82)
    Tensor kernel = _kernel.view({outch, inch, maxk});
    
    kernel_tm = otter::empty({outch / 4, inch / 8, 32 * maxk}, otter::ScalarType::Byte);
    
    auto kernel_a = kernel.accessor<signed char, 3>();
    auto kernel_tm_a = kernel_tm.accessor<signed char, 3>();

    for (int q = 0; q + 3 < outch; q += 4) {
        signed char* g00 = kernel_tm_a[q / 4].data();

        for (int p = 0; p + 7 < inch; p += 8) {
            for (int k = 0; k < maxk; k++) {
#if __ARM_FEATURE_DOTPROD
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        const signed char* k00 = kernel_a[q + i][p + j].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
                for (int i = 0; i < 4; i++) {
                    for (int j = 4; j < 8; j++) {
                        const signed char* k00 = kernel_a[q + i][p + j].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
#else
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 8; j++) {
                        const signed char* k00 = kernel_a[q + i][p + j].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
#endif
            }
        }
    }
}

void im2col_sgemm_conv2d_int8_pack1to4_impl_neon(
    const Tensor& im2col,
    const Tensor& kernel_tf_,
    const Tensor& bias_,
    int64_t input_channels,
    int64_t output_channels,
    Tensor& output) {
    
    const int size = im2col.size(2);
    const int maxk = im2col.size(1);
    const int inch = im2col.size(0);

    const int outch = output.size(1);
    
    auto output_a = output.accessor<int, 4, 4>()[0];
    auto im2col_a = im2col.accessor<signed char, 3>();
    auto kernel_a = kernel_tf_.accessor<signed char, 3>();

    // permute
    Tensor tmp;
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
    if (inch >= 8)
    {
        if (size >= 16)
            tmp = otter::empty({size / 16 + (size % 16) / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, inch / 8 + (inch % 8) / 4 + inch % 4, 16 * maxk}, otter::ScalarType::Byte8);
        else if (size >= 8)
            tmp = otter::empty({size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, inch / 8 + (inch % 8) / 4 + inch % 4, 8 * maxk}, otter::ScalarType::Byte8);
        else if (size >= 4)
            tmp = otter::empty({size / 4 + (size % 4) / 2 + size % 2, inch / 8 + (inch % 8) / 4 + inch % 4, 4 * maxk}, otter::ScalarType::Byte8);
        else if (size >= 2)
            tmp = otter::empty({size / 2 + size % 2, inch / 8 + (inch % 8) / 4 + inch % 4, 2 * maxk}, otter::ScalarType::Byte8);
        else
            tmp = otter::empty({size, inch / 8 + (inch % 8) / 4 + inch % 4, maxk}, otter::ScalarType::Byte8);
    }
    else if (inch >= 4)
    {
        if (size >= 16)
            tmp = otter::empty({size / 16 + (size % 16) / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, inch / 4 + inch % 4, 16 * maxk}, otter::ScalarType::Byte4);
        else if (size >= 8)
            tmp = otter::empty({size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, inch / 4 + inch % 4, 8 * maxk}, otter::ScalarType::Byte4);
        else if (size >= 4)
            tmp = otter::empty({size / 4 + (size % 4) / 2 + size % 2, inch / 4 + inch % 4, 4 * maxk}, otter::ScalarType::Byte4);
        else if (size >= 2)
            tmp = otter::empty({size / 2 + size % 2, inch / 4 + inch % 4, 2 * maxk}, otter::ScalarType::Byte4);
        else
            tmp = otter::empty({size, inch / 4 + inch % 4, maxk}, otter::ScalarType::Byte4);
    }
    else
    {
        if (size >= 16)
            tmp = otter::empty({size / 16 + (size % 16) / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, inch, 16 * maxk}, otter::ScalarType::Byte);
        else if (size >= 8)
            tmp = otter::empty({size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, inch, 8 * maxk}, otter::ScalarType::Byte);
        else if (size >= 4)
            tmp = otter::empty({size / 4 + (size % 4) / 2 + size % 2, inch, 4 * maxk}, otter::ScalarType::Byte);
        else if (size >= 2)
            tmp = otter::empty({size / 2 + size % 2, inch, 2 * maxk}, otter::ScalarType::Byte);
        else
            tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Byte8);
    }
#else  // __ARM_FEATURE_DOTPROD
    if (inch >= 8)
    {
        if (size >= 4)
            tmp = otter::empty({size / 4 + (size % 4) / 2 + size % 2, inch / 8 + (inch % 8) / 4 + inch % 4, 4 * maxk}, otter::ScalarType::Byte8);
        else if (size >= 2)
            tmp = otter::empty({size / 2 + size % 2, inch / 8 + (inch % 8) / 4 + inch % 4, 2 * maxk}, otter::ScalarType::Byte8);
        else
            tmp = otter::empty({size, inch / 8 + (inch % 8) / 4 + inch % 4, maxk}, otter::ScalarType::Byte8);
    }
    else if (inch >= 4)
    {
        if (size >= 4)
            tmp = otter::empty({size / 4 + (size % 4) / 2 + size % 2, inch / 4 + inch % 4, 4 * maxk}, otter::ScalarType::Byte4);
        else if (size >= 2)
            tmp = otter::empty({size / 2 + size % 2, inch / 4 + inch % 4, 2 * maxk}, otter::ScalarType::Byte4);
        else
            tmp = otter::empty({size, inch / 4 + inch % 4, maxk}, otter::ScalarType::Byte4);
    }
    else
    {
        if (size >= 4)
            tmp = otter::empty({size / 4 + (size % 4) / 2 + size % 2, inch, 4 * maxk}, otter::ScalarType::Byte);
        else if (size >= 2)
            tmp = otter::empty({size / 2 + size % 2, inch, 2 * maxk}, otter::ScalarType::Byte);
        else
            tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Byte);
    }
#endif // __ARM_FEATURE_DOTPROD
#else  // __aarch64__
    if (inch >= 8)
    {
        if (size >= 2)
            tmp = otter::empty({size / 2 + size % 2, inch / 8 + (inch % 8) / 4 + inch % 4, 2 * maxk}, otter::ScalarType::Byte8);
        else
            tmp = otter::empty({size, inch / 8 + (inch % 8) / 4 + inch % 4, maxk}, otter::ScalarType::Byte8);
    }
    else if (inch >= 4)
    {
        if (size >= 2)
            tmp = otter::empty({size / 2 + size % 2, inch / 4 + inch % 4, 2 * maxk}, otter::ScalarType::Byte4);
        else
            tmp = otter::empty({size, inch / 4 + inch % 4, maxk}, otter::ScalarType::Byte4);
    }
    else
    {
        if (size >= 2)
            tmp = otter::empty({size / 2 + size % 2, inch, 2 * maxk}, otter::ScalarType::Byte);
        else
            tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Byte);
    }
#endif // __aarch64__
    
    auto tmp_ra = tmp.raw_accessor<signed char, 3>();
    
    {
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
        int nn_size = size >> 4;
        int remain_size_start = 0;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end))
            {
                int i = remain_size_start + ii * 16;

                signed char* tmpptr = tmp_ra[i / 16].data();

                int q = 0;
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q + 0].data() + i;
                    const signed char* img1 = (const signed char*)im2col_a[q + 1].data() + i;
                    const signed char* img2 = (const signed char*)im2col_a[q + 2].data() + i;
                    const signed char* img3 = (const signed char*)im2col_a[q + 3].data() + i;
                    const signed char* img4 = (const signed char*)im2col_a[q + 4].data() + i;
                    const signed char* img5 = (const signed char*)im2col_a[q + 5].data() + i;
                    const signed char* img6 = (const signed char*)im2col_a[q + 6].data() + i;
                    const signed char* img7 = (const signed char*)im2col_a[q + 7].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        asm volatile(
                            "ld1    {v0.16b}, [%0]              \n"
                            "ld1    {v1.16b}, [%1]              \n"
                            "ld1    {v2.16b}, [%2]              \n"
                            "ld1    {v3.16b}, [%3]              \n"
                            "ld1    {v4.16b}, [%4]              \n"
                            "ld1    {v5.16b}, [%5]              \n"
                            "ld1    {v6.16b}, [%6]              \n"
                            "ld1    {v7.16b}, [%7]              \n"
                            "st4    {v0.16b, v1.16b, v2.16b, v3.16b}, [%8], #64 \n"
                            "st4    {v4.16b, v5.16b, v6.16b, v7.16b}, [%8], #64 \n"
                            : "=r"(img0), // %0
                            "=r"(img1),
                            "=r"(img2),
                            "=r"(img3),
                            "=r"(img4),
                            "=r"(img5),
                            "=r"(img6),
                            "=r"(img7),
                            "=r"(tmpptr) // %8
                            : "0"(img0),
                            "1"(img1),
                            "2"(img2),
                            "3"(img3),
                            "4"(img4),
                            "5"(img5),
                            "6"(img6),
                            "7"(img7),
                            "8"(tmpptr)
                            : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
                        img0 += size;
                        img1 += size;
                        img2 += size;
                        img3 += size;
                        img4 += size;
                        img5 += size;
                        img6 += size;
                        img7 += size;
                    }
                }
                for (; q + 3 < inch; q += 4)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q + 0].data() + i;
                    const signed char* img1 = (const signed char*)im2col_a[q + 1].data() + i;
                    const signed char* img2 = (const signed char*)im2col_a[q + 2].data() + i;
                    const signed char* img3 = (const signed char*)im2col_a[q + 3].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        asm volatile(
                            "ld1    {v0.16b}, [%0]              \n"
                            "ld1    {v1.16b}, [%1]              \n"
                            "ld1    {v2.16b}, [%2]              \n"
                            "ld1    {v3.16b}, [%3]              \n"
                            "st4    {v0.16b, v1.16b, v2.16b, v3.16b}, [%4], #64 \n"
                            : "=r"(img0), // %0
                            "=r"(img1),
                            "=r"(img2),
                            "=r"(img3),
                            "=r"(tmpptr) // %4
                            : "0"(img0),
                            "1"(img1),
                            "2"(img2),
                            "3"(img3),
                            "4"(tmpptr)
                            : "memory", "v0", "v1", "v2", "v3");
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
                        asm volatile(
                            "prfm   pldl1keep, [%0, #128]   \n"
                            "ld1    {v0.16b}, [%0]          \n"
                            "st1    {v0.16b}, [%1], #16     \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0");
                        img0 += size;
                    }
                }
            }
        });

        remain_size_start += nn_size << 4;
        nn_size = (size - remain_size_start) >> 3;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end))
            {
                int i = remain_size_start + ii * 8;

                signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8].data();

                int q = 0;
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q + 0].data() + i;
                    const signed char* img1 = (const signed char*)im2col_a[q + 1].data() + i;
                    const signed char* img2 = (const signed char*)im2col_a[q + 2].data() + i;
                    const signed char* img3 = (const signed char*)im2col_a[q + 3].data() + i;
                    const signed char* img4 = (const signed char*)im2col_a[q + 4].data() + i;
                    const signed char* img5 = (const signed char*)im2col_a[q + 5].data() + i;
                    const signed char* img6 = (const signed char*)im2col_a[q + 6].data() + i;
                    const signed char* img7 = (const signed char*)im2col_a[q + 7].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        asm volatile(
                            "ld1    {v0.8b}, [%0]               \n"
                            "ld1    {v1.8b}, [%1]               \n"
                            "ld1    {v2.8b}, [%2]               \n"
                            "ld1    {v3.8b}, [%3]               \n"
                            "ld1    {v4.8b}, [%4]               \n"
                            "ld1    {v5.8b}, [%5]               \n"
                            "ld1    {v6.8b}, [%6]               \n"
                            "ld1    {v7.8b}, [%7]               \n"
                            "st4    {v0.8b, v1.8b, v2.8b, v3.8b}, [%8], #32 \n"
                            "st4    {v4.8b, v5.8b, v6.8b, v7.8b}, [%8], #32 \n"
                            : "=r"(img0), // %0
                            "=r"(img1),
                            "=r"(img2),
                            "=r"(img3),
                            "=r"(img4),
                            "=r"(img5),
                            "=r"(img6),
                            "=r"(img7),
                            "=r"(tmpptr) // %8
                            : "0"(img0),
                            "1"(img1),
                            "2"(img2),
                            "3"(img3),
                            "4"(img4),
                            "5"(img5),
                            "6"(img6),
                            "7"(img7),
                            "8"(tmpptr)
                            : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
                        img0 += size;
                        img1 += size;
                        img2 += size;
                        img3 += size;
                        img4 += size;
                        img5 += size;
                        img6 += size;
                        img7 += size;
                    }
                }
                for (; q + 3 < inch; q += 4)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q + 0].data() + i;
                    const signed char* img1 = (const signed char*)im2col_a[q + 1].data() + i;
                    const signed char* img2 = (const signed char*)im2col_a[q + 2].data() + i;
                    const signed char* img3 = (const signed char*)im2col_a[q + 3].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        asm volatile(
                            "ld1    {v0.8b}, [%0]               \n"
                            "ld1    {v1.8b}, [%1]               \n"
                            "ld1    {v2.8b}, [%2]               \n"
                            "ld1    {v3.8b}, [%3]               \n"
                            "st4    {v0.8b, v1.8b, v2.8b, v3.8b}, [%4], #32 \n"
                            : "=r"(img0), // %0
                            "=r"(img1),
                            "=r"(img2),
                            "=r"(img3),
                            "=r"(tmpptr) // %4
                            : "0"(img0),
                            "1"(img1),
                            "2"(img2),
                            "3"(img3),
                            "4"(tmpptr)
                            : "memory", "v0", "v1", "v2", "v3");
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
                        asm volatile(
                            "prfm   pldl1keep, [%0, #64]    \n"
                            "ld1    {v0.8b}, [%0]           \n"
                            "st1    {v0.8b}, [%1], #8       \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0");
                        img0 += size;
                    }
                }
            }
        });

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;
#else  // __ARM_FEATURE_DOTPROD
        int remain_size_start = 0;
        int nn_size = (size - remain_size_start) >> 2;
#endif // __ARM_FEATURE_DOTPROD

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end))
            {
                int i = remain_size_start + ii * 4;

    #if __ARM_FEATURE_DOTPROD
                signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4].data();
    #else
                signed char* tmpptr = tmp_ra[i / 4].data();
    #endif

                int q = 0;
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q + 0].data() + i;
                    const signed char* img1 = (const signed char*)im2col_a[q + 1].data() + i;
                    const signed char* img2 = (const signed char*)im2col_a[q + 2].data() + i;
                    const signed char* img3 = (const signed char*)im2col_a[q + 3].data() + i;
                    const signed char* img4 = (const signed char*)im2col_a[q + 4].data() + i;
                    const signed char* img5 = (const signed char*)im2col_a[q + 5].data() + i;
                    const signed char* img6 = (const signed char*)im2col_a[q + 6].data() + i;
                    const signed char* img7 = (const signed char*)im2col_a[q + 7].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
    #if __ARM_FEATURE_DOTPROD
                        tmpptr[0] = img0[0];
                        tmpptr[1] = img1[0];
                        tmpptr[2] = img2[0];
                        tmpptr[3] = img3[0];
                        tmpptr[4] = img0[1];
                        tmpptr[5] = img1[1];
                        tmpptr[6] = img2[1];
                        tmpptr[7] = img3[1];
                        tmpptr += 8;

                        tmpptr[0] = img0[2];
                        tmpptr[1] = img1[2];
                        tmpptr[2] = img2[2];
                        tmpptr[3] = img3[2];
                        tmpptr[4] = img0[3];
                        tmpptr[5] = img1[3];
                        tmpptr[6] = img2[3];
                        tmpptr[7] = img3[3];
                        tmpptr += 8;

                        tmpptr[0] = img4[0];
                        tmpptr[1] = img5[0];
                        tmpptr[2] = img6[0];
                        tmpptr[3] = img7[0];
                        tmpptr[4] = img4[1];
                        tmpptr[5] = img5[1];
                        tmpptr[6] = img6[1];
                        tmpptr[7] = img7[1];
                        tmpptr += 8;

                        tmpptr[0] = img4[2];
                        tmpptr[1] = img5[2];
                        tmpptr[2] = img6[2];
                        tmpptr[3] = img7[2];
                        tmpptr[4] = img4[3];
                        tmpptr[5] = img5[3];
                        tmpptr[6] = img6[3];
                        tmpptr[7] = img7[3];
                        tmpptr += 8;
    #else
                        tmpptr[0] = img0[0];
                        tmpptr[1] = img1[0];
                        tmpptr[2] = img2[0];
                        tmpptr[3] = img3[0];
                        tmpptr[4] = img4[0];
                        tmpptr[5] = img5[0];
                        tmpptr[6] = img6[0];
                        tmpptr[7] = img7[0];
                        tmpptr += 8;

                        tmpptr[0] = img0[1];
                        tmpptr[1] = img1[1];
                        tmpptr[2] = img2[1];
                        tmpptr[3] = img3[1];
                        tmpptr[4] = img4[1];
                        tmpptr[5] = img5[1];
                        tmpptr[6] = img6[1];
                        tmpptr[7] = img7[1];
                        tmpptr += 8;

                        tmpptr[0] = img0[2];
                        tmpptr[1] = img1[2];
                        tmpptr[2] = img2[2];
                        tmpptr[3] = img3[2];
                        tmpptr[4] = img4[2];
                        tmpptr[5] = img5[2];
                        tmpptr[6] = img6[2];
                        tmpptr[7] = img7[2];
                        tmpptr += 8;

                        tmpptr[0] = img0[3];
                        tmpptr[1] = img1[3];
                        tmpptr[2] = img2[3];
                        tmpptr[3] = img3[3];
                        tmpptr[4] = img4[3];
                        tmpptr[5] = img5[3];
                        tmpptr[6] = img6[3];
                        tmpptr[7] = img7[3];
                        tmpptr += 8;
    #endif // __ARM_FEATURE_DOTPROD

                        img0 += size;
                        img1 += size;
                        img2 += size;
                        img3 += size;
                        img4 += size;
                        img5 += size;
                        img6 += size;
                        img7 += size;
                    }
                }
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

                        tmpptr[0] = img0[2];
                        tmpptr[1] = img1[2];
                        tmpptr[2] = img2[2];
                        tmpptr[3] = img3[2];
                        tmpptr[4] = img0[3];
                        tmpptr[5] = img1[3];
                        tmpptr[6] = img2[3];
                        tmpptr[7] = img3[3];
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

    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
                signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2].data();
    #else
                signed char* tmpptr = tmp_ra[i / 4 + (i % 4) / 2].data();
    #endif
    #else
                signed char* tmpptr = tmp_ra[i / 2].data();
    #endif

                int q = 0;
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q + 0].data() + i;
                    const signed char* img1 = (const signed char*)im2col_a[q + 1].data() + i;
                    const signed char* img2 = (const signed char*)im2col_a[q + 2].data() + i;
                    const signed char* img3 = (const signed char*)im2col_a[q + 3].data() + i;
                    const signed char* img4 = (const signed char*)im2col_a[q + 4].data() + i;
                    const signed char* img5 = (const signed char*)im2col_a[q + 5].data() + i;
                    const signed char* img6 = (const signed char*)im2col_a[q + 6].data() + i;
                    const signed char* img7 = (const signed char*)im2col_a[q + 7].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
    #if __ARM_FEATURE_DOTPROD
                        tmpptr[0] = img0[0];
                        tmpptr[1] = img1[0];
                        tmpptr[2] = img2[0];
                        tmpptr[3] = img3[0];
                        tmpptr[4] = img0[1];
                        tmpptr[5] = img1[1];
                        tmpptr[6] = img2[1];
                        tmpptr[7] = img3[1];
                        tmpptr += 8;

                        tmpptr[0] = img4[0];
                        tmpptr[1] = img5[0];
                        tmpptr[2] = img6[0];
                        tmpptr[3] = img7[0];
                        tmpptr[4] = img4[1];
                        tmpptr[5] = img5[1];
                        tmpptr[6] = img6[1];
                        tmpptr[7] = img7[1];
                        tmpptr += 8;
    #else
                        tmpptr[0] = img0[0];
                        tmpptr[1] = img1[0];
                        tmpptr[2] = img2[0];
                        tmpptr[3] = img3[0];
                        tmpptr[4] = img4[0];
                        tmpptr[5] = img5[0];
                        tmpptr[6] = img6[0];
                        tmpptr[7] = img7[0];
                        tmpptr += 8;

                        tmpptr[0] = img0[1];
                        tmpptr[1] = img1[1];
                        tmpptr[2] = img2[1];
                        tmpptr[3] = img3[1];
                        tmpptr[4] = img4[1];
                        tmpptr[5] = img5[1];
                        tmpptr[6] = img6[1];
                        tmpptr[7] = img7[1];
                        tmpptr += 8;
    #endif // __ARM_FEATURE_DOTPROD

                        img0 += size;
                        img1 += size;
                        img2 += size;
                        img3 += size;
                        img4 += size;
                        img5 += size;
                        img6 += size;
                        img7 += size;
                    }
                }
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
    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
                signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2].data();
    #else
                signed char* tmpptr = tmp_ra[i / 4 + (i % 4) / 2 + i % 2].data();
    #endif
    #else
                signed char* tmpptr = tmp_ra[i / 2 + i % 2].data();
    #endif

                int q = 0;
                for (; q + 7 < inch; q += 8)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q + 0].data() + i;
                    const signed char* img1 = (const signed char*)im2col_a[q + 1].data() + i;
                    const signed char* img2 = (const signed char*)im2col_a[q + 2].data() + i;
                    const signed char* img3 = (const signed char*)im2col_a[q + 3].data() + i;
                    const signed char* img4 = (const signed char*)im2col_a[q + 4].data() + i;
                    const signed char* img5 = (const signed char*)im2col_a[q + 5].data() + i;
                    const signed char* img6 = (const signed char*)im2col_a[q + 6].data() + i;
                    const signed char* img7 = (const signed char*)im2col_a[q + 7].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        tmpptr[0] = img0[0];
                        tmpptr[1] = img1[0];
                        tmpptr[2] = img2[0];
                        tmpptr[3] = img3[0];
                        tmpptr[4] = img4[0];
                        tmpptr[5] = img5[0];
                        tmpptr[6] = img6[0];
                        tmpptr[7] = img7[0];
                        tmpptr += 8;

                        img0 += size;
                        img1 += size;
                        img2 += size;
                        img3 += size;
                        img4 += size;
                        img5 += size;
                        img6 += size;
                        img7 += size;
                    }
                }
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
        for (const auto p : otter::irange(begin, end))
        {
            int* outptr0 = output_a[p].data();

            int i = 0;
    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
            for (; i + 15 < size; i += 16)
            {
                const signed char* tmpptr = tmp_ra[i / 16].data();
                const signed char* kptr0 = kernel_a[p].data();

                int nn = (inch / 8) * maxk;
                int nn4 = ((inch % 8) / 4) * maxk;
                int nn1 = (inch % 4) * maxk;

                asm volatile(
                    "eor    v16.16b, v16.16b, v16.16b   \n"
                    "eor    v17.16b, v17.16b, v17.16b   \n"
                    "eor    v18.16b, v18.16b, v18.16b   \n"
                    "eor    v19.16b, v19.16b, v19.16b   \n"
                    "eor    v20.16b, v20.16b, v20.16b   \n"
                    "eor    v21.16b, v21.16b, v21.16b   \n"
                    "eor    v22.16b, v22.16b, v22.16b   \n"
                    "eor    v23.16b, v23.16b, v23.16b   \n"
                    "eor    v24.16b, v24.16b, v24.16b   \n"
                    "eor    v25.16b, v25.16b, v25.16b   \n"
                    "eor    v26.16b, v26.16b, v26.16b   \n"
                    "eor    v27.16b, v27.16b, v27.16b   \n"
                    "eor    v28.16b, v28.16b, v28.16b   \n"
                    "eor    v29.16b, v29.16b, v29.16b   \n"
                    "eor    v30.16b, v30.16b, v30.16b   \n"
                    "eor    v31.16b, v31.16b, v31.16b   \n"

                    "cmp    %w1, #0                     \n"
                    "beq    1f                          \n"

                    "ld1    {v8.16b}, [%5], #16         \n" // _w0123_l

                    "ld1    {v0.16b}, [%4], #16         \n" // _val0123_l

                    "0:                                 \n"

                    "ld1    {v1.16b}, [%4], #16         \n" // _val4567_l

                    "sdot   v16.4s, v8.16b, v0.4b[0]    \n"
                    "sdot   v17.4s, v8.16b, v0.4b[1]    \n"
                    "sdot   v18.4s, v8.16b, v0.4b[2]    \n"
                    "sdot   v19.4s, v8.16b, v0.4b[3]    \n"

                    "ld1    {v2.16b}, [%4], #16         \n" // _val891011_l

                    "sdot   v20.4s, v8.16b, v1.4b[0]    \n"
                    "sdot   v21.4s, v8.16b, v1.4b[1]    \n"
                    "sdot   v22.4s, v8.16b, v1.4b[2]    \n"
                    "sdot   v23.4s, v8.16b, v1.4b[3]    \n"

                    "ld1    {v3.16b}, [%4], #16         \n" // _val12131415_l

                    "sdot   v24.4s, v8.16b, v2.4b[0]    \n"
                    "sdot   v25.4s, v8.16b, v2.4b[1]    \n"

                    "ld1    {v9.16b}, [%5], #16         \n" // _w0123_h

                    "sdot   v26.4s, v8.16b, v2.4b[2]    \n"
                    "sdot   v27.4s, v8.16b, v2.4b[3]    \n"

                    "ld1    {v4.16b}, [%4], #16         \n" // _val0123_h

                    "sdot   v28.4s, v8.16b, v3.4b[0]    \n"
                    "sdot   v29.4s, v8.16b, v3.4b[1]    \n"
                    "sdot   v30.4s, v8.16b, v3.4b[2]    \n"
                    "sdot   v31.4s, v8.16b, v3.4b[3]    \n"

                    "ld1    {v5.16b}, [%4], #16         \n" // _val4567_h

                    "sdot   v16.4s, v9.16b, v4.4b[0]    \n"
                    "sdot   v17.4s, v9.16b, v4.4b[1]    \n"
                    "sdot   v18.4s, v9.16b, v4.4b[2]    \n"
                    "sdot   v19.4s, v9.16b, v4.4b[3]    \n"

                    "ld1    {v6.16b}, [%4], #16         \n" // _val891011_h

                    "sdot   v20.4s, v9.16b, v5.4b[0]    \n"
                    "sdot   v21.4s, v9.16b, v5.4b[1]    \n"
                    "sdot   v22.4s, v9.16b, v5.4b[2]    \n"
                    "sdot   v23.4s, v9.16b, v5.4b[3]    \n"

                    "ld1    {v7.16b}, [%4], #16         \n" // _val12131415_h

                    "sdot   v24.4s, v9.16b, v6.4b[0]    \n"
                    "sdot   v25.4s, v9.16b, v6.4b[1]    \n"

                    "ld1    {v8.16b}, [%5], #16         \n" // _w0123_l

                    "sdot   v26.4s, v9.16b, v6.4b[2]    \n"
                    "sdot   v27.4s, v9.16b, v6.4b[3]    \n"

                    "ld1    {v0.16b}, [%4], #16         \n" // _val0123_l

                    "sdot   v28.4s, v9.16b, v7.4b[0]    \n"
                    "sdot   v29.4s, v9.16b, v7.4b[1]    \n"

                    "subs   %w1, %w1, #1                \n"

                    "sdot   v30.4s, v9.16b, v7.4b[2]    \n"
                    "sdot   v31.4s, v9.16b, v7.4b[3]    \n"

                    "bne    0b                          \n"

                    "sub    %4, %4, #16                 \n"
                    "sub    %5, %5, #16                 \n"

                    "1:                                 \n"

                    "cmp    %w2, #0                     \n"
                    "beq    3f                          \n"

                    "2:                                 \n"

                    "ld1    {v8.16b}, [%5], #16         \n"

                    "ld1    {v0.16b, v1.16b, v2.16b, v3.16b}, [%4], #64 \n"

                    "sdot   v16.4s, v8.16b, v0.4b[0]    \n"
                    "sdot   v17.4s, v8.16b, v0.4b[1]    \n"
                    "sdot   v18.4s, v8.16b, v0.4b[2]    \n"
                    "sdot   v19.4s, v8.16b, v0.4b[3]    \n"
                    "sdot   v20.4s, v8.16b, v1.4b[0]    \n"
                    "sdot   v21.4s, v8.16b, v1.4b[1]    \n"
                    "sdot   v22.4s, v8.16b, v1.4b[2]    \n"
                    "sdot   v23.4s, v8.16b, v1.4b[3]    \n"
                    "sdot   v24.4s, v8.16b, v2.4b[0]    \n"
                    "sdot   v25.4s, v8.16b, v2.4b[1]    \n"
                    "sdot   v26.4s, v8.16b, v2.4b[2]    \n"
                    "sdot   v27.4s, v8.16b, v2.4b[3]    \n"
                    "sdot   v28.4s, v8.16b, v3.4b[0]    \n"
                    "sdot   v29.4s, v8.16b, v3.4b[1]    \n"

                    "subs   %w2, %w2, #1                \n"

                    "sdot   v30.4s, v8.16b, v3.4b[2]    \n"
                    "sdot   v31.4s, v8.16b, v3.4b[3]    \n"

                    "bne    2b                          \n"

                    "3:                                 \n"

                    "lsr    w4, %w3, #2                 \n" // w4 = nn1 >> 2
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "4:                                 \n"

                    "ld1    {v8.8b, v9.8b}, [%5], #16   \n"

                    "ld4    {v0.16b, v1.16b, v2.16b, v3.16b}, [%4], #64 \n"

                    "uzp1   v10.8b, v8.8b, v9.8b        \n"
                    "uzp2   v11.8b, v8.8b, v9.8b        \n"

                    "uzp1   v4.16b, v0.16b, v1.16b      \n"
                    "uzp2   v5.16b, v0.16b, v1.16b      \n"
                    "uzp1   v6.16b, v2.16b, v3.16b      \n"
                    "uzp2   v7.16b, v2.16b, v3.16b      \n"

                    "uzp1   v8.8b, v10.8b, v11.8b       \n"
                    "uzp2   v9.8b, v10.8b, v11.8b       \n"

                    "uzp1   v0.16b, v4.16b, v5.16b      \n" // 0 1 4 5
                    "uzp2   v1.16b, v4.16b, v5.16b      \n" // 8 9 c d

                    "mov    v8.d[1], v9.d[0]            \n" // _w

                    "uzp1   v2.16b, v6.16b, v7.16b      \n" // 2 3 6 7
                    "uzp2   v3.16b, v6.16b, v7.16b      \n" // a b e f

                    "sdot   v16.4s, v8.16b, v0.4b[0]    \n"
                    "sdot   v17.4s, v8.16b, v0.4b[1]    \n"
                    "sdot   v18.4s, v8.16b, v2.4b[0]    \n"
                    "sdot   v19.4s, v8.16b, v2.4b[1]    \n"
                    "sdot   v20.4s, v8.16b, v0.4b[2]    \n"
                    "sdot   v21.4s, v8.16b, v0.4b[3]    \n"
                    "sdot   v22.4s, v8.16b, v2.4b[2]    \n"
                    "sdot   v23.4s, v8.16b, v2.4b[3]    \n"
                    "sdot   v24.4s, v8.16b, v1.4b[0]    \n"
                    "sdot   v25.4s, v8.16b, v1.4b[1]    \n"
                    "sdot   v26.4s, v8.16b, v3.4b[0]    \n"
                    "sdot   v27.4s, v8.16b, v3.4b[1]    \n"
                    "sdot   v28.4s, v8.16b, v1.4b[2]    \n"
                    "sdot   v29.4s, v8.16b, v1.4b[3]    \n"
                    "sdot   v30.4s, v8.16b, v3.4b[2]    \n"
                    "sdot   v31.4s, v8.16b, v3.4b[3]    \n"

                    "subs   w4, w4, #1                  \n"
                    "bne    4b                          \n"

                    "5:                                 \n"

                    "and    w4, %w3, #3                 \n" // w4 = remain = nn1 & 3
                    "cmp    w4, #0                      \n" // w4 > 0
                    "beq    7f                          \n"

                    "6:                                 \n"

                    "ld1    {v1.8b}, [%5]               \n"
                    "ld1    {v0.16b}, [%4]              \n"

                    "sshll  v1.8h, v1.8b, #0            \n"
                    "sshll  v2.8h, v0.8b, #0            \n"
                    "sshll2 v3.8h, v0.16b, #0           \n"

                    "smlal  v16.4s, v1.4h, v2.h[0]      \n"
                    "smlal  v17.4s, v1.4h, v2.h[1]      \n"
                    "smlal  v18.4s, v1.4h, v2.h[2]      \n"
                    "smlal  v19.4s, v1.4h, v2.h[3]      \n"
                    "smlal  v20.4s, v1.4h, v2.h[4]      \n"
                    "smlal  v21.4s, v1.4h, v2.h[5]      \n"
                    "smlal  v22.4s, v1.4h, v2.h[6]      \n"
                    "smlal  v23.4s, v1.4h, v2.h[7]      \n"
                    "smlal  v24.4s, v1.4h, v3.h[0]      \n"
                    "smlal  v25.4s, v1.4h, v3.h[1]      \n"
                    "smlal  v26.4s, v1.4h, v3.h[2]      \n"
                    "smlal  v27.4s, v1.4h, v3.h[3]      \n"
                    "smlal  v28.4s, v1.4h, v3.h[4]      \n"
                    "smlal  v29.4s, v1.4h, v3.h[5]      \n"
                    "smlal  v30.4s, v1.4h, v3.h[6]      \n"
                    "smlal  v31.4s, v1.4h, v3.h[7]      \n"

                    "add    %4, %4, #16                 \n"
                    "add    %5, %5, #4                  \n"

                    "subs   w4, w4, #1                  \n"
                    "bne    6b                          \n"

                    "7:                                 \n"

                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%0], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%0], #64 \n"
                    : "=r"(outptr0),
                    "=r"(nn),
                    "=r"(nn4),
                    "=r"(nn1),
                    "=r"(tmpptr),
                    "=r"(kptr0)
                    : "0"(outptr0),
                    "1"(nn),
                    "2"(nn4),
                    "3"(nn1),
                    "4"(tmpptr),
                    "5"(kptr0)
                    : "memory", "x4", "x5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            for (; i + 7 < size; i += 8)
            {
                const signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8].data();
                const signed char* kptr0 = kernel_a[p].data();

                int nn = (inch / 8) * maxk;
                int nn4 = ((inch % 8) / 4) * maxk;
                int nn1 = (inch % 4) * maxk;

                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                int32x4_t _sum4 = vdupq_n_s32(0);
                int32x4_t _sum5 = vdupq_n_s32(0);
                int32x4_t _sum6 = vdupq_n_s32(0);
                int32x4_t _sum7 = vdupq_n_s32(0);

                for (int j = 0; j < nn; j++)
                {
                    int8x16_t _val0123_l = vld1q_s8(tmpptr);
                    int8x16_t _val4567_l = vld1q_s8(tmpptr + 16);

                    int8x16_t _w0123_l = vld1q_s8(kptr0);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0123_l, _val0123_l, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0123_l, _val0123_l, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _w0123_l, _val0123_l, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _w0123_l, _val0123_l, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _w0123_l, _val4567_l, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _w0123_l, _val4567_l, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _w0123_l, _val4567_l, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _w0123_l, _val4567_l, 3);

                    int8x16_t _val0123_h = vld1q_s8(tmpptr + 32);
                    int8x16_t _val4567_h = vld1q_s8(tmpptr + 48);

                    int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0123_h, _val0123_h, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0123_h, _val0123_h, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _w0123_h, _val0123_h, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _w0123_h, _val0123_h, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _w0123_h, _val4567_h, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _w0123_h, _val4567_h, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _w0123_h, _val4567_h, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _w0123_h, _val4567_h, 3);

                    tmpptr += 64;
                    kptr0 += 32;
                }

                for (int j = 0; j < nn4; j++)
                {
                    int8x16_t _val0123 = vld1q_s8(tmpptr);
                    int8x16_t _val4567 = vld1q_s8(tmpptr + 16);
                    int8x16_t _w0 = vld1q_s8(kptr0);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0, _val0123, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0, _val0123, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _w0, _val0123, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _w0, _val0123, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _w0, _val4567, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _w0, _val4567, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _w0, _val4567, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _w0, _val4567, 3);

                    tmpptr += 32;
                    kptr0 += 16;
                }

                int j = 0;
                for (; j + 3 < nn1; j += 4)
                {
                    int8x8x4_t _val4 = vld4_s8(tmpptr);

                    int8x8x2_t _val0145 = vuzp_s8(_val4.val[0], _val4.val[1]);
                    int8x8x2_t _val2367 = vuzp_s8(_val4.val[2], _val4.val[3]);

                    int8x16_t _val0123 = vcombine_s8(_val0145.val[0], _val2367.val[0]);
                    int8x16_t _val4567 = vcombine_s8(_val0145.val[1], _val2367.val[1]);

                    int8x16_t _w = vld1q_s8(kptr0);

                    int8x8x2_t _w01 = vuzp_s8(vget_low_s8(_w), vget_high_s8(_w));
                    int8x8x2_t _w0123 = vuzp_s8(_w01.val[0], _w01.val[1]);
                    int8x16_t _w0123f = vcombine_s8(_w0123.val[0], _w0123.val[1]);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0123f, _val0123, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0123f, _val0123, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _w0123f, _val0123, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _w0123f, _val0123, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _w0123f, _val4567, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _w0123f, _val4567, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _w0123f, _val4567, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _w0123f, _val4567, 3);

                    tmpptr += 32;
                    kptr0 += 16;
                }
                for (; j < nn1; j++)
                {
                    int16x4_t _val0 = vdup_n_s16(tmpptr[0]);
                    int16x4_t _val1 = vdup_n_s16(tmpptr[1]);
                    int16x4_t _val2 = vdup_n_s16(tmpptr[2]);
                    int16x4_t _val3 = vdup_n_s16(tmpptr[3]);
                    int16x4_t _val4 = vdup_n_s16(tmpptr[4]);
                    int16x4_t _val5 = vdup_n_s16(tmpptr[5]);
                    int16x4_t _val6 = vdup_n_s16(tmpptr[6]);
                    int16x4_t _val7 = vdup_n_s16(tmpptr[7]);

                    int16x4_t _w0123;
                    _w0123 = vset_lane_s16(kptr0[0], _w0123, 0);
                    _w0123 = vset_lane_s16(kptr0[1], _w0123, 1);
                    _w0123 = vset_lane_s16(kptr0[2], _w0123, 2);
                    _w0123 = vset_lane_s16(kptr0[3], _w0123, 3);

                    _sum0 = vmlal_s16(_sum0, _val0, _w0123);
                    _sum1 = vmlal_s16(_sum1, _val1, _w0123);
                    _sum2 = vmlal_s16(_sum2, _val2, _w0123);
                    _sum3 = vmlal_s16(_sum3, _val3, _w0123);
                    _sum4 = vmlal_s16(_sum4, _val4, _w0123);
                    _sum5 = vmlal_s16(_sum5, _val5, _w0123);
                    _sum6 = vmlal_s16(_sum6, _val6, _w0123);
                    _sum7 = vmlal_s16(_sum7, _val7, _w0123);

                    tmpptr += 8;
                    kptr0 += 4;
                }

                vst1q_s32(outptr0, _sum0);
                vst1q_s32(outptr0 + 4, _sum1);
                vst1q_s32(outptr0 + 8, _sum2);
                vst1q_s32(outptr0 + 12, _sum3);
                vst1q_s32(outptr0 + 16, _sum4);
                vst1q_s32(outptr0 + 20, _sum5);
                vst1q_s32(outptr0 + 24, _sum6);
                vst1q_s32(outptr0 + 28, _sum7);
                outptr0 += 32;
            }
    #endif
            for (; i + 3 < size; i += 4)
            {
    #if __ARM_FEATURE_DOTPROD
                const signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4].data();
    #else
                const signed char* tmpptr = tmp_ra[i / 4].data();
    #endif
                const signed char* kptr0 = kernel_a[p].data();

                int nn = (inch / 8) * maxk;
                int nn4 = ((inch % 8) / 4) * maxk;
                int nn1 = (inch % 4) * maxk;
    #if __ARM_FEATURE_DOTPROD
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);

                for (int j = 0; j < nn; j++)
                {
                    int8x16_t _val0123_l = vld1q_s8(tmpptr);
                    int8x16_t _w0123_l = vld1q_s8(kptr0);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0123_l, _val0123_l, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0123_l, _val0123_l, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _w0123_l, _val0123_l, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _w0123_l, _val0123_l, 3);

                    int8x16_t _val0123_h = vld1q_s8(tmpptr + 16);
                    int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0123_h, _val0123_h, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0123_h, _val0123_h, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _w0123_h, _val0123_h, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _w0123_h, _val0123_h, 3);

                    tmpptr += 32;
                    kptr0 += 32;
                }

                for (int j = 0; j < nn4; j++)
                {
                    int8x16_t _val0123 = vld1q_s8(tmpptr);
                    int8x16_t _w0 = vld1q_s8(kptr0);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0, _val0123, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0, _val0123, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _w0, _val0123, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _w0, _val0123, 3);

                    tmpptr += 16;
                    kptr0 += 16;
                }

                int j = 0;
                for (; j + 3 < nn1; j += 4)
                {
                    int8x16_t _val = vld1q_s8(tmpptr);

                    int8x8x2_t _val01 = vuzp_s8(vget_low_s8(_val), vget_high_s8(_val));
                    int8x8x2_t _val0123 = vuzp_s8(_val01.val[0], _val01.val[1]);
                    int8x16_t _val0123f = vcombine_s8(_val0123.val[0], _val0123.val[1]);

                    int8x16_t _w = vld1q_s8(kptr0);

                    int8x8x2_t _w01 = vuzp_s8(vget_low_s8(_w), vget_high_s8(_w));
                    int8x8x2_t _w0123 = vuzp_s8(_w01.val[0], _w01.val[1]);
                    int8x16_t _w0123f = vcombine_s8(_w0123.val[0], _w0123.val[1]);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0123f, _val0123f, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0123f, _val0123f, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _w0123f, _val0123f, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _w0123f, _val0123f, 3);

                    tmpptr += 16;
                    kptr0 += 16;
                }
                for (; j < nn1; j++)
                {
                    int16x4_t _val0 = vdup_n_s16(tmpptr[0]);
                    int16x4_t _val1 = vdup_n_s16(tmpptr[1]);
                    int16x4_t _val2 = vdup_n_s16(tmpptr[2]);
                    int16x4_t _val3 = vdup_n_s16(tmpptr[3]);

                    int16x4_t _w0123;
                    _w0123 = vset_lane_s16(kptr0[0], _w0123, 0);
                    _w0123 = vset_lane_s16(kptr0[1], _w0123, 1);
                    _w0123 = vset_lane_s16(kptr0[2], _w0123, 2);
                    _w0123 = vset_lane_s16(kptr0[3], _w0123, 3);

                    _sum0 = vmlal_s16(_sum0, _val0, _w0123);
                    _sum1 = vmlal_s16(_sum1, _val1, _w0123);
                    _sum2 = vmlal_s16(_sum2, _val2, _w0123);
                    _sum3 = vmlal_s16(_sum3, _val3, _w0123);

                    tmpptr += 4;
                    kptr0 += 4;
                }

                vst1q_s32(outptr0, _sum0);
                vst1q_s32(outptr0 + 4, _sum1);
                vst1q_s32(outptr0 + 8, _sum2);
                vst1q_s32(outptr0 + 12, _sum3);
                outptr0 += 16;
    #else  // __ARM_FEATURE_DOTPROD
                asm volatile(
                    "eor    v0.16b, v0.16b, v0.16b      \n"
                    "eor    v1.16b, v1.16b, v1.16b      \n"
                    "eor    v2.16b, v2.16b, v2.16b      \n"
                    "eor    v3.16b, v3.16b, v3.16b      \n"

                    "cmp    %w1, #0                     \n"
                    "beq    3f                          \n"

                    "eor    v4.16b, v4.16b, v4.16b      \n"
                    "eor    v5.16b, v5.16b, v5.16b      \n"
                    "eor    v6.16b, v6.16b, v6.16b      \n"
                    "eor    v7.16b, v7.16b, v7.16b      \n"
                    "eor    v8.16b, v8.16b, v8.16b      \n"
                    "eor    v9.16b, v9.16b, v9.16b      \n"
                    "eor    v10.16b, v10.16b, v10.16b   \n"
                    "eor    v11.16b, v11.16b, v11.16b   \n"
                    "eor    v12.16b, v12.16b, v12.16b   \n"
                    "eor    v13.16b, v13.16b, v13.16b   \n"
                    "eor    v14.16b, v14.16b, v14.16b   \n"
                    "eor    v15.16b, v15.16b, v15.16b   \n"

                    "prfm   pldl1keep, [%4, #128]       \n"

                    "prfm   pldl1keep, [%5, #256]       \n"

                    "lsr    w4, %w1, #1                 \n" // w4 = nn >> 1
                    "cmp    w4, #0                      \n"
                    "beq    1f                          \n"

                    "prfm   pldl1keep, [%5, #512]       \n"

                    "add    x5, %4, #16                 \n"

                    "prfm   pldl1keep, [x5, #128]       \n"

                    "ld1    {v16.16b}, [%4]             \n" // val L H
                    "ld1    {v20.16b, v21.16b, v22.16b, v23.16b}, [%5], #64 \n"
                    "add    %4, %4, #32                 \n"
                    "ext    v17.16b, v16.16b, v16.16b, #8 \n" // val H L

                    "ld1    {v18.16b}, [%4]             \n"
                    "add    %4, %4, #32                 \n"

                    "0:                                 \n"

                    "smull  v24.8h, v16.8b,  v20.8b     \n"
                    "prfm   pldl1keep, [%5, #256]       \n"
                    "smull2 v25.8h, v17.16b, v20.16b    \n"
                    "prfm   pldl1keep, [%5, #512]       \n"
                    "smull  v26.8h, v16.8b,  v21.8b     \n"
                    "subs   w4, w4, #1                  \n"
                    "smull2 v27.8h, v17.16b, v21.16b    \n"
                    "ext    v19.16b, v18.16b, v18.16b, #8 \n" // val H L

                    "smlal  v24.8h, v18.8b,  v22.8b     \n"
                    "smlal2 v25.8h, v19.16b, v22.16b    \n"
                    "smlal  v26.8h, v18.8b,  v23.8b     \n"
                    "smlal2 v27.8h, v19.16b, v23.16b    \n"

                    "smull2 v29.8h, v16.16b, v20.16b    \n"
                    "sadalp v0.4s, v24.8h               \n"
                    "smull  v28.8h, v17.8b,  v20.8b     \n"
                    "sadalp v1.4s, v25.8h               \n"
                    "smull2 v31.8h, v16.16b, v21.16b    \n"
                    "ld1    {v16.16b}, [x5]             \n" // val L H
                    "smull  v30.8h, v17.8b,  v21.8b     \n"
                    "add    x5, x5, #32                 \n"
                    "smlal2 v29.8h, v18.16b, v22.16b    \n"
                    "sadalp v2.4s, v26.8h               \n"
                    "smlal  v28.8h, v19.8b,  v22.8b     \n"
                    "sadalp v3.4s, v27.8h               \n"
                    "smlal2 v31.8h, v18.16b, v23.16b    \n"
                    "ld1    {v18.16b}, [x5]             \n"
                    "smlal  v30.8h, v19.8b,  v23.8b     \n"
                    "ext    v17.16b, v16.16b, v16.16b, #8 \n" // val H L

                    "smull  v24.8h, v16.8b,  v20.8b     \n"
                    "add    x5, x5, #32                 \n"
                    "smull2 v25.8h, v17.16b, v20.16b    \n"
                    "prfm   pldl1keep, [x5, #128]       \n"
                    "smull  v26.8h, v16.8b,  v21.8b     \n"
                    "prfm   pldl1keep, [x5, #384]       \n"
                    "smull2 v27.8h, v17.16b, v21.16b    \n"
                    "ext    v19.16b, v18.16b, v18.16b, #8 \n" // val H L

                    "smlal  v24.8h, v18.8b,  v22.8b     \n"
                    "sadalp v5.4s, v29.8h               \n"
                    "smlal2 v25.8h, v19.16b, v22.16b    \n"
                    "sadalp v4.4s, v28.8h               \n"
                    "smlal  v26.8h, v18.8b,  v23.8b     \n"
                    "sadalp v7.4s, v31.8h               \n"
                    "smlal2 v27.8h, v19.16b, v23.16b    \n"
                    "sadalp v6.4s, v30.8h               \n"

                    "smull2 v29.8h, v16.16b, v20.16b    \n"
                    "sadalp v8.4s, v24.8h               \n"
                    "smull  v28.8h, v17.8b,  v20.8b     \n"
                    "sadalp v9.4s, v25.8h               \n"
                    "smull2 v31.8h, v16.16b, v21.16b    \n"
                    "ld1    {v16.16b}, [%4]             \n" // val L H
                    "smull  v30.8h, v17.8b,  v21.8b     \n"
                    "add    %4, %4, #32                 \n"
                    "smlal2 v29.8h, v18.16b, v22.16b    \n"
                    "sadalp v10.4s, v26.8h              \n"
                    "smlal  v28.8h, v19.8b,  v22.8b     \n"
                    "sadalp v11.4s, v27.8h              \n"
                    "smlal2 v31.8h, v18.16b, v23.16b    \n"
                    "ld1    {v18.16b}, [%4]             \n"
                    "smlal  v30.8h, v19.8b,  v23.8b     \n"
                    "add    %4, %4, #32                 \n"
                    "ld1    {v20.16b, v21.16b, v22.16b, v23.16b}, [%5], #64 \n"

                    "sadalp v13.4s, v29.8h              \n"
                    "prfm   pldl1keep, [%4, #128]       \n"
                    "sadalp v12.4s, v28.8h              \n"
                    "prfm   pldl1keep, [%4, #384]       \n"
                    "sadalp v15.4s, v31.8h              \n"
                    "ext    v17.16b, v16.16b, v16.16b, #8 \n" // val H L

                    "sadalp v14.4s, v30.8h              \n"

                    "bne    0b                          \n"

                    "sub    %4, %4, #64                 \n"
                    "sub    %5, %5, #64                 \n"

                    "1:                                 \n"
                    "and    w4, %w1, #1                 \n" // w4 = remain = nn & 1
                    "cmp    w4, #0                      \n" // w4 > 0
                    "beq    2f                          \n"

                    "ld1    {v16.8b, v17.8b}, [%4], #16 \n"
                    "ld1    {v20.8b, v21.8b, v22.8b, v23.8b}, [%5], #32 \n"

                    "smull  v24.8h, v16.8b, v20.8b      \n"
                    "smull  v25.8h, v16.8b, v21.8b      \n"
                    "smull  v26.8h, v16.8b, v22.8b      \n"
                    "ld1    {v18.8b, v19.8b}, [%4], #16 \n"
                    "smull  v27.8h, v16.8b, v23.8b      \n"
                    "sadalp v0.4s, v24.8h               \n"
                    "smull  v28.8h, v17.8b, v20.8b      \n"
                    "sadalp v1.4s, v25.8h               \n"
                    "smull  v29.8h, v17.8b, v21.8b      \n"
                    "sadalp v2.4s, v26.8h               \n"
                    "smull  v30.8h, v17.8b, v22.8b      \n"
                    "sadalp v3.4s, v27.8h               \n"
                    "smull  v31.8h, v17.8b, v23.8b      \n"
                    "sadalp v4.4s, v28.8h               \n"
                    "smull  v24.8h, v18.8b, v20.8b      \n"
                    "sadalp v5.4s, v29.8h               \n"
                    "smull  v25.8h, v18.8b, v21.8b      \n"
                    "sadalp v6.4s, v30.8h               \n"
                    "smull  v26.8h, v18.8b, v22.8b      \n"
                    "sadalp v7.4s, v31.8h               \n"
                    "smull  v27.8h, v18.8b, v23.8b      \n"
                    "sadalp v8.4s, v24.8h               \n"
                    "smull  v28.8h, v19.8b, v20.8b      \n"
                    "sadalp v9.4s, v25.8h               \n"
                    "smull  v29.8h, v19.8b, v21.8b      \n"
                    "sadalp v10.4s, v26.8h              \n"
                    "smull  v30.8h, v19.8b, v22.8b      \n"
                    "sadalp v11.4s, v27.8h              \n"
                    "smull  v31.8h, v19.8b, v23.8b      \n"

                    "sadalp v12.4s, v28.8h              \n"
                    "sadalp v13.4s, v29.8h              \n"
                    "sadalp v14.4s, v30.8h              \n"
                    "sadalp v15.4s, v31.8h              \n"

                    "2:                                 \n"

                    "addp   v0.4s, v0.4s, v1.4s         \n"
                    "addp   v2.4s, v2.4s, v3.4s         \n"
                    "addp   v4.4s, v4.4s, v5.4s         \n"
                    "addp   v6.4s, v6.4s, v7.4s         \n"
                    "addp   v8.4s, v8.4s, v9.4s         \n"
                    "addp   v10.4s, v10.4s, v11.4s      \n"
                    "addp   v12.4s, v12.4s, v13.4s      \n"
                    "addp   v14.4s, v14.4s, v15.4s      \n"

                    "addp   v0.4s, v0.4s, v2.4s         \n"
                    "addp   v1.4s, v4.4s, v6.4s         \n"
                    "addp   v2.4s, v8.4s, v10.4s        \n"
                    "addp   v3.4s, v12.4s, v14.4s       \n"

                    "3:                                 \n"

                    "cmp    %w2, #0                     \n"
                    "beq    7f                          \n"

                    "eor    v8.16b, v8.16b, v8.16b      \n"
                    "eor    v9.16b, v9.16b, v9.16b      \n"
                    "eor    v10.16b, v10.16b, v10.16b   \n"
                    "eor    v11.16b, v11.16b, v11.16b   \n"
                    "eor    v12.16b, v12.16b, v12.16b   \n"
                    "eor    v13.16b, v13.16b, v13.16b   \n"
                    "eor    v14.16b, v14.16b, v14.16b   \n"
                    "eor    v15.16b, v15.16b, v15.16b   \n"

                    "lsr    w4, %w2, #1                 \n" // w4 = nn4 >> 1
                    "cmp    w4, #0                      \n"
                    "beq    5f                          \n"

                    "4:                                 \n"

                    "ld1    {v16.8b, v17.8b}, [%4], #16 \n"
                    "ld1    {v22.8b, v23.8b}, [%5], #16 \n"

                    "zip1   v18.2s, v16.2s, v16.2s      \n" // _val00
                    "zip2   v19.2s, v16.2s, v16.2s      \n" // _val11

                    "smull  v24.8h, v18.8b, v22.8b      \n"
                    "smull  v25.8h, v18.8b, v23.8b      \n"

                    "zip1   v20.2s, v17.2s, v17.2s      \n" // _val22

                    "smull  v26.8h, v19.8b, v22.8b      \n"
                    "smull  v27.8h, v19.8b, v23.8b      \n"

                    "zip2   v21.2s, v17.2s, v17.2s      \n" // _val33

                    "smull  v28.8h, v20.8b, v22.8b      \n"
                    "smull  v29.8h, v20.8b, v23.8b      \n"

                    "ld1    {v16.8b, v17.8b}, [%4], #16 \n"

                    "smull  v30.8h, v21.8b, v22.8b      \n"
                    "smull  v31.8h, v21.8b, v23.8b      \n"

                    "ld1    {v22.8b, v23.8b}, [%5], #16 \n"

                    "zip1   v18.2s, v16.2s, v16.2s      \n" // _val44
                    "zip2   v19.2s, v16.2s, v16.2s      \n" // _val55

                    "smlal  v24.8h, v18.8b, v22.8b      \n"
                    "smlal  v25.8h, v18.8b, v23.8b      \n"

                    "zip1   v20.2s, v17.2s, v17.2s      \n" // _val66

                    "smlal  v26.8h, v19.8b, v22.8b      \n"
                    "smlal  v27.8h, v19.8b, v23.8b      \n"

                    "zip2   v21.2s, v17.2s, v17.2s      \n" // _val77

                    "sadalp v8.4s, v24.8h               \n"
                    "smlal  v28.8h, v20.8b, v22.8b      \n"
                    "sadalp v9.4s, v25.8h               \n"
                    "smlal  v29.8h, v20.8b, v23.8b      \n"
                    "sadalp v10.4s, v26.8h              \n"
                    "smlal  v30.8h, v21.8b, v22.8b      \n"
                    "sadalp v11.4s, v27.8h              \n"
                    "smlal  v31.8h, v21.8b, v23.8b      \n"
                    "sadalp v12.4s, v28.8h              \n"
                    "sadalp v13.4s, v29.8h              \n"

                    "subs   w4, w4, #1                  \n"

                    "sadalp v14.4s, v30.8h              \n"
                    "sadalp v15.4s, v31.8h              \n"

                    "bne    4b                          \n"

                    "5:                                 \n"

                    "and    w4, %w2, #1                 \n" // w4 = remain = nn4 & 1
                    "cmp    w4, #0                      \n" // w4 > 0
                    "beq    6f                          \n"

                    "ld1    {v16.8b, v17.8b}, [%4], #16 \n"
                    "ld1    {v22.8b, v23.8b}, [%5], #16 \n"

                    "zip1   v18.2s, v16.2s, v16.2s      \n" // _val00
                    "zip2   v19.2s, v16.2s, v16.2s      \n" // _val11

                    "smull  v24.8h, v18.8b, v22.8b      \n"
                    "smull  v25.8h, v18.8b, v23.8b      \n"

                    "zip1   v20.2s, v17.2s, v17.2s      \n" // _val22

                    "smull  v26.8h, v19.8b, v22.8b      \n"
                    "smull  v27.8h, v19.8b, v23.8b      \n"

                    "zip2   v21.2s, v17.2s, v17.2s      \n" // _val33

                    "sadalp v8.4s, v24.8h               \n"
                    "smull  v28.8h, v20.8b, v22.8b      \n"
                    "sadalp v9.4s, v25.8h               \n"
                    "smull  v29.8h, v20.8b, v23.8b      \n"
                    "sadalp v10.4s, v26.8h              \n"
                    "smull  v30.8h, v21.8b, v22.8b      \n"
                    "sadalp v11.4s, v27.8h              \n"
                    "smull  v31.8h, v21.8b, v23.8b      \n"
                    "sadalp v12.4s, v28.8h              \n"
                    "sadalp v13.4s, v29.8h              \n"
                    "sadalp v14.4s, v30.8h              \n"
                    "sadalp v15.4s, v31.8h              \n"

                    "6:                                 \n"

                    "addp   v8.4s, v8.4s, v9.4s         \n"
                    "addp   v10.4s, v10.4s, v11.4s      \n"
                    "addp   v12.4s, v12.4s, v13.4s      \n"
                    "addp   v14.4s, v14.4s, v15.4s      \n"

                    "add    v0.4s, v0.4s, v8.4s         \n"
                    "add    v1.4s, v1.4s, v10.4s        \n"
                    "add    v2.4s, v2.4s, v12.4s        \n"
                    "add    v3.4s, v3.4s, v14.4s        \n"

                    "7:                                 \n"

                    "lsr    w4, %w3, #2                 \n" // w4 = nn1 >> 2
                    "cmp    w4, #0                      \n"
                    "beq    9f                          \n"

                    "8:                                 \n"

                    "ld1    {v8.16b}, [%4], #16         \n"
                    "ld1    {v9.16b}, [%5], #16         \n"

                    "sshll  v4.8h, v8.8b, #0            \n"
                    "sshll2 v5.8h, v8.16b, #0           \n"
                    "sshll  v6.8h, v9.8b, #0            \n"
                    "sshll2 v7.8h, v9.16b, #0           \n"

                    "smlal  v0.4s, v6.4h, v4.h[0]       \n"
                    "smlal  v1.4s, v6.4h, v4.h[1]       \n"
                    "smlal  v2.4s, v6.4h, v4.h[2]       \n"
                    "smlal  v3.4s, v6.4h, v4.h[3]       \n"
                    "smlal2 v0.4s, v6.8h, v4.h[4]       \n"
                    "smlal2 v1.4s, v6.8h, v4.h[5]       \n"
                    "smlal2 v2.4s, v6.8h, v4.h[6]       \n"
                    "smlal2 v3.4s, v6.8h, v4.h[7]       \n"
                    "smlal  v0.4s, v7.4h, v5.h[0]       \n"
                    "smlal  v1.4s, v7.4h, v5.h[1]       \n"
                    "smlal  v2.4s, v7.4h, v5.h[2]       \n"
                    "smlal  v3.4s, v7.4h, v5.h[3]       \n"
                    "smlal2 v0.4s, v7.8h, v5.h[4]       \n"
                    "smlal2 v1.4s, v7.8h, v5.h[5]       \n"
                    "smlal2 v2.4s, v7.8h, v5.h[6]       \n"
                    "smlal2 v3.4s, v7.8h, v5.h[7]       \n"

                    "subs   w4, w4, #1                  \n"
                    "bne    8b                          \n"

                    "9:                                 \n"

                    "and    w4, %w3, #3                 \n" // w4 = nn1 & 3
                    "cmp    w4, #0                      \n" // w4 > 0
                    "beq    11f                         \n"

                    "10:                                \n"

                    "ld1    {v4.8b}, [%4]               \n"
                    "ld1    {v6.8b}, [%5]               \n"

                    "sshll  v4.8h, v4.8b, #0            \n"
                    "sshll  v6.8h, v6.8b, #0            \n"

                    "smlal  v0.4s, v6.4h, v4.h[0]       \n"
                    "smlal  v1.4s, v6.4h, v4.h[1]       \n"
                    "smlal  v2.4s, v6.4h, v4.h[2]       \n"
                    "smlal  v3.4s, v6.4h, v4.h[3]       \n"

                    "add    %4, %4, #4                  \n"
                    "add    %5, %5, #4                  \n"

                    "subs   w4, w4, #1                  \n"
                    "bne    10b                         \n"

                    "11:                                \n"

                    "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"

                    : "=r"(outptr0),
                    "=r"(nn),
                    "=r"(nn4),
                    "=r"(nn1),
                    "=r"(tmpptr),
                    "=r"(kptr0)
                    : "0"(outptr0),
                    "1"(nn),
                    "2"(nn4),
                    "3"(nn1),
                    "4"(tmpptr),
                    "5"(kptr0)
                    : "memory", "x4", "x5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    #endif // __ARM_FEATURE_DOTPROD
            }
    #endif // __aarch64__
            for (; i + 1 < size; i += 2)
            {
    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
                const signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2].data();
    #else
                const signed char* tmpptr = tmp_ra[i / 4 + (i % 4) / 2].data();
    #endif
    #else
                const signed char* tmpptr = tmp_ra[i / 2].data();
    #endif
                const signed char* kptr0 = kernel_a[p].data();

                int nn = (inch / 8) * maxk;
                int nn4 = ((inch % 8) / 4) * maxk;
                int nn1 = (inch % 4) * maxk;
    #if __aarch64__
                int32x4_t _sum00 = vdupq_n_s32(0);
                int32x4_t _sum10 = vdupq_n_s32(0);
    #if __ARM_FEATURE_DOTPROD
                for (int j = 0; j < nn; j++)
                {
                    int8x16_t _val01_l_h = vld1q_s8(tmpptr);
                    int8x16_t _w0123_l = vld1q_s8(kptr0);

                    _sum00 = vdotq_laneq_s32(_sum00, _w0123_l, _val01_l_h, 0);
                    _sum10 = vdotq_laneq_s32(_sum10, _w0123_l, _val01_l_h, 1);

                    int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

                    _sum00 = vdotq_laneq_s32(_sum00, _w0123_h, _val01_l_h, 2);
                    _sum10 = vdotq_laneq_s32(_sum10, _w0123_h, _val01_l_h, 3);

                    tmpptr += 16;
                    kptr0 += 32;
                }

                if (nn4 > 0)
                {
                    int j = 0;
                    for (; j + 1 < nn4; j += 2)
                    {
                        int8x16_t _val0123 = vld1q_s8(tmpptr);
                        int8x16_t _w0 = vld1q_s8(kptr0);

                        _sum00 = vdotq_laneq_s32(_sum00, _w0, _val0123, 0);
                        _sum10 = vdotq_laneq_s32(_sum10, _w0, _val0123, 1);

                        int8x16_t _w1 = vld1q_s8(kptr0 + 16);

                        _sum00 = vdotq_laneq_s32(_sum00, _w1, _val0123, 2);
                        _sum10 = vdotq_laneq_s32(_sum10, _w1, _val0123, 3);

                        tmpptr += 16;
                        kptr0 += 32;
                    }
                    for (; j < nn4; j++)
                    {
                        int8x8_t _val01 = vld1_s8(tmpptr);
                        int8x16_t _w0 = vld1q_s8(kptr0);

                        _sum00 = vdotq_lane_s32(_sum00, _w0, _val01, 0);
                        _sum10 = vdotq_lane_s32(_sum10, _w0, _val01, 1);

                        tmpptr += 8;
                        kptr0 += 16;
                    }
                }
    #else  // __ARM_FEATURE_DOTPROD
                if (nn > 0)
                {
                    int32x4_t _sum01 = vdupq_n_s32(0);
                    int32x4_t _sum02 = vdupq_n_s32(0);
                    int32x4_t _sum03 = vdupq_n_s32(0);
                    int32x4_t _sum11 = vdupq_n_s32(0);
                    int32x4_t _sum12 = vdupq_n_s32(0);
                    int32x4_t _sum13 = vdupq_n_s32(0);

                    int j = 0;
                    for (; j + 1 < nn; j += 2)
                    {
                        int8x16_t _val0 = vld1q_s8(tmpptr);
                        int8x16_t _val1 = vld1q_s8(tmpptr + 16);

                        int8x16_t _w01 = vld1q_s8(kptr0);
                        int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                        int16x8_t _wv00 = vmull_s8(vget_low_s8(_val0), vget_low_s8(_w01));
                        int16x8_t _wv01 = vmull_s8(vget_low_s8(_val0), vget_high_s8(_w01));
                        int16x8_t _wv02 = vmull_s8(vget_low_s8(_val0), vget_low_s8(_w23));
                        int16x8_t _wv03 = vmull_s8(vget_low_s8(_val0), vget_high_s8(_w23));

                        int16x8_t _wv10 = vmull_s8(vget_high_s8(_val0), vget_low_s8(_w01));
                        int16x8_t _wv11 = vmull_s8(vget_high_s8(_val0), vget_high_s8(_w01));
                        int16x8_t _wv12 = vmull_s8(vget_high_s8(_val0), vget_low_s8(_w23));
                        int16x8_t _wv13 = vmull_s8(vget_high_s8(_val0), vget_high_s8(_w23));

                        int8x16_t _w45 = vld1q_s8(kptr0 + 32);
                        int8x16_t _w67 = vld1q_s8(kptr0 + 48);

                        _wv00 = vmlal_s8(_wv00, vget_low_s8(_val1), vget_low_s8(_w45));
                        _wv01 = vmlal_s8(_wv01, vget_low_s8(_val1), vget_high_s8(_w45));
                        _wv02 = vmlal_s8(_wv02, vget_low_s8(_val1), vget_low_s8(_w67));
                        _wv03 = vmlal_s8(_wv03, vget_low_s8(_val1), vget_high_s8(_w67));

                        _wv10 = vmlal_s8(_wv10, vget_high_s8(_val1), vget_low_s8(_w45));
                        _wv11 = vmlal_s8(_wv11, vget_high_s8(_val1), vget_high_s8(_w45));
                        _wv12 = vmlal_s8(_wv12, vget_high_s8(_val1), vget_low_s8(_w67));
                        _wv13 = vmlal_s8(_wv13, vget_high_s8(_val1), vget_high_s8(_w67));

                        _sum00 = vpadalq_s16(_sum00, _wv00);
                        _sum01 = vpadalq_s16(_sum01, _wv01);
                        _sum02 = vpadalq_s16(_sum02, _wv02);
                        _sum03 = vpadalq_s16(_sum03, _wv03);
                        _sum10 = vpadalq_s16(_sum10, _wv10);
                        _sum11 = vpadalq_s16(_sum11, _wv11);
                        _sum12 = vpadalq_s16(_sum12, _wv12);
                        _sum13 = vpadalq_s16(_sum13, _wv13);

                        tmpptr += 32;
                        kptr0 += 64;
                    }
                    for (; j < nn; j++)
                    {
                        int8x16_t _val = vld1q_s8(tmpptr);

                        int8x16_t _w01 = vld1q_s8(kptr0);
                        int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                        int16x8_t _wv00 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w01));
                        int16x8_t _wv01 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w01));
                        int16x8_t _wv02 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w23));
                        int16x8_t _wv03 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w23));
                        int16x8_t _wv10 = vmull_s8(vget_high_s8(_val), vget_low_s8(_w01));
                        int16x8_t _wv11 = vmull_s8(vget_high_s8(_val), vget_high_s8(_w01));
                        int16x8_t _wv12 = vmull_s8(vget_high_s8(_val), vget_low_s8(_w23));
                        int16x8_t _wv13 = vmull_s8(vget_high_s8(_val), vget_high_s8(_w23));

                        _sum00 = vpadalq_s16(_sum00, _wv00);
                        _sum01 = vpadalq_s16(_sum01, _wv01);
                        _sum02 = vpadalq_s16(_sum02, _wv02);
                        _sum03 = vpadalq_s16(_sum03, _wv03);
                        _sum10 = vpadalq_s16(_sum10, _wv10);
                        _sum11 = vpadalq_s16(_sum11, _wv11);
                        _sum12 = vpadalq_s16(_sum12, _wv12);
                        _sum13 = vpadalq_s16(_sum13, _wv13);

                        tmpptr += 16;
                        kptr0 += 32;
                    }

                    int32x4_t _s001 = vpaddq_s32(_sum00, _sum01);
                    int32x4_t _s023 = vpaddq_s32(_sum02, _sum03);
                    int32x4_t _s101 = vpaddq_s32(_sum10, _sum11);
                    int32x4_t _s123 = vpaddq_s32(_sum12, _sum13);

                    _sum00 = vpaddq_s32(_s001, _s023);
                    _sum10 = vpaddq_s32(_s101, _s123);
                }

                if (nn4 > 0)
                {
                    int32x4_t _sum100 = vdupq_n_s32(0);
                    int32x4_t _sum101 = vdupq_n_s32(0);
                    int32x4_t _sum110 = vdupq_n_s32(0);
                    int32x4_t _sum111 = vdupq_n_s32(0);

                    int j = 0;
                    for (; j + 1 < nn4; j += 2)
                    {
                        int8x16_t _val0123 = vld1q_s8(tmpptr);

                        int32x4x2_t _val00221133 = vzipq_s32(vreinterpretq_s32_s8(_val0123), vreinterpretq_s32_s8(_val0123));
                        int8x8_t _val00 = vreinterpret_s8_s32(vget_low_s32(_val00221133.val[0]));
                        int8x8_t _val11 = vreinterpret_s8_s32(vget_high_s32(_val00221133.val[0]));
                        int8x8_t _val22 = vreinterpret_s8_s32(vget_low_s32(_val00221133.val[1]));
                        int8x8_t _val33 = vreinterpret_s8_s32(vget_high_s32(_val00221133.val[1]));

                        int8x16_t _w01 = vld1q_s8(kptr0);
                        int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                        int16x8_t _wv00 = vmull_s8(_val00, vget_low_s8(_w01));
                        int16x8_t _wv01 = vmull_s8(_val00, vget_high_s8(_w01));
                        int16x8_t _wv10 = vmull_s8(_val11, vget_low_s8(_w01));
                        int16x8_t _wv11 = vmull_s8(_val11, vget_high_s8(_w01));

                        _wv00 = vmlal_s8(_wv00, _val22, vget_low_s8(_w23));
                        _wv01 = vmlal_s8(_wv01, _val22, vget_high_s8(_w23));
                        _wv10 = vmlal_s8(_wv10, _val33, vget_low_s8(_w23));
                        _wv11 = vmlal_s8(_wv11, _val33, vget_high_s8(_w23));

                        _sum100 = vpadalq_s16(_sum100, _wv00);
                        _sum101 = vpadalq_s16(_sum101, _wv01);
                        _sum110 = vpadalq_s16(_sum110, _wv10);
                        _sum111 = vpadalq_s16(_sum111, _wv11);

                        tmpptr += 16;
                        kptr0 += 32;
                    }
                    for (; j < nn4; j++)
                    {
                        int8x8_t _val01 = vld1_s8(tmpptr);
                        int32x2x2_t _val0011 = vzip_s32(vreinterpret_s32_s8(_val01), vreinterpret_s32_s8(_val01));
                        int8x8_t _val00 = vreinterpret_s8_s32(_val0011.val[0]);
                        int8x8_t _val11 = vreinterpret_s8_s32(_val0011.val[1]);

                        int8x16_t _w01 = vld1q_s8(kptr0);

                        int16x8_t _wv00 = vmull_s8(_val00, vget_low_s8(_w01));
                        int16x8_t _wv01 = vmull_s8(_val00, vget_high_s8(_w01));
                        int16x8_t _wv10 = vmull_s8(_val11, vget_low_s8(_w01));
                        int16x8_t _wv11 = vmull_s8(_val11, vget_high_s8(_w01));

                        _sum100 = vpadalq_s16(_sum100, _wv00);
                        _sum101 = vpadalq_s16(_sum101, _wv01);
                        _sum110 = vpadalq_s16(_sum110, _wv10);
                        _sum111 = vpadalq_s16(_sum111, _wv11);

                        tmpptr += 8;
                        kptr0 += 16;
                    }

                    int32x4_t _s001 = vpaddq_s32(_sum100, _sum101);
                    int32x4_t _s101 = vpaddq_s32(_sum110, _sum111);

                    _sum00 = vaddq_s32(_sum00, _s001);
                    _sum10 = vaddq_s32(_sum10, _s101);
                }
    #endif // __ARM_FEATURE_DOTPROD

                int j = 0;
                for (; j + 3 < nn1; j += 4)
                {
                    int16x8_t _val01234567 = vmovl_s8(vld1_s8(tmpptr));

                    int8x16_t _w = vld1q_s8(kptr0);
                    int16x8_t _w01234567 = vmovl_s8(vget_low_s8(_w));
                    int16x8_t _w89abcdef = vmovl_s8(vget_high_s8(_w));
                    int16x4_t _w0123 = vget_low_s16(_w01234567);
                    int16x4_t _w4567 = vget_high_s16(_w01234567);
                    int16x4_t _w89ab = vget_low_s16(_w89abcdef);
                    int16x4_t _wcdef = vget_high_s16(_w89abcdef);

                    _sum00 = vmlal_laneq_s16(_sum00, _w0123, _val01234567, 0);
                    _sum10 = vmlal_laneq_s16(_sum10, _w0123, _val01234567, 1);
                    _sum00 = vmlal_laneq_s16(_sum00, _w4567, _val01234567, 2);
                    _sum10 = vmlal_laneq_s16(_sum10, _w4567, _val01234567, 3);
                    _sum00 = vmlal_laneq_s16(_sum00, _w89ab, _val01234567, 4);
                    _sum10 = vmlal_laneq_s16(_sum10, _w89ab, _val01234567, 5);
                    _sum00 = vmlal_laneq_s16(_sum00, _wcdef, _val01234567, 6);
                    _sum10 = vmlal_laneq_s16(_sum10, _wcdef, _val01234567, 7);

                    tmpptr += 8;
                    kptr0 += 16;
                }
                for (; j < nn1; j++)
                {
                    int16x4_t _val0 = vdup_n_s16(tmpptr[0]);
                    int16x4_t _val1 = vdup_n_s16(tmpptr[1]);

                    int16x4_t _w0123;
                    _w0123 = vset_lane_s16(kptr0[0], _w0123, 0);
                    _w0123 = vset_lane_s16(kptr0[1], _w0123, 1);
                    _w0123 = vset_lane_s16(kptr0[2], _w0123, 2);
                    _w0123 = vset_lane_s16(kptr0[3], _w0123, 3);

                    _sum00 = vmlal_s16(_sum00, _val0, _w0123);
                    _sum10 = vmlal_s16(_sum10, _val1, _w0123);

                    tmpptr += 2;
                    kptr0 += 4;
                }

                vst1q_s32(outptr0, _sum00);
                vst1q_s32(outptr0 + 4, _sum10);
                outptr0 += 8;
    #else  // __aarch64__
                asm volatile(
                    "veor       q0, q0              \n"
                    "veor       q1, q1              \n"
                    "veor       q2, q2              \n"
                    "veor       q3, q3              \n"
                    "veor       q4, q4              \n"
                    "veor       q5, q5              \n"
                    "veor       q6, q6              \n"
                    "veor       q7, q7              \n"

                    "cmp        %1, #0              \n"
                    "beq        3f                  \n"

                    "pld        [%4, #256]          \n"

                    "lsr        r4, %1, #1          \n" // r4 = nn = size >> 1
                    "cmp        r4, #0              \n"
                    "beq        1f                  \n"

                    "add        r5, %5, #16         \n"
                    "pld        [%5, #128]          \n"
                    "mov        r6, #32             \n"
                    "pld        [%5, #384]          \n"

                    "vld1.s8    {d20-d21}, [%5 :128], r6 \n" // _w01

                    "vld1.s8    {d16-d19}, [%4 :128]! \n" // _val0 _val1

                    "vld1.s8    {d22-d23}, [%5 :128], r6 \n" // _w45

                    "0:                             \n"

                    "vmull.s8   q12, d16, d20       \n"
                    "pld        [%4, #256]          \n"
                    "vmull.s8   q13, d16, d21       \n"
                    "pld        [%5, #384]          \n"
                    "vmull.s8   q14, d17, d20       \n"
                    "vmull.s8   q15, d17, d21       \n"
                    "vld1.s8    {d20-d21}, [r5 :128], r6 \n" // _w23

                    "vmlal.s8   q12, d18, d22       \n"
                    "vmlal.s8   q13, d18, d23       \n"
                    "subs       r4, r4, #1          \n"
                    "vmlal.s8   q14, d19, d22       \n"
                    "vmlal.s8   q15, d19, d23       \n"
                    "vld1.s8    {d22-d23}, [r5 :128], r6 \n" // _w67

                    "vpadal.s16 q0, q12             \n"
                    "vmull.s8   q12, d16, d20       \n"
                    "vpadal.s16 q1, q13             \n"
                    "vmull.s8   q13, d16, d21       \n"
                    "vpadal.s16 q4, q14             \n"
                    "vmull.s8   q14, d17, d20       \n"
                    "vpadal.s16 q5, q15             \n"
                    "vmull.s8   q15, d17, d21       \n"
                    "vld1.s8    {d16-d17}, [%4 :128]! \n" // _val0

                    "vmlal.s8   q12, d18, d22       \n"
                    "vld1.s8    {d20-d21}, [%5 :128], r6 \n" // _w01
                    "vmlal.s8   q13, d18, d23       \n"
                    "pld        [r5, #128]          \n"
                    "vmlal.s8   q14, d19, d22       \n"
                    "pld        [r5, #384]          \n"
                    "vmlal.s8   q15, d19, d23       \n"
                    "vld1.s8    {d18-d19}, [%4 :128]! \n" // _val1

                    "vpadal.s16 q2, q12             \n"
                    "vld1.s8    {d22-d23}, [%5 :128], r6 \n" // _w45
                    "vpadal.s16 q3, q13             \n"
                    "pld        [%4, #128]          \n"
                    "vpadal.s16 q6, q14             \n"
                    "pld        [%5, #128]          \n"
                    "vpadal.s16 q7, q15             \n"

                    "bne        0b                  \n"

                    "sub        %4, %4, #32         \n"
                    "sub        %5, %5, #64         \n"

                    "1:                             \n"
                    "and        r4, %1, #1          \n" // r4 = remain = size & 1
                    "cmp        r4, #0              \n" // r4 > 0
                    "beq        2f                  \n"

                    "vld1.s8    {d16-d17}, [%4 :128]! \n" // _val
                    "vld1.s8    {d20-d21}, [%5 :128]! \n" // _w01

                    "vmull.s8   q12, d16, d20       \n"

                    "vld1.s8    {d22-d23}, [%5 :128]! \n" // _w23
                    "vmull.s8   q13, d16, d21       \n"
                    "vmull.s8   q14, d17, d20       \n"
                    "vmull.s8   q15, d17, d21       \n"

                    "vpadal.s16 q0, q12             \n"
                    "vmull.s8   q12, d16, d22       \n"
                    "vpadal.s16 q1, q13             \n"
                    "vmull.s8   q13, d16, d23       \n"
                    "vpadal.s16 q4, q14             \n"
                    "vmull.s8   q14, d17, d22       \n"
                    "vpadal.s16 q5, q15             \n"
                    "vmull.s8   q15, d17, d23       \n"

                    "vpadal.s16 q2, q12             \n"
                    "vpadal.s16 q3, q13             \n"
                    "vpadal.s16 q6, q14             \n"
                    "vpadal.s16 q7, q15             \n"

                    "2:                             \n"

                    "vpadd.s32  d16, d0, d1         \n"
                    "vpadd.s32  d17, d2, d3         \n"
                    "vpadd.s32  d18, d4, d5         \n"
                    "vpadd.s32  d19, d6, d7         \n"
                    "vpadd.s32  d20, d8, d9         \n"
                    "vpadd.s32  d21, d10, d11       \n"
                    "vpadd.s32  d22, d12, d13       \n"
                    "vpadd.s32  d23, d14, d15       \n"

                    "vpadd.s32  d0, d16, d17        \n"
                    "vpadd.s32  d1, d18, d19        \n"
                    "vpadd.s32  d2, d20, d21        \n"
                    "vpadd.s32  d3, d22, d23        \n"

                    "3:                             \n"

                    "cmp        %2, #0              \n"
                    "beq        7f                  \n"

                    "veor       q2, q2              \n"
                    "veor       q3, q3              \n"
                    "veor       q4, q4              \n"
                    "veor       q5, q5              \n"

                    "lsr        r4, %2, #1          \n" // r4 = nn4 >> 1
                    "cmp        r4, #0              \n"
                    "beq        5f                  \n"

                    "4:                             \n"

                    "vld1.s8    {d16-d17}, [%4]!    \n" // _val0123
                    "vld1.s8    {d20-d23}, [%5]!    \n" // _w01 _w23

                    "vmov.s8    q9, q8              \n"
                    "vtrn.s32   q8, q9              \n" // _val00 _val22 _val11 _val33

                    "vmull.s8   q12, d16, d20       \n"
                    "vmull.s8   q13, d16, d21       \n"
                    "vmull.s8   q14, d18, d20       \n"
                    "vmull.s8   q15, d18, d21       \n"

                    "vmlal.s8   q12, d17, d22       \n"
                    "vmlal.s8   q13, d17, d23       \n"
                    "vmlal.s8   q14, d19, d22       \n"
                    "vmlal.s8   q15, d19, d23       \n"

                    "vpadal.s16 q2, q12             \n"
                    "vpadal.s16 q3, q13             \n"
                    "vpadal.s16 q4, q14             \n"
                    "vpadal.s16 q5, q15             \n"

                    "subs       r4, r4, #1          \n"
                    "bne        4b                  \n"

                    "5:                             \n"

                    "and        r4, %2, #1          \n" // r4 = nn4 & 1
                    "cmp        r4, #0              \n" // r4 > 0
                    "beq        6f                  \n"

                    "vld1.s8    {d16}, [%4]!        \n" // _val01
                    "vld1.s8    {d18-d19}, [%5]!    \n" // _w01

                    "vmov.s8    d17, d16            \n"
                    "vtrn.s32   d16, d17            \n" // _val00 _val11

                    "vmull.s8   q12, d16, d18       \n"
                    "vmull.s8   q13, d16, d19       \n"
                    "vmull.s8   q14, d17, d18       \n"
                    "vmull.s8   q15, d17, d19       \n"

                    "vpadal.s16 q2, q12             \n"
                    "vpadal.s16 q3, q13             \n"
                    "vpadal.s16 q4, q14             \n"
                    "vpadal.s16 q5, q15             \n"

                    "6:                             \n"

                    "vpadd.s32  d16, d4, d5         \n"
                    "vpadd.s32  d17, d6, d7         \n"
                    "vpadd.s32  d18, d8, d9         \n"
                    "vpadd.s32  d19, d10, d11       \n"

                    "vadd.s32   q0, q0, q8          \n"
                    "vadd.s32   q1, q1, q9          \n"

                    "7:                             \n"

                    "lsr        r4, %3, #2          \n" // r4 = nn1 >> 2
                    "cmp        r4, #0              \n"
                    "beq        9f                  \n"

                    "8:                             \n"

                    "vld1.s8    {d4}, [%4]!         \n"
                    "vmovl.s8   q2, d4              \n"

                    "vld1.s8    {d10-d11}, [%5]!    \n"
                    "vmovl.s8   q3, d10             \n"
                    "vmovl.s8   q4, d11             \n"

                    "vmlal.s16  q0, d6, d4[0]       \n"
                    "vmlal.s16  q1, d6, d4[1]       \n"
                    "vmlal.s16  q0, d7, d4[2]       \n"
                    "vmlal.s16  q1, d7, d4[3]       \n"
                    "vmlal.s16  q0, d8, d5[0]       \n"
                    "vmlal.s16  q1, d8, d5[1]       \n"
                    "vmlal.s16  q0, d9, d5[2]       \n"
                    "vmlal.s16  q1, d9, d5[3]       \n"

                    "subs       r4, r4, #1          \n"
                    "bne        8b                  \n"

                    "9:                             \n"

                    "and        r4, %3, #3          \n" // r4 = nn1 & 3
                    "cmp        r4, #0              \n" // w4 > 0
                    "beq        11f                 \n"

                    "10:                            \n"

                    "vld1.s8    {d4[]}, [%4]!       \n"
                    "vld1.s8    {d6[]}, [%4]!       \n"
                    "vmovl.s8   q2, d4              \n"
                    "vmovl.s8   q3, d6              \n"

                    "vld1.s8    {d8}, [%5]          \n"
                    "vmovl.s8   q4, d8              \n"

                    "vmlal.s16  q0, d4, d8          \n"
                    "vmlal.s16  q1, d6, d8          \n"

                    "add        %5, %5, #4          \n"

                    "subs       r4, r4, #1          \n"
                    "bne        10b                 \n"

                    "11:                            \n"

                    "vst1.s32   {d0-d3}, [%0 :128]! \n"

                    : "=r"(outptr0),
                    "=r"(nn),
                    "=r"(nn4),
                    "=r"(nn1),
                    "=r"(tmpptr),
                    "=r"(kptr0)
                    : "0"(outptr0),
                    "1"(nn),
                    "2"(nn4),
                    "3"(nn1),
                    "4"(tmpptr),
                    "5"(kptr0)
                    : "memory", "r4", "r5", "r6", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
    #endif // __aarch64__
            }
            for (; i < size; i++)
            {
    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
                const signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2].data();
    #else
                const signed char* tmpptr = tmp_ra[i / 4 + (i % 4) / 2 + i % 2].data();
    #endif
    #else
                const signed char* tmpptr = tmp_ra[i / 2 + i % 2].data();
    #endif
                const signed char* kptr0 = kernel_a[p].data();

                int nn = (inch / 8) * maxk;
                int nn4 = ((inch % 8) / 4) * maxk;
                int nn1 = (inch % 4) * maxk;

                int32x4_t _sum0 = vdupq_n_s32(0);
    #if __ARM_FEATURE_DOTPROD
                for (int j = 0; j < nn; j++)
                {
                    int8x8_t _val0_l_h = vld1_s8(tmpptr);

                    int8x16_t _w0123_l = vld1q_s8(kptr0);

                    _sum0 = vdotq_lane_s32(_sum0, _w0123_l, _val0_l_h, 0);

                    int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

                    _sum0 = vdotq_lane_s32(_sum0, _w0123_h, _val0_l_h, 1);

                    tmpptr += 8;
                    kptr0 += 32;
                }

                if (nn4 > 0)
                {
                    int j = 0;
                    for (; j + 1 < nn4; j += 2)
                    {
                        int8x8_t _val01 = vld1_s8(tmpptr);

                        int8x16_t _w0 = vld1q_s8(kptr0);

                        _sum0 = vdotq_lane_s32(_sum0, _w0, _val01, 0);

                        int8x16_t _w1 = vld1q_s8(kptr0 + 16);

                        _sum0 = vdotq_lane_s32(_sum0, _w1, _val01, 1);

                        tmpptr += 8;
                        kptr0 += 32;
                    }
                    for (; j < nn4; j++)
                    {
                        int8x8_t _val_xxx = vld1_s8(tmpptr);

                        int8x16_t _w0 = vld1q_s8(kptr0);

                        _sum0 = vdotq_lane_s32(_sum0, _w0, _val_xxx, 0);

                        tmpptr += 4;
                        kptr0 += 16;
                    }
                }
    #else // __ARM_FEATURE_DOTPROD
                if (nn > 0)
                {
                    int32x4_t _sum1 = vdupq_n_s32(0);
                    int32x4_t _sum2 = vdupq_n_s32(0);
                    int32x4_t _sum3 = vdupq_n_s32(0);

                    int j = 0;
                    for (; j + 1 < nn; j += 2)
                    {
                        int8x16_t _val = vld1q_s8(tmpptr);

                        int8x16_t _w01 = vld1q_s8(kptr0);
                        int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                        int16x8_t _wv0 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w01));
                        int16x8_t _wv1 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w01));
                        int16x8_t _wv2 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w23));
                        int16x8_t _wv3 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w23));

                        int8x16_t _w45 = vld1q_s8(kptr0 + 32);
                        int8x16_t _w67 = vld1q_s8(kptr0 + 48);

                        _wv0 = vmlal_s8(_wv0, vget_high_s8(_val), vget_low_s8(_w45));
                        _wv1 = vmlal_s8(_wv1, vget_high_s8(_val), vget_high_s8(_w45));
                        _wv2 = vmlal_s8(_wv2, vget_high_s8(_val), vget_low_s8(_w67));
                        _wv3 = vmlal_s8(_wv3, vget_high_s8(_val), vget_high_s8(_w67));

                        _sum0 = vpadalq_s16(_sum0, _wv0);
                        _sum1 = vpadalq_s16(_sum1, _wv1);
                        _sum2 = vpadalq_s16(_sum2, _wv2);
                        _sum3 = vpadalq_s16(_sum3, _wv3);

                        tmpptr += 16;
                        kptr0 += 64;
                    }
                    for (; j < nn; j++)
                    {
                        int8x8_t _val = vld1_s8(tmpptr);

                        int8x16_t _w01 = vld1q_s8(kptr0);
                        int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                        int16x8_t _wv0 = vmull_s8(_val, vget_low_s8(_w01));
                        int16x8_t _wv1 = vmull_s8(_val, vget_high_s8(_w01));
                        int16x8_t _wv2 = vmull_s8(_val, vget_low_s8(_w23));
                        int16x8_t _wv3 = vmull_s8(_val, vget_high_s8(_w23));

                        _sum0 = vpadalq_s16(_sum0, _wv0);
                        _sum1 = vpadalq_s16(_sum1, _wv1);
                        _sum2 = vpadalq_s16(_sum2, _wv2);
                        _sum3 = vpadalq_s16(_sum3, _wv3);

                        tmpptr += 8;
                        kptr0 += 32;
                    }

    #if __aarch64__
                    int32x4_t _s01 = vpaddq_s32(_sum0, _sum1);
                    int32x4_t _s23 = vpaddq_s32(_sum2, _sum3);

                    _sum0 = vpaddq_s32(_s01, _s23);
    #else
                    int32x2_t _s01_low = vpadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                    int32x2_t _s01_high = vpadd_s32(vget_low_s32(_sum1), vget_high_s32(_sum1));
                    int32x2_t _s23_low = vpadd_s32(vget_low_s32(_sum2), vget_high_s32(_sum2));
                    int32x2_t _s23_high = vpadd_s32(vget_low_s32(_sum3), vget_high_s32(_sum3));

                    _sum0 = vcombine_s32(vpadd_s32(_s01_low, _s01_high), vpadd_s32(_s23_low, _s23_high));
    #endif
                }

                if (nn4 > 0)
                {
                    int32x4_t _sum10 = vdupq_n_s32(0);
                    int32x4_t _sum11 = vdupq_n_s32(0);

                    int j = 0;
                    for (; j + 1 < nn4; j += 2)
                    {
                        int8x8_t _val01 = vld1_s8(tmpptr);
                        int32x2x2_t _val0011 = vzip_s32(vreinterpret_s32_s8(_val01), vreinterpret_s32_s8(_val01));
                        int8x8_t _val00 = vreinterpret_s8_s32(_val0011.val[0]);
                        int8x8_t _val11 = vreinterpret_s8_s32(_val0011.val[1]);

                        int8x16_t _w0 = vld1q_s8(kptr0);
                        int8x16_t _w1 = vld1q_s8(kptr0 + 16);

                        int16x8_t _wv0 = vmull_s8(_val00, vget_low_s8(_w0));
                        int16x8_t _wv1 = vmull_s8(_val00, vget_high_s8(_w0));

                        _wv0 = vmlal_s8(_wv0, _val11, vget_low_s8(_w1));
                        _wv1 = vmlal_s8(_wv1, _val11, vget_high_s8(_w1));

                        _sum10 = vpadalq_s16(_sum10, _wv0);
                        _sum11 = vpadalq_s16(_sum11, _wv1);

                        tmpptr += 8;
                        kptr0 += 32;
                    }
                    for (; j < nn4; j++)
                    {
                        int8x8_t _val_xxx = vld1_s8(tmpptr);
                        int8x8_t _val_val = vreinterpret_s8_s32(vzip_s32(vreinterpret_s32_s8(_val_xxx), vreinterpret_s32_s8(_val_xxx)).val[0]);

                        int8x16_t _w0 = vld1q_s8(kptr0);

                        int16x8_t _wv0 = vmull_s8(_val_val, vget_low_s8(_w0));
                        int16x8_t _wv1 = vmull_s8(_val_val, vget_high_s8(_w0));

                        _sum10 = vpadalq_s16(_sum10, _wv0);
                        _sum11 = vpadalq_s16(_sum11, _wv1);

                        tmpptr += 4;
                        kptr0 += 16;
                    }

    #if __aarch64__
                    int32x4_t _s01 = vpaddq_s32(_sum10, _sum11);
    #else
                    int32x2_t _s01_low = vpadd_s32(vget_low_s32(_sum10), vget_high_s32(_sum10));
                    int32x2_t _s01_high = vpadd_s32(vget_low_s32(_sum11), vget_high_s32(_sum11));

                    int32x4_t _s01 = vcombine_s32(_s01_low, _s01_high);
    #endif

                    _sum0 = vaddq_s32(_sum0, _s01);
                }
    #endif // __ARM_FEATURE_DOTPROD

                int32x4_t _sum1 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 3 < nn1; j += 4)
                {
                    int16x4_t _val0123 = vget_low_s16(vmovl_s8(vld1_s8(tmpptr)));

                    int8x16_t _w = vld1q_s8(kptr0);
                    int16x8_t _w01234567 = vmovl_s8(vget_low_s8(_w));
                    int16x8_t _w89abcdef = vmovl_s8(vget_high_s8(_w));
                    int16x4_t _w0123 = vget_low_s16(_w01234567);
                    int16x4_t _w4567 = vget_high_s16(_w01234567);
                    int16x4_t _w89ab = vget_low_s16(_w89abcdef);
                    int16x4_t _wcdef = vget_high_s16(_w89abcdef);

                    _sum0 = vmlal_lane_s16(_sum0, _w0123, _val0123, 0);
                    _sum1 = vmlal_lane_s16(_sum1, _w4567, _val0123, 1);
                    _sum0 = vmlal_lane_s16(_sum0, _w89ab, _val0123, 2);
                    _sum1 = vmlal_lane_s16(_sum1, _wcdef, _val0123, 3);

                    tmpptr += 4;
                    kptr0 += 16;
                }
                for (; j < nn1; j++)
                {
                    int16x4_t _val = vdup_n_s16(tmpptr[0]);

                    int16x4_t _w0123;
                    _w0123 = vset_lane_s16(kptr0[0], _w0123, 0);
                    _w0123 = vset_lane_s16(kptr0[1], _w0123, 1);
                    _w0123 = vset_lane_s16(kptr0[2], _w0123, 2);
                    _w0123 = vset_lane_s16(kptr0[3], _w0123, 3);

                    _sum0 = vmlal_s16(_sum0, _val, _w0123);

                    tmpptr += 1;
                    kptr0 += 4;
                }

                _sum0 = vaddq_s32(_sum0, _sum1);

                vst1q_s32(outptr0, _sum0);
                outptr0 += 4;
            }
        }
    });
}

void im2col_sgemm_conv2d_int8_pack8to1_impl_neon(
    const Tensor& im2col,
    const Tensor& kernel_tf_,
    const Tensor& bias_,
    int64_t input_channels,
    int64_t output_channels,
    Tensor& output) {
    
    const int size = im2col.size(2);
    const int maxk = im2col.size(1);
    const int inch = im2col.size(0);

    const int outch = output.size(1);
    
    auto output_a = output.accessor<int, 4>()[0];
    auto im2col_a = im2col.accessor<signed char, 3, 8>();
    auto kernel_a = kernel_tf_.accessor<signed char, 3>();

    // permute
    Tensor tmp;
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
    if (size >= 16)
        tmp = otter::empty({size / 16 + (size % 16) / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, inch, 16 * maxk}, otter::ScalarType::Byte8);
    else if (size >= 8)
        tmp = otter::empty({size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, inch, 8 * maxk}, otter::ScalarType::Byte8);
    else if (size >= 4)
        tmp = otter::empty({size / 4 + (size % 4) / 2 + size % 2, inch, 4 * maxk}, otter::ScalarType::Byte8);
    else if (size >= 2)
        tmp = otter::empty({size / 2 + size % 2, inch, 2 * maxk}, otter::ScalarType::Byte8);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Byte8);
#else  // __ARM_FEATURE_DOTPROD
    if (size >= 4)
        tmp = otter::empty({size / 4 + (size % 4) / 2 + size % 2, inch, 4 * maxk}, otter::ScalarType::Byte8);
    else if (size >= 2)
        tmp = otter::empty({size / 2 + size % 2, inch, 2 * maxk}, otter::ScalarType::Byte8);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Byte8);
#endif // __ARM_FEATURE_DOTPROD
#else  // __aarch64__
    if (size >= 2)
        tmp = otter::empty({size / 2 + size % 2, inch, 2 * maxk}, otter::ScalarType::Byte8);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Byte8);
#endif // __aarch64__
    
    auto tmp_ra = tmp.raw_accessor<signed char, 3>();
    
    {
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
        int nn_size = size >> 4;
        int remain_size_start = 0;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end))
            {
                int i = remain_size_start + ii * 16;

                signed char* tmpptr = tmp_ra[i / 16].data();

                for (int q = 0; q < inch; q++)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++)
                    {
                        // split pack8to1 to pack4
                        asm volatile(
                            "prfm   pldl1keep, [%0, #512]       \n"
                            "ld2    {v0.4s, v1.4s}, [%0], #32   \n"
                            "ld2    {v2.4s, v3.4s}, [%0], #32   \n"
                            "ld2    {v4.4s, v5.4s}, [%0], #32   \n"
                            "ld2    {v6.4s, v7.4s}, [%0]        \n"
                            "sub    %0, %0, #96                 \n"
                            "st1    {v0.16b}, [%1], #16         \n"
                            "st1    {v2.16b}, [%1], #16         \n"
                            "st1    {v4.16b}, [%1], #16         \n"
                            "st1    {v6.16b}, [%1], #16         \n"
                            "st1    {v1.16b}, [%1], #16         \n"
                            "st1    {v3.16b}, [%1], #16         \n"
                            "st1    {v5.16b}, [%1], #16         \n"
                            "st1    {v7.16b}, [%1], #16         \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
                        img0 += size * 8;
                    }
                }
            }
        });

        remain_size_start += nn_size << 4;
        nn_size = (size - remain_size_start) >> 3;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end))
            {
                int i = remain_size_start + ii * 8;

                signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8].data();

                for (int q = 0; q < inch; q++)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++)
                    {
                        asm volatile(
                            "prfm   pldl1keep, [%0, #512]       \n"
                            "ld2    {v0.4s, v1.4s}, [%0], #32   \n"
                            "ld2    {v2.4s, v3.4s}, [%0]        \n"
                            "sub    %0, %0, #32                 \n"
                            "st1    {v0.16b}, [%1], #16         \n"
                            "st1    {v2.16b}, [%1], #16         \n"
                            "st1    {v1.16b}, [%1], #16         \n"
                            "st1    {v3.16b}, [%1], #16         \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0", "v1", "v2", "v3");
                        img0 += size * 8;
                    }
                }
            }
        });

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;
#else  // __ARM_FEATURE_DOTPROD
        int remain_size_start = 0;
        int nn_size = (size - remain_size_start) >> 2;
#endif // __ARM_FEATURE_DOTPROD

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
        for (const auto ii : otter::irange(begin, end))
            {
                int i = remain_size_start + ii * 4;

    #if __ARM_FEATURE_DOTPROD
                signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4].data();
    #else
                signed char* tmpptr = tmp_ra[i / 4].data();
    #endif

                for (int q = 0; q < inch; q++)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++)
                    {
    #if __ARM_FEATURE_DOTPROD
                        asm volatile(
                            "prfm   pldl1keep, [%0, #256]       \n"
                            "ld2    {v0.4s, v1.4s}, [%0]        \n"
                            "st1    {v0.4s, v1.4s}, [%1], #32   \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0", "v1");
    #else
                        asm volatile(
                            "prfm   pldl1keep, [%0, #256]       \n"
                            "ld1    {v0.16b, v1.16b}, [%0]      \n"
                            "st1    {v0.16b, v1.16b}, [%1], #32 \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0", "v1");
    #endif // __ARM_FEATURE_DOTPROD
                        img0 += size * 8;
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

    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
                signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2].data();
    #else
                signed char* tmpptr = tmp_ra[i / 4 + (i % 4) / 2].data();
    #endif
    #else
                signed char* tmpptr = tmp_ra[i / 2].data();
    #endif

                for (int q = 0; q < inch; q++)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++)
                    {
    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
                        asm volatile(
                            "prfm   pldl1keep, [%0, #128]   \n"
                            "ld2    {v0.2s, v1.2s}, [%0]    \n"
                            "st1    {v0.2s, v1.2s}, [%1], #16 \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0", "v1");
    #else
                        asm volatile(
                            "prfm   pldl1keep, [%0, #128]   \n"
                            "ld1    {v0.16b}, [%0]          \n"
                            "st1    {v0.16b}, [%1], #16     \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0");
    #endif // __ARM_FEATURE_DOTPROD
    #else
                        asm volatile(
                            "pld        [%0, #128]          \n"
                            "vld1.s8    {d0-d1}, [%0 :64]   \n"
                            "vst1.s8    {d0-d1}, [%1 :64]!  \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "q0");
    #endif
                        img0 += size * 8;
                    }
                }
            }
        });

        remain_size_start += nn_size << 1;

        otter::parallel_for(remain_size_start, size, 0, [&](int64_t begin, int64_t end) {
        for (const auto i : otter::irange(begin, end))
            {
    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
                signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2].data();
    #else
                signed char* tmpptr = tmp_ra[i / 4 + (i % 4) / 2 + i % 2].data();
    #endif
    #else
                signed char* tmpptr = tmp_ra[i / 2 + i % 2].data();
    #endif

                for (int q = 0; q < inch; q++)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++)
                    {
    #if __aarch64__
                        asm volatile(
                            "prfm   pldl1keep, [%0, #64]    \n"
                            "ld1    {v0.8b}, [%0]           \n"
                            "st1    {v0.8b}, [%1], #8       \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0");
    #else
                        asm volatile(
                            "pld        [%0, #64]           \n"
                            "vld1.s8    {d0}, [%0 :64]      \n"
                            "vst1.s8    {d0}, [%1 :64]!     \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "d0");
    #endif
                        img0 += size * 8;
                    }
                }
            }
        });
    }

    int nn_outch = 0;
    int remain_outch_start = 0;

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
    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
            for (; i + 15 < size; i += 16)
            {
                const signed char* tmpptr = tmp_ra[i / 16].data();
                const signed char* kptr0 = kernel_a[p / 4].data();

                int nn = inch * maxk; // inch always > 0

                asm volatile(
                    "ld1    {v24.16b}, [%6], #16        \n" // _w0123_l

                    "eor    v0.16b, v0.16b, v0.16b      \n"
                    "eor    v1.16b, v1.16b, v1.16b      \n"

                    "ld1    {v16.16b}, [%5], #16        \n" // _val0123_l

                    "eor    v2.16b, v2.16b, v2.16b      \n"
                    "eor    v3.16b, v3.16b, v3.16b      \n"
                    "eor    v4.16b, v4.16b, v4.16b      \n"
                    "eor    v5.16b, v5.16b, v5.16b      \n"
                    "eor    v6.16b, v6.16b, v6.16b      \n"
                    "eor    v7.16b, v7.16b, v7.16b      \n"
                    "eor    v8.16b, v8.16b, v8.16b      \n"
                    "eor    v9.16b, v9.16b, v9.16b      \n"
                    "eor    v10.16b, v10.16b, v10.16b   \n"
                    "eor    v11.16b, v11.16b, v11.16b   \n"
                    "eor    v12.16b, v12.16b, v12.16b   \n"
                    "eor    v13.16b, v13.16b, v13.16b   \n"
                    "eor    v14.16b, v14.16b, v14.16b   \n"
                    "eor    v15.16b, v15.16b, v15.16b   \n"

                    "0:                                 \n"

                    "ld1    {v17.16b}, [%5], #16        \n" // _val4567_l

                    "sdot   v0.4s, v24.16b, v16.4b[0]   \n"
                    "sdot   v1.4s, v24.16b, v16.4b[1]   \n"
                    "sdot   v2.4s, v24.16b, v16.4b[2]   \n"
                    "sdot   v3.4s, v24.16b, v16.4b[3]   \n"

                    "ld1    {v18.16b}, [%5], #16        \n" // _val891011_l

                    "sdot   v4.4s, v24.16b, v17.4b[0]   \n"
                    "sdot   v5.4s, v24.16b, v17.4b[1]   \n"
                    "sdot   v6.4s, v24.16b, v17.4b[2]   \n"
                    "sdot   v7.4s, v24.16b, v17.4b[3]   \n"

                    "ld1    {v19.16b}, [%5], #16        \n" // _val12131415_l

                    "sdot   v8.4s, v24.16b, v18.4b[0]   \n"
                    "sdot   v9.4s, v24.16b, v18.4b[1]   \n"

                    "ld1    {v25.16b}, [%6], #16        \n" // _w0123_h

                    "sdot   v10.4s, v24.16b, v18.4b[2]  \n"
                    "sdot   v11.4s, v24.16b, v18.4b[3]  \n"

                    "ld1    {v20.16b}, [%5], #16        \n" // _val0123_h

                    "sdot   v12.4s, v24.16b, v19.4b[0]  \n"
                    "sdot   v13.4s, v24.16b, v19.4b[1]  \n"
                    "sdot   v14.4s, v24.16b, v19.4b[2]  \n"
                    "sdot   v15.4s, v24.16b, v19.4b[3]  \n"

                    "ld1    {v21.16b}, [%5], #16        \n" // _val4567_h

                    "sdot   v0.4s, v25.16b, v20.4b[0]   \n"
                    "sdot   v1.4s, v25.16b, v20.4b[1]   \n"
                    "sdot   v2.4s, v25.16b, v20.4b[2]   \n"
                    "sdot   v3.4s, v25.16b, v20.4b[3]   \n"

                    "ld1    {v22.16b}, [%5], #16        \n" // _val891011_h

                    "sdot   v4.4s, v25.16b, v21.4b[0]   \n"
                    "sdot   v5.4s, v25.16b, v21.4b[1]   \n"
                    "sdot   v6.4s, v25.16b, v21.4b[2]   \n"
                    "sdot   v7.4s, v25.16b, v21.4b[3]   \n"

                    "ld1    {v23.16b}, [%5], #16        \n" // _val12131415_h

                    "sdot   v8.4s, v25.16b, v22.4b[0]   \n"
                    "sdot   v9.4s, v25.16b, v22.4b[1]   \n"

                    "ld1    {v24.16b}, [%6], #16        \n" // _w0123_l

                    "sdot   v10.4s, v25.16b, v22.4b[2]  \n"
                    "sdot   v11.4s, v25.16b, v22.4b[3]  \n"

                    "ld1    {v16.16b}, [%5], #16        \n" // _val0123_l

                    "sdot   v12.4s, v25.16b, v23.4b[0]  \n"
                    "sdot   v13.4s, v25.16b, v23.4b[1]  \n"

                    "subs   %w4, %w4, #1                \n"

                    "sdot   v14.4s, v25.16b, v23.4b[2]  \n"
                    "sdot   v15.4s, v25.16b, v23.4b[3]  \n"

                    "bne    0b                          \n"

                    "sub    %5, %5, #16                 \n"
                    "sub    %6, %6, #16                 \n"

                    // transpose 4x16
                    "trn1   v16.4s, v0.4s, v1.4s        \n"
                    "trn2   v17.4s, v0.4s, v1.4s        \n"
                    "trn1   v18.4s, v2.4s, v3.4s        \n"
                    "trn2   v19.4s, v2.4s, v3.4s        \n"
                    "trn1   v20.4s, v4.4s, v5.4s        \n"
                    "trn2   v21.4s, v4.4s, v5.4s        \n"
                    "trn1   v22.4s, v6.4s, v7.4s        \n"
                    "trn2   v23.4s, v6.4s, v7.4s        \n"
                    "trn1   v24.4s, v8.4s, v9.4s        \n"
                    "trn2   v25.4s, v8.4s, v9.4s        \n"
                    "trn1   v26.4s, v10.4s, v11.4s      \n"
                    "trn2   v27.4s, v10.4s, v11.4s      \n"
                    "trn1   v28.4s, v12.4s, v13.4s      \n"
                    "trn2   v29.4s, v12.4s, v13.4s      \n"
                    "trn1   v30.4s, v14.4s, v15.4s      \n"
                    "trn2   v31.4s, v14.4s, v15.4s      \n"

                    "trn1   v0.2d, v16.2d, v18.2d       \n"
                    "trn2   v8.2d, v16.2d, v18.2d       \n"
                    "trn1   v4.2d, v17.2d, v19.2d       \n"
                    "trn2   v12.2d, v17.2d, v19.2d      \n"

                    "trn1   v1.2d, v20.2d, v22.2d       \n"
                    "trn2   v9.2d, v20.2d, v22.2d       \n"
                    "trn1   v5.2d, v21.2d, v23.2d       \n"
                    "trn2   v13.2d, v21.2d, v23.2d      \n"

                    "trn1   v2.2d, v24.2d, v26.2d       \n"
                    "trn2   v10.2d, v24.2d, v26.2d      \n"
                    "trn1   v6.2d, v25.2d, v27.2d       \n"
                    "trn2   v14.2d, v25.2d, v27.2d      \n"

                    "trn1   v3.2d, v28.2d, v30.2d       \n"
                    "trn2   v11.2d, v28.2d, v30.2d      \n"
                    "trn1   v7.2d, v29.2d, v31.2d       \n"
                    "trn2   v15.2d, v29.2d, v31.2d      \n"

                    "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                    "st1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n"
                    "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%2], #64 \n"
                    "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%3], #64 \n"
                    : "=r"(outptr0),
                    "=r"(outptr1),
                    "=r"(outptr2),
                    "=r"(outptr3),
                    "=r"(nn),
                    "=r"(tmpptr),
                    "=r"(kptr0)
                    : "0"(outptr0),
                    "1"(outptr1),
                    "2"(outptr2),
                    "3"(outptr3),
                    "4"(nn),
                    "5"(tmpptr),
                    "6"(kptr0)
                    : "memory", "x4", "x5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            for (; i + 7 < size; i += 8)
            {
                const signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8].data();
                const signed char* kptr0 = kernel_a[p / 4].data();

                int nn = inch * maxk; // inch always > 0

                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                int32x4_t _sum4 = vdupq_n_s32(0);
                int32x4_t _sum5 = vdupq_n_s32(0);
                int32x4_t _sum6 = vdupq_n_s32(0);
                int32x4_t _sum7 = vdupq_n_s32(0);

                for (int j = 0; j < nn; j++)
                {
                    int8x16_t _val0123_l = vld1q_s8(tmpptr);
                    int8x16_t _val4567_l = vld1q_s8(tmpptr + 16);

                    int8x16_t _w0123_l = vld1q_s8(kptr0);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0123_l, _val0123_l, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0123_l, _val0123_l, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _w0123_l, _val0123_l, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _w0123_l, _val0123_l, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _w0123_l, _val4567_l, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _w0123_l, _val4567_l, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _w0123_l, _val4567_l, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _w0123_l, _val4567_l, 3);

                    int8x16_t _val0123_h = vld1q_s8(tmpptr + 32);
                    int8x16_t _val4567_h = vld1q_s8(tmpptr + 48);

                    int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0123_h, _val0123_h, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0123_h, _val0123_h, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _w0123_h, _val0123_h, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _w0123_h, _val0123_h, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _w0123_h, _val4567_h, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _w0123_h, _val4567_h, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _w0123_h, _val4567_h, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _w0123_h, _val4567_h, 3);

                    tmpptr += 64;
                    kptr0 += 32;
                }

                // transpose 4x8
                int32x4x2_t _s01 = vtrnq_s32(_sum0, _sum1);
                int32x4x2_t _s23 = vtrnq_s32(_sum2, _sum3);
                int32x4x2_t _s45 = vtrnq_s32(_sum4, _sum5);
                int32x4x2_t _s67 = vtrnq_s32(_sum6, _sum7);
                _sum0 = vcombine_s32(vget_low_s32(_s01.val[0]), vget_low_s32(_s23.val[0]));
                _sum1 = vcombine_s32(vget_low_s32(_s01.val[1]), vget_low_s32(_s23.val[1]));
                _sum2 = vcombine_s32(vget_high_s32(_s01.val[0]), vget_high_s32(_s23.val[0]));
                _sum3 = vcombine_s32(vget_high_s32(_s01.val[1]), vget_high_s32(_s23.val[1]));
                _sum4 = vcombine_s32(vget_low_s32(_s45.val[0]), vget_low_s32(_s67.val[0]));
                _sum5 = vcombine_s32(vget_low_s32(_s45.val[1]), vget_low_s32(_s67.val[1]));
                _sum6 = vcombine_s32(vget_high_s32(_s45.val[0]), vget_high_s32(_s67.val[0]));
                _sum7 = vcombine_s32(vget_high_s32(_s45.val[1]), vget_high_s32(_s67.val[1]));

                vst1q_s32(outptr0, _sum0);
                vst1q_s32(outptr1, _sum1);
                vst1q_s32(outptr2, _sum2);
                vst1q_s32(outptr3, _sum3);
                vst1q_s32(outptr0 + 4, _sum4);
                vst1q_s32(outptr1 + 4, _sum5);
                vst1q_s32(outptr2 + 4, _sum6);
                vst1q_s32(outptr3 + 4, _sum7);
                outptr0 += 8;
                outptr1 += 8;
                outptr2 += 8;
                outptr3 += 8;
            }
    #endif
            for (; i + 3 < size; i += 4)
            {
    #if __ARM_FEATURE_DOTPROD
                const signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4].data();
    #else
                const signed char* tmpptr = tmp_ra[i / 4].data();
    #endif
                const signed char* kptr0 = kernel_a[p / 4].data();

                int nn = inch * maxk; // inch always > 0

    #if __ARM_FEATURE_DOTPROD
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);

                for (int j = 0; j < nn; j++)
                {
                    int8x16_t _val0123_l = vld1q_s8(tmpptr);
                    int8x16_t _w0123_l = vld1q_s8(kptr0);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0123_l, _val0123_l, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0123_l, _val0123_l, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _w0123_l, _val0123_l, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _w0123_l, _val0123_l, 3);

                    int8x16_t _val0123_h = vld1q_s8(tmpptr + 16);
                    int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0123_h, _val0123_h, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0123_h, _val0123_h, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _w0123_h, _val0123_h, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _w0123_h, _val0123_h, 3);

                    tmpptr += 32;
                    kptr0 += 32;
                }

                // transpose 4x4
                int32x4x2_t _s01 = vtrnq_s32(_sum0, _sum1);
                int32x4x2_t _s23 = vtrnq_s32(_sum2, _sum3);
                _sum0 = vcombine_s32(vget_low_s32(_s01.val[0]), vget_low_s32(_s23.val[0]));
                _sum1 = vcombine_s32(vget_low_s32(_s01.val[1]), vget_low_s32(_s23.val[1]));
                _sum2 = vcombine_s32(vget_high_s32(_s01.val[0]), vget_high_s32(_s23.val[0]));
                _sum3 = vcombine_s32(vget_high_s32(_s01.val[1]), vget_high_s32(_s23.val[1]));

                vst1q_s32(outptr0, _sum0);
                vst1q_s32(outptr1, _sum1);
                vst1q_s32(outptr2, _sum2);
                vst1q_s32(outptr3, _sum3);
                outptr0 += 4;
                outptr1 += 4;
                outptr2 += 4;
                outptr3 += 4;
    #else  // __ARM_FEATURE_DOTPROD
                asm volatile(
                    "eor    v0.16b, v0.16b, v0.16b      \n"
                    "eor    v1.16b, v1.16b, v1.16b      \n"
                    "eor    v2.16b, v2.16b, v2.16b      \n"
                    "eor    v3.16b, v3.16b, v3.16b      \n"
                    "eor    v4.16b, v4.16b, v4.16b      \n"
                    "eor    v5.16b, v5.16b, v5.16b      \n"
                    "eor    v6.16b, v6.16b, v6.16b      \n"
                    "eor    v7.16b, v7.16b, v7.16b      \n"
                    "eor    v8.16b, v8.16b, v8.16b      \n"
                    "eor    v9.16b, v9.16b, v9.16b      \n"
                    "eor    v10.16b, v10.16b, v10.16b   \n"
                    "eor    v11.16b, v11.16b, v11.16b   \n"
                    "eor    v12.16b, v12.16b, v12.16b   \n"
                    "eor    v13.16b, v13.16b, v13.16b   \n"
                    "eor    v14.16b, v14.16b, v14.16b   \n"
                    "eor    v15.16b, v15.16b, v15.16b   \n"

                    "prfm   pldl1keep, [%5, #128]       \n"

                    "prfm   pldl1keep, [%6, #256]       \n"

                    "lsr    w4, %w4, #1                 \n" // w4 = nn >> 1
                    "cmp    w4, #0                      \n"
                    "beq    1f                          \n"

                    "prfm   pldl1keep, [%6, #512]       \n"

                    "add    x5, %5, #16                 \n"

                    "prfm   pldl1keep, [x5, #128]       \n"

                    "ld1    {v16.16b}, [%5]             \n" // val L H
                    "ld1    {v20.16b, v21.16b, v22.16b, v23.16b}, [%6], #64 \n"
                    "add    %5, %5, #32                 \n"
                    "ext    v17.16b, v16.16b, v16.16b, #8 \n" // val H L

                    "ld1    {v18.16b}, [%5]             \n"
                    "add    %5, %5, #32                 \n"

                    "0:                                 \n"

                    "smull  v24.8h, v16.8b,  v20.8b     \n"
                    "prfm   pldl1keep, [%6, #256]       \n"
                    "smull2 v25.8h, v17.16b, v20.16b    \n"
                    "prfm   pldl1keep, [%6, #512]       \n"
                    "smull  v26.8h, v16.8b,  v21.8b     \n"
                    "subs   w4, w4, #1                  \n"
                    "smull2 v27.8h, v17.16b, v21.16b    \n"
                    "ext    v19.16b, v18.16b, v18.16b, #8 \n" // val H L

                    "smlal  v24.8h, v18.8b,  v22.8b     \n"
                    "smlal2 v25.8h, v19.16b, v22.16b    \n"
                    "smlal  v26.8h, v18.8b,  v23.8b     \n"
                    "smlal2 v27.8h, v19.16b, v23.16b    \n"

                    "smull2 v29.8h, v16.16b, v20.16b    \n"
                    "sadalp v0.4s, v24.8h               \n"
                    "smull  v28.8h, v17.8b,  v20.8b     \n"
                    "sadalp v1.4s, v25.8h               \n"
                    "smull2 v31.8h, v16.16b, v21.16b    \n"
                    "ld1    {v16.16b}, [x5]             \n" // val L H
                    "smull  v30.8h, v17.8b,  v21.8b     \n"
                    "add    x5, x5, #32                 \n"
                    "smlal2 v29.8h, v18.16b, v22.16b    \n"
                    "sadalp v2.4s, v26.8h               \n"
                    "smlal  v28.8h, v19.8b,  v22.8b     \n"
                    "sadalp v3.4s, v27.8h               \n"
                    "smlal2 v31.8h, v18.16b, v23.16b    \n"
                    "ld1    {v18.16b}, [x5]             \n"
                    "smlal  v30.8h, v19.8b,  v23.8b     \n"
                    "ext    v17.16b, v16.16b, v16.16b, #8 \n" // val H L

                    "smull  v24.8h, v16.8b,  v20.8b     \n"
                    "add    x5, x5, #32                 \n"
                    "smull2 v25.8h, v17.16b, v20.16b    \n"
                    "prfm   pldl1keep, [x5, #128]       \n"
                    "smull  v26.8h, v16.8b,  v21.8b     \n"
                    "prfm   pldl1keep, [x5, #384]       \n"
                    "smull2 v27.8h, v17.16b, v21.16b    \n"
                    "ext    v19.16b, v18.16b, v18.16b, #8 \n" // val H L

                    "smlal  v24.8h, v18.8b,  v22.8b     \n"
                    "sadalp v5.4s, v29.8h               \n"
                    "smlal2 v25.8h, v19.16b, v22.16b    \n"
                    "sadalp v4.4s, v28.8h               \n"
                    "smlal  v26.8h, v18.8b,  v23.8b     \n"
                    "sadalp v7.4s, v31.8h               \n"
                    "smlal2 v27.8h, v19.16b, v23.16b    \n"
                    "sadalp v6.4s, v30.8h               \n"

                    "smull2 v29.8h, v16.16b, v20.16b    \n"
                    "sadalp v8.4s, v24.8h               \n"
                    "smull  v28.8h, v17.8b,  v20.8b     \n"
                    "sadalp v9.4s, v25.8h               \n"
                    "smull2 v31.8h, v16.16b, v21.16b    \n"
                    "ld1    {v16.16b}, [%5]             \n" // val L H
                    "smull  v30.8h, v17.8b,  v21.8b     \n"
                    "add    %5, %5, #32                 \n"
                    "smlal2 v29.8h, v18.16b, v22.16b    \n"
                    "sadalp v10.4s, v26.8h              \n"
                    "smlal  v28.8h, v19.8b,  v22.8b     \n"
                    "sadalp v11.4s, v27.8h              \n"
                    "smlal2 v31.8h, v18.16b, v23.16b    \n"
                    "ld1    {v18.16b}, [%5]             \n"
                    "smlal  v30.8h, v19.8b,  v23.8b     \n"
                    "add    %5, %5, #32                 \n"
                    "ld1    {v20.16b, v21.16b, v22.16b, v23.16b}, [%6], #64 \n"

                    "sadalp v13.4s, v29.8h              \n"
                    "prfm   pldl1keep, [%5, #128]       \n"
                    "sadalp v12.4s, v28.8h              \n"
                    "prfm   pldl1keep, [%5, #384]       \n"
                    "sadalp v15.4s, v31.8h              \n"
                    "ext    v17.16b, v16.16b, v16.16b, #8 \n" // val H L

                    "sadalp v14.4s, v30.8h              \n"

                    "bne    0b                          \n"

                    "sub    %5, %5, #64                 \n"
                    "sub    %6, %6, #64                 \n"

                    "1:                                 \n"
                    "and    w4, %w4, #1                 \n" // w4 = remain = nn & 1
                    "cmp    w4, #0                      \n" // w4 > 0
                    "beq    2f                          \n"

                    "ld1    {v16.8b, v17.8b}, [%5], #16 \n"
                    "ld1    {v20.8b, v21.8b, v22.8b, v23.8b}, [%6], #32 \n"

                    "smull  v24.8h, v16.8b, v20.8b      \n"
                    "smull  v25.8h, v16.8b, v21.8b      \n"
                    "smull  v26.8h, v16.8b, v22.8b      \n"
                    "ld1    {v18.8b, v19.8b}, [%5], #16 \n"
                    "smull  v27.8h, v16.8b, v23.8b      \n"
                    "sadalp v0.4s, v24.8h               \n"
                    "smull  v28.8h, v17.8b, v20.8b      \n"
                    "sadalp v1.4s, v25.8h               \n"
                    "smull  v29.8h, v17.8b, v21.8b      \n"
                    "sadalp v2.4s, v26.8h               \n"
                    "smull  v30.8h, v17.8b, v22.8b      \n"
                    "sadalp v3.4s, v27.8h               \n"
                    "smull  v31.8h, v17.8b, v23.8b      \n"
                    "sadalp v4.4s, v28.8h               \n"
                    "smull  v24.8h, v18.8b, v20.8b      \n"
                    "sadalp v5.4s, v29.8h               \n"
                    "smull  v25.8h, v18.8b, v21.8b      \n"
                    "sadalp v6.4s, v30.8h               \n"
                    "smull  v26.8h, v18.8b, v22.8b      \n"
                    "sadalp v7.4s, v31.8h               \n"
                    "smull  v27.8h, v18.8b, v23.8b      \n"
                    "sadalp v8.4s, v24.8h               \n"
                    "smull  v28.8h, v19.8b, v20.8b      \n"
                    "sadalp v9.4s, v25.8h               \n"
                    "smull  v29.8h, v19.8b, v21.8b      \n"
                    "sadalp v10.4s, v26.8h              \n"
                    "smull  v30.8h, v19.8b, v22.8b      \n"
                    "sadalp v11.4s, v27.8h              \n"
                    "smull  v31.8h, v19.8b, v23.8b      \n"

                    "sadalp v12.4s, v28.8h              \n"
                    "sadalp v13.4s, v29.8h              \n"
                    "sadalp v14.4s, v30.8h              \n"
                    "sadalp v15.4s, v31.8h              \n"

                    "2:                                 \n"

                    "addp   v0.4s, v0.4s, v4.4s         \n"
                    "addp   v1.4s, v1.4s, v5.4s         \n"
                    "addp   v2.4s, v2.4s, v6.4s         \n"
                    "addp   v3.4s, v3.4s, v7.4s         \n"
                    "addp   v8.4s, v8.4s, v12.4s        \n"
                    "addp   v9.4s, v9.4s, v13.4s        \n"
                    "addp   v10.4s, v10.4s, v14.4s      \n"
                    "addp   v11.4s, v11.4s, v15.4s      \n"

                    "addp   v0.4s, v0.4s, v8.4s         \n"
                    "addp   v1.4s, v1.4s, v9.4s         \n"
                    "addp   v2.4s, v2.4s, v10.4s        \n"
                    "addp   v3.4s, v3.4s, v11.4s        \n"

                    "st1    {v0.4s}, [%0], #16          \n"
                    "st1    {v1.4s}, [%1], #16          \n"
                    "st1    {v2.4s}, [%2], #16          \n"
                    "st1    {v3.4s}, [%3], #16          \n"

                    : "=r"(outptr0),
                    "=r"(outptr1),
                    "=r"(outptr2),
                    "=r"(outptr3),
                    "=r"(nn),
                    "=r"(tmpptr),
                    "=r"(kptr0)
                    : "0"(outptr0),
                    "1"(outptr1),
                    "2"(outptr2),
                    "3"(outptr3),
                    "4"(nn),
                    "5"(tmpptr),
                    "6"(kptr0)
                    : "memory", "x4", "x5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    #endif // __ARM_FEATURE_DOTPROD
            }
    #endif // __aarch64__
            for (; i + 1 < size; i += 2)
            {
    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
                const signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2].data();
    #else
                const signed char* tmpptr = tmp_ra[i / 4 + (i % 4) / 2].data();
    #endif
    #else
                const signed char* tmpptr = tmp_ra[i / 2].data();
    #endif
                const signed char* kptr0 = kernel_a[p / 4].data();

                int nn = inch * maxk; // inch always > 0

    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);

                for (int j = 0; j < nn; j++)
                {
                    int8x16_t _val01_l_h = vld1q_s8(tmpptr);
                    int8x16_t _w0123_l = vld1q_s8(kptr0);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0123_l, _val01_l_h, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0123_l, _val01_l_h, 1);

                    int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0123_h, _val01_l_h, 2);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0123_h, _val01_l_h, 3);

                    tmpptr += 16;
                    kptr0 += 32;
                }

                vst1q_lane_s32(outptr0, _sum0, 0);
                vst1q_lane_s32(outptr1, _sum0, 1);
                vst1q_lane_s32(outptr2, _sum0, 2);
                vst1q_lane_s32(outptr3, _sum0, 3);
                vst1q_lane_s32(outptr0 + 1, _sum1, 0);
                vst1q_lane_s32(outptr1 + 1, _sum1, 1);
                vst1q_lane_s32(outptr2 + 1, _sum1, 2);
                vst1q_lane_s32(outptr3 + 1, _sum1, 3);
                outptr0 += 2;
                outptr1 += 2;
                outptr2 += 2;
                outptr3 += 2;
    #else  // __ARM_FEATURE_DOTPROD
                int32x4_t _sum00 = vdupq_n_s32(0);
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum02 = vdupq_n_s32(0);
                int32x4_t _sum03 = vdupq_n_s32(0);
                int32x4_t _sum10 = vdupq_n_s32(0);
                int32x4_t _sum11 = vdupq_n_s32(0);
                int32x4_t _sum12 = vdupq_n_s32(0);
                int32x4_t _sum13 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 1 < nn; j += 2)
                {
                    int8x16_t _val0 = vld1q_s8(tmpptr);
                    int8x16_t _val1 = vld1q_s8(tmpptr + 16);

                    int8x16_t _w01 = vld1q_s8(kptr0);
                    int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                    int16x8_t _wv00 = vmull_s8(vget_low_s8(_val0), vget_low_s8(_w01));
                    int16x8_t _wv01 = vmull_s8(vget_low_s8(_val0), vget_high_s8(_w01));
                    int16x8_t _wv02 = vmull_s8(vget_low_s8(_val0), vget_low_s8(_w23));
                    int16x8_t _wv03 = vmull_s8(vget_low_s8(_val0), vget_high_s8(_w23));

                    int16x8_t _wv10 = vmull_s8(vget_high_s8(_val0), vget_low_s8(_w01));
                    int16x8_t _wv11 = vmull_s8(vget_high_s8(_val0), vget_high_s8(_w01));
                    int16x8_t _wv12 = vmull_s8(vget_high_s8(_val0), vget_low_s8(_w23));
                    int16x8_t _wv13 = vmull_s8(vget_high_s8(_val0), vget_high_s8(_w23));

                    int8x16_t _w45 = vld1q_s8(kptr0 + 32);
                    int8x16_t _w67 = vld1q_s8(kptr0 + 48);

                    _wv00 = vmlal_s8(_wv00, vget_low_s8(_val1), vget_low_s8(_w45));
                    _wv01 = vmlal_s8(_wv01, vget_low_s8(_val1), vget_high_s8(_w45));
                    _wv02 = vmlal_s8(_wv02, vget_low_s8(_val1), vget_low_s8(_w67));
                    _wv03 = vmlal_s8(_wv03, vget_low_s8(_val1), vget_high_s8(_w67));

                    _wv10 = vmlal_s8(_wv10, vget_high_s8(_val1), vget_low_s8(_w45));
                    _wv11 = vmlal_s8(_wv11, vget_high_s8(_val1), vget_high_s8(_w45));
                    _wv12 = vmlal_s8(_wv12, vget_high_s8(_val1), vget_low_s8(_w67));
                    _wv13 = vmlal_s8(_wv13, vget_high_s8(_val1), vget_high_s8(_w67));

                    _sum00 = vpadalq_s16(_sum00, _wv00);
                    _sum01 = vpadalq_s16(_sum01, _wv01);
                    _sum02 = vpadalq_s16(_sum02, _wv02);
                    _sum03 = vpadalq_s16(_sum03, _wv03);
                    _sum10 = vpadalq_s16(_sum10, _wv10);
                    _sum11 = vpadalq_s16(_sum11, _wv11);
                    _sum12 = vpadalq_s16(_sum12, _wv12);
                    _sum13 = vpadalq_s16(_sum13, _wv13);

                    tmpptr += 32;
                    kptr0 += 64;
                }
                for (; j < nn; j++)
                {
                    int8x16_t _val = vld1q_s8(tmpptr);

                    int8x16_t _w01 = vld1q_s8(kptr0);
                    int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                    int16x8_t _wv00 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w01));
                    int16x8_t _wv01 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w01));
                    int16x8_t _wv02 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w23));
                    int16x8_t _wv03 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w23));
                    int16x8_t _wv10 = vmull_s8(vget_high_s8(_val), vget_low_s8(_w01));
                    int16x8_t _wv11 = vmull_s8(vget_high_s8(_val), vget_high_s8(_w01));
                    int16x8_t _wv12 = vmull_s8(vget_high_s8(_val), vget_low_s8(_w23));
                    int16x8_t _wv13 = vmull_s8(vget_high_s8(_val), vget_high_s8(_w23));

                    _sum00 = vpadalq_s16(_sum00, _wv00);
                    _sum01 = vpadalq_s16(_sum01, _wv01);
                    _sum02 = vpadalq_s16(_sum02, _wv02);
                    _sum03 = vpadalq_s16(_sum03, _wv03);
                    _sum10 = vpadalq_s16(_sum10, _wv10);
                    _sum11 = vpadalq_s16(_sum11, _wv11);
                    _sum12 = vpadalq_s16(_sum12, _wv12);
                    _sum13 = vpadalq_s16(_sum13, _wv13);

                    tmpptr += 16;
                    kptr0 += 32;
                }

                int32x4_t _s001 = vpaddq_s32(_sum00, _sum01);
                int32x4_t _s023 = vpaddq_s32(_sum02, _sum03);
                int32x4_t _s101 = vpaddq_s32(_sum10, _sum11);
                int32x4_t _s123 = vpaddq_s32(_sum12, _sum13);

                int32x4_t _sum0 = vpaddq_s32(_s001, _s023);
                int32x4_t _sum1 = vpaddq_s32(_s101, _s123);

                vst1q_lane_s32(outptr0, _sum0, 0);
                vst1q_lane_s32(outptr1, _sum0, 1);
                vst1q_lane_s32(outptr2, _sum0, 2);
                vst1q_lane_s32(outptr3, _sum0, 3);
                vst1q_lane_s32(outptr0 + 1, _sum1, 0);
                vst1q_lane_s32(outptr1 + 1, _sum1, 1);
                vst1q_lane_s32(outptr2 + 1, _sum1, 2);
                vst1q_lane_s32(outptr3 + 1, _sum1, 3);
                outptr0 += 2;
                outptr1 += 2;
                outptr2 += 2;
                outptr3 += 2;
    #endif // __ARM_FEATURE_DOTPROD
    #else  // __aarch64__
                asm volatile(
                    "veor       q0, q0              \n"
                    "veor       q1, q1              \n"
                    "veor       q2, q2              \n"
                    "veor       q3, q3              \n"
                    "veor       q4, q4              \n"
                    "veor       q5, q5              \n"
                    "veor       q6, q6              \n"
                    "veor       q7, q7              \n"

                    "pld        [%5, #256]          \n"

                    "lsr        r4, %4, #1          \n" // r4 = nn = size >> 1
                    "cmp        r4, #0              \n"
                    "beq        1f                  \n"

                    "add        r5, %6, #16         \n"
                    "pld        [%6, #128]          \n"
                    "mov        r6, #32             \n"
                    "pld        [%6, #384]          \n"

                    "vld1.s8    {d20-d21}, [%6 :128], r6 \n" // _w01

                    "vld1.s8    {d16-d19}, [%5 :128]! \n" // _val0 _val1

                    "vld1.s8    {d22-d23}, [%6 :128], r6 \n" // _w45

                    "0:                             \n"

                    "vmull.s8   q12, d16, d20       \n"
                    "pld        [%5, #256]          \n"
                    "vmull.s8   q13, d16, d21       \n"
                    "pld        [%6, #384]          \n"
                    "vmull.s8   q14, d17, d20       \n"
                    "vmull.s8   q15, d17, d21       \n"
                    "vld1.s8    {d20-d21}, [r5 :128], r6 \n" // _w23

                    "vmlal.s8   q12, d18, d22       \n"
                    "vmlal.s8   q13, d18, d23       \n"
                    "subs       r4, r4, #1          \n"
                    "vmlal.s8   q14, d19, d22       \n"
                    "vmlal.s8   q15, d19, d23       \n"
                    "vld1.s8    {d22-d23}, [r5 :128], r6 \n" // _w67

                    "vpadal.s16 q0, q12             \n"
                    "vmull.s8   q12, d16, d20       \n"
                    "vpadal.s16 q1, q13             \n"
                    "vmull.s8   q13, d16, d21       \n"
                    "vpadal.s16 q4, q14             \n"
                    "vmull.s8   q14, d17, d20       \n"
                    "vpadal.s16 q5, q15             \n"
                    "vmull.s8   q15, d17, d21       \n"
                    "vld1.s8    {d16-d17}, [%5 :128]! \n" // _val0

                    "vmlal.s8   q12, d18, d22       \n"
                    "vld1.s8    {d20-d21}, [%6 :128], r6 \n" // _w01
                    "vmlal.s8   q13, d18, d23       \n"
                    "pld        [r5, #128]          \n"
                    "vmlal.s8   q14, d19, d22       \n"
                    "pld        [r5, #384]          \n"
                    "vmlal.s8   q15, d19, d23       \n"
                    "vld1.s8    {d18-d19}, [%5 :128]! \n" // _val1

                    "vpadal.s16 q2, q12             \n"
                    "vld1.s8    {d22-d23}, [%6 :128], r6 \n" // _w45
                    "vpadal.s16 q3, q13             \n"
                    "pld        [%5, #128]          \n"
                    "vpadal.s16 q6, q14             \n"
                    "pld        [%6, #128]          \n"
                    "vpadal.s16 q7, q15             \n"

                    "bne        0b                  \n"

                    "sub        %5, %5, #32         \n"
                    "sub        %6, %6, #64         \n"

                    "1:                             \n"
                    "and        r4, %4, #1          \n" // r4 = remain = size & 1
                    "cmp        r4, #0              \n" // r4 > 0
                    "beq        2f                  \n"

                    "vld1.s8    {d16-d17}, [%5 :128]! \n" // _val
                    "vld1.s8    {d20-d21}, [%6 :128]! \n" // _w01

                    "vmull.s8   q12, d16, d20       \n"

                    "vld1.s8    {d22-d23}, [%6 :128]! \n" // _w23
                    "vmull.s8   q13, d16, d21       \n"
                    "vmull.s8   q14, d17, d20       \n"
                    "vmull.s8   q15, d17, d21       \n"

                    "vpadal.s16 q0, q12             \n"
                    "vmull.s8   q12, d16, d22       \n"
                    "vpadal.s16 q1, q13             \n"
                    "vmull.s8   q13, d16, d23       \n"
                    "vpadal.s16 q4, q14             \n"
                    "vmull.s8   q14, d17, d22       \n"
                    "vpadal.s16 q5, q15             \n"
                    "vmull.s8   q15, d17, d23       \n"

                    "vpadal.s16 q2, q12             \n"
                    "vpadal.s16 q3, q13             \n"
                    "vpadal.s16 q6, q14             \n"
                    "vpadal.s16 q7, q15             \n"

                    "2:                             \n"

                    "vpadd.s32  d16, d0, d1         \n"
                    "vpadd.s32  d17, d2, d3         \n"
                    "vpadd.s32  d18, d4, d5         \n"
                    "vpadd.s32  d19, d6, d7         \n"
                    "vpadd.s32  d20, d8, d9         \n"
                    "vpadd.s32  d21, d10, d11       \n"
                    "vpadd.s32  d22, d12, d13       \n"
                    "vpadd.s32  d23, d14, d15       \n"

                    "vpadd.s32  d0, d16, d17        \n"
                    "vpadd.s32  d1, d18, d19        \n"
                    "vpadd.s32  d2, d20, d21        \n"
                    "vpadd.s32  d3, d22, d23        \n"

                    "vst1.s32   {d0[0]}, [%0]!      \n"
                    "vst1.s32   {d0[1]}, [%1]!      \n"
                    "vst1.s32   {d1[0]}, [%2]!      \n"
                    "vst1.s32   {d1[1]}, [%3]!      \n"
                    "vst1.s32   {d2[0]}, [%0]!      \n"
                    "vst1.s32   {d2[1]}, [%1]!      \n"
                    "vst1.s32   {d3[0]}, [%2]!      \n"
                    "vst1.s32   {d3[1]}, [%3]!      \n"

                    : "=r"(outptr0),
                    "=r"(outptr1),
                    "=r"(outptr2),
                    "=r"(outptr3),
                    "=r"(nn),
                    "=r"(tmpptr),
                    "=r"(kptr0)
                    : "0"(outptr0),
                    "1"(outptr1),
                    "2"(outptr2),
                    "3"(outptr3),
                    "4"(nn),
                    "5"(tmpptr),
                    "6"(kptr0)
                    : "memory", "r4", "r5", "r6", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
    #endif // __aarch64__
            }
            for (; i < size; i++)
            {
    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
                const signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2].data();
    #else
                const signed char* tmpptr = tmp_ra[i / 4 + (i % 4) / 2 + i % 2].data();
    #endif
    #else
                const signed char* tmpptr = tmp_ra[i / 2 + i % 2].data();
    #endif
                const signed char* kptr0 = kernel_a[p / 4].data();

                int nn = inch * maxk; // inch always > 0

    #if __ARM_FEATURE_DOTPROD
                int32x4_t _sum0 = vdupq_n_s32(0);

                for (int j = 0; j < nn; j++)
                {
                    int8x8_t _val0_l_h = vld1_s8(tmpptr);

                    int8x16_t _w0123_l = vld1q_s8(kptr0);

                    _sum0 = vdotq_lane_s32(_sum0, _w0123_l, _val0_l_h, 0);

                    int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

                    _sum0 = vdotq_lane_s32(_sum0, _w0123_h, _val0_l_h, 1);

                    tmpptr += 8;
                    kptr0 += 32;
                }

                vst1q_lane_s32(outptr0, _sum0, 0);
                vst1q_lane_s32(outptr1, _sum0, 1);
                vst1q_lane_s32(outptr2, _sum0, 2);
                vst1q_lane_s32(outptr3, _sum0, 3);
                outptr0 += 1;
                outptr1 += 1;
                outptr2 += 1;
                outptr3 += 1;
    #else // __ARM_FEATURE_DOTPROD
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 1 < nn; j += 2)
                {
                    int8x16_t _val = vld1q_s8(tmpptr);

                    int8x16_t _w01 = vld1q_s8(kptr0);
                    int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                    int16x8_t _wv0 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w01));
                    int16x8_t _wv1 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w01));
                    int16x8_t _wv2 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w23));
                    int16x8_t _wv3 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w23));

                    int8x16_t _w45 = vld1q_s8(kptr0 + 32);
                    int8x16_t _w67 = vld1q_s8(kptr0 + 48);

                    _wv0 = vmlal_s8(_wv0, vget_high_s8(_val), vget_low_s8(_w45));
                    _wv1 = vmlal_s8(_wv1, vget_high_s8(_val), vget_high_s8(_w45));
                    _wv2 = vmlal_s8(_wv2, vget_high_s8(_val), vget_low_s8(_w67));
                    _wv3 = vmlal_s8(_wv3, vget_high_s8(_val), vget_high_s8(_w67));

                    _sum0 = vpadalq_s16(_sum0, _wv0);
                    _sum1 = vpadalq_s16(_sum1, _wv1);
                    _sum2 = vpadalq_s16(_sum2, _wv2);
                    _sum3 = vpadalq_s16(_sum3, _wv3);

                    tmpptr += 16;
                    kptr0 += 64;
                }
                for (; j < nn; j++)
                {
                    int8x8_t _val = vld1_s8(tmpptr);

                    int8x16_t _w01 = vld1q_s8(kptr0);
                    int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                    int16x8_t _wv0 = vmull_s8(_val, vget_low_s8(_w01));
                    int16x8_t _wv1 = vmull_s8(_val, vget_high_s8(_w01));
                    int16x8_t _wv2 = vmull_s8(_val, vget_low_s8(_w23));
                    int16x8_t _wv3 = vmull_s8(_val, vget_high_s8(_w23));

                    _sum0 = vpadalq_s16(_sum0, _wv0);
                    _sum1 = vpadalq_s16(_sum1, _wv1);
                    _sum2 = vpadalq_s16(_sum2, _wv2);
                    _sum3 = vpadalq_s16(_sum3, _wv3);

                    tmpptr += 8;
                    kptr0 += 32;
                }

    #if __aarch64__
                int32x4_t _s01 = vpaddq_s32(_sum0, _sum1);
                int32x4_t _s23 = vpaddq_s32(_sum2, _sum3);

                _sum0 = vpaddq_s32(_s01, _s23);
    #else
                int32x2_t _s01_low = vpadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                int32x2_t _s01_high = vpadd_s32(vget_low_s32(_sum1), vget_high_s32(_sum1));
                int32x2_t _s23_low = vpadd_s32(vget_low_s32(_sum2), vget_high_s32(_sum2));
                int32x2_t _s23_high = vpadd_s32(vget_low_s32(_sum3), vget_high_s32(_sum3));

                _sum0 = vcombine_s32(vpadd_s32(_s01_low, _s01_high), vpadd_s32(_s23_low, _s23_high));
    #endif

                vst1q_lane_s32(outptr0, _sum0, 0);
                vst1q_lane_s32(outptr1, _sum0, 1);
                vst1q_lane_s32(outptr2, _sum0, 2);
                vst1q_lane_s32(outptr3, _sum0, 3);
                outptr0 += 1;
                outptr1 += 1;
                outptr2 += 1;
                outptr3 += 1;
    #endif // __ARM_FEATURE_DOTPROD
            }
        }
    });

    remain_outch_start += nn_outch << 2;

    otter::parallel_for(remain_outch_start, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end))
        {
            int* outptr0 = output_a[p].data();

            int i = 0;
    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
            for (; i + 15 < size; i += 16)
            {
                const signed char* tmpptr = tmp_ra[i / 16].data();
                const signed char* kptr0 = kernel_a[p / 4 + p % 4].data();

                int nn = inch * maxk; // inch always > 0

                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);

                int j = 0;
                for (; j < nn; j++)
                {
                    int8x16_t _val0123_l = vld1q_s8(tmpptr);
                    int8x16_t _val4567_l = vld1q_s8(tmpptr + 16);
                    int8x16_t _val89ab_l = vld1q_s8(tmpptr + 32);
                    int8x16_t _valcdef_l = vld1q_s8(tmpptr + 48);
                    int8x16_t _val0123_h = vld1q_s8(tmpptr + 64);
                    int8x16_t _val4567_h = vld1q_s8(tmpptr + 80);
                    int8x16_t _val89ab_h = vld1q_s8(tmpptr + 96);
                    int8x16_t _valcdef_h = vld1q_s8(tmpptr + 112);
                    int8x8_t _w_lh = vld1_s8(kptr0);

                    _sum0 = vdotq_lane_s32(_sum0, _val0123_l, _w_lh, 0);
                    _sum1 = vdotq_lane_s32(_sum1, _val4567_l, _w_lh, 0);
                    _sum2 = vdotq_lane_s32(_sum2, _val89ab_l, _w_lh, 0);
                    _sum3 = vdotq_lane_s32(_sum3, _valcdef_l, _w_lh, 0);
                    _sum0 = vdotq_lane_s32(_sum0, _val0123_h, _w_lh, 1);
                    _sum1 = vdotq_lane_s32(_sum1, _val4567_h, _w_lh, 1);
                    _sum2 = vdotq_lane_s32(_sum2, _val89ab_h, _w_lh, 1);
                    _sum3 = vdotq_lane_s32(_sum3, _valcdef_h, _w_lh, 1);

                    tmpptr += 128;
                    kptr0 += 8;
                }

                vst1q_s32(outptr0, _sum0);
                vst1q_s32(outptr0 + 4, _sum1);
                vst1q_s32(outptr0 + 8, _sum2);
                vst1q_s32(outptr0 + 12, _sum3);
                outptr0 += 16;
            }
            for (; i + 7 < size; i += 8)
            {
                const signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8].data();
                const signed char* kptr0 = kernel_a[p / 4 + p % 4].data();

                int nn = inch * maxk; // inch always > 0

                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);

                int j = 0;
                for (; j < nn; j++)
                {
                    int8x16_t _val0123_l = vld1q_s8(tmpptr);
                    int8x16_t _val4567_l = vld1q_s8(tmpptr + 16);
                    int8x16_t _val0123_h = vld1q_s8(tmpptr + 32);
                    int8x16_t _val4567_h = vld1q_s8(tmpptr + 48);
                    int8x8_t _w_lh = vld1_s8(kptr0);

                    _sum0 = vdotq_lane_s32(_sum0, _val0123_l, _w_lh, 0);
                    _sum1 = vdotq_lane_s32(_sum1, _val4567_l, _w_lh, 0);
                    _sum2 = vdotq_lane_s32(_sum2, _val0123_h, _w_lh, 1);
                    _sum3 = vdotq_lane_s32(_sum3, _val4567_h, _w_lh, 1);

                    tmpptr += 64;
                    kptr0 += 8;
                }

                _sum0 = vaddq_s32(_sum0, _sum2);
                _sum1 = vaddq_s32(_sum1, _sum3);

                vst1q_s32(outptr0, _sum0);
                vst1q_s32(outptr0 + 4, _sum1);
                outptr0 += 8;
            }
    #endif // __ARM_FEATURE_DOTPROD
            for (; i + 3 < size; i += 4)
            {
    #if __ARM_FEATURE_DOTPROD
                const signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4].data();
    #else
                const signed char* tmpptr = tmp_ra[i / 4].data();
    #endif
                const signed char* kptr0 = kernel_a[p / 4 + p % 4].data();

                int nn = inch * maxk; // inch always > 0

    #if __ARM_FEATURE_DOTPROD
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);

                int j = 0;
                for (; j < nn; j++)
                {
                    int8x16_t _val0123_l = vld1q_s8(tmpptr);
                    int8x16_t _val0123_h = vld1q_s8(tmpptr + 16);
                    int8x8_t _w_lh = vld1_s8(kptr0);

                    _sum0 = vdotq_lane_s32(_sum0, _val0123_l, _w_lh, 0);
                    _sum1 = vdotq_lane_s32(_sum1, _val0123_h, _w_lh, 1);

                    tmpptr += 32;
                    kptr0 += 8;
                }

                _sum0 = vaddq_s32(_sum0, _sum1);

                vst1q_s32(outptr0, _sum0);
                outptr0 += 4;
    #else  // __ARM_FEATURE_DOTPROD
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                int32x4_t _sum4 = vdupq_n_s32(0);
                int32x4_t _sum5 = vdupq_n_s32(0);
                int32x4_t _sum6 = vdupq_n_s32(0);
                int32x4_t _sum7 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 1 < nn; j += 2)
                {
                    int8x16_t _val0 = vld1q_s8(tmpptr);
                    int8x16_t _val1 = vld1q_s8(tmpptr + 16);
                    int8x16_t _val2 = vld1q_s8(tmpptr + 32);
                    int8x16_t _val3 = vld1q_s8(tmpptr + 48);
                    int8x16_t _w = vld1q_s8(kptr0);

                    int16x8_t _s0 = vmull_s8(vget_low_s8(_val0), vget_low_s8(_w));
                    int16x8_t _s1 = vmull_s8(vget_high_s8(_val0), vget_low_s8(_w));
                    int16x8_t _s2 = vmull_s8(vget_low_s8(_val1), vget_low_s8(_w));
                    int16x8_t _s3 = vmull_s8(vget_high_s8(_val1), vget_low_s8(_w));

                    _s0 = vmlal_s8(_s0, vget_low_s8(_val2), vget_high_s8(_w));
                    _s1 = vmlal_s8(_s1, vget_high_s8(_val2), vget_high_s8(_w));
                    _s2 = vmlal_s8(_s2, vget_low_s8(_val3), vget_high_s8(_w));
                    _s3 = vmlal_s8(_s3, vget_high_s8(_val3), vget_high_s8(_w));

                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));
                    _sum4 = vaddw_s16(_sum4, vget_low_s16(_s2));
                    _sum5 = vaddw_s16(_sum5, vget_high_s16(_s2));
                    _sum6 = vaddw_s16(_sum6, vget_low_s16(_s3));
                    _sum7 = vaddw_s16(_sum7, vget_high_s16(_s3));

                    tmpptr += 64;
                    kptr0 += 16;
                }
                for (; j < nn; j++)
                {
                    int8x16_t _val0 = vld1q_s8(tmpptr);
                    int8x16_t _val1 = vld1q_s8(tmpptr + 16);
                    int8x8_t _w = vld1_s8(kptr0);

                    int16x8_t _s0 = vmull_s8(vget_low_s8(_val0), _w);
                    int16x8_t _s1 = vmull_s8(vget_high_s8(_val0), _w);
                    int16x8_t _s2 = vmull_s8(vget_low_s8(_val1), _w);
                    int16x8_t _s3 = vmull_s8(vget_high_s8(_val1), _w);

                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));
                    _sum4 = vaddw_s16(_sum4, vget_low_s16(_s2));
                    _sum5 = vaddw_s16(_sum5, vget_high_s16(_s2));
                    _sum6 = vaddw_s16(_sum6, vget_low_s16(_s3));
                    _sum7 = vaddw_s16(_sum7, vget_high_s16(_s3));

                    tmpptr += 32;
                    kptr0 += 8;
                }

                _sum0 = vaddq_s32(_sum0, _sum1);
                _sum2 = vaddq_s32(_sum2, _sum3);
                _sum4 = vaddq_s32(_sum4, _sum5);
                _sum6 = vaddq_s32(_sum6, _sum7);

                int32x2_t _s0 = vadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                int32x2_t _s2 = vadd_s32(vget_low_s32(_sum2), vget_high_s32(_sum2));
                int32x2_t _s4 = vadd_s32(vget_low_s32(_sum4), vget_high_s32(_sum4));
                int32x2_t _s6 = vadd_s32(vget_low_s32(_sum6), vget_high_s32(_sum6));
                int32x2_t _ss0 = vpadd_s32(_s0, _s2);
                int32x2_t _ss1 = vpadd_s32(_s4, _s6);
                int32x4_t _ss = vcombine_s32(_ss0, _ss1);

                vst1q_s32(outptr0, _ss);
                outptr0 += 4;
    #endif // __ARM_FEATURE_DOTPROD
            }
    #endif // __aarch64__
            for (; i + 1 < size; i += 2)
            {
    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
                const signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2].data();
    #else
                const signed char* tmpptr = tmp_ra[i / 4 + (i % 4) / 2].data();
    #endif
    #else
                const signed char* tmpptr = tmp_ra[i / 2].data();
    #endif
                const signed char* kptr0 = kernel_a[p / 4 + p % 4].data();

                int nn = inch * maxk; // inch always > 0

    #if __ARM_FEATURE_DOTPROD
                int32x2_t _sum0 = vdup_n_s32(0);
                int32x2_t _sum1 = vdup_n_s32(0);

                int j = 0;
                for (; j < nn; j++)
                {
                    int8x16_t _val01_lh = vld1q_s8(tmpptr);
                    int8x8_t _w_lh = vld1_s8(kptr0);

                    _sum0 = vdot_lane_s32(_sum0, vget_low_s8(_val01_lh), _w_lh, 0);
                    _sum1 = vdot_lane_s32(_sum1, vget_high_s8(_val01_lh), _w_lh, 1);

                    tmpptr += 16;
                    kptr0 += 8;
                }

                int32x2_t _sum = vadd_s32(_sum0, _sum1);

                vst1_s32(outptr0, _sum);
                outptr0 += 2;
    #else  // __ARM_FEATURE_DOTPROD
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 1 < nn; j += 2)
                {
                    int8x16_t _val0 = vld1q_s8(tmpptr);
                    int8x16_t _val1 = vld1q_s8(tmpptr + 16);
                    int8x16_t _w = vld1q_s8(kptr0);

                    int16x8_t _s0 = vmull_s8(vget_low_s8(_val0), vget_low_s8(_w));
                    int16x8_t _s1 = vmull_s8(vget_high_s8(_val0), vget_low_s8(_w));

                    _s0 = vmlal_s8(_s0, vget_low_s8(_val1), vget_high_s8(_w));
                    _s1 = vmlal_s8(_s1, vget_high_s8(_val1), vget_high_s8(_w));

                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));

                    tmpptr += 32;
                    kptr0 += 16;
                }
                for (; j < nn; j++)
                {
                    int8x16_t _val = vld1q_s8(tmpptr);
                    int8x8_t _w = vld1_s8(kptr0);

                    int16x8_t _s0 = vmull_s8(vget_low_s8(_val), _w);
                    int16x8_t _s1 = vmull_s8(vget_high_s8(_val), _w);

                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                    _sum2 = vaddw_s16(_sum2, vget_low_s16(_s1));
                    _sum3 = vaddw_s16(_sum3, vget_high_s16(_s1));

                    tmpptr += 16;
                    kptr0 += 8;
                }

                _sum0 = vaddq_s32(_sum0, _sum1);
                _sum2 = vaddq_s32(_sum2, _sum3);

                int32x2_t _s0 = vadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                int32x2_t _s2 = vadd_s32(vget_low_s32(_sum2), vget_high_s32(_sum2));
                int32x2_t _ss = vpadd_s32(_s0, _s2);

                vst1_s32(outptr0, _ss);
                outptr0 += 2;
    #endif // __ARM_FEATURE_DOTPROD
            }
            for (; i < size; i++)
            {
    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
                const signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2].data();
    #else
                const signed char* tmpptr = tmp_ra[i / 4 + (i % 4) / 2 + i % 2].data();
    #endif
    #else
                const signed char* tmpptr = tmp_ra[i / 2 + i % 2].data();
    #endif
                const signed char* kptr0 = kernel_a[p / 4 + p % 4].data();

                int nn = inch * maxk; // inch always > 0

    #if __ARM_FEATURE_DOTPROD
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x2_t _sum1 = vdup_n_s32(0);

                int j = 0;
                for (; j + 1 < nn; j += 2)
                {
                    int8x16_t _val = vld1q_s8(tmpptr);
                    int8x16_t _w = vld1q_s8(kptr0);

                    _sum0 = vdotq_s32(_sum0, _val, _w);

                    tmpptr += 16;
                    kptr0 += 16;
                }
                for (; j < nn; j++)
                {
                    int8x8_t _val = vld1_s8(tmpptr);
                    int8x8_t _w = vld1_s8(kptr0);

                    _sum1 = vdot_s32(_sum1, _val, _w);

                    tmpptr += 8;
                    kptr0 += 8;
                }

                int sum = vaddvq_s32(_sum0) + vaddv_s32(_sum1);

                outptr0[0] = sum;
                outptr0 += 1;
    #else // __ARM_FEATURE_DOTPROD
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 1 < nn; j += 2)
                {
                    int8x16_t _val = vld1q_s8(tmpptr);
                    int8x16_t _w = vld1q_s8(kptr0);

                    int16x8_t _s8 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w));
                    _s8 = vmlal_s8(_s8, vget_high_s8(_val), vget_high_s8(_w));

                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s8));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s8));

                    tmpptr += 16;
                    kptr0 += 16;
                }
                for (; j < nn; j++)
                {
                    int8x8_t _val = vld1_s8(tmpptr);
                    int8x8_t _w = vld1_s8(kptr0);

                    int16x8_t _s8 = vmull_s8(_val, _w);

                    _sum0 = vaddw_s16(_sum0, vget_low_s16(_s8));
                    _sum1 = vaddw_s16(_sum1, vget_high_s16(_s8));

                    tmpptr += 8;
                    kptr0 += 8;
                }

                int32x4_t _sum = vaddq_s32(_sum0, _sum1);
    #if __aarch64__
                int sum = vaddvq_s32(_sum); // dot
    #else
                int32x2_t _ss = vadd_s32(vget_low_s32(_sum), vget_high_s32(_sum));
                _ss = vpadd_s32(_ss, _ss);
                int sum = vget_lane_s32(_ss, 0);
    #endif

                outptr0[0] = sum;
                outptr0 += 1;
    #endif // __ARM_FEATURE_DOTPROD
            }
        }
    });
}

void im2col_sgemm_conv2d_int8_pack8to4_impl_neon(
    const Tensor& im2col,
    const Tensor& kernel_tf_,
    const Tensor& bias_,
    int64_t input_channels,
    int64_t output_channels,
    Tensor& output) {
    
    const int size = im2col.size(2);
    const int maxk = im2col.size(1);
    const int inch = im2col.size(0);

    const int outch = output.size(1);
    
    auto output_a = output.accessor<int, 4, 4>()[0];
    auto im2col_a = im2col.accessor<signed char, 3, 8>();
    auto kernel_a = kernel_tf_.accessor<signed char, 3>();

    // permute
    Tensor tmp;
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
    if (size >= 16)
        tmp = otter::empty({size / 16 + (size % 16) / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, inch, 16 * maxk}, otter::ScalarType::Byte8);
    else if (size >= 8)
        tmp = otter::empty({size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, inch, 8 * maxk}, otter::ScalarType::Byte8);
    else if (size >= 4)
        tmp = otter::empty({size / 4 + (size % 4) / 2 + size % 2, inch, 4 * maxk}, otter::ScalarType::Byte8);
    else if (size >= 2)
        tmp = otter::empty({size / 2 + size % 2, inch, 2 * maxk}, otter::ScalarType::Byte8);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Byte8);
#else  // __ARM_FEATURE_DOTPROD
    if (size >= 4)
        tmp = otter::empty({size / 4 + (size % 4) / 2 + size % 2, inch, 4 * maxk}, otter::ScalarType::Byte8);
    else if (size >= 2)
        tmp = otter::empty({size / 2 + size % 2, inch, 2 * maxk}, otter::ScalarType::Byte8);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Byte8);
#endif // __ARM_FEATURE_DOTPROD
#else  // __aarch64__
    if (size >= 2)
        tmp = otter::empty({size / 2 + size % 2, inch, 2 * maxk}, otter::ScalarType::Byte8);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Byte8);
#endif // __aarch64__
    
    auto tmp_ra = tmp.raw_accessor<signed char, 3>();
    
    {
#if __aarch64__
#if __ARM_FEATURE_DOTPROD
        int nn_size = size >> 4;
        int remain_size_start = 0;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end))
            {
                int i = remain_size_start + ii * 16;

                signed char* tmpptr = tmp_ra[i / 16].data();

                for (int q = 0; q < inch; q++)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++)
                    {
                        // split pack8 to pack4
                        asm volatile(
                            "prfm   pldl1keep, [%0, #512]       \n"
                            "ld2    {v0.4s, v1.4s}, [%0], #32   \n"
                            "ld2    {v2.4s, v3.4s}, [%0], #32   \n"
                            "ld2    {v4.4s, v5.4s}, [%0], #32   \n"
                            "ld2    {v6.4s, v7.4s}, [%0]        \n"
                            "sub    %0, %0, #96                 \n"
                            "st1    {v0.16b}, [%1], #16         \n"
                            "st1    {v2.16b}, [%1], #16         \n"
                            "st1    {v4.16b}, [%1], #16         \n"
                            "st1    {v6.16b}, [%1], #16         \n"
                            "st1    {v1.16b}, [%1], #16         \n"
                            "st1    {v3.16b}, [%1], #16         \n"
                            "st1    {v5.16b}, [%1], #16         \n"
                            "st1    {v7.16b}, [%1], #16         \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
                        img0 += size * 8;
                    }
                }
            }
        });

        remain_size_start += nn_size << 4;
        nn_size = (size - remain_size_start) >> 3;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end))
            {
                int i = remain_size_start + ii * 8;

                signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8].data();

                for (int q = 0; q < inch; q++)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++)
                    {
                        asm volatile(
                            "prfm   pldl1keep, [%0, #512]       \n"
                            "ld2    {v0.4s, v1.4s}, [%0], #32   \n"
                            "ld2    {v2.4s, v3.4s}, [%0]        \n"
                            "sub    %0, %0, #32                 \n"
                            "st1    {v0.16b}, [%1], #16         \n"
                            "st1    {v2.16b}, [%1], #16         \n"
                            "st1    {v1.16b}, [%1], #16         \n"
                            "st1    {v3.16b}, [%1], #16         \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0", "v1", "v2", "v3");
                        img0 += size * 8;
                    }
                }
            }
        });

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;
#else  // __ARM_FEATURE_DOTPROD
        int remain_size_start = 0;
        int nn_size = (size - remain_size_start) >> 2;
#endif // __ARM_FEATURE_DOTPROD

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end))
            {
                int i = remain_size_start + ii * 4;

    #if __ARM_FEATURE_DOTPROD
                signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4].data();
    #else
                signed char* tmpptr = tmp_ra[i / 4].data();
    #endif

                for (int q = 0; q < inch; q++)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++)
                    {
    #if __ARM_FEATURE_DOTPROD
                        asm volatile(
                            "prfm   pldl1keep, [%0, #256]       \n"
                            "ld2    {v0.4s, v1.4s}, [%0]        \n"
                            "st1    {v0.4s, v1.4s}, [%1], #32   \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0", "v1");
    #else
                        asm volatile(
                            "prfm   pldl1keep, [%0, #256]       \n"
                            "ld1    {v0.16b, v1.16b}, [%0]      \n"
                            "st1    {v0.16b, v1.16b}, [%1], #32 \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0", "v1");
    #endif // __ARM_FEATURE_DOTPROD
                        img0 += size * 8;
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

    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
                signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2].data();
    #else
                signed char* tmpptr = tmp_ra[i / 4 + (i % 4) / 2].data();
    #endif
    #else
                signed char* tmpptr = tmp_ra[i / 2].data();
    #endif

                for (int q = 0; q < inch; q++)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++)
                    {
    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
                        asm volatile(
                            "prfm   pldl1keep, [%0, #128]   \n"
                            "ld2    {v0.2s, v1.2s}, [%0]    \n"
                            "st1    {v0.2s, v1.2s}, [%1], #16 \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0", "v1");
    #else
                        asm volatile(
                            "prfm   pldl1keep, [%0, #128]   \n"
                            "ld1    {v0.16b}, [%0]          \n"
                            "st1    {v0.16b}, [%1], #16     \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0");
    #endif // __ARM_FEATURE_DOTPROD
    #else
                        asm volatile(
                            "pld        [%0, #128]          \n"
                            "vld1.s8    {d0-d1}, [%0 :64]   \n"
                            "vst1.s8    {d0-d1}, [%1 :64]!  \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "q0");
    #endif
                        img0 += size * 8;
                    }
                }
            }
        });

        remain_size_start += nn_size << 1;

        otter::parallel_for(remain_size_start, size, 0, [&](int64_t begin, int64_t end) {
            for (const auto i : otter::irange(begin, end))
            {
    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
                signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2].data();
    #else
                signed char* tmpptr = tmp_ra[i / 4 + (i % 4) / 2 + i % 2].data();
    #endif
    #else
                signed char* tmpptr = tmp_ra[i / 2 + i % 2].data();
    #endif

                for (int q = 0; q < inch; q++)
                {
                    const signed char* img0 = (const signed char*)im2col_a[q].data() + i * 8;

                    for (int k = 0; k < maxk; k++)
                    {
    #if __aarch64__
                        asm volatile(
                            "prfm   pldl1keep, [%0, #64]    \n"
                            "ld1    {v0.8b}, [%0]           \n"
                            "st1    {v0.8b}, [%1], #8       \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0");
    #else
                        asm volatile(
                            "pld        [%0, #64]           \n"
                            "vld1.s8    {d0}, [%0 :64]      \n"
                            "vst1.s8    {d0}, [%1 :64]!     \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "d0");
    #endif
                        img0 += size * 8;
                    }
                }
            }
        });
    }

    otter::parallel_for(0, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            int* outptr0 = output_a[p].data();

            int i = 0;
    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
            for (; i + 15 < size; i += 16)
            {
                const signed char* tmpptr = tmp_ra[i / 16].data();
                const signed char* kptr0 = kernel_a[p].data();

                int nn = inch * maxk; // inch always > 0

                asm volatile(
                    "ld1    {v24.16b}, [%3], #16        \n" // _w0123_l

                    "eor    v0.16b, v0.16b, v0.16b      \n"
                    "eor    v1.16b, v1.16b, v1.16b      \n"

                    "ld1    {v16.16b}, [%2], #16        \n" // _val0123_l

                    "eor    v2.16b, v2.16b, v2.16b      \n"
                    "eor    v3.16b, v3.16b, v3.16b      \n"
                    "eor    v4.16b, v4.16b, v4.16b      \n"
                    "eor    v5.16b, v5.16b, v5.16b      \n"
                    "eor    v6.16b, v6.16b, v6.16b      \n"
                    "eor    v7.16b, v7.16b, v7.16b      \n"
                    "eor    v8.16b, v8.16b, v8.16b      \n"
                    "eor    v9.16b, v9.16b, v9.16b      \n"
                    "eor    v10.16b, v10.16b, v10.16b   \n"
                    "eor    v11.16b, v11.16b, v11.16b   \n"
                    "eor    v12.16b, v12.16b, v12.16b   \n"
                    "eor    v13.16b, v13.16b, v13.16b   \n"
                    "eor    v14.16b, v14.16b, v14.16b   \n"
                    "eor    v15.16b, v15.16b, v15.16b   \n"

                    "0:                                 \n"

                    "ld1    {v17.16b}, [%2], #16        \n" // _val4567_l

                    "sdot   v0.4s, v24.16b, v16.4b[0]   \n"
                    "sdot   v1.4s, v24.16b, v16.4b[1]   \n"
                    "sdot   v2.4s, v24.16b, v16.4b[2]   \n"
                    "sdot   v3.4s, v24.16b, v16.4b[3]   \n"

                    "ld1    {v18.16b}, [%2], #16        \n" // _val891011_l

                    "sdot   v4.4s, v24.16b, v17.4b[0]   \n"
                    "sdot   v5.4s, v24.16b, v17.4b[1]   \n"
                    "sdot   v6.4s, v24.16b, v17.4b[2]   \n"
                    "sdot   v7.4s, v24.16b, v17.4b[3]   \n"

                    "ld1    {v19.16b}, [%2], #16        \n" // _val12131415_l

                    "sdot   v8.4s, v24.16b, v18.4b[0]   \n"
                    "sdot   v9.4s, v24.16b, v18.4b[1]   \n"

                    "ld1    {v25.16b}, [%3], #16        \n" // _w0123_h

                    "sdot   v10.4s, v24.16b, v18.4b[2]  \n"
                    "sdot   v11.4s, v24.16b, v18.4b[3]  \n"

                    "ld1    {v20.16b}, [%2], #16        \n" // _val0123_h

                    "sdot   v12.4s, v24.16b, v19.4b[0]  \n"
                    "sdot   v13.4s, v24.16b, v19.4b[1]  \n"
                    "sdot   v14.4s, v24.16b, v19.4b[2]  \n"
                    "sdot   v15.4s, v24.16b, v19.4b[3]  \n"

                    "ld1    {v21.16b}, [%2], #16        \n" // _val4567_h

                    "sdot   v0.4s, v25.16b, v20.4b[0]   \n"
                    "sdot   v1.4s, v25.16b, v20.4b[1]   \n"
                    "sdot   v2.4s, v25.16b, v20.4b[2]   \n"
                    "sdot   v3.4s, v25.16b, v20.4b[3]   \n"

                    "ld1    {v22.16b}, [%2], #16        \n" // _val891011_h

                    "sdot   v4.4s, v25.16b, v21.4b[0]   \n"
                    "sdot   v5.4s, v25.16b, v21.4b[1]   \n"
                    "sdot   v6.4s, v25.16b, v21.4b[2]   \n"
                    "sdot   v7.4s, v25.16b, v21.4b[3]   \n"

                    "ld1    {v23.16b}, [%2], #16        \n" // _val12131415_h

                    "sdot   v8.4s, v25.16b, v22.4b[0]   \n"
                    "sdot   v9.4s, v25.16b, v22.4b[1]   \n"

                    "ld1    {v24.16b}, [%3], #16        \n" // _w0123_l

                    "sdot   v10.4s, v25.16b, v22.4b[2]  \n"
                    "sdot   v11.4s, v25.16b, v22.4b[3]  \n"

                    "ld1    {v16.16b}, [%2], #16        \n" // _val0123_l

                    "sdot   v12.4s, v25.16b, v23.4b[0]  \n"
                    "sdot   v13.4s, v25.16b, v23.4b[1]  \n"

                    "subs   %w1, %w1, #1                \n"

                    "sdot   v14.4s, v25.16b, v23.4b[2]  \n"
                    "sdot   v15.4s, v25.16b, v23.4b[3]  \n"

                    "bne    0b                          \n"

                    "sub    %2, %2, #16                 \n"
                    "sub    %3, %3, #16                 \n"

                    "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                    "st1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64 \n"
                    "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0], #64 \n"
                    "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%0], #64 \n"
                    : "=r"(outptr0),
                    "=r"(nn),
                    "=r"(tmpptr),
                    "=r"(kptr0)
                    : "0"(outptr0),
                    "1"(nn),
                    "2"(tmpptr),
                    "3"(kptr0)
                    : "memory", "x4", "x5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            for (; i + 7 < size; i += 8)
            {
                const signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8].data();
                const signed char* kptr0 = kernel_a[p].data();

                int nn = inch * maxk; // inch always > 0

                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);
                int32x4_t _sum4 = vdupq_n_s32(0);
                int32x4_t _sum5 = vdupq_n_s32(0);
                int32x4_t _sum6 = vdupq_n_s32(0);
                int32x4_t _sum7 = vdupq_n_s32(0);

                for (int j = 0; j < nn; j++)
                {
                    int8x16_t _val0123_l = vld1q_s8(tmpptr);
                    int8x16_t _val4567_l = vld1q_s8(tmpptr + 16);

                    int8x16_t _w0123_l = vld1q_s8(kptr0);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0123_l, _val0123_l, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0123_l, _val0123_l, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _w0123_l, _val0123_l, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _w0123_l, _val0123_l, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _w0123_l, _val4567_l, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _w0123_l, _val4567_l, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _w0123_l, _val4567_l, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _w0123_l, _val4567_l, 3);

                    int8x16_t _val0123_h = vld1q_s8(tmpptr + 32);
                    int8x16_t _val4567_h = vld1q_s8(tmpptr + 48);

                    int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0123_h, _val0123_h, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0123_h, _val0123_h, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _w0123_h, _val0123_h, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _w0123_h, _val0123_h, 3);
                    _sum4 = vdotq_laneq_s32(_sum4, _w0123_h, _val4567_h, 0);
                    _sum5 = vdotq_laneq_s32(_sum5, _w0123_h, _val4567_h, 1);
                    _sum6 = vdotq_laneq_s32(_sum6, _w0123_h, _val4567_h, 2);
                    _sum7 = vdotq_laneq_s32(_sum7, _w0123_h, _val4567_h, 3);

                    tmpptr += 64;
                    kptr0 += 32;
                }

                vst1q_s32(outptr0, _sum0);
                vst1q_s32(outptr0 + 4, _sum1);
                vst1q_s32(outptr0 + 8, _sum2);
                vst1q_s32(outptr0 + 12, _sum3);
                vst1q_s32(outptr0 + 16, _sum4);
                vst1q_s32(outptr0 + 20, _sum5);
                vst1q_s32(outptr0 + 24, _sum6);
                vst1q_s32(outptr0 + 28, _sum7);
                outptr0 += 32;
            }
    #endif
            for (; i + 3 < size; i += 4)
            {
    #if __ARM_FEATURE_DOTPROD
                const signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4].data();
    #else
                const signed char* tmpptr = tmp_ra[i / 4].data();
    #endif
                const signed char* kptr0 = kernel_a[p].data();

                int nn = inch * maxk; // inch always > 0

    #if __ARM_FEATURE_DOTPROD
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);

                for (int j = 0; j < nn; j++)
                {
                    int8x16_t _val0123_l = vld1q_s8(tmpptr);
                    int8x16_t _w0123_l = vld1q_s8(kptr0);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0123_l, _val0123_l, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0123_l, _val0123_l, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _w0123_l, _val0123_l, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _w0123_l, _val0123_l, 3);

                    int8x16_t _val0123_h = vld1q_s8(tmpptr + 16);
                    int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0123_h, _val0123_h, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0123_h, _val0123_h, 1);
                    _sum2 = vdotq_laneq_s32(_sum2, _w0123_h, _val0123_h, 2);
                    _sum3 = vdotq_laneq_s32(_sum3, _w0123_h, _val0123_h, 3);

                    tmpptr += 32;
                    kptr0 += 32;
                }

                vst1q_s32(outptr0, _sum0);
                vst1q_s32(outptr0 + 4, _sum1);
                vst1q_s32(outptr0 + 8, _sum2);
                vst1q_s32(outptr0 + 12, _sum3);
                outptr0 += 16;
    #else  // __ARM_FEATURE_DOTPROD
                asm volatile(
                    "eor    v0.16b, v0.16b, v0.16b      \n"
                    "eor    v1.16b, v1.16b, v1.16b      \n"
                    "eor    v2.16b, v2.16b, v2.16b      \n"
                    "eor    v3.16b, v3.16b, v3.16b      \n"
                    "eor    v4.16b, v4.16b, v4.16b      \n"
                    "eor    v5.16b, v5.16b, v5.16b      \n"
                    "eor    v6.16b, v6.16b, v6.16b      \n"
                    "eor    v7.16b, v7.16b, v7.16b      \n"
                    "eor    v8.16b, v8.16b, v8.16b      \n"
                    "eor    v9.16b, v9.16b, v9.16b      \n"
                    "eor    v10.16b, v10.16b, v10.16b   \n"
                    "eor    v11.16b, v11.16b, v11.16b   \n"
                    "eor    v12.16b, v12.16b, v12.16b   \n"
                    "eor    v13.16b, v13.16b, v13.16b   \n"
                    "eor    v14.16b, v14.16b, v14.16b   \n"
                    "eor    v15.16b, v15.16b, v15.16b   \n"

                    "prfm   pldl1keep, [%2, #128]       \n"

                    "prfm   pldl1keep, [%3, #256]       \n"

                    "lsr    w4, %w1, #1                 \n" // w4 = nn >> 1
                    "cmp    w4, #0                      \n"
                    "beq    1f                          \n"

                    "prfm   pldl1keep, [%3, #512]       \n"

                    "add    x5, %2, #16                 \n"

                    "prfm   pldl1keep, [x5, #128]       \n"

                    "ld1    {v16.16b}, [%2]             \n" // val L H
                    "ld1    {v20.16b, v21.16b, v22.16b, v23.16b}, [%3], #64 \n"
                    "add    %2, %2, #32                 \n"
                    "ext    v17.16b, v16.16b, v16.16b, #8 \n" // val H L

                    "ld1    {v18.16b}, [%2]             \n"
                    "add    %2, %2, #32                 \n"

                    "0:                                 \n"

                    "smull  v24.8h, v16.8b,  v20.8b     \n"
                    "prfm   pldl1keep, [%3, #256]       \n"
                    "smull2 v25.8h, v17.16b, v20.16b    \n"
                    "prfm   pldl1keep, [%3, #512]       \n"
                    "smull  v26.8h, v16.8b,  v21.8b     \n"
                    "subs   w4, w4, #1                  \n"
                    "smull2 v27.8h, v17.16b, v21.16b    \n"
                    "ext    v19.16b, v18.16b, v18.16b, #8 \n" // val H L

                    "smlal  v24.8h, v18.8b,  v22.8b     \n"
                    "smlal2 v25.8h, v19.16b, v22.16b    \n"
                    "smlal  v26.8h, v18.8b,  v23.8b     \n"
                    "smlal2 v27.8h, v19.16b, v23.16b    \n"

                    "smull2 v29.8h, v16.16b, v20.16b    \n"
                    "sadalp v0.4s, v24.8h               \n"
                    "smull  v28.8h, v17.8b,  v20.8b     \n"
                    "sadalp v1.4s, v25.8h               \n"
                    "smull2 v31.8h, v16.16b, v21.16b    \n"
                    "ld1    {v16.16b}, [x5]             \n" // val L H
                    "smull  v30.8h, v17.8b,  v21.8b     \n"
                    "add    x5, x5, #32                 \n"
                    "smlal2 v29.8h, v18.16b, v22.16b    \n"
                    "sadalp v2.4s, v26.8h               \n"
                    "smlal  v28.8h, v19.8b,  v22.8b     \n"
                    "sadalp v3.4s, v27.8h               \n"
                    "smlal2 v31.8h, v18.16b, v23.16b    \n"
                    "ld1    {v18.16b}, [x5]             \n"
                    "smlal  v30.8h, v19.8b,  v23.8b     \n"
                    "ext    v17.16b, v16.16b, v16.16b, #8 \n" // val H L

                    "smull  v24.8h, v16.8b,  v20.8b     \n"
                    "add    x5, x5, #32                 \n"
                    "smull2 v25.8h, v17.16b, v20.16b    \n"
                    "prfm   pldl1keep, [x5, #128]       \n"
                    "smull  v26.8h, v16.8b,  v21.8b     \n"
                    "prfm   pldl1keep, [x5, #384]       \n"
                    "smull2 v27.8h, v17.16b, v21.16b    \n"
                    "ext    v19.16b, v18.16b, v18.16b, #8 \n" // val H L

                    "smlal  v24.8h, v18.8b,  v22.8b     \n"
                    "sadalp v5.4s, v29.8h               \n"
                    "smlal2 v25.8h, v19.16b, v22.16b    \n"
                    "sadalp v4.4s, v28.8h               \n"
                    "smlal  v26.8h, v18.8b,  v23.8b     \n"
                    "sadalp v7.4s, v31.8h               \n"
                    "smlal2 v27.8h, v19.16b, v23.16b    \n"
                    "sadalp v6.4s, v30.8h               \n"

                    "smull2 v29.8h, v16.16b, v20.16b    \n"
                    "sadalp v8.4s, v24.8h               \n"
                    "smull  v28.8h, v17.8b,  v20.8b     \n"
                    "sadalp v9.4s, v25.8h               \n"
                    "smull2 v31.8h, v16.16b, v21.16b    \n"
                    "ld1    {v16.16b}, [%2]             \n" // val L H
                    "smull  v30.8h, v17.8b,  v21.8b     \n"
                    "add    %2, %2, #32                 \n"
                    "smlal2 v29.8h, v18.16b, v22.16b    \n"
                    "sadalp v10.4s, v26.8h              \n"
                    "smlal  v28.8h, v19.8b,  v22.8b     \n"
                    "sadalp v11.4s, v27.8h              \n"
                    "smlal2 v31.8h, v18.16b, v23.16b    \n"
                    "ld1    {v18.16b}, [%2]             \n"
                    "smlal  v30.8h, v19.8b,  v23.8b     \n"
                    "add    %2, %2, #32                 \n"
                    "ld1    {v20.16b, v21.16b, v22.16b, v23.16b}, [%3], #64 \n"

                    "sadalp v13.4s, v29.8h              \n"
                    "prfm   pldl1keep, [%2, #128]       \n"
                    "sadalp v12.4s, v28.8h              \n"
                    "prfm   pldl1keep, [%2, #384]       \n"
                    "sadalp v15.4s, v31.8h              \n"
                    "ext    v17.16b, v16.16b, v16.16b, #8 \n" // val H L

                    "sadalp v14.4s, v30.8h              \n"

                    "bne    0b                          \n"

                    "sub    %2, %2, #64                 \n"
                    "sub    %3, %3, #64                 \n"

                    "1:                                 \n"
                    "and    w4, %w1, #1                 \n" // w4 = remain = nn & 1
                    "cmp    w4, #0                      \n" // w4 > 0
                    "beq    2f                          \n"

                    "ld1    {v16.8b, v17.8b}, [%2], #16 \n"
                    "ld1    {v20.8b, v21.8b, v22.8b, v23.8b}, [%3], #32 \n"

                    "smull  v24.8h, v16.8b, v20.8b      \n"
                    "smull  v25.8h, v16.8b, v21.8b      \n"
                    "smull  v26.8h, v16.8b, v22.8b      \n"
                    "ld1    {v18.8b, v19.8b}, [%2], #16 \n"
                    "smull  v27.8h, v16.8b, v23.8b      \n"
                    "sadalp v0.4s, v24.8h               \n"
                    "smull  v28.8h, v17.8b, v20.8b      \n"
                    "sadalp v1.4s, v25.8h               \n"
                    "smull  v29.8h, v17.8b, v21.8b      \n"
                    "sadalp v2.4s, v26.8h               \n"
                    "smull  v30.8h, v17.8b, v22.8b      \n"
                    "sadalp v3.4s, v27.8h               \n"
                    "smull  v31.8h, v17.8b, v23.8b      \n"
                    "sadalp v4.4s, v28.8h               \n"
                    "smull  v24.8h, v18.8b, v20.8b      \n"
                    "sadalp v5.4s, v29.8h               \n"
                    "smull  v25.8h, v18.8b, v21.8b      \n"
                    "sadalp v6.4s, v30.8h               \n"
                    "smull  v26.8h, v18.8b, v22.8b      \n"
                    "sadalp v7.4s, v31.8h               \n"
                    "smull  v27.8h, v18.8b, v23.8b      \n"
                    "sadalp v8.4s, v24.8h               \n"
                    "smull  v28.8h, v19.8b, v20.8b      \n"
                    "sadalp v9.4s, v25.8h               \n"
                    "smull  v29.8h, v19.8b, v21.8b      \n"
                    "sadalp v10.4s, v26.8h              \n"
                    "smull  v30.8h, v19.8b, v22.8b      \n"
                    "sadalp v11.4s, v27.8h              \n"
                    "smull  v31.8h, v19.8b, v23.8b      \n"

                    "sadalp v12.4s, v28.8h              \n"
                    "sadalp v13.4s, v29.8h              \n"
                    "sadalp v14.4s, v30.8h              \n"
                    "sadalp v15.4s, v31.8h              \n"

                    "2:                                 \n"

                    "addp   v0.4s, v0.4s, v1.4s         \n"
                    "addp   v2.4s, v2.4s, v3.4s         \n"
                    "addp   v4.4s, v4.4s, v5.4s         \n"
                    "addp   v6.4s, v6.4s, v7.4s         \n"
                    "addp   v8.4s, v8.4s, v9.4s         \n"
                    "addp   v10.4s, v10.4s, v11.4s      \n"
                    "addp   v12.4s, v12.4s, v13.4s      \n"
                    "addp   v14.4s, v14.4s, v15.4s      \n"

                    "addp   v0.4s, v0.4s, v2.4s         \n"
                    "addp   v1.4s, v4.4s, v6.4s         \n"
                    "addp   v2.4s, v8.4s, v10.4s        \n"
                    "addp   v3.4s, v12.4s, v14.4s       \n"

                    "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"

                    : "=r"(outptr0),
                    "=r"(nn),
                    "=r"(tmpptr),
                    "=r"(kptr0)
                    : "0"(outptr0),
                    "1"(nn),
                    "2"(tmpptr),
                    "3"(kptr0)
                    : "memory", "x4", "x5", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    #endif // __ARM_FEATURE_DOTPROD
            }
    #endif // __aarch64__
            for (; i + 1 < size; i += 2)
            {
    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
                const signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2].data();
    #else
                const signed char* tmpptr = tmp_ra[i / 4 + (i % 4) / 2].data();
    #endif
    #else
                const signed char* tmpptr = tmp_ra[i / 2].data();
    #endif
                const signed char* kptr0 = kernel_a[p].data();

                int nn = inch * maxk; // inch always > 0

    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);

                for (int j = 0; j < nn; j++)
                {
                    int8x16_t _val01_l_h = vld1q_s8(tmpptr);
                    int8x16_t _w0123_l = vld1q_s8(kptr0);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0123_l, _val01_l_h, 0);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0123_l, _val01_l_h, 1);

                    int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

                    _sum0 = vdotq_laneq_s32(_sum0, _w0123_h, _val01_l_h, 2);
                    _sum1 = vdotq_laneq_s32(_sum1, _w0123_h, _val01_l_h, 3);

                    tmpptr += 16;
                    kptr0 += 32;
                }

                vst1q_s32(outptr0, _sum0);
                vst1q_s32(outptr0 + 4, _sum1);
                outptr0 += 8;
    #else  // __ARM_FEATURE_DOTPROD
                int32x4_t _sum00 = vdupq_n_s32(0);
                int32x4_t _sum01 = vdupq_n_s32(0);
                int32x4_t _sum02 = vdupq_n_s32(0);
                int32x4_t _sum03 = vdupq_n_s32(0);
                int32x4_t _sum10 = vdupq_n_s32(0);
                int32x4_t _sum11 = vdupq_n_s32(0);
                int32x4_t _sum12 = vdupq_n_s32(0);
                int32x4_t _sum13 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 1 < nn; j += 2)
                {
                    int8x16_t _val0 = vld1q_s8(tmpptr);
                    int8x16_t _val1 = vld1q_s8(tmpptr + 16);

                    int8x16_t _w01 = vld1q_s8(kptr0);
                    int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                    int16x8_t _wv00 = vmull_s8(vget_low_s8(_val0), vget_low_s8(_w01));
                    int16x8_t _wv01 = vmull_s8(vget_low_s8(_val0), vget_high_s8(_w01));
                    int16x8_t _wv02 = vmull_s8(vget_low_s8(_val0), vget_low_s8(_w23));
                    int16x8_t _wv03 = vmull_s8(vget_low_s8(_val0), vget_high_s8(_w23));

                    int16x8_t _wv10 = vmull_s8(vget_high_s8(_val0), vget_low_s8(_w01));
                    int16x8_t _wv11 = vmull_s8(vget_high_s8(_val0), vget_high_s8(_w01));
                    int16x8_t _wv12 = vmull_s8(vget_high_s8(_val0), vget_low_s8(_w23));
                    int16x8_t _wv13 = vmull_s8(vget_high_s8(_val0), vget_high_s8(_w23));

                    int8x16_t _w45 = vld1q_s8(kptr0 + 32);
                    int8x16_t _w67 = vld1q_s8(kptr0 + 48);

                    _wv00 = vmlal_s8(_wv00, vget_low_s8(_val1), vget_low_s8(_w45));
                    _wv01 = vmlal_s8(_wv01, vget_low_s8(_val1), vget_high_s8(_w45));
                    _wv02 = vmlal_s8(_wv02, vget_low_s8(_val1), vget_low_s8(_w67));
                    _wv03 = vmlal_s8(_wv03, vget_low_s8(_val1), vget_high_s8(_w67));

                    _wv10 = vmlal_s8(_wv10, vget_high_s8(_val1), vget_low_s8(_w45));
                    _wv11 = vmlal_s8(_wv11, vget_high_s8(_val1), vget_high_s8(_w45));
                    _wv12 = vmlal_s8(_wv12, vget_high_s8(_val1), vget_low_s8(_w67));
                    _wv13 = vmlal_s8(_wv13, vget_high_s8(_val1), vget_high_s8(_w67));

                    _sum00 = vpadalq_s16(_sum00, _wv00);
                    _sum01 = vpadalq_s16(_sum01, _wv01);
                    _sum02 = vpadalq_s16(_sum02, _wv02);
                    _sum03 = vpadalq_s16(_sum03, _wv03);
                    _sum10 = vpadalq_s16(_sum10, _wv10);
                    _sum11 = vpadalq_s16(_sum11, _wv11);
                    _sum12 = vpadalq_s16(_sum12, _wv12);
                    _sum13 = vpadalq_s16(_sum13, _wv13);

                    tmpptr += 32;
                    kptr0 += 64;
                }
                for (; j < nn; j++)
                {
                    int8x16_t _val = vld1q_s8(tmpptr);

                    int8x16_t _w01 = vld1q_s8(kptr0);
                    int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                    int16x8_t _wv00 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w01));
                    int16x8_t _wv01 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w01));
                    int16x8_t _wv02 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w23));
                    int16x8_t _wv03 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w23));
                    int16x8_t _wv10 = vmull_s8(vget_high_s8(_val), vget_low_s8(_w01));
                    int16x8_t _wv11 = vmull_s8(vget_high_s8(_val), vget_high_s8(_w01));
                    int16x8_t _wv12 = vmull_s8(vget_high_s8(_val), vget_low_s8(_w23));
                    int16x8_t _wv13 = vmull_s8(vget_high_s8(_val), vget_high_s8(_w23));

                    _sum00 = vpadalq_s16(_sum00, _wv00);
                    _sum01 = vpadalq_s16(_sum01, _wv01);
                    _sum02 = vpadalq_s16(_sum02, _wv02);
                    _sum03 = vpadalq_s16(_sum03, _wv03);
                    _sum10 = vpadalq_s16(_sum10, _wv10);
                    _sum11 = vpadalq_s16(_sum11, _wv11);
                    _sum12 = vpadalq_s16(_sum12, _wv12);
                    _sum13 = vpadalq_s16(_sum13, _wv13);

                    tmpptr += 16;
                    kptr0 += 32;
                }

                int32x4_t _s001 = vpaddq_s32(_sum00, _sum01);
                int32x4_t _s023 = vpaddq_s32(_sum02, _sum03);
                int32x4_t _s101 = vpaddq_s32(_sum10, _sum11);
                int32x4_t _s123 = vpaddq_s32(_sum12, _sum13);

                int32x4_t _s00123 = vpaddq_s32(_s001, _s023);
                int32x4_t _s10123 = vpaddq_s32(_s101, _s123);

                vst1q_s32(outptr0, _s00123);
                vst1q_s32(outptr0 + 4, _s10123);
                outptr0 += 8;
    #endif // __ARM_FEATURE_DOTPROD
    #else  // __aarch64__
                asm volatile(
                    "veor       q0, q0              \n"
                    "veor       q1, q1              \n"
                    "veor       q2, q2              \n"
                    "veor       q3, q3              \n"
                    "veor       q4, q4              \n"
                    "veor       q5, q5              \n"
                    "veor       q6, q6              \n"
                    "veor       q7, q7              \n"

                    "pld        [%2, #256]          \n"

                    "lsr        r4, %1, #1          \n" // r4 = nn = size >> 1
                    "cmp        r4, #0              \n"
                    "beq        1f                  \n"

                    "add        r5, %3, #16         \n"
                    "pld        [%3, #128]          \n"
                    "mov        r6, #32             \n"
                    "pld        [%3, #384]          \n"

                    "vld1.s8    {d20-d21}, [%3 :128], r6 \n" // _w01

                    "vld1.s8    {d16-d19}, [%2 :128]! \n" // _val0 _val1

                    "vld1.s8    {d22-d23}, [%3 :128], r6 \n" // _w45

                    "0:                             \n"

                    "vmull.s8   q12, d16, d20       \n"
                    "pld        [%2, #256]          \n"
                    "vmull.s8   q13, d16, d21       \n"
                    "pld        [%3, #384]          \n"
                    "vmull.s8   q14, d17, d20       \n"
                    "vmull.s8   q15, d17, d21       \n"
                    "vld1.s8    {d20-d21}, [r5 :128], r6 \n" // _w23

                    "vmlal.s8   q12, d18, d22       \n"
                    "vmlal.s8   q13, d18, d23       \n"
                    "subs       r4, r4, #1          \n"
                    "vmlal.s8   q14, d19, d22       \n"
                    "vmlal.s8   q15, d19, d23       \n"
                    "vld1.s8    {d22-d23}, [r5 :128], r6 \n" // _w67

                    "vpadal.s16 q0, q12             \n"
                    "vmull.s8   q12, d16, d20       \n"
                    "vpadal.s16 q1, q13             \n"
                    "vmull.s8   q13, d16, d21       \n"
                    "vpadal.s16 q4, q14             \n"
                    "vmull.s8   q14, d17, d20       \n"
                    "vpadal.s16 q5, q15             \n"
                    "vmull.s8   q15, d17, d21       \n"
                    "vld1.s8    {d16-d17}, [%2 :128]! \n" // _val0

                    "vmlal.s8   q12, d18, d22       \n"
                    "vld1.s8    {d20-d21}, [%3 :128], r6 \n" // _w01
                    "vmlal.s8   q13, d18, d23       \n"
                    "pld        [r5, #128]          \n"
                    "vmlal.s8   q14, d19, d22       \n"
                    "pld        [r5, #384]          \n"
                    "vmlal.s8   q15, d19, d23       \n"
                    "vld1.s8    {d18-d19}, [%2 :128]! \n" // _val1

                    "vpadal.s16 q2, q12             \n"
                    "vld1.s8    {d22-d23}, [%3 :128], r6 \n" // _w45
                    "vpadal.s16 q3, q13             \n"
                    "pld        [%2, #128]          \n"
                    "vpadal.s16 q6, q14             \n"
                    "pld        [%3, #128]          \n"
                    "vpadal.s16 q7, q15             \n"

                    "bne        0b                  \n"

                    "sub        %2, %2, #32         \n"
                    "sub        %3, %3, #64         \n"

                    "1:                             \n"
                    "and        r4, %1, #1          \n" // r4 = remain = size & 1
                    "cmp        r4, #0              \n" // r4 > 0
                    "beq        2f                  \n"

                    "vld1.s8    {d16-d17}, [%2 :128]! \n" // _val
                    "vld1.s8    {d20-d21}, [%3 :128]! \n" // _w01

                    "vmull.s8   q12, d16, d20       \n"

                    "vld1.s8    {d22-d23}, [%3 :128]! \n" // _w23
                    "vmull.s8   q13, d16, d21       \n"
                    "vmull.s8   q14, d17, d20       \n"
                    "vmull.s8   q15, d17, d21       \n"

                    "vpadal.s16 q0, q12             \n"
                    "vmull.s8   q12, d16, d22       \n"
                    "vpadal.s16 q1, q13             \n"
                    "vmull.s8   q13, d16, d23       \n"
                    "vpadal.s16 q4, q14             \n"
                    "vmull.s8   q14, d17, d22       \n"
                    "vpadal.s16 q5, q15             \n"
                    "vmull.s8   q15, d17, d23       \n"

                    "vpadal.s16 q2, q12             \n"
                    "vpadal.s16 q3, q13             \n"
                    "vpadal.s16 q6, q14             \n"
                    "vpadal.s16 q7, q15             \n"

                    "2:                             \n"

                    "vpadd.s32  d16, d0, d1         \n"
                    "vpadd.s32  d17, d2, d3         \n"
                    "vpadd.s32  d18, d4, d5         \n"
                    "vpadd.s32  d19, d6, d7         \n"
                    "vpadd.s32  d20, d8, d9         \n"
                    "vpadd.s32  d21, d10, d11       \n"
                    "vpadd.s32  d22, d12, d13       \n"
                    "vpadd.s32  d23, d14, d15       \n"

                    "vpadd.s32  d0, d16, d17        \n"
                    "vpadd.s32  d1, d18, d19        \n"
                    "vpadd.s32  d2, d20, d21        \n"
                    "vpadd.s32  d3, d22, d23        \n"

                    "vst1.s32   {d0-d3}, [%0 :128]! \n"

                    : "=r"(outptr0),
                    "=r"(nn),
                    "=r"(tmpptr),
                    "=r"(kptr0)
                    : "0"(outptr0),
                    "1"(nn),
                    "2"(tmpptr),
                    "3"(kptr0)
                    : "memory", "r4", "r5", "r6", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
    #endif // __aarch64__
            }
            for (; i < size; i++)
            {
    #if __aarch64__
    #if __ARM_FEATURE_DOTPROD
                const signed char* tmpptr = tmp_ra[i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2].data();
    #else
                const signed char* tmpptr = tmp_ra[i / 4 + (i % 4) / 2 + i % 2].data();
    #endif
    #else
                const signed char* tmpptr = tmp_ra[i / 2 + i % 2].data();
    #endif
                const signed char* kptr0 = kernel_a[p].data();

                int nn = inch * maxk; // inch always > 0

    #if __ARM_FEATURE_DOTPROD
                int32x4_t _sum0 = vdupq_n_s32(0);

                for (int j = 0; j < nn; j++)
                {
                    int8x8_t _val0_l_h = vld1_s8(tmpptr);

                    int8x16_t _w0123_l = vld1q_s8(kptr0);

                    _sum0 = vdotq_lane_s32(_sum0, _w0123_l, _val0_l_h, 0);

                    int8x16_t _w0123_h = vld1q_s8(kptr0 + 16);

                    _sum0 = vdotq_lane_s32(_sum0, _w0123_h, _val0_l_h, 1);

                    tmpptr += 8;
                    kptr0 += 32;
                }

                vst1q_s32(outptr0, _sum0);
                outptr0 += 4;
    #else // __ARM_FEATURE_DOTPROD
                int32x4_t _sum0 = vdupq_n_s32(0);
                int32x4_t _sum1 = vdupq_n_s32(0);
                int32x4_t _sum2 = vdupq_n_s32(0);
                int32x4_t _sum3 = vdupq_n_s32(0);

                int j = 0;
                for (; j + 1 < nn; j += 2)
                {
                    int8x16_t _val = vld1q_s8(tmpptr);

                    int8x16_t _w01 = vld1q_s8(kptr0);
                    int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                    int16x8_t _wv0 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w01));
                    int16x8_t _wv1 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w01));
                    int16x8_t _wv2 = vmull_s8(vget_low_s8(_val), vget_low_s8(_w23));
                    int16x8_t _wv3 = vmull_s8(vget_low_s8(_val), vget_high_s8(_w23));

                    int8x16_t _w45 = vld1q_s8(kptr0 + 32);
                    int8x16_t _w67 = vld1q_s8(kptr0 + 48);

                    _wv0 = vmlal_s8(_wv0, vget_high_s8(_val), vget_low_s8(_w45));
                    _wv1 = vmlal_s8(_wv1, vget_high_s8(_val), vget_high_s8(_w45));
                    _wv2 = vmlal_s8(_wv2, vget_high_s8(_val), vget_low_s8(_w67));
                    _wv3 = vmlal_s8(_wv3, vget_high_s8(_val), vget_high_s8(_w67));

                    _sum0 = vpadalq_s16(_sum0, _wv0);
                    _sum1 = vpadalq_s16(_sum1, _wv1);
                    _sum2 = vpadalq_s16(_sum2, _wv2);
                    _sum3 = vpadalq_s16(_sum3, _wv3);

                    tmpptr += 16;
                    kptr0 += 64;
                }
                for (; j < nn; j++)
                {
                    int8x8_t _val = vld1_s8(tmpptr);

                    int8x16_t _w01 = vld1q_s8(kptr0);
                    int8x16_t _w23 = vld1q_s8(kptr0 + 16);

                    int16x8_t _wv0 = vmull_s8(_val, vget_low_s8(_w01));
                    int16x8_t _wv1 = vmull_s8(_val, vget_high_s8(_w01));
                    int16x8_t _wv2 = vmull_s8(_val, vget_low_s8(_w23));
                    int16x8_t _wv3 = vmull_s8(_val, vget_high_s8(_w23));

                    _sum0 = vpadalq_s16(_sum0, _wv0);
                    _sum1 = vpadalq_s16(_sum1, _wv1);
                    _sum2 = vpadalq_s16(_sum2, _wv2);
                    _sum3 = vpadalq_s16(_sum3, _wv3);

                    tmpptr += 8;
                    kptr0 += 32;
                }

    #if __aarch64__
                int32x4_t _s01 = vpaddq_s32(_sum0, _sum1);
                int32x4_t _s23 = vpaddq_s32(_sum2, _sum3);

                int32x4_t _s0123 = vpaddq_s32(_s01, _s23);
    #else
                int32x2_t _s01_low = vpadd_s32(vget_low_s32(_sum0), vget_high_s32(_sum0));
                int32x2_t _s01_high = vpadd_s32(vget_low_s32(_sum1), vget_high_s32(_sum1));
                int32x2_t _s23_low = vpadd_s32(vget_low_s32(_sum2), vget_high_s32(_sum2));
                int32x2_t _s23_high = vpadd_s32(vget_low_s32(_sum3), vget_high_s32(_sum3));

                int32x4_t _s0123 = vcombine_s32(vpadd_s32(_s01_low, _s01_high), vpadd_s32(_s23_low, _s23_high));
    #endif

                vst1q_s32(outptr0, _s0123);
                outptr0 += 4;
    #endif // __ARM_FEATURE_DOTPROD
            }
        }
    });
}

Tensor& sgemm_conv2d_int8_pack1to4_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
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
    
    int outw = output.size(3);
    int outh = output.size(2);
    int outch = output.size(1);
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;
    
    Tensor kernel_tf;
    if (weight_o.defined())
        kernel_tf = weight_o;
    else
        convolution_im2col_sgemm_transform_kernel_pack1to4_int8_neon(weight, kernel_tf, inch, outch * 4, kernel_w, kernel_h);
    
    Tensor im2col = otter::im2col_cpu(self, kernel_size, stride, padding, dilation).view({inch, maxk, size});
    
    im2col_sgemm_conv2d_int8_pack1to4_impl_neon(im2col, kernel_tf, bias, inch, outch, output);
    
    return output;
}
    
Tensor sgemm_conv2d_int8_pack1to4_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, otter::ScalarType::Int4);
    
    return sgemm_conv2d_int8_pack1to4_neon_out(self, weight, weight_o, weight_int8_scales, bias, kernel_size, stride, padding, dilation, output);
}

Tensor& sgemm_conv2d_int8_pack8to1_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
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
        convolution_im2col_sgemm_transform_kernel_pack8to1_int8_neon(weight, kernel_tf, inch * 8, outch, kernel_w, kernel_h);
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    
    Tensor im2col = otter::empty({inch, maxk, size}, otter::ScalarType::Byte8);
    
    auto im2col_a = im2col.accessor<signed char, 3, 8>();
    auto input_a = input.accessor<signed char, 3, 8>();
    
    {
        const int gap = (w * stride_h - outw * stride_w) * 8;

        otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
            for (const auto p : otter::irange(begin, end)) {
                const auto img = input_a[p];
                signed char* ptr = im2col_a[p].data();

                for (int u = 0; u < kernel_h; u++) {
                    for (int v = 0; v < kernel_w; v++) {
                        const signed char* sptr = img[dilation_h * u].data() + dilation_w * v * 8;

                        for (int i = 0; i < outh; i++) {
                            int j = 0;
                            for (; j + 3 < outw; j += 4) {
                                int8x8_t _val0 = vld1_s8(sptr);
                                int8x8_t _val1 = vld1_s8(sptr + stride_w * 8);
                                int8x8_t _val2 = vld1_s8(sptr + stride_w * 16);
                                int8x8_t _val3 = vld1_s8(sptr + stride_w * 24);
                                vst1_s8(ptr, _val0);
                                vst1_s8(ptr + 8, _val1);
                                vst1_s8(ptr + 16, _val2);
                                vst1_s8(ptr + 24, _val3);

                                sptr += stride_w * 32;
                                ptr += 32;
                            }
                            for (; j + 1 < outw; j += 2)
                            {
                                int8x8_t _val0 = vld1_s8(sptr);
                                int8x8_t _val1 = vld1_s8(sptr + stride_w * 8);
                                vst1_s8(ptr, _val0);
                                vst1_s8(ptr + 8, _val1);

                                sptr += stride_w * 16;
                                ptr += 16;
                            }
                            for (; j < outw; j++)
                            {
                                int8x8_t _val = vld1_s8(sptr);
                                vst1_s8(ptr, _val);

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
    
    im2col_sgemm_conv2d_int8_pack8to1_impl_neon(im2col, kernel_tf, bias, inch, outch, output);
    
    return output;
}
    
Tensor sgemm_conv2d_int8_pack8to1_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, otter::ScalarType::Int);
    
    return sgemm_conv2d_int8_pack8to1_neon_out(self, weight, weight_o, weight_int8_scales, bias, kernel_size, stride, padding, dilation, output);
}

Tensor& sgemm_conv2d_int8_pack8to4_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
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
        convolution_im2col_sgemm_transform_kernel_pack8to4_int8_neon(weight, kernel_tf, inch * 8, outch * 4, kernel_w, kernel_h);
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    
    Tensor im2col = otter::empty({inch, maxk, size}, otter::ScalarType::Byte8);
    
    auto im2col_a = im2col.accessor<signed char, 3, 8>();
    auto input_a = input.accessor<signed char, 3, 8>();
    
    {
        const int gap = (w * stride_h - outw * stride_w) * 8;

        otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
            for (const auto p : otter::irange(begin, end)) {
                const auto img = input_a[p];
                signed char* ptr = im2col_a[p].data();

                for (int u = 0; u < kernel_h; u++) {
                    for (int v = 0; v < kernel_w; v++) {
                        const signed char* sptr = img[dilation_h * u].data() + dilation_w * v * 8;

                        for (int i = 0; i < outh; i++)
                        {
                            int j = 0;
                            for (; j + 3 < outw; j += 4)
                            {
                                int8x8_t _val0 = vld1_s8(sptr);
                                int8x8_t _val1 = vld1_s8(sptr + stride_w * 8);
                                int8x8_t _val2 = vld1_s8(sptr + stride_w * 16);
                                int8x8_t _val3 = vld1_s8(sptr + stride_w * 24);
                                vst1_s8(ptr, _val0);
                                vst1_s8(ptr + 8, _val1);
                                vst1_s8(ptr + 16, _val2);
                                vst1_s8(ptr + 24, _val3);

                                sptr += stride_w * 32;
                                ptr += 32;
                            }
                            for (; j + 1 < outw; j += 2)
                            {
                                int8x8_t _val0 = vld1_s8(sptr);
                                int8x8_t _val1 = vld1_s8(sptr + stride_w * 8);
                                vst1_s8(ptr, _val0);
                                vst1_s8(ptr + 8, _val1);

                                sptr += stride_w * 16;
                                ptr += 16;
                            }
                            for (; j < outw; j++)
                            {
                                int8x8_t _val = vld1_s8(sptr);
                                vst1_s8(ptr, _val);

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
    
    im2col_sgemm_conv2d_int8_pack8to4_impl_neon(im2col, kernel_tf, bias, inch, outch, output);
    
    return output;
}
    
Tensor sgemm_conv2d_int8_pack8to4_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    auto output = otter::empty({}, otter::ScalarType::Int4);
    
    return sgemm_conv2d_int8_pack8to4_neon_out(self, weight, weight_o, weight_int8_scales, bias, kernel_size, stride, padding, dilation, output);
}

Tensor& sgemm_conv2d_1x1s1_int8_pack1to4_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
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
        convolution_im2col_sgemm_transform_kernel_pack8to4_int8_neon(weight, kernel_tf, inch, outch * 4, 1, 1);
    
    im2col_sgemm_conv2d_int8_pack1to4_impl_neon(im2col, kernel_tf, bias, inch, outch, output);
    
    return output;
}
    
Tensor sgemm_conv2d_1x1s1_int8_pack1to4_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, otter::ScalarType::Int4);
    
    return sgemm_conv2d_1x1s1_int8_pack1to4_neon_out(self, weight, weight_o, weight_int8_scales, bias, padding, output);
}

Tensor& sgemm_conv2d_1x1s1_int8_pack8to1_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
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
        convolution_im2col_sgemm_transform_kernel_pack8to4_int8_neon(weight, kernel_tf, inch * 8, outch, 1, 1);
    
    im2col_sgemm_conv2d_int8_pack8to1_impl_neon(im2col, kernel_tf, bias, inch, outch, output);
    
    return output;
}
    
Tensor sgemm_conv2d_1x1s1_int8_pack8to1_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, otter::ScalarType::Int);
    
    return sgemm_conv2d_1x1s1_int8_pack8to1_neon_out(self, weight, weight_o, weight_int8_scales, bias, padding, output);
}

Tensor& sgemm_conv2d_1x1s1_int8_pack8to4_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
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
        convolution_im2col_sgemm_transform_kernel_pack8to4_int8_neon(weight, kernel_tf, inch * 8, outch * 4, 1, 1);
    
    im2col_sgemm_conv2d_int8_pack8to4_impl_neon(im2col, kernel_tf, bias, inch, outch, output);
    
    return output;
}
    
Tensor sgemm_conv2d_1x1s1_int8_pack8to4_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& weight_int8_scales,
    const Tensor& bias,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, otter::ScalarType::Int4);
    
    return sgemm_conv2d_1x1s1_int8_pack8to4_neon_out(self, weight, weight_o, weight_int8_scales, bias, padding, output);
}
#endif

}   // end namespace otter
