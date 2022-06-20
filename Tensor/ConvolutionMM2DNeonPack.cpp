//
//  ConvolutionMM2DNeonPack.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/17.
//

#include "ConvolutionMM2DNeonPack.hpp"
#include "TensorFactory.hpp"
#include "Parallel.hpp"
#include "VecIntrinsic.hpp"
#include "ConvolutionUtils.hpp"
#include "Padding.hpp"
#include "im2col.hpp"

namespace otter {

#if __ARM_NEON__

void im2col_sgemm_conv2d_pack4_impl_neon(const Tensor& im2col, Tensor& output_, const Tensor& kernel, const Tensor& _bias) {
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
#if __aarch64__
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
#else
    if (size >= 8)
        tmp = otter::empty({size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, inch, 8 * maxk}, otter::ScalarType::Float4);
    else if (size >= 4)
        tmp = otter::empty({size / 4 + (size % 4) / 2 + size % 2, inch, 4 * maxk}, otter::ScalarType::Float4);
    else if (size >= 2)
        tmp = otter::empty({size / 2 + size % 2, inch, 2 * maxk}, otter::ScalarType::Float4);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Float4);
#endif
    
    auto tmp_a = tmp.accessor<float, 3, 4>();
    {
#if __aarch64__
        int nn_size = size / 12;
        int remain_size_start = 0;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 12;

                float* tmpptr = tmp_a[i / 12].data();

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++) {
                        asm volatile(
                            "prfm   pldl1keep, [%0, #512]       \n"
                            "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                            "prfm   pldl1keep, [%0, #512]       \n"
                            "ld4    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64 \n"
                            "prfm   pldl1keep, [%0, #512]       \n"
                            "ld4    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0] \n"
                            "st1    {v0.4s}, [%1], #16          \n"
                            "st1    {v4.4s}, [%1], #16          \n"
                            "st1    {v8.4s}, [%1], #16          \n"
                            "sub    %0, %0, #128                \n"
                            "st1    {v1.4s}, [%1], #16          \n"
                            "st1    {v5.4s}, [%1], #16          \n"
                            "st1    {v9.4s}, [%1], #16          \n"
                            "st1    {v2.4s}, [%1], #16          \n"
                            "st1    {v6.4s}, [%1], #16          \n"
                            "st1    {v10.4s}, [%1], #16         \n"
                            "st1    {v3.4s}, [%1], #16          \n"
                            "st1    {v7.4s}, [%1], #16          \n"
                            "st1    {v11.4s}, [%1], #16         \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
                        img0 += size * 4;
                    }
                }
            }
        });

        remain_size_start += nn_size * 12;
        nn_size = (size - remain_size_start) >> 3;
#else
        int nn_size = size >> 3;
        int remain_size_start = 0;
#endif

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 8;

    #if __aarch64__
                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8].data();
    #else
                float* tmpptr = tmp_a[i / 8].data();
    #endif

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++) {
    #if __aarch64__
                        asm volatile(
                            "prfm   pldl1keep, [%0, #512]       \n"
                            "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                            "prfm   pldl1keep, [%0, #512]       \n"
                            "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0] \n"
                            "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"
                            "sub    %0, %0, #64                 \n"
                            "st1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
    #else
                        asm volatile(
                            "pld        [%0, #512]          \n"
                            "vldm       %0!, {d0-d7}        \n"
                            "pld        [%0, #512]          \n"
                            "vldm       %0, {d16-d23}       \n"

                            // transpose 8x4
                            "vtrn.32    q0, q1              \n"
                            "vtrn.32    q2, q3              \n"
                            "vtrn.32    q8, q9              \n"
                            "vtrn.32    q10, q11            \n"
                            "vswp       d1, d4              \n"
                            "vswp       d3, d6              \n"
                            "vswp       d17, d20            \n"
                            "vswp       d19, d22            \n"
                            "vswp       q1, q8              \n"
                            "vswp       q3, q10             \n"

                            "vst1.f32   {d0-d3}, [%1 :128]! \n"
                            "vst1.f32   {d16-d19}, [%1 :128]! \n"
                            "sub        %0, %0, #64         \n"
                            "vst1.f32   {d4-d7}, [%1 :128]! \n"
                            "vst1.f32   {d20-d23}, [%1 :128]! \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
    #endif // __aarch64__
                        img0 += size * 4;
                    }
                }
            }
        });

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 4;

    #if __aarch64__
                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4].data();
    #else
                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();
    #endif

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++) {
    #if __aarch64__
                        asm volatile(
                            "prfm   pldl1keep, [%0, #512]       \n"
                            "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0] \n"
                            "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0", "v1", "v2", "v3");
    #else
                        asm volatile(
                            "pld        [%0, #512]          \n"
                            "vldm       %0, {d0-d7}         \n"
                            "vstm       %1!, {d0-d7}        \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "q0", "q1", "q2", "q3");
    #endif // __aarch64__
                        img0 += size * 4;
                    }
                }
            }
        });

        remain_size_start += nn_size << 2;
        nn_size = (size - remain_size_start) >> 1;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 2;

    #if __aarch64__
                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2].data();
    #else
                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + (i % 4) / 2].data();
    #endif

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++) {
    #if __aarch64__
                        asm volatile(
                            "prfm   pldl1keep, [%0, #256]       \n"
                            "ld1    {v0.4s, v1.4s}, [%0]        \n"
                            "st1    {v0.4s, v1.4s}, [%1], #32   \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0", "v1");
    #else
                        asm volatile(
                            "pld        [%0, #256]          \n"
                            "vld1.f32   {d0-d3}, [%0 :128]  \n"
                            "vst1.f32   {d0-d3}, [%1 :128]! \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "q0", "q1");
    #endif // __aarch64__
                        img0 += size * 4;
                    }
                }
            }
        });

        remain_size_start += nn_size << 1;

        otter::parallel_for(remain_size_start, size, 0, [&](int64_t begin, int64_t end) {
            for (const auto i : otter::irange(begin, end)) {
    #if __aarch64__
                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2].data();
    #else
                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2].data();
    #endif

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++) {
    #if __aarch64__
                        asm volatile(
                            "prfm   pldl1keep, [%0, #128]       \n"
                            "ld1    {v0.4s}, [%0]               \n"
                            "st1    {v0.4s}, [%1], #16          \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0");
    #else
                        asm volatile(
                            "pld        [%0, #128]          \n"
                            "vld1.f32   {d0-d1}, [%0 :128]  \n"
                            "vst1.f32   {d0-d1}, [%1 :128]! \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "q0");
    #endif // __aarch64__
                        img0 += size * 4;
                    }
                }
            }
        });
    }

    int remain_outch_start = 0;

#if __aarch64__
    int nn_outch = outch >> 1;
    remain_outch_start = nn_outch << 1;

    otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto pp : otter::irange(begin, end)) {
            int p = pp * 2;

            float* outptr0 = output_a[p].data();
            float* outptr1 = output_a[p + 1].data();

            const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + p * 4 : zeros;

            int i = 0;
            for (; i + 11 < size; i += 12) {
                const float* tmpptr = tmp_a[i / 12].data();
                const float* kptr0 = kernel_a[p / 2].data();

                int nn = inch * maxk; // inch always > 0

                asm volatile(
                    "ld1    {v0.4s, v1.4s}, [%10]       \n"
                    "mov    v8.16b, v0.16b              \n"
                    "mov    v9.16b, v0.16b              \n"
                    "mov    v10.16b, v0.16b             \n"
                    "mov    v11.16b, v0.16b             \n"
                    "mov    v12.16b, v0.16b             \n"
                    "mov    v13.16b, v0.16b             \n"
                    "mov    v14.16b, v0.16b             \n"
                    "mov    v15.16b, v0.16b             \n"
                    "mov    v16.16b, v0.16b             \n"
                    "mov    v17.16b, v0.16b             \n"
                    "mov    v18.16b, v0.16b             \n"
                    "mov    v19.16b, v0.16b             \n"
                    "mov    v20.16b, v1.16b             \n"
                    "mov    v21.16b, v1.16b             \n"
                    "mov    v22.16b, v1.16b             \n"
                    "mov    v23.16b, v1.16b             \n"
                    "mov    v24.16b, v1.16b             \n"
                    "mov    v25.16b, v1.16b             \n"
                    "mov    v26.16b, v1.16b             \n"
                    "mov    v27.16b, v1.16b             \n"
                    "mov    v28.16b, v1.16b             \n"
                    "mov    v29.16b, v1.16b             \n"
                    "mov    v30.16b, v1.16b             \n"
                    "mov    v31.16b, v1.16b             \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64 \n" // w0011_01

                    "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                    "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                    "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                    "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                    "fmla   v16.4s, v4.4s, v2.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v2.s[3]      \n"

                    "fmla   v20.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v21.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v22.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v23.4s, v5.4s, v0.s[3]      \n"
                    "fmla   v24.4s, v5.4s, v1.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v1.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v1.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v1.s[3]      \n"
                    "fmla   v28.4s, v5.4s, v2.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v2.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v2.s[3]      \n"

                    "fmla   v8.4s, v6.4s, v3.s[0]       \n"
                    "fmla   v9.4s, v6.4s, v3.s[1]       \n"
                    "fmla   v10.4s, v6.4s, v3.s[2]      \n"
                    "fmla   v11.4s, v6.4s, v3.s[3]      \n"

                    "fmla   v20.4s, v7.4s, v3.s[0]      \n"
                    "fmla   v21.4s, v7.4s, v3.s[1]      \n"
                    "fmla   v22.4s, v7.4s, v3.s[2]      \n"
                    "fmla   v23.4s, v7.4s, v3.s[3]      \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n"

                    "fmla   v12.4s, v6.4s, v0.s[0]      \n"
                    "fmla   v13.4s, v6.4s, v0.s[1]      \n"
                    "fmla   v14.4s, v6.4s, v0.s[2]      \n"
                    "fmla   v15.4s, v6.4s, v0.s[3]      \n"
                    "fmla   v16.4s, v6.4s, v1.s[0]      \n"
                    "fmla   v17.4s, v6.4s, v1.s[1]      \n"
                    "fmla   v18.4s, v6.4s, v1.s[2]      \n"
                    "fmla   v19.4s, v6.4s, v1.s[3]      \n"

                    "fmla   v24.4s, v7.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v7.4s, v0.s[1]      \n"
                    "fmla   v26.4s, v7.4s, v0.s[2]      \n"
                    "fmla   v27.4s, v7.4s, v0.s[3]      \n"
                    "fmla   v28.4s, v7.4s, v1.s[0]      \n"
                    "fmla   v29.4s, v7.4s, v1.s[1]      \n"
                    "fmla   v30.4s, v7.4s, v1.s[2]      \n"
                    "fmla   v31.4s, v7.4s, v1.s[3]      \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%4], #64 \n" // w2233_01

                    "fmla   v8.4s, v4.4s, v2.s[0]       \n"
                    "fmla   v9.4s, v4.4s, v2.s[1]       \n"
                    "fmla   v10.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v2.s[3]      \n"
                    "fmla   v12.4s, v4.4s, v3.s[0]      \n"
                    "fmla   v13.4s, v4.4s, v3.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v3.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v3.s[3]      \n"

                    "fmla   v20.4s, v5.4s, v2.s[0]      \n"
                    "fmla   v21.4s, v5.4s, v2.s[1]      \n"
                    "fmla   v22.4s, v5.4s, v2.s[2]      \n"
                    "fmla   v23.4s, v5.4s, v2.s[3]      \n"
                    "fmla   v24.4s, v5.4s, v3.s[0]      \n"
                    "fmla   v25.4s, v5.4s, v3.s[1]      \n"
                    "fmla   v26.4s, v5.4s, v3.s[2]      \n"
                    "fmla   v27.4s, v5.4s, v3.s[3]      \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n"

                    "fmla   v16.4s, v4.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v0.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v0.s[3]      \n"

                    "fmla   v28.4s, v5.4s, v0.s[0]      \n"
                    "fmla   v29.4s, v5.4s, v0.s[1]      \n"
                    "fmla   v30.4s, v5.4s, v0.s[2]      \n"
                    "fmla   v31.4s, v5.4s, v0.s[3]      \n"

                    "fmla   v8.4s, v6.4s, v1.s[0]       \n"
                    "fmla   v9.4s, v6.4s, v1.s[1]       \n"
                    "fmla   v10.4s, v6.4s, v1.s[2]      \n"
                    "fmla   v11.4s, v6.4s, v1.s[3]      \n"
                    "fmla   v12.4s, v6.4s, v2.s[0]      \n"
                    "fmla   v13.4s, v6.4s, v2.s[1]      \n"
                    "fmla   v14.4s, v6.4s, v2.s[2]      \n"
                    "fmla   v15.4s, v6.4s, v2.s[3]      \n"
                    "fmla   v16.4s, v6.4s, v3.s[0]      \n"
                    "fmla   v17.4s, v6.4s, v3.s[1]      \n"
                    "fmla   v18.4s, v6.4s, v3.s[2]      \n"
                    "fmla   v19.4s, v6.4s, v3.s[3]      \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v20.4s, v7.4s, v1.s[0]      \n"
                    "fmla   v21.4s, v7.4s, v1.s[1]      \n"
                    "fmla   v22.4s, v7.4s, v1.s[2]      \n"
                    "fmla   v23.4s, v7.4s, v1.s[3]      \n"
                    "fmla   v24.4s, v7.4s, v2.s[0]      \n"
                    "fmla   v25.4s, v7.4s, v2.s[1]      \n"
                    "fmla   v26.4s, v7.4s, v2.s[2]      \n"
                    "fmla   v27.4s, v7.4s, v2.s[3]      \n"
                    "fmla   v28.4s, v7.4s, v3.s[0]      \n"
                    "fmla   v29.4s, v7.4s, v3.s[1]      \n"
                    "fmla   v30.4s, v7.4s, v3.s[2]      \n"
                    "fmla   v31.4s, v7.4s, v3.s[3]      \n"

                    "bne    0b                          \n"

                    "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%1], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%2], #64 \n"
                    "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%1], #64 \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%2], #64 \n"
                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%2], #64 \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(tmpptr),  // %3
                    "=r"(kptr0)    // %4
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(tmpptr),
                    "4"(kptr0),
                    "r"(biasptr) // %10
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            for (; i + 7 < size; i += 8)
            {
                const float* tmpptr = tmp_a[i / 12 + (i % 12) / 8].data();
                const float* kptr0 = kernel_a[p / 2].data();

                int nn = inch * maxk; // inch always > 0

                asm volatile(
                    "ld1    {v0.4s, v1.4s}, [%10]       \n"
                    "mov    v16.16b, v0.16b             \n"
                    "mov    v17.16b, v0.16b             \n"
                    "mov    v18.16b, v0.16b             \n"
                    "mov    v19.16b, v0.16b             \n"
                    "mov    v20.16b, v0.16b             \n"
                    "mov    v21.16b, v0.16b             \n"
                    "mov    v22.16b, v0.16b             \n"
                    "mov    v23.16b, v0.16b             \n"
                    "mov    v24.16b, v1.16b             \n"
                    "mov    v25.16b, v1.16b             \n"
                    "mov    v26.16b, v1.16b             \n"
                    "mov    v27.16b, v1.16b             \n"
                    "mov    v28.16b, v1.16b             \n"
                    "mov    v29.16b, v1.16b             \n"
                    "mov    v30.16b, v1.16b             \n"
                    "mov    v31.16b, v1.16b             \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n" // r0 r1 r2 r3

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%4], #64 \n" // w0011_01

                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v8.4s, v1.s[0]      \n"
                    "fmla   v18.4s, v8.4s, v2.s[0]      \n"
                    "fmla   v19.4s, v8.4s, v3.s[0]      \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%3], #64 \n" // r4 r5 r6 r7

                    "fmla   v20.4s, v8.4s, v4.s[0]      \n"
                    "fmla   v21.4s, v8.4s, v5.s[0]      \n"
                    "fmla   v22.4s, v8.4s, v6.s[0]      \n"
                    "fmla   v23.4s, v8.4s, v7.s[0]      \n"

                    "fmla   v24.4s, v9.4s, v0.s[0]      \n"
                    "fmla   v25.4s, v9.4s, v1.s[0]      \n"
                    "fmla   v26.4s, v9.4s, v2.s[0]      \n"
                    "fmla   v27.4s, v9.4s, v3.s[0]      \n"
                    "fmla   v28.4s, v9.4s, v4.s[0]      \n"
                    "fmla   v29.4s, v9.4s, v5.s[0]      \n"
                    "fmla   v30.4s, v9.4s, v6.s[0]      \n"
                    "fmla   v31.4s, v9.4s, v7.s[0]      \n"

                    "fmla   v16.4s, v10.4s, v0.s[1]     \n"
                    "fmla   v17.4s, v10.4s, v1.s[1]     \n"
                    "fmla   v18.4s, v10.4s, v2.s[1]     \n"
                    "fmla   v19.4s, v10.4s, v3.s[1]     \n"
                    "fmla   v20.4s, v10.4s, v4.s[1]     \n"
                    "fmla   v21.4s, v10.4s, v5.s[1]     \n"
                    "fmla   v22.4s, v10.4s, v6.s[1]     \n"
                    "fmla   v23.4s, v10.4s, v7.s[1]     \n"

                    "fmla   v24.4s, v11.4s, v0.s[1]     \n"
                    "fmla   v25.4s, v11.4s, v1.s[1]     \n"
                    "fmla   v26.4s, v11.4s, v2.s[1]     \n"
                    "fmla   v27.4s, v11.4s, v3.s[1]     \n"
                    "fmla   v28.4s, v11.4s, v4.s[1]     \n"
                    "fmla   v29.4s, v11.4s, v5.s[1]     \n"
                    "fmla   v30.4s, v11.4s, v6.s[1]     \n"
                    "fmla   v31.4s, v11.4s, v7.s[1]     \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%4], #64 \n" // w2233_01

                    "fmla   v16.4s, v12.4s, v0.s[2]     \n"
                    "fmla   v17.4s, v12.4s, v1.s[2]     \n"
                    "fmla   v18.4s, v12.4s, v2.s[2]     \n"
                    "fmla   v19.4s, v12.4s, v3.s[2]     \n"
                    "fmla   v20.4s, v12.4s, v4.s[2]     \n"
                    "fmla   v21.4s, v12.4s, v5.s[2]     \n"
                    "fmla   v22.4s, v12.4s, v6.s[2]     \n"
                    "fmla   v23.4s, v12.4s, v7.s[2]     \n"

                    "fmla   v24.4s, v13.4s, v0.s[2]     \n"
                    "fmla   v25.4s, v13.4s, v1.s[2]     \n"
                    "fmla   v26.4s, v13.4s, v2.s[2]     \n"
                    "fmla   v27.4s, v13.4s, v3.s[2]     \n"
                    "fmla   v28.4s, v13.4s, v4.s[2]     \n"
                    "fmla   v29.4s, v13.4s, v5.s[2]     \n"
                    "fmla   v30.4s, v13.4s, v6.s[2]     \n"
                    "fmla   v31.4s, v13.4s, v7.s[2]     \n"

                    "fmla   v16.4s, v14.4s, v0.s[3]     \n"
                    "fmla   v17.4s, v14.4s, v1.s[3]     \n"
                    "fmla   v18.4s, v14.4s, v2.s[3]     \n"
                    "fmla   v19.4s, v14.4s, v3.s[3]     \n"
                    "fmla   v20.4s, v14.4s, v4.s[3]     \n"
                    "fmla   v21.4s, v14.4s, v5.s[3]     \n"
                    "fmla   v22.4s, v14.4s, v6.s[3]     \n"
                    "fmla   v23.4s, v14.4s, v7.s[3]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v24.4s, v15.4s, v0.s[3]     \n"
                    "fmla   v25.4s, v15.4s, v1.s[3]     \n"
                    "fmla   v26.4s, v15.4s, v2.s[3]     \n"
                    "fmla   v27.4s, v15.4s, v3.s[3]     \n"
                    "fmla   v28.4s, v15.4s, v4.s[3]     \n"
                    "fmla   v29.4s, v15.4s, v5.s[3]     \n"
                    "fmla   v30.4s, v15.4s, v6.s[3]     \n"
                    "fmla   v31.4s, v15.4s, v7.s[3]     \n"

                    "bne    0b                          \n"

                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
                    "st1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%2], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%1], #64 \n"
                    "st1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%2], #64 \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(tmpptr),  // %3
                    "=r"(kptr0)    // %4
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(tmpptr),
                    "4"(kptr0),
                    "r"(biasptr) // %10
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            for (; i + 3 < size; i += 4)
            {
                const float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4].data();
                const float* kptr0 = kernel_a[p / 2].data();

                int nn = inch * maxk; // inch always > 0

                asm volatile(
                    "ld1    {v0.4s, v1.4s}, [%10]       \n"
                    "mov    v16.16b, v0.16b             \n"
                    "mov    v17.16b, v0.16b             \n"
                    "mov    v18.16b, v0.16b             \n"
                    "mov    v19.16b, v0.16b             \n"
                    "mov    v20.16b, v1.16b             \n"
                    "mov    v21.16b, v1.16b             \n"
                    "mov    v22.16b, v1.16b             \n"
                    "mov    v23.16b, v1.16b             \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n" // r0 r1 r2 r3

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%4], #64 \n" // w0011_01

                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v8.4s, v1.s[0]      \n"
                    "fmla   v18.4s, v8.4s, v2.s[0]      \n"
                    "fmla   v19.4s, v8.4s, v3.s[0]      \n"

                    "fmla   v20.4s, v9.4s, v0.s[0]      \n"
                    "fmla   v21.4s, v9.4s, v1.s[0]      \n"
                    "fmla   v22.4s, v9.4s, v2.s[0]      \n"
                    "fmla   v23.4s, v9.4s, v3.s[0]      \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%4], #64 \n" // w2233_01

                    "fmla   v16.4s, v10.4s, v0.s[1]     \n"
                    "fmla   v17.4s, v10.4s, v1.s[1]     \n"
                    "fmla   v18.4s, v10.4s, v2.s[1]     \n"
                    "fmla   v19.4s, v10.4s, v3.s[1]     \n"

                    "fmla   v20.4s, v11.4s, v0.s[1]     \n"
                    "fmla   v21.4s, v11.4s, v1.s[1]     \n"
                    "fmla   v22.4s, v11.4s, v2.s[1]     \n"
                    "fmla   v23.4s, v11.4s, v3.s[1]     \n"

                    "fmla   v16.4s, v12.4s, v0.s[2]     \n"
                    "fmla   v17.4s, v12.4s, v1.s[2]     \n"
                    "fmla   v18.4s, v12.4s, v2.s[2]     \n"
                    "fmla   v19.4s, v12.4s, v3.s[2]     \n"

                    "fmla   v20.4s, v13.4s, v0.s[2]     \n"
                    "fmla   v21.4s, v13.4s, v1.s[2]     \n"
                    "fmla   v22.4s, v13.4s, v2.s[2]     \n"
                    "fmla   v23.4s, v13.4s, v3.s[2]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v16.4s, v14.4s, v0.s[3]     \n"
                    "fmla   v17.4s, v14.4s, v1.s[3]     \n"
                    "fmla   v18.4s, v14.4s, v2.s[3]     \n"
                    "fmla   v19.4s, v14.4s, v3.s[3]     \n"

                    "fmla   v20.4s, v15.4s, v0.s[3]     \n"
                    "fmla   v21.4s, v15.4s, v1.s[3]     \n"
                    "fmla   v22.4s, v15.4s, v2.s[3]     \n"
                    "fmla   v23.4s, v15.4s, v3.s[3]     \n"

                    "bne    0b                          \n"

                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%2], #64 \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(tmpptr),  // %3
                    "=r"(kptr0)    // %4
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(tmpptr),
                    "4"(kptr0),
                    "r"(biasptr) // %10
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
            }
            for (; i + 1 < size; i += 2)
            {
                const float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2].data();
                const float* kptr0 = kernel_a[p / 2].data();

                int nn = inch * maxk; // inch always > 0

                asm volatile(
                    "ld1    {v0.4s, v1.4s}, [%10]       \n"
                    "mov    v16.16b, v0.16b             \n"
                    "mov    v17.16b, v0.16b             \n"
                    "mov    v18.16b, v1.16b             \n"
                    "mov    v19.16b, v1.16b             \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%3, #256]       \n"
                    "ld1    {v0.4s, v1.4s}, [%3], #32   \n" // r0 r1

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%4], #64 \n" // w0011_01

                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v8.4s, v1.s[0]      \n"
                    "fmla   v18.4s, v9.4s, v0.s[0]     \n"
                    "fmla   v19.4s, v9.4s, v1.s[0]     \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%4], #64 \n" // w2233_01

                    "fmla   v16.4s, v10.4s, v0.s[1]      \n"
                    "fmla   v17.4s, v10.4s, v1.s[1]      \n"
                    "fmla   v18.4s, v11.4s, v0.s[1]     \n"
                    "fmla   v19.4s, v11.4s, v1.s[1]     \n"

                    "fmla   v16.4s, v12.4s, v0.s[2]     \n"
                    "fmla   v17.4s, v12.4s, v1.s[2]     \n"
                    "fmla   v18.4s, v13.4s, v0.s[2]     \n"
                    "fmla   v19.4s, v13.4s, v1.s[2]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v16.4s, v14.4s, v0.s[3]     \n"
                    "fmla   v17.4s, v14.4s, v1.s[3]     \n"
                    "fmla   v18.4s, v15.4s, v0.s[3]     \n"
                    "fmla   v19.4s, v15.4s, v1.s[3]     \n"

                    "bne    0b                          \n"

                    "st1    {v16.4s, v17.4s}, [%1], #32 \n"
                    "st1    {v18.4s, v19.4s}, [%2], #32 \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(tmpptr),  // %3
                    "=r"(kptr0)    // %4
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(tmpptr),
                    "4"(kptr0),
                    "r"(biasptr) // %10
                    : "cc", "memory", "v0", "v1", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
            }
            for (; i < size; i++)
            {
                const float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2].data();
                const float* kptr0 = kernel_a[p / 2].data();

                int nn = inch * maxk; // inch always > 0

                asm volatile(
                    "ld1    {v16.4s, v17.4s}, [%10]     \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%3, #128]       \n"
                    "ld1    {v0.4s}, [%3], #16          \n" // r0

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%4], #64 \n" // w0011_01

                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v9.4s, v0.s[0]      \n"

                    "prfm   pldl1keep, [%4, #512]       \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%4], #64 \n" // w2233_01

                    "fmla   v16.4s, v10.4s, v0.s[1]     \n"
                    "fmla   v17.4s, v11.4s, v0.s[1]     \n"

                    "fmla   v16.4s, v12.4s, v0.s[2]     \n"
                    "fmla   v17.4s, v13.4s, v0.s[2]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v16.4s, v14.4s, v0.s[3]     \n"
                    "fmla   v17.4s, v15.4s, v0.s[3]     \n"

                    "bne    0b                          \n"

                    "st1    {v16.4s}, [%1], #16         \n"
                    "st1    {v17.4s}, [%2], #16         \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(tmpptr),  // %3
                    "=r"(kptr0)    // %4
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(tmpptr),
                    "4"(kptr0),
                    "r"(biasptr) // %10
                    : "cc", "memory", "v0", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17");
            }
        }
    });
#endif // __aarch64__

    otter::parallel_for(remain_outch_start, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            float* outptr0 = output_a[p].data();

            const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + p * 4 : zeros;

            int i = 0;
    #if __aarch64__
            for (; i + 11 < size; i += 12) {
                const float* tmpptr = tmp_a[i / 12].data();
                const float* kptr0 = kernel_a[p / 2 + p % 2].data();

                int nn = inch * maxk; // inch always > 0

                asm volatile(
                    "ld1    {v0.4s}, [%8]               \n"
                    "mov    v8.16b, v0.16b              \n"
                    "mov    v9.16b, v0.16b              \n"
                    "mov    v10.16b, v0.16b             \n"
                    "mov    v11.16b, v0.16b             \n"
                    "mov    v12.16b, v0.16b             \n"
                    "mov    v13.16b, v0.16b             \n"
                    "mov    v14.16b, v0.16b             \n"
                    "mov    v15.16b, v0.16b             \n"
                    "mov    v16.16b, v0.16b             \n"
                    "mov    v17.16b, v0.16b             \n"
                    "mov    v18.16b, v0.16b             \n"
                    "mov    v19.16b, v0.16b             \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%3], #64 \n" // w0123_0

                    "fmla   v8.4s, v4.4s, v0.s[0]       \n"
                    "fmla   v9.4s, v4.4s, v0.s[1]       \n"
                    "fmla   v10.4s, v4.4s, v0.s[2]      \n"
                    "fmla   v11.4s, v4.4s, v0.s[3]      \n"
                    "fmla   v12.4s, v4.4s, v1.s[0]      \n"
                    "fmla   v13.4s, v4.4s, v1.s[1]      \n"
                    "fmla   v14.4s, v4.4s, v1.s[2]      \n"
                    "fmla   v15.4s, v4.4s, v1.s[3]      \n"
                    "fmla   v16.4s, v4.4s, v2.s[0]      \n"
                    "fmla   v17.4s, v4.4s, v2.s[1]      \n"
                    "fmla   v18.4s, v4.4s, v2.s[2]      \n"
                    "fmla   v19.4s, v4.4s, v2.s[3]      \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%2], #64 \n"

                    "fmla   v8.4s, v5.4s, v3.s[0]       \n"
                    "fmla   v9.4s, v5.4s, v3.s[1]       \n"
                    "fmla   v10.4s, v5.4s, v3.s[2]      \n"
                    "fmla   v11.4s, v5.4s, v3.s[3]      \n"
                    "fmla   v12.4s, v5.4s, v20.s[0]     \n"
                    "fmla   v13.4s, v5.4s, v20.s[1]     \n"
                    "fmla   v14.4s, v5.4s, v20.s[2]     \n"
                    "fmla   v15.4s, v5.4s, v20.s[3]     \n"
                    "fmla   v16.4s, v5.4s, v21.s[0]     \n"
                    "fmla   v17.4s, v5.4s, v21.s[1]     \n"
                    "fmla   v18.4s, v5.4s, v21.s[2]     \n"
                    "fmla   v19.4s, v5.4s, v21.s[3]     \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%2], #64 \n"

                    "fmla   v8.4s, v6.4s, v22.s[0]      \n"
                    "fmla   v9.4s, v6.4s, v22.s[1]      \n"
                    "fmla   v10.4s, v6.4s, v22.s[2]     \n"
                    "fmla   v11.4s, v6.4s, v22.s[3]     \n"
                    "fmla   v12.4s, v6.4s, v23.s[0]     \n"
                    "fmla   v13.4s, v6.4s, v23.s[1]     \n"
                    "fmla   v14.4s, v6.4s, v23.s[2]     \n"
                    "fmla   v15.4s, v6.4s, v23.s[3]     \n"
                    "fmla   v16.4s, v6.4s, v24.s[0]     \n"
                    "fmla   v17.4s, v6.4s, v24.s[1]     \n"
                    "fmla   v18.4s, v6.4s, v24.s[2]     \n"
                    "fmla   v19.4s, v6.4s, v24.s[3]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v8.4s, v7.4s, v25.s[0]      \n"
                    "fmla   v9.4s, v7.4s, v25.s[1]      \n"
                    "fmla   v10.4s, v7.4s, v25.s[2]     \n"
                    "fmla   v11.4s, v7.4s, v25.s[3]     \n"
                    "fmla   v12.4s, v7.4s, v26.s[0]     \n"
                    "fmla   v13.4s, v7.4s, v26.s[1]     \n"
                    "fmla   v14.4s, v7.4s, v26.s[2]     \n"
                    "fmla   v15.4s, v7.4s, v26.s[3]     \n"
                    "fmla   v16.4s, v7.4s, v27.s[0]     \n"
                    "fmla   v17.4s, v7.4s, v27.s[1]     \n"
                    "fmla   v18.4s, v7.4s, v27.s[2]     \n"
                    "fmla   v19.4s, v7.4s, v27.s[3]     \n"

                    "bne    0b                          \n"

                    "st1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%1], #64 \n"
                    "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%1], #64 \n"
                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(tmpptr),  // %2
                    "=r"(kptr0)    // %3
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(tmpptr),
                    "3"(kptr0),
                    "r"(biasptr) // %8
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
            }
    #endif // __aarch64__
            for (; i + 7 < size; i += 8)
            {
    #if __aarch64__
                const float* tmpptr = tmp_a[i / 12 + (i % 12) / 8].data();
                const float* kptr0 = kernel_a[p / 2 + p % 2].data();
    #else
                const float* tmpptr = tmp_a[i / 8].data();
                const float* kptr0 = kernel_a[p].data();
    #endif

                int nn = inch * maxk; // inch always > 0

    #if __aarch64__
                asm volatile(
                    "ld1    {v0.4s}, [%8]               \n"
                    "mov    v16.16b, v0.16b             \n"
                    "mov    v17.16b, v0.16b             \n"
                    "mov    v18.16b, v0.16b             \n"
                    "mov    v19.16b, v0.16b             \n"
                    "mov    v20.16b, v0.16b             \n"
                    "mov    v21.16b, v0.16b             \n"
                    "mov    v22.16b, v0.16b             \n"
                    "mov    v23.16b, v0.16b             \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n" // r0 r1 r2 r3

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%3], #64 \n" // w0123

                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v8.4s, v1.s[0]      \n"
                    "fmla   v18.4s, v8.4s, v2.s[0]      \n"
                    "fmla   v19.4s, v8.4s, v3.s[0]      \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2], #64 \n" // r4 r5 r6 r7

                    "fmla   v20.4s, v8.4s, v4.s[0]      \n"
                    "fmla   v21.4s, v8.4s, v5.s[0]      \n"
                    "fmla   v22.4s, v8.4s, v6.s[0]      \n"
                    "fmla   v23.4s, v8.4s, v7.s[0]      \n"

                    "fmla   v16.4s, v9.4s, v0.s[1]      \n"
                    "fmla   v17.4s, v9.4s, v1.s[1]      \n"
                    "fmla   v18.4s, v9.4s, v2.s[1]      \n"
                    "fmla   v19.4s, v9.4s, v3.s[1]      \n"
                    "fmla   v20.4s, v9.4s, v4.s[1]      \n"
                    "fmla   v21.4s, v9.4s, v5.s[1]      \n"
                    "fmla   v22.4s, v9.4s, v6.s[1]      \n"
                    "fmla   v23.4s, v9.4s, v7.s[1]      \n"

                    "fmla   v16.4s, v10.4s, v0.s[2]     \n"
                    "fmla   v17.4s, v10.4s, v1.s[2]     \n"
                    "fmla   v18.4s, v10.4s, v2.s[2]     \n"
                    "fmla   v19.4s, v10.4s, v3.s[2]     \n"
                    "fmla   v20.4s, v10.4s, v4.s[2]     \n"
                    "fmla   v21.4s, v10.4s, v5.s[2]     \n"
                    "fmla   v22.4s, v10.4s, v6.s[2]     \n"
                    "fmla   v23.4s, v10.4s, v7.s[2]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v16.4s, v11.4s, v0.s[3]     \n"
                    "fmla   v17.4s, v11.4s, v1.s[3]     \n"
                    "fmla   v18.4s, v11.4s, v2.s[3]     \n"
                    "fmla   v19.4s, v11.4s, v3.s[3]     \n"
                    "fmla   v20.4s, v11.4s, v4.s[3]     \n"
                    "fmla   v21.4s, v11.4s, v5.s[3]     \n"
                    "fmla   v22.4s, v11.4s, v6.s[3]     \n"
                    "fmla   v23.4s, v11.4s, v7.s[3]     \n"

                    "bne    0b                          \n"

                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
                    "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%1], #64 \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(tmpptr),  // %2
                    "=r"(kptr0)    // %3
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(tmpptr),
                    "3"(kptr0),
                    "r"(biasptr) // %8
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
    #else
                asm volatile(
                    "vld1.f32   {d0-d1}, [%8]   \n"
                    "vmov       q8, q0          \n"
                    "vmov       q9, q0          \n"
                    "vmov       q10, q0         \n"
                    "vmov       q11, q0         \n"
                    "vmov       q12, q0         \n"
                    "vmov       q13, q0         \n"
                    "vmov       q14, q0         \n"
                    "vmov       q15, q0         \n"

                    "0:                         \n"

                    "pld        [%2, #512]      \n"
                    "vldm       %2!, {d0-d7}    \n"

                    "pld        [%3, #512]      \n"
                    "vldm       %3!, {d8-d15}   \n"

                    "vmla.f32   q8, q4, d0[0]   \n"
                    "vmla.f32   q9, q4, d0[1]   \n"
                    "vmla.f32   q10, q4, d1[0]  \n"
                    "vmla.f32   q11, q4, d1[1]  \n"
                    "vmla.f32   q12, q4, d2[0]  \n"
                    "vmla.f32   q13, q4, d2[1]  \n"
                    "vmla.f32   q14, q4, d3[0]  \n"
                    "vmla.f32   q15, q4, d3[1]  \n"

                    "vmla.f32   q8, q5, d4[0]   \n"
                    "vmla.f32   q9, q5, d4[1]   \n"
                    "vmla.f32   q10, q5, d5[0]  \n"
                    "vmla.f32   q11, q5, d5[1]  \n"
                    "vmla.f32   q12, q5, d6[0]  \n"
                    "vmla.f32   q13, q5, d6[1]  \n"
                    "vmla.f32   q14, q5, d7[0]  \n"
                    "vmla.f32   q15, q5, d7[1]  \n"

                    "pld        [%2, #512]      \n"
                    "vldm       %2!, {d0-d7}    \n"

                    "vmla.f32   q8, q6, d0[0]   \n"
                    "vmla.f32   q9, q6, d0[1]   \n"
                    "vmla.f32   q10, q6, d1[0]  \n"
                    "vmla.f32   q11, q6, d1[1]  \n"
                    "vmla.f32   q12, q6, d2[0]  \n"
                    "vmla.f32   q13, q6, d2[1]  \n"
                    "vmla.f32   q14, q6, d3[0]  \n"
                    "vmla.f32   q15, q6, d3[1]  \n"

                    "subs       %0, %0, #1      \n"

                    "vmla.f32   q8, q7, d4[0]   \n"
                    "vmla.f32   q9, q7, d4[1]   \n"
                    "vmla.f32   q10, q7, d5[0]  \n"
                    "vmla.f32   q11, q7, d5[1]  \n"
                    "vmla.f32   q12, q7, d6[0]  \n"
                    "vmla.f32   q13, q7, d6[1]  \n"
                    "vmla.f32   q14, q7, d7[0]  \n"
                    "vmla.f32   q15, q7, d7[1]  \n"

                    "bne        0b              \n"

                    "vstm       %1!, {d16-d23}  \n"
                    "vstm       %1!, {d24-d31}  \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(tmpptr),  // %2
                    "=r"(kptr0)    // %3
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(tmpptr),
                    "3"(kptr0),
                    "r"(biasptr) // %8
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
    #endif
            }
            for (; i + 3 < size; i += 4)
            {
    #if __aarch64__
                const float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4].data();
                const float* kptr0 = kernel_a[p / 2 + p % 2].data();
    #else
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();
                const float* kptr0 = kernel_a[p].data();
    #endif

                int nn = inch * maxk; // inch always > 0

    #if __aarch64__
                asm volatile(
                    "ld1    {v0.4s}, [%8]               \n"
                    "mov    v16.16b, v0.16b             \n"
                    "mov    v17.16b, v0.16b             \n"
                    "mov    v18.16b, v0.16b             \n"
                    "mov    v19.16b, v0.16b             \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%2, #512]       \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n" // r0 r1 r2 r3

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%3], #64 \n" // w0123

                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v8.4s, v1.s[0]      \n"
                    "fmla   v18.4s, v8.4s, v2.s[0]      \n"
                    "fmla   v19.4s, v8.4s, v3.s[0]      \n"

                    "fmla   v16.4s, v9.4s, v0.s[1]      \n"
                    "fmla   v17.4s, v9.4s, v1.s[1]      \n"
                    "fmla   v18.4s, v9.4s, v2.s[1]      \n"
                    "fmla   v19.4s, v9.4s, v3.s[1]      \n"

                    "fmla   v16.4s, v10.4s, v0.s[2]     \n"
                    "fmla   v17.4s, v10.4s, v1.s[2]     \n"
                    "fmla   v18.4s, v10.4s, v2.s[2]     \n"
                    "fmla   v19.4s, v10.4s, v3.s[2]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v16.4s, v11.4s, v0.s[3]     \n"
                    "fmla   v17.4s, v11.4s, v1.s[3]     \n"
                    "fmla   v18.4s, v11.4s, v2.s[3]     \n"
                    "fmla   v19.4s, v11.4s, v3.s[3]     \n"

                    "bne    0b                          \n"

                    "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(tmpptr),  // %2
                    "=r"(kptr0)    // %3
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(tmpptr),
                    "3"(kptr0),
                    "r"(biasptr) // %8
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19");
    #else
                asm volatile(
                    "vld1.f32   {d0-d1}, [%8]   \n"
                    "vmov       q8, q0          \n"
                    "vmov       q9, q0          \n"
                    "vmov       q10, q0         \n"
                    "vmov       q11, q0         \n"

                    "0:                         \n"

                    "pld        [%2, #512]      \n"
                    "vldm       %2!, {d0-d7}    \n"

                    "pld        [%3, #512]      \n"
                    "vldm       %3!, {d8-d15}   \n"

                    "vmla.f32   q8, q4, d0[0]   \n"
                    "vmla.f32   q9, q4, d2[0]   \n"
                    "vmla.f32   q10, q4, d4[0]  \n"
                    "vmla.f32   q11, q4, d6[0]  \n"

                    "vmla.f32   q8, q5, d0[1]   \n"
                    "vmla.f32   q9, q5, d2[1]   \n"
                    "vmla.f32   q10, q5, d4[1]  \n"
                    "vmla.f32   q11, q5, d6[1]  \n"

                    "vmla.f32   q8, q6, d1[0]   \n"
                    "vmla.f32   q9, q6, d3[0]   \n"
                    "vmla.f32   q10, q6, d5[0]  \n"
                    "vmla.f32   q11, q6, d7[0]  \n"

                    "subs       %0, %0, #1      \n"

                    "vmla.f32   q8, q7, d1[1]   \n"
                    "vmla.f32   q9, q7, d3[1]   \n"
                    "vmla.f32   q10, q7, d5[1]  \n"
                    "vmla.f32   q11, q7, d7[1]  \n"

                    "bne        0b              \n"

                    "vstm       %1!, {d16-d23}  \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(tmpptr),  // %2
                    "=r"(kptr0)    // %3
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(tmpptr),
                    "3"(kptr0),
                    "r"(biasptr) // %8
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
    #endif
            }
            for (; i + 1 < size; i += 2)
            {
    #if __aarch64__
                const float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2].data();
                const float* kptr0 = kernel_a[p / 2 + p % 2].data();
    #else
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + (i % 4) / 2].data();
                const float* kptr0 = kernel_a[p].data();
    #endif

                int nn = inch * maxk; // inch always > 0

    #if __aarch64__
                asm volatile(
                    "ld1    {v0.4s}, [%8]               \n"
                    "mov    v16.16b, v0.16b             \n"
                    "mov    v17.16b, v0.16b             \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%2, #256]       \n"
                    "ld1    {v0.4s, v1.4s}, [%2], #32   \n" // r0 r1

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%3], #64 \n" // w0123

                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "fmla   v17.4s, v8.4s, v1.s[0]      \n"

                    "fmla   v16.4s, v9.4s, v0.s[1]      \n"
                    "fmla   v17.4s, v9.4s, v1.s[1]      \n"

                    "fmla   v16.4s, v10.4s, v0.s[2]     \n"
                    "fmla   v17.4s, v10.4s, v1.s[2]     \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v16.4s, v11.4s, v0.s[3]     \n"
                    "fmla   v17.4s, v11.4s, v1.s[3]     \n"

                    "bne    0b                          \n"

                    "st1    {v16.4s, v17.4s}, [%1], #32 \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(tmpptr),  // %2
                    "=r"(kptr0)    // %3
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(tmpptr),
                    "3"(kptr0),
                    "r"(biasptr) // %8
                    : "cc", "memory", "v0", "v1", "v8", "v9", "v10", "v11", "v16", "v17");
    #else
                asm volatile(
                    "vld1.f32   {d0-d1}, [%8]   \n"
                    "vmov       q8, q0          \n"
                    "vmov       q9, q0          \n"

                    "0:                         \n"

                    "pld        [%2, #256]      \n"
                    "vld1.f32   {d0-d3}, [%2 :128]! \n"

                    "pld        [%3, #512]      \n"
                    "vldm       %3!, {d8-d15}   \n"

                    "vmla.f32   q8, q4, d0[0]   \n"
                    "vmla.f32   q9, q4, d2[0]   \n"

                    "vmla.f32   q8, q5, d0[1]   \n"
                    "vmla.f32   q9, q5, d2[1]   \n"

                    "vmla.f32   q8, q6, d1[0]   \n"
                    "vmla.f32   q9, q6, d3[0]   \n"

                    "subs       %0, %0, #1      \n"

                    "vmla.f32   q8, q7, d1[1]   \n"
                    "vmla.f32   q9, q7, d3[1]   \n"

                    "bne        0b              \n"

                    "vst1.f32   {d16-d19}, [%1 :128]! \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(tmpptr),  // %2
                    "=r"(kptr0)    // %3
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(tmpptr),
                    "3"(kptr0),
                    "r"(biasptr) // %8
                    : "cc", "memory", "q0", "q1", "q4", "q5", "q6", "q7", "q8", "q9");
    #endif
            }
            for (; i < size; i++)
            {
    #if __aarch64__
                const float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + (i % 12 % 4) / 2 + i % 12 % 2].data();
                const float* kptr0 = kernel_a[p / 2 + p % 2].data();
    #else
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2].data();
                const float* kptr0 = kernel_a[p].data();
    #endif

                int nn = inch * maxk; // inch always > 0

    #if __aarch64__
                asm volatile(
                    "ld1    {v16.4s}, [%8]              \n"

                    "0:                                 \n"

                    "prfm   pldl1keep, [%2, #128]       \n"
                    "ld1    {v0.4s}, [%2], #16          \n" // r0

                    "prfm   pldl1keep, [%3, #512]       \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%3], #64 \n" // w0123

                    "fmla   v16.4s, v8.4s, v0.s[0]      \n"
                    "fmla   v16.4s, v9.4s, v0.s[1]      \n"

                    "subs   %w0, %w0, #1                \n"

                    "fmla   v16.4s, v10.4s, v0.s[2]     \n"
                    "fmla   v16.4s, v11.4s, v0.s[3]     \n"

                    "bne    0b                          \n"

                    "st1    {v16.4s}, [%1], #16         \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(tmpptr),  // %2
                    "=r"(kptr0)    // %3
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(tmpptr),
                    "3"(kptr0),
                    "r"(biasptr) // %8
                    : "cc", "memory", "v0", "v8", "v9", "v10", "v11", "v16");
    #else
                asm volatile(
                    "vld1.f32   {d16-d17}, [%8] \n"

                    "0:                         \n"

                    "pld        [%2, #128]      \n"
                    "vld1.f32   {d0-d1}, [%2 :128]! \n"

                    "pld        [%3, #512]      \n"
                    "vldm       %3!, {d8-d15}   \n"

                    "vmla.f32   q8, q4, d0[0]   \n"
                    "vmla.f32   q8, q5, d0[1]   \n"

                    "subs       %0, %0, #1      \n"

                    "vmla.f32   q8, q6, d1[0]   \n"
                    "vmla.f32   q8, q7, d1[1]   \n"

                    "bne        0b              \n"

                    "vst1.f32   {d16-d17}, [%1 :128]! \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(tmpptr),  // %2
                    "=r"(kptr0)    // %3
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(tmpptr),
                    "3"(kptr0),
                    "r"(biasptr) // %8
                    : "cc", "memory", "q0", "q4", "q5", "q6", "q7", "q8");
    #endif
            }
        }
    });
}

void im2col_sgemm_conv2d_pack4to1_impl_neon(const Tensor& im2col, Tensor& output_, const Tensor& kernel, const Tensor& _bias) {
    const int size = im2col.size(2);
    const int maxk = im2col.size(1);
    const int inch = im2col.size(0);

    const int outch = output_.size(1);

    const float* bias = (_bias.defined()) ? _bias.data_ptr<float>() : nullptr;
    
    auto output_a = output_.accessor<float, 4>()[0];
    auto im2col_a = im2col.accessor<float, 3, 4>();
    auto kernel_a = kernel.accessor<float, 3>();
    
    // permute
    Tensor tmp;
#if __aarch64__
    if (size >= 12)
        tmp = otter::empty({size / 12 + (size % 12) / 8 + (size % 12 % 8) / 4 + size % 12 % 4, inch, 12 * maxk}, otter::ScalarType::Float4);
    else if (size >= 8)
        tmp = otter::empty({size / 8 + (size % 8) / 4 + size % 4, inch, 8 * maxk}, otter::ScalarType::Float4);
    else if (size >= 4)
        tmp = otter::empty({size / 4 + size % 4, inch, 4 * maxk}, otter::ScalarType::Float4);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Float4);
#else
    if (size >= 8)
        tmp = otter::empty({size / 8 + (size % 8) / 4 + size % 4, inch, 8 * maxk}, otter::ScalarType::Float4);
    else if (size >= 4)
        tmp = otter::empty({size / 4 + size % 4, inch, 4 * maxk}, otter::ScalarType::Float4);
    else
        tmp = otter::empty({size, inch, maxk}, otter::ScalarType::Float4);
#endif
    
    auto tmp_a = tmp.accessor<float, 3, 4>();
    {
#if __aarch64__
        int nn_size = size / 12;
        int remain_size_start = 0;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 12;

                float* tmpptr = tmp_a[i / 12].data();

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++) {
                        // transpose 4x12
                        asm volatile(
                            "prfm   pldl1keep, [%0, #512]       \n"
                            "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                            "prfm   pldl1keep, [%0, #512]       \n"
                            "ld4    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0], #64 \n"
                            "prfm   pldl1keep, [%0, #512]       \n"
                            "ld4    {v8.4s, v9.4s, v10.4s, v11.4s}, [%0] \n"
                            "st1    {v0.4s}, [%1], #16          \n"
                            "st1    {v4.4s}, [%1], #16          \n"
                            "st1    {v8.4s}, [%1], #16          \n"
                            "sub    %0, %0, #128                \n"
                            "st1    {v1.4s}, [%1], #16          \n"
                            "st1    {v5.4s}, [%1], #16          \n"
                            "st1    {v9.4s}, [%1], #16          \n"
                            "st1    {v2.4s}, [%1], #16          \n"
                            "st1    {v6.4s}, [%1], #16          \n"
                            "st1    {v10.4s}, [%1], #16         \n"
                            "st1    {v3.4s}, [%1], #16          \n"
                            "st1    {v7.4s}, [%1], #16          \n"
                            "st1    {v11.4s}, [%1], #16         \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
                        img0 += size * 4;
                    }
                }
            }
        });

        remain_size_start += nn_size * 12;
        nn_size = (size - remain_size_start) >> 3;
#else
        int nn_size = size >> 3;
        int remain_size_start = 0;
#endif

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 8;

    #if __aarch64__
                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8].data();
    #else
                float* tmpptr = tmp_a[i / 8].data();
    #endif

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++) {
                        // transpose 4x8
    #if __aarch64__
                        asm volatile(
                            "prfm   pldl1keep, [%0, #512]       \n"
                            "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
                            "prfm   pldl1keep, [%0, #512]       \n"
                            "ld4    {v4.4s, v5.4s, v6.4s, v7.4s}, [%0] \n"
                            "sub    %0, %0, #64                 \n"
                            "st1    {v0.4s}, [%1], #16          \n"
                            "st1    {v4.4s}, [%1], #16          \n"
                            "st1    {v1.4s}, [%1], #16          \n"
                            "st1    {v5.4s}, [%1], #16          \n"
                            "st1    {v2.4s}, [%1], #16          \n"
                            "st1    {v6.4s}, [%1], #16          \n"
                            "st1    {v3.4s}, [%1], #16          \n"
                            "st1    {v7.4s}, [%1], #16          \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
    #else
                        asm volatile(
                            "pld        [%0, #512]          \n"
                            "vldm       %0!, {d0-d7}        \n"
                            "pld        [%0, #512]          \n"
                            "vldm       %0, {d16-d23}       \n"

                            // transpose 8x4
                            "vtrn.32    q0, q1              \n"
                            "vtrn.32    q2, q3              \n"
                            "vtrn.32    q8, q9              \n"
                            "vtrn.32    q10, q11            \n"
                            "vswp       d1, d4              \n"
                            "vswp       d3, d6              \n"
                            "vswp       d17, d20            \n"
                            "vswp       d19, d22            \n"
                            "vswp       q1, q8              \n"
                            "vswp       q3, q10             \n"

                            "vst1.f32   {d0-d3}, [%1 :128]! \n"
                            "vst1.f32   {d16-d19}, [%1 :128]! \n"
                            "sub        %0, %0, #64         \n"
                            "vst1.f32   {d4-d7}, [%1 :128]! \n"
                            "vst1.f32   {d20-d23}, [%1 :128]! \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
    #endif // __aarch64__
                        img0 += size * 4;
                    }
                }
            }
        });

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        otter::parallel_for(0, nn_size, 0, [&](int64_t begin, int64_t end) {
            for (const auto ii : otter::irange(begin, end)) {
                int i = remain_size_start + ii * 4;

    #if __aarch64__
                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4].data();
    #else
                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();
    #endif

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++) {
                        // transpose 4x4
    #if __aarch64__
                        asm volatile(
                            "prfm   pldl1keep, [%0, #512]       \n"
                            "ld4    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0] \n"
                            "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0", "v1", "v2", "v3");
    #else
                        asm volatile(
                            "pld        [%0, #256]          \n"
                            "vld4.f32   {d0-d3}, [%0 :128]! \n"
                            "pld        [%0, #256]          \n"
                            "vld4.f32   {d4-d7}, [%0 :128]  \n"
                            "sub        %0, %0, #32         \n"
                            "vswp       d1, d4              \n"
                            "vswp       d3, d6              \n"
                            "vst1.f32   {d0-d1}, [%1 :128]! \n"
                            "vst1.f32   {d4-d5}, [%1 :128]! \n"
                            "vst1.f32   {d2-d3}, [%1 :128]! \n"
                            "vst1.f32   {d6-d7}, [%1 :128]! \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "q0", "q1", "q2", "q3");
    #endif
                        img0 += size * 4;
                    }
                }
            }
        });

        remain_size_start += nn_size << 2;

        otter::parallel_for(remain_size_start, size, 0, [&](int64_t begin, int64_t end) {
            for (const auto i : otter::irange(begin, end)) {
    #if __aarch64__
                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4].data();
    #else
                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();
    #endif

                for (int q = 0; q < inch; q++) {
                    const float* img0 = (const float*)im2col_a[q].data() + i * 4;

                    for (int k = 0; k < maxk; k++) {
    #if __aarch64__
                        asm volatile(
                            "prfm   pldl1keep, [%0, #128]       \n"
                            "ld1    {v0.4s}, [%0]               \n"
                            "st1    {v0.4s}, [%1], #16          \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "v0");
    #else
                        asm volatile(
                            "pld        [%0, #128]          \n"
                            "vld1.f32   {d0-d1}, [%0 :128]  \n"
                            "vst1.f32   {d0-d1}, [%1 :128]! \n"
                            : "=r"(img0),  // %0
                            "=r"(tmpptr) // %1
                            : "0"(img0),
                            "1"(tmpptr)
                            : "memory", "q0");
    #endif // __aarch64__
                        img0 += size * 4;
                    }
                }
            }
        });
    }

    int nn_outch = 0;
    int remain_outch_start = 0;

#if __aarch64__
    nn_outch = outch >> 3;

    otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto pp : otter::irange(begin, end)) {
            int p = pp * 8;

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
            for (; i + 11 < size; i += 12) {
                float* tmpptr = tmp_a[i / 12].data();
                const float* kptr = (const float*)kernel_a[p / 8].data();

                int nn = inch * maxk; // inch always > 0

                asm volatile(
                    "ld1    {v30.4s, v31.4s}, [%22] \n"
                    "dup    v8.4s, v30.s[0]         \n"
                    "dup    v9.4s, v30.s[0]         \n"
                    "dup    v10.4s, v30.s[0]        \n"
                    "dup    v11.4s, v30.s[1]        \n"
                    "dup    v12.4s, v30.s[1]        \n"
                    "dup    v13.4s, v30.s[1]        \n"
                    "dup    v14.4s, v30.s[2]        \n"
                    "dup    v15.4s, v30.s[2]        \n"
                    "dup    v16.4s, v30.s[2]        \n"
                    "dup    v17.4s, v30.s[3]        \n"
                    "dup    v18.4s, v30.s[3]        \n"
                    "dup    v19.4s, v30.s[3]        \n"
                    "dup    v20.4s, v31.s[0]        \n"
                    "dup    v21.4s, v31.s[0]        \n"
                    "dup    v22.4s, v31.s[0]        \n"
                    "dup    v23.4s, v31.s[1]        \n"
                    "dup    v24.4s, v31.s[1]        \n"
                    "dup    v25.4s, v31.s[1]        \n"
                    "dup    v26.4s, v31.s[2]        \n"
                    "dup    v27.4s, v31.s[2]        \n"
                    "dup    v28.4s, v31.s[2]        \n"
                    "dup    v29.4s, v31.s[3]        \n"
                    "dup    v30.4s, v31.s[3]        \n"
                    "dup    v31.4s, v31.s[3]        \n"

                    "0:                             \n"

                    "prfm   pldl1keep, [%9, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64 \n"

                    "prfm   pldl1keep, [%10, #512]  \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%10], #64 \n"

                    "subs   %w0, %w0, #1            \n"

                    "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                    "fmla   v11.4s, v0.4s, v4.s[1]  \n"
                    "fmla   v14.4s, v0.4s, v4.s[2]  \n"
                    "fmla   v17.4s, v0.4s, v4.s[3]  \n"
                    "fmla   v20.4s, v0.4s, v5.s[0]  \n"
                    "fmla   v23.4s, v0.4s, v5.s[1]  \n"
                    "fmla   v26.4s, v0.4s, v5.s[2]  \n"
                    "fmla   v29.4s, v0.4s, v5.s[3]  \n"

                    "fmla   v9.4s, v1.4s, v4.s[0]   \n"
                    "fmla   v12.4s, v1.4s, v4.s[1]  \n"
                    "fmla   v15.4s, v1.4s, v4.s[2]  \n"
                    "fmla   v18.4s, v1.4s, v4.s[3]  \n"
                    "fmla   v21.4s, v1.4s, v5.s[0]  \n"
                    "fmla   v24.4s, v1.4s, v5.s[1]  \n"
                    "fmla   v27.4s, v1.4s, v5.s[2]  \n"
                    "fmla   v30.4s, v1.4s, v5.s[3]  \n"

                    "fmla   v10.4s, v2.4s, v4.s[0]  \n"
                    "fmla   v13.4s, v2.4s, v4.s[1]  \n"
                    "fmla   v16.4s, v2.4s, v4.s[2]  \n"
                    "fmla   v19.4s, v2.4s, v4.s[3]  \n"
                    "fmla   v22.4s, v2.4s, v5.s[0]  \n"
                    "fmla   v25.4s, v2.4s, v5.s[1]  \n"
                    "fmla   v28.4s, v2.4s, v5.s[2]  \n"
                    "fmla   v31.4s, v2.4s, v5.s[3]  \n"

                    "fmla   v8.4s, v3.4s, v6.s[0]   \n"
                    "fmla   v11.4s, v3.4s, v6.s[1]  \n"
                    "fmla   v14.4s, v3.4s, v6.s[2]  \n"
                    "fmla   v17.4s, v3.4s, v6.s[3]  \n"
                    "fmla   v20.4s, v3.4s, v7.s[0]  \n"
                    "fmla   v23.4s, v3.4s, v7.s[1]  \n"
                    "fmla   v26.4s, v3.4s, v7.s[2]  \n"
                    "fmla   v29.4s, v3.4s, v7.s[3]  \n"

                    "prfm   pldl1keep, [%9, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64 \n"

                    "fmla   v9.4s, v0.4s, v6.s[0]   \n"
                    "fmla   v12.4s, v0.4s, v6.s[1]  \n"
                    "fmla   v15.4s, v0.4s, v6.s[2]  \n"
                    "fmla   v18.4s, v0.4s, v6.s[3]  \n"
                    "fmla   v21.4s, v0.4s, v7.s[0]  \n"
                    "fmla   v24.4s, v0.4s, v7.s[1]  \n"
                    "fmla   v27.4s, v0.4s, v7.s[2]  \n"
                    "fmla   v30.4s, v0.4s, v7.s[3]  \n"

                    "fmla   v10.4s, v1.4s, v6.s[0]  \n"
                    "fmla   v13.4s, v1.4s, v6.s[1]  \n"
                    "fmla   v16.4s, v1.4s, v6.s[2]  \n"
                    "fmla   v19.4s, v1.4s, v6.s[3]  \n"
                    "fmla   v22.4s, v1.4s, v7.s[0]  \n"
                    "fmla   v25.4s, v1.4s, v7.s[1]  \n"
                    "fmla   v28.4s, v1.4s, v7.s[2]  \n"
                    "fmla   v31.4s, v1.4s, v7.s[3]  \n"

                    "prfm   pldl1keep, [%10, #512]  \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%10], #64 \n"

                    "fmla   v8.4s, v2.4s, v4.s[0]   \n"
                    "fmla   v11.4s, v2.4s, v4.s[1]  \n"
                    "fmla   v14.4s, v2.4s, v4.s[2]  \n"
                    "fmla   v17.4s, v2.4s, v4.s[3]  \n"
                    "fmla   v20.4s, v2.4s, v5.s[0]  \n"
                    "fmla   v23.4s, v2.4s, v5.s[1]  \n"
                    "fmla   v26.4s, v2.4s, v5.s[2]  \n"
                    "fmla   v29.4s, v2.4s, v5.s[3]  \n"

                    "fmla   v9.4s, v3.4s, v4.s[0]   \n"
                    "fmla   v12.4s, v3.4s, v4.s[1]  \n"
                    "fmla   v15.4s, v3.4s, v4.s[2]  \n"
                    "fmla   v18.4s, v3.4s, v4.s[3]  \n"
                    "fmla   v21.4s, v3.4s, v5.s[0]  \n"
                    "fmla   v24.4s, v3.4s, v5.s[1]  \n"
                    "fmla   v27.4s, v3.4s, v5.s[2]  \n"
                    "fmla   v30.4s, v3.4s, v5.s[3]  \n"

                    "prfm   pldl1keep, [%9, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64 \n"

                    "fmla   v10.4s, v0.4s, v4.s[0]  \n"
                    "fmla   v13.4s, v0.4s, v4.s[1]  \n"
                    "fmla   v16.4s, v0.4s, v4.s[2]  \n"
                    "fmla   v19.4s, v0.4s, v4.s[3]  \n"
                    "fmla   v22.4s, v0.4s, v5.s[0]  \n"
                    "fmla   v25.4s, v0.4s, v5.s[1]  \n"
                    "fmla   v28.4s, v0.4s, v5.s[2]  \n"
                    "fmla   v31.4s, v0.4s, v5.s[3]  \n"

                    "fmla   v8.4s, v1.4s, v6.s[0]   \n"
                    "fmla   v11.4s, v1.4s, v6.s[1]  \n"
                    "fmla   v14.4s, v1.4s, v6.s[2]  \n"
                    "fmla   v17.4s, v1.4s, v6.s[3]  \n"
                    "fmla   v20.4s, v1.4s, v7.s[0]  \n"
                    "fmla   v23.4s, v1.4s, v7.s[1]  \n"
                    "fmla   v26.4s, v1.4s, v7.s[2]  \n"
                    "fmla   v29.4s, v1.4s, v7.s[3]  \n"

                    "fmla   v9.4s, v2.4s, v6.s[0]   \n"
                    "fmla   v12.4s, v2.4s, v6.s[1]  \n"
                    "fmla   v15.4s, v2.4s, v6.s[2]  \n"
                    "fmla   v18.4s, v2.4s, v6.s[3]  \n"
                    "fmla   v21.4s, v2.4s, v7.s[0]  \n"
                    "fmla   v24.4s, v2.4s, v7.s[1]  \n"
                    "fmla   v27.4s, v2.4s, v7.s[2]  \n"
                    "fmla   v30.4s, v2.4s, v7.s[3]  \n"

                    "fmla   v10.4s, v3.4s, v6.s[0]  \n"
                    "fmla   v13.4s, v3.4s, v6.s[1]  \n"
                    "fmla   v16.4s, v3.4s, v6.s[2]  \n"
                    "fmla   v19.4s, v3.4s, v6.s[3]  \n"
                    "fmla   v22.4s, v3.4s, v7.s[0]  \n"
                    "fmla   v25.4s, v3.4s, v7.s[1]  \n"
                    "fmla   v28.4s, v3.4s, v7.s[2]  \n"
                    "fmla   v31.4s, v3.4s, v7.s[3]  \n"

                    "bne    0b                      \n"

                    "st1    {v8.4s, v9.4s, v10.4s}, [%1], #48 \n"
                    "st1    {v11.4s, v12.4s, v13.4s}, [%2], #48 \n"
                    "st1    {v14.4s, v15.4s, v16.4s}, [%3], #48 \n"
                    "st1    {v17.4s, v18.4s, v19.4s}, [%4], #48 \n"
                    "st1    {v20.4s, v21.4s, v22.4s}, [%5], #48 \n"
                    "st1    {v23.4s, v24.4s, v25.4s}, [%6], #48 \n"
                    "st1    {v26.4s, v27.4s, v28.4s}, [%7], #48 \n"
                    "st1    {v29.4s, v30.4s, v31.4s}, [%8], #48 \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(outptr2), // %3
                    "=r"(outptr3), // %4
                    "=r"(outptr4), // %5
                    "=r"(outptr5), // %6
                    "=r"(outptr6), // %7
                    "=r"(outptr7), // %8
                    "=r"(tmpptr),  // %9
                    "=r"(kptr)     // %10
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(outptr2),
                    "4"(outptr3),
                    "5"(outptr4),
                    "6"(outptr5),
                    "7"(outptr6),
                    "8"(outptr7),
                    "9"(tmpptr),
                    "10"(kptr),
                    "r"(biasptr) // %22
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            for (; i + 7 < size; i += 8)
            {
                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8].data();
                const float* kptr = (const float*)kernel_a[p / 8].data();

                int nn = inch * maxk; // inch always > 0

                asm volatile(
                    "ld1    {v30.4s, v31.4s}, [%22] \n"
                    "dup    v16.4s, v30.s[0]        \n"
                    "dup    v17.4s, v30.s[0]        \n"
                    "dup    v18.4s, v30.s[1]        \n"
                    "dup    v19.4s, v30.s[1]        \n"
                    "dup    v20.4s, v30.s[2]        \n"
                    "dup    v21.4s, v30.s[2]        \n"
                    "dup    v22.4s, v30.s[3]        \n"
                    "dup    v23.4s, v30.s[3]        \n"
                    "dup    v24.4s, v31.s[0]        \n"
                    "dup    v25.4s, v31.s[0]        \n"
                    "dup    v26.4s, v31.s[1]        \n"
                    "dup    v27.4s, v31.s[1]        \n"
                    "dup    v28.4s, v31.s[2]        \n"
                    "dup    v29.4s, v31.s[2]        \n"
                    "dup    v30.4s, v31.s[3]        \n"
                    "dup    v31.4s, v31.s[3]        \n"

                    "0:                             \n"

                    "prfm   pldl1keep, [%9, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64 \n"

                    "prfm   pldl1keep, [%10, #512]  \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%10], #64 \n"

                    "subs   %w0, %w0, #1            \n"

                    "fmla   v16.4s, v0.4s, v4.s[0]  \n"
                    "fmla   v18.4s, v0.4s, v4.s[1]  \n"
                    "fmla   v20.4s, v0.4s, v4.s[2]  \n"
                    "fmla   v22.4s, v0.4s, v4.s[3]  \n"
                    "fmla   v24.4s, v0.4s, v5.s[0]  \n"
                    "fmla   v26.4s, v0.4s, v5.s[1]  \n"
                    "fmla   v28.4s, v0.4s, v5.s[2]  \n"
                    "fmla   v30.4s, v0.4s, v5.s[3]  \n"
                    "fmla   v17.4s, v1.4s, v4.s[0]  \n"
                    "fmla   v19.4s, v1.4s, v4.s[1]  \n"
                    "fmla   v21.4s, v1.4s, v4.s[2]  \n"
                    "fmla   v23.4s, v1.4s, v4.s[3]  \n"
                    "fmla   v25.4s, v1.4s, v5.s[0]  \n"
                    "fmla   v27.4s, v1.4s, v5.s[1]  \n"
                    "fmla   v29.4s, v1.4s, v5.s[2]  \n"
                    "fmla   v31.4s, v1.4s, v5.s[3]  \n"

                    "fmla   v16.4s, v2.4s, v6.s[0]  \n"
                    "fmla   v18.4s, v2.4s, v6.s[1]  \n"
                    "fmla   v20.4s, v2.4s, v6.s[2]  \n"
                    "fmla   v22.4s, v2.4s, v6.s[3]  \n"
                    "fmla   v24.4s, v2.4s, v7.s[0]  \n"
                    "fmla   v26.4s, v2.4s, v7.s[1]  \n"
                    "fmla   v28.4s, v2.4s, v7.s[2]  \n"
                    "fmla   v30.4s, v2.4s, v7.s[3]  \n"
                    "fmla   v17.4s, v3.4s, v6.s[0]  \n"
                    "fmla   v19.4s, v3.4s, v6.s[1]  \n"
                    "fmla   v21.4s, v3.4s, v6.s[2]  \n"
                    "fmla   v23.4s, v3.4s, v6.s[3]  \n"
                    "fmla   v25.4s, v3.4s, v7.s[0]  \n"
                    "fmla   v27.4s, v3.4s, v7.s[1]  \n"
                    "fmla   v29.4s, v3.4s, v7.s[2]  \n"
                    "fmla   v31.4s, v3.4s, v7.s[3]  \n"

                    "prfm   pldl1keep, [%9, #512]   \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%9], #64 \n"

                    "prfm   pldl1keep, [%10, #512]  \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%10], #64 \n"

                    "fmla   v16.4s, v12.4s, v8.s[0] \n"
                    "fmla   v18.4s, v12.4s, v8.s[1] \n"
                    "fmla   v20.4s, v12.4s, v8.s[2] \n"
                    "fmla   v22.4s, v12.4s, v8.s[3] \n"
                    "fmla   v24.4s, v12.4s, v9.s[0] \n"
                    "fmla   v26.4s, v12.4s, v9.s[1] \n"
                    "fmla   v28.4s, v12.4s, v9.s[2] \n"
                    "fmla   v30.4s, v12.4s, v9.s[3] \n"
                    "fmla   v17.4s, v13.4s, v8.s[0] \n"
                    "fmla   v19.4s, v13.4s, v8.s[1] \n"
                    "fmla   v21.4s, v13.4s, v8.s[2] \n"
                    "fmla   v23.4s, v13.4s, v8.s[3] \n"
                    "fmla   v25.4s, v13.4s, v9.s[0] \n"
                    "fmla   v27.4s, v13.4s, v9.s[1] \n"
                    "fmla   v29.4s, v13.4s, v9.s[2] \n"
                    "fmla   v31.4s, v13.4s, v9.s[3] \n"

                    "fmla   v16.4s, v14.4s, v10.s[0] \n"
                    "fmla   v18.4s, v14.4s, v10.s[1] \n"
                    "fmla   v20.4s, v14.4s, v10.s[2] \n"
                    "fmla   v22.4s, v14.4s, v10.s[3] \n"
                    "fmla   v24.4s, v14.4s, v11.s[0] \n"
                    "fmla   v26.4s, v14.4s, v11.s[1] \n"
                    "fmla   v28.4s, v14.4s, v11.s[2] \n"
                    "fmla   v30.4s, v14.4s, v11.s[3] \n"
                    "fmla   v17.4s, v15.4s, v10.s[0] \n"
                    "fmla   v19.4s, v15.4s, v10.s[1] \n"
                    "fmla   v21.4s, v15.4s, v10.s[2] \n"
                    "fmla   v23.4s, v15.4s, v10.s[3] \n"
                    "fmla   v25.4s, v15.4s, v11.s[0] \n"
                    "fmla   v27.4s, v15.4s, v11.s[1] \n"
                    "fmla   v29.4s, v15.4s, v11.s[2] \n"
                    "fmla   v31.4s, v15.4s, v11.s[3] \n"

                    "bne    0b                      \n"

                    "st1    {v16.4s, v17.4s}, [%1], #32 \n"
                    "st1    {v18.4s, v19.4s}, [%2], #32 \n"
                    "st1    {v20.4s, v21.4s}, [%3], #32 \n"
                    "st1    {v22.4s, v23.4s}, [%4], #32 \n"
                    "st1    {v24.4s, v25.4s}, [%5], #32 \n"
                    "st1    {v26.4s, v27.4s}, [%6], #32 \n"
                    "st1    {v28.4s, v29.4s}, [%7], #32 \n"
                    "st1    {v30.4s, v31.4s}, [%8], #32 \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(outptr2), // %3
                    "=r"(outptr3), // %4
                    "=r"(outptr4), // %5
                    "=r"(outptr5), // %6
                    "=r"(outptr6), // %7
                    "=r"(outptr7), // %8
                    "=r"(tmpptr),  // %9
                    "=r"(kptr)     // %10
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(outptr2),
                    "4"(outptr3),
                    "5"(outptr4),
                    "6"(outptr5),
                    "7"(outptr6),
                    "8"(outptr7),
                    "9"(tmpptr),
                    "10"(kptr),
                    "r"(biasptr) // %22
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
            }
            for (; i + 3 < size; i += 4)
            {
                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4].data();
                const float* kptr = (const float*)kernel_a[p / 8].data();

                int nn = inch * maxk; // inch always > 0

                asm volatile(
                    "ld1    {v22.4s, v23.4s}, [%22] \n"
                    "dup    v16.4s, v22.s[0]        \n"
                    "dup    v17.4s, v22.s[1]        \n"
                    "dup    v18.4s, v22.s[2]        \n"
                    "dup    v19.4s, v22.s[3]        \n"
                    "dup    v20.4s, v23.s[0]        \n"
                    "dup    v21.4s, v23.s[1]        \n"
                    "dup    v22.4s, v23.s[2]        \n"
                    "dup    v23.4s, v23.s[3]        \n"

                    "0:                             \n"

                    "prfm   pldl1keep, [%9, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%9], #64 \n"

                    "prfm   pldl1keep, [%10, #512]  \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%10], #64 \n"

                    "subs   %w0, %w0, #1            \n"

                    "fmla   v16.4s, v0.4s, v4.s[0]  \n"
                    "fmla   v17.4s, v0.4s, v4.s[1]  \n"
                    "fmla   v18.4s, v0.4s, v4.s[2]  \n"
                    "fmla   v19.4s, v0.4s, v4.s[3]  \n"
                    "fmla   v20.4s, v0.4s, v5.s[0]  \n"
                    "fmla   v21.4s, v0.4s, v5.s[1]  \n"
                    "fmla   v22.4s, v0.4s, v5.s[2]  \n"
                    "fmla   v23.4s, v0.4s, v5.s[3]  \n"

                    "prfm   pldl1keep, [%10, #512]  \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%10], #64 \n"

                    "fmla   v16.4s, v1.4s, v6.s[0]  \n"
                    "fmla   v17.4s, v1.4s, v6.s[1]  \n"
                    "fmla   v18.4s, v1.4s, v6.s[2]  \n"
                    "fmla   v19.4s, v1.4s, v6.s[3]  \n"
                    "fmla   v20.4s, v1.4s, v7.s[0]  \n"
                    "fmla   v21.4s, v1.4s, v7.s[1]  \n"
                    "fmla   v22.4s, v1.4s, v7.s[2]  \n"
                    "fmla   v23.4s, v1.4s, v7.s[3]  \n"

                    "fmla   v16.4s, v2.4s, v8.s[0]  \n"
                    "fmla   v17.4s, v2.4s, v8.s[1]  \n"
                    "fmla   v18.4s, v2.4s, v8.s[2]  \n"
                    "fmla   v19.4s, v2.4s, v8.s[3]  \n"
                    "fmla   v20.4s, v2.4s, v9.s[0]  \n"
                    "fmla   v21.4s, v2.4s, v9.s[1]  \n"
                    "fmla   v22.4s, v2.4s, v9.s[2]  \n"
                    "fmla   v23.4s, v2.4s, v9.s[3]  \n"

                    "fmla   v16.4s, v3.4s, v10.s[0] \n"
                    "fmla   v17.4s, v3.4s, v10.s[1] \n"
                    "fmla   v18.4s, v3.4s, v10.s[2] \n"
                    "fmla   v19.4s, v3.4s, v10.s[3] \n"
                    "fmla   v20.4s, v3.4s, v11.s[0] \n"
                    "fmla   v21.4s, v3.4s, v11.s[1] \n"
                    "fmla   v22.4s, v3.4s, v11.s[2] \n"
                    "fmla   v23.4s, v3.4s, v11.s[3] \n"

                    "bne    0b                      \n"

                    "st1    {v16.4s}, [%1], #16     \n"
                    "st1    {v17.4s}, [%2], #16     \n"
                    "st1    {v18.4s}, [%3], #16     \n"
                    "st1    {v19.4s}, [%4], #16     \n"
                    "st1    {v20.4s}, [%5], #16     \n"
                    "st1    {v21.4s}, [%6], #16     \n"
                    "st1    {v22.4s}, [%7], #16     \n"
                    "st1    {v23.4s}, [%8], #16     \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(outptr2), // %3
                    "=r"(outptr3), // %4
                    "=r"(outptr4), // %5
                    "=r"(outptr5), // %6
                    "=r"(outptr6), // %7
                    "=r"(outptr7), // %8
                    "=r"(tmpptr),  // %9
                    "=r"(kptr)     // %10
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(outptr2),
                    "4"(outptr3),
                    "5"(outptr4),
                    "6"(outptr5),
                    "7"(outptr6),
                    "8"(outptr7),
                    "9"(tmpptr),
                    "10"(kptr),
                    "r"(biasptr) // %22
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
            }
            for (; i < size; i++)
            {
                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4].data();
                const float* kptr = (const float*)kernel_a[p / 8].data();

                int nn = inch * maxk; // inch always > 0

                asm volatile(
                    "ld1    {v16.4s, v17.4s}, [%22] \n"
                    "eor    v18.16b, v18.16b, v18.16b \n"
                    "eor    v19.16b, v19.16b, v19.16b \n"

                    "0:                             \n"

                    "prfm   pldl1keep, [%9, #128]   \n"
                    "ld1    {v0.4s}, [%9], #16      \n"

                    "prfm   pldl1keep, [%10, #512]  \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%10], #64 \n"

                    "subs   %w0, %w0, #1            \n"

                    "fmla   v16.4s, v4.4s, v0.s[0]  \n"
                    "fmla   v17.4s, v5.4s, v0.s[0]  \n"
                    "fmla   v18.4s, v6.4s, v0.s[1]  \n"
                    "fmla   v19.4s, v7.4s, v0.s[1]  \n"

                    "prfm   pldl1keep, [%10, #512]  \n"
                    "ld1    {v8.4s, v9.4s, v10.4s, v11.4s}, [%10], #64 \n"

                    "fmla   v16.4s, v8.4s, v0.s[2]  \n"
                    "fmla   v17.4s, v9.4s, v0.s[2]  \n"
                    "fmla   v18.4s, v10.4s, v0.s[3] \n"
                    "fmla   v19.4s, v11.4s, v0.s[3] \n"

                    "bne    0b                      \n"

                    "fadd   v16.4s, v16.4s, v18.4s  \n"
                    "fadd   v17.4s, v17.4s, v19.4s  \n"

                    "st1    {v16.s}[0], [%1], #4    \n"
                    "st1    {v16.s}[1], [%2], #4    \n"
                    "st1    {v16.s}[2], [%3], #4    \n"
                    "st1    {v16.s}[3], [%4], #4    \n"
                    "st1    {v17.s}[0], [%5], #4    \n"
                    "st1    {v17.s}[1], [%6], #4    \n"
                    "st1    {v17.s}[2], [%7], #4    \n"
                    "st1    {v17.s}[3], [%8], #4    \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(outptr2), // %3
                    "=r"(outptr3), // %4
                    "=r"(outptr4), // %5
                    "=r"(outptr5), // %6
                    "=r"(outptr6), // %7
                    "=r"(outptr7), // %8
                    "=r"(tmpptr),  // %9
                    "=r"(kptr)     // %10
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(outptr2),
                    "4"(outptr3),
                    "5"(outptr4),
                    "6"(outptr5),
                    "7"(outptr6),
                    "8"(outptr7),
                    "9"(tmpptr),
                    "10"(kptr),
                    "r"(biasptr) // %22
                    : "cc", "memory", "v0", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19");
            }
        }
    });

    remain_outch_start += nn_outch << 3;
    nn_outch = (outch - remain_outch_start) >> 2;
#else  // __aarch64__
    nn_outch = outch >> 2;
#endif // __aarch64__

    otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto pp : otter::irange(begin, end)) {
            int p = remain_outch_start + pp * 4;

            float* outptr0 = output_a[p + 0].data();
            float* outptr1 = output_a[p + 1].data();
            float* outptr2 = output_a[p + 2].data();
            float* outptr3 = output_a[p + 3].data();

            const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + p : zeros;

            int i = 0;
    #if __aarch64__
            for (; i + 11 < size; i += 12) {
                float* tmpptr = tmp_a[i / 12].data();
                const float* kptr = (const float*)kernel_a[p / 8 + (p % 8) / 4].data();

                int nn = inch * maxk; // inch always > 0

                asm volatile(
                    "ld1    {v19.4s}, [%14]         \n"
                    "dup    v8.4s, v19.s[0]         \n"
                    "dup    v9.4s, v19.s[0]         \n"
                    "dup    v10.4s, v19.s[0]        \n"
                    "dup    v11.4s, v19.s[1]        \n"
                    "dup    v12.4s, v19.s[1]        \n"
                    "dup    v13.4s, v19.s[1]        \n"
                    "dup    v14.4s, v19.s[2]        \n"
                    "dup    v15.4s, v19.s[2]        \n"
                    "dup    v16.4s, v19.s[2]        \n"
                    "dup    v17.4s, v19.s[3]        \n"
                    "dup    v18.4s, v19.s[3]        \n"
                    "dup    v19.4s, v19.s[3]        \n"

                    "0:                             \n"

                    "prfm   pldl1keep, [%5, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64 \n"

                    "prfm   pldl1keep, [%6, #512]   \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%6], #64 \n"

                    "subs   %w0, %w0, #1            \n"

                    "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                    "fmla   v11.4s, v0.4s, v4.s[1]  \n"
                    "fmla   v14.4s, v0.4s, v4.s[2]  \n"
                    "fmla   v17.4s, v0.4s, v4.s[3]  \n"
                    "fmla   v9.4s, v1.4s, v4.s[0]   \n"
                    "fmla   v12.4s, v1.4s, v4.s[1]  \n"
                    "fmla   v15.4s, v1.4s, v4.s[2]  \n"
                    "fmla   v18.4s, v1.4s, v4.s[3]  \n"
                    "fmla   v10.4s, v2.4s, v4.s[0]  \n"
                    "fmla   v13.4s, v2.4s, v4.s[1]  \n"
                    "fmla   v16.4s, v2.4s, v4.s[2]  \n"
                    "fmla   v19.4s, v2.4s, v4.s[3]  \n"

                    "prfm   pldl1keep, [%5, #512]   \n"
                    "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%5], #64 \n"

                    "fmla   v8.4s, v3.4s, v5.s[0]   \n"
                    "fmla   v11.4s, v3.4s, v5.s[1]  \n"
                    "fmla   v14.4s, v3.4s, v5.s[2]  \n"
                    "fmla   v17.4s, v3.4s, v5.s[3]  \n"
                    "fmla   v9.4s, v20.4s, v5.s[0]  \n"
                    "fmla   v12.4s, v20.4s, v5.s[1] \n"
                    "fmla   v15.4s, v20.4s, v5.s[2] \n"
                    "fmla   v18.4s, v20.4s, v5.s[3] \n"
                    "fmla   v10.4s, v21.4s, v5.s[0] \n"
                    "fmla   v13.4s, v21.4s, v5.s[1] \n"
                    "fmla   v16.4s, v21.4s, v5.s[2] \n"
                    "fmla   v19.4s, v21.4s, v5.s[3] \n"

                    "prfm   pldl1keep, [%5, #512]   \n"
                    "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%5], #64 \n"

                    "fmla   v8.4s, v22.4s, v6.s[0]  \n"
                    "fmla   v11.4s, v22.4s, v6.s[1] \n"
                    "fmla   v14.4s, v22.4s, v6.s[2] \n"
                    "fmla   v17.4s, v22.4s, v6.s[3] \n"
                    "fmla   v9.4s, v23.4s, v6.s[0]  \n"
                    "fmla   v12.4s, v23.4s, v6.s[1] \n"
                    "fmla   v15.4s, v23.4s, v6.s[2] \n"
                    "fmla   v18.4s, v23.4s, v6.s[3] \n"
                    "fmla   v10.4s, v24.4s, v6.s[0] \n"
                    "fmla   v13.4s, v24.4s, v6.s[1] \n"
                    "fmla   v16.4s, v24.4s, v6.s[2] \n"
                    "fmla   v19.4s, v24.4s, v6.s[3] \n"

                    "fmla   v8.4s, v25.4s, v7.s[0]  \n"
                    "fmla   v11.4s, v25.4s, v7.s[1] \n"
                    "fmla   v14.4s, v25.4s, v7.s[2] \n"
                    "fmla   v17.4s, v25.4s, v7.s[3] \n"
                    "fmla   v9.4s, v26.4s, v7.s[0]  \n"
                    "fmla   v12.4s, v26.4s, v7.s[1] \n"
                    "fmla   v15.4s, v26.4s, v7.s[2] \n"
                    "fmla   v18.4s, v26.4s, v7.s[3] \n"
                    "fmla   v10.4s, v27.4s, v7.s[0] \n"
                    "fmla   v13.4s, v27.4s, v7.s[1] \n"
                    "fmla   v16.4s, v27.4s, v7.s[2] \n"
                    "fmla   v19.4s, v27.4s, v7.s[3] \n"

                    "bne    0b                      \n"

                    "st1    {v8.4s, v9.4s, v10.4s}, [%1], #48 \n"
                    "st1    {v11.4s, v12.4s, v13.4s}, [%2], #48 \n"
                    "st1    {v14.4s, v15.4s, v16.4s}, [%3], #48 \n"
                    "st1    {v17.4s, v18.4s, v19.4s}, [%4], #48 \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(outptr2), // %3
                    "=r"(outptr3), // %4
                    "=r"(tmpptr),  // %5
                    "=r"(kptr)     // %6
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(outptr2),
                    "4"(outptr3),
                    "5"(tmpptr),
                    "6"(kptr),
                    "r"(biasptr) // %14
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27");
            }
    #endif // __aarch64__
            for (; i + 7 < size; i += 8) {
    #if __aarch64__
                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8].data();
                const float* kptr = (const float*)kernel_a[p / 8 + (p % 8) / 4].data();
    #else
                float* tmpptr = tmp_a[i / 8].data();
                const float* kptr = (const float*)kernel_a[p / 4].data();
    #endif

                int nn = inch * maxk; // inch always > 0

    #if __aarch64__
                asm volatile(
                    "ld1    {v15.4s}, [%14]         \n"
                    "dup    v8.4s, v15.s[0]         \n"
                    "dup    v9.4s, v15.s[0]         \n"
                    "dup    v10.4s, v15.s[1]        \n"
                    "dup    v11.4s, v15.s[1]        \n"
                    "dup    v12.4s, v15.s[2]        \n"
                    "dup    v13.4s, v15.s[2]        \n"
                    "dup    v14.4s, v15.s[3]        \n"
                    "dup    v15.4s, v15.s[3]        \n"

                    "0:                             \n"

                    "prfm   pldl1keep, [%5, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64 \n"

                    "prfm   pldl1keep, [%6, #512]   \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%6], #64 \n"

                    "subs   %w0, %w0, #1            \n"

                    "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                    "fmla   v10.4s, v0.4s, v4.s[1]  \n"
                    "fmla   v12.4s, v0.4s, v4.s[2]  \n"
                    "fmla   v14.4s, v0.4s, v4.s[3]  \n"
                    "fmla   v9.4s, v1.4s, v4.s[0]   \n"
                    "fmla   v11.4s, v1.4s, v4.s[1]  \n"
                    "fmla   v13.4s, v1.4s, v4.s[2]  \n"
                    "fmla   v15.4s, v1.4s, v4.s[3]  \n"

                    "fmla   v8.4s, v2.4s, v5.s[0]   \n"
                    "fmla   v10.4s, v2.4s, v5.s[1]  \n"
                    "fmla   v12.4s, v2.4s, v5.s[2]  \n"
                    "fmla   v14.4s, v2.4s, v5.s[3]  \n"
                    "fmla   v9.4s, v3.4s, v5.s[0]   \n"
                    "fmla   v11.4s, v3.4s, v5.s[1]  \n"
                    "fmla   v13.4s, v3.4s, v5.s[2]  \n"
                    "fmla   v15.4s, v3.4s, v5.s[3]  \n"

                    "prfm   pldl1keep, [%5, #512]   \n"
                    "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%5], #64 \n"

                    "fmla   v8.4s, v16.4s, v6.s[0]  \n"
                    "fmla   v10.4s, v16.4s, v6.s[1] \n"
                    "fmla   v12.4s, v16.4s, v6.s[2] \n"
                    "fmla   v14.4s, v16.4s, v6.s[3] \n"
                    "fmla   v9.4s, v17.4s, v6.s[0]  \n"
                    "fmla   v11.4s, v17.4s, v6.s[1] \n"
                    "fmla   v13.4s, v17.4s, v6.s[2] \n"
                    "fmla   v15.4s, v17.4s, v6.s[3] \n"

                    "fmla   v8.4s, v18.4s, v7.s[0]  \n"
                    "fmla   v10.4s, v18.4s, v7.s[1] \n"
                    "fmla   v12.4s, v18.4s, v7.s[2] \n"
                    "fmla   v14.4s, v18.4s, v7.s[3] \n"
                    "fmla   v9.4s, v19.4s, v7.s[0]  \n"
                    "fmla   v11.4s, v19.4s, v7.s[1] \n"
                    "fmla   v13.4s, v19.4s, v7.s[2] \n"
                    "fmla   v15.4s, v19.4s, v7.s[3] \n"

                    "bne    0b                      \n"

                    "st1    {v8.4s, v9.4s}, [%1], #32 \n"
                    "st1    {v10.4s, v11.4s}, [%2], #32 \n"
                    "st1    {v12.4s, v13.4s}, [%3], #32 \n"
                    "st1    {v14.4s, v15.4s}, [%4], #32 \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(outptr2), // %3
                    "=r"(outptr3), // %4
                    "=r"(tmpptr),  // %5
                    "=r"(kptr)     // %6
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(outptr2),
                    "4"(outptr3),
                    "5"(tmpptr),
                    "6"(kptr),
                    "r"(biasptr) // %14
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
    #else  // __aarch64__
                asm volatile(
                    "vld1.f32   {d30-d31}, [%14] \n"
                    "vdup.f32   q8, d30[0]      \n"
                    "vdup.f32   q9, d30[0]      \n"
                    "vdup.f32   q10, d30[1]     \n"
                    "vdup.f32   q11, d30[1]     \n"
                    "vdup.f32   q12, d31[0]     \n"
                    "vdup.f32   q13, d31[0]     \n"
                    "vdup.f32   q14, d31[1]     \n"
                    "vdup.f32   q15, d31[1]     \n"

                    "0:                         \n"

                    "pld        [%5, #512]      \n"
                    "vldm       %5!, {d0-d7}    \n"

                    "pld        [%6, #512]      \n"
                    "vldm       %6!, {d8-d15}   \n"

                    "vmla.f32   q8, q0, d8[0]   \n"
                    "vmla.f32   q10, q0, d8[1]  \n"
                    "vmla.f32   q12, q0, d9[0]  \n"
                    "vmla.f32   q14, q0, d9[1]  \n"
                    "vmla.f32   q9, q1, d8[0]   \n"
                    "vmla.f32   q11, q1, d8[1]  \n"
                    "vmla.f32   q13, q1, d9[0]  \n"
                    "vmla.f32   q15, q1, d9[1]  \n"

                    "vmla.f32   q8, q2, d10[0]  \n"
                    "vmla.f32   q10, q2, d10[1] \n"
                    "vmla.f32   q12, q2, d11[0] \n"
                    "vmla.f32   q14, q2, d11[1] \n"
                    "vmla.f32   q9, q3, d10[0]  \n"
                    "vmla.f32   q11, q3, d10[1] \n"
                    "vmla.f32   q13, q3, d11[0] \n"
                    "vmla.f32   q15, q3, d11[1] \n"

                    "pld        [%5, #512]      \n"
                    "vldm       %5!, {d0-d7}    \n"

                    "vmla.f32   q8, q0, d12[0]  \n"
                    "vmla.f32   q10, q0, d12[1] \n"
                    "vmla.f32   q12, q0, d13[0] \n"
                    "vmla.f32   q14, q0, d13[1] \n"
                    "vmla.f32   q9, q1, d12[0]  \n"
                    "vmla.f32   q11, q1, d12[1] \n"
                    "vmla.f32   q13, q1, d13[0] \n"
                    "vmla.f32   q15, q1, d13[1] \n"

                    "subs       %0, %0, #1      \n"

                    "vmla.f32   q8, q2, d14[0]  \n"
                    "vmla.f32   q10, q2, d14[1] \n"
                    "vmla.f32   q12, q2, d15[0] \n"
                    "vmla.f32   q14, q2, d15[1] \n"
                    "vmla.f32   q9, q3, d14[0]  \n"
                    "vmla.f32   q11, q3, d14[1] \n"
                    "vmla.f32   q13, q3, d15[0] \n"
                    "vmla.f32   q15, q3, d15[1] \n"

                    "bne        0b              \n"

                    "vst1.f32   {d16-d19}, [%1 :128]! \n"
                    "vst1.f32   {d20-d23}, [%2 :128]! \n"
                    "vst1.f32   {d24-d27}, [%3 :128]! \n"
                    "vst1.f32   {d28-d31}, [%4 :128]! \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(outptr2), // %3
                    "=r"(outptr3), // %4
                    "=r"(tmpptr),  // %5
                    "=r"(kptr)     // %6
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(outptr2),
                    "4"(outptr3),
                    "5"(tmpptr),
                    "6"(kptr),
                    "r"(biasptr) // %14
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
    #endif // __aarch64__
            }
            for (; i + 3 < size; i += 4)
            {
    #if __aarch64__
                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4].data();
                const float* kptr = (const float*)kernel_a[p / 8 + (p % 8) / 4].data();
    #else
                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();
                const float* kptr = (const float*)kernel_a[p / 4].data();
    #endif

                int nn = inch * maxk; // inch always > 0

    #if __aarch64__
                asm volatile(
                    "ld1    {v11.4s}, [%14]         \n"
                    "dup    v8.4s, v11.s[0]         \n"
                    "dup    v9.4s, v11.s[1]         \n"
                    "dup    v10.4s, v11.s[2]        \n"
                    "dup    v11.4s, v11.s[3]        \n"

                    "0:                             \n"

                    "prfm   pldl1keep, [%5, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%5], #64 \n"

                    "prfm   pldl1keep, [%6, #512]   \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%6], #64 \n"

                    "subs   %w0, %w0, #1            \n"

                    "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                    "fmla   v9.4s, v0.4s, v4.s[1]   \n"
                    "fmla   v10.4s, v0.4s, v4.s[2]  \n"
                    "fmla   v11.4s, v0.4s, v4.s[3]  \n"

                    "fmla   v8.4s, v1.4s, v5.s[0]   \n"
                    "fmla   v9.4s, v1.4s, v5.s[1]   \n"
                    "fmla   v10.4s, v1.4s, v5.s[2]  \n"
                    "fmla   v11.4s, v1.4s, v5.s[3]  \n"

                    "fmla   v8.4s, v2.4s, v6.s[0]   \n"
                    "fmla   v9.4s, v2.4s, v6.s[1]   \n"
                    "fmla   v10.4s, v2.4s, v6.s[2]  \n"
                    "fmla   v11.4s, v2.4s, v6.s[3]  \n"

                    "fmla   v8.4s, v3.4s, v7.s[0]   \n"
                    "fmla   v9.4s, v3.4s, v7.s[1]   \n"
                    "fmla   v10.4s, v3.4s, v7.s[2]  \n"
                    "fmla   v11.4s, v3.4s, v7.s[3]  \n"

                    "bne    0b                      \n"

                    "st1    {v8.4s}, [%1], #16      \n"
                    "st1    {v9.4s}, [%2], #16      \n"
                    "st1    {v10.4s}, [%3], #16     \n"
                    "st1    {v11.4s}, [%4], #16     \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(outptr2), // %3
                    "=r"(outptr3), // %4
                    "=r"(tmpptr),  // %5
                    "=r"(kptr)     // %6
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(outptr2),
                    "4"(outptr3),
                    "5"(tmpptr),
                    "6"(kptr),
                    "r"(biasptr) // %14
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
    #else  // __aarch64__
                asm volatile(
                    "vld1.f32   {d22-d23}, [%14] \n"
                    "vdup.f32   q8, d22[0]      \n"
                    "vdup.f32   q9, d22[1]      \n"
                    "vdup.f32   q10, d23[0]     \n"
                    "vdup.f32   q11, d23[1]     \n"

                    "0:                         \n"

                    "pld        [%5, #512]      \n"
                    "vldm       %5!, {d0-d7}    \n"

                    "pld        [%6, #512]      \n"
                    "vldm       %6!, {d8-d15}   \n"

                    "vmla.f32   q8, q0, d8[0]   \n"
                    "vmla.f32   q9, q0, d8[1]   \n"
                    "vmla.f32   q10, q0, d9[0]  \n"
                    "vmla.f32   q11, q0, d9[1]  \n"

                    "vmla.f32   q8, q1, d10[0]  \n"
                    "vmla.f32   q9, q1, d10[1]  \n"
                    "vmla.f32   q10, q1, d11[0] \n"
                    "vmla.f32   q11, q1, d11[1] \n"

                    "subs       %0, %0, #1      \n"

                    "vmla.f32   q8, q2, d12[0]  \n"
                    "vmla.f32   q9, q2, d12[1]  \n"
                    "vmla.f32   q10, q2, d13[0] \n"
                    "vmla.f32   q11, q2, d13[1] \n"

                    "vmla.f32   q8, q3, d14[0]  \n"
                    "vmla.f32   q9, q3, d14[1]  \n"
                    "vmla.f32   q10, q3, d15[0] \n"
                    "vmla.f32   q11, q3, d15[1] \n"

                    "bne        0b              \n"

                    "vst1.f32   {d16-d17}, [%1 :128]! \n"
                    "vst1.f32   {d18-d19}, [%2 :128]! \n"
                    "vst1.f32   {d20-d21}, [%3 :128]! \n"
                    "vst1.f32   {d22-d23}, [%4 :128]! \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(outptr2), // %3
                    "=r"(outptr3), // %4
                    "=r"(tmpptr),  // %5
                    "=r"(kptr)     // %6
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(outptr2),
                    "4"(outptr3),
                    "5"(tmpptr),
                    "6"(kptr),
                    "r"(biasptr) // %14
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
    #endif // __aarch64__
            }
            for (; i < size; i++)
            {
    #if __aarch64__
                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4].data();
                const float* kptr = (const float*)kernel_a[p / 8 + (p % 8) / 4].data();
    #else
                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();
                const float* kptr = (const float*)kernel_a[p / 4].data();
    #endif

                int nn = inch * maxk; // inch always > 0

    #if __aarch64__
                asm volatile(
                    "ld1    {v8.4s}, [%14]          \n"
                    "eor    v9.16b, v9.16b, v9.16b  \n"
                    "eor    v10.16b, v10.16b, v10.16b \n"
                    "eor    v11.16b, v11.16b, v11.16b \n"

                    "0:                             \n"

                    "prfm   pldl1keep, [%5, #128]   \n"
                    "ld1    {v0.4s}, [%5], #16      \n"

                    "prfm   pldl1keep, [%6, #512]   \n"
                    "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%6], #64 \n"

                    "subs   %w0, %w0, #1            \n"

                    "fmla   v8.4s, v4.4s, v0.s[0]   \n"
                    "fmla   v9.4s, v5.4s, v0.s[1]   \n"
                    "fmla   v10.4s, v6.4s, v0.s[2]  \n"
                    "fmla   v11.4s, v7.4s, v0.s[3]  \n"

                    "bne    0b                      \n"

                    "fadd   v8.4s, v8.4s, v9.4s     \n"
                    "fadd   v10.4s, v10.4s, v11.4s  \n"
                    "fadd   v8.4s, v8.4s, v10.4s    \n"

                    "st1    {v8.s}[0], [%1], #4     \n"
                    "st1    {v8.s}[1], [%2], #4     \n"
                    "st1    {v8.s}[2], [%3], #4     \n"
                    "st1    {v8.s}[3], [%4], #4     \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(outptr2), // %3
                    "=r"(outptr3), // %4
                    "=r"(tmpptr),  // %5
                    "=r"(kptr)     // %6
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(outptr2),
                    "4"(outptr3),
                    "5"(tmpptr),
                    "6"(kptr),
                    "r"(biasptr) // %14
                    : "cc", "memory", "v0", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11");
    #else  // __aarch64__
                asm volatile(
                    "vld1.f32   {d16-d17}, [%14] \n"
                    "veor       q9, q9          \n"
                    "veor       q10, q10        \n"
                    "veor       q11, q11        \n"

                    "0:                         \n"

                    "pld        [%5, #128]      \n"
                    "vld1.f32   {d0-d1}, [%5]!  \n"

                    "pld        [%6, #512]      \n"
                    "vldm       %6!, {d8-d15}   \n"

                    "subs       %0, %0, #1      \n"

                    "vmla.f32   q8, q4, d0[0]   \n"
                    "vmla.f32   q9, q5, d0[1]   \n"
                    "vmla.f32   q10, q6, d1[0]  \n"
                    "vmla.f32   q11, q7, d1[1]  \n"

                    "bne        0b              \n"

                    "vadd.f32   q8, q8, q9      \n"
                    "vadd.f32   q10, q10, q11   \n"
                    "vadd.f32   q8, q8, q10     \n"

                    "vst1.f32   {d16[0]}, [%1]! \n"
                    "vst1.f32   {d16[1]}, [%2]! \n"
                    "vst1.f32   {d17[0]}, [%3]! \n"
                    "vst1.f32   {d17[1]}, [%4]! \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(outptr1), // %2
                    "=r"(outptr2), // %3
                    "=r"(outptr3), // %4
                    "=r"(tmpptr),  // %5
                    "=r"(kptr)     // %6
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(outptr1),
                    "3"(outptr2),
                    "4"(outptr3),
                    "5"(tmpptr),
                    "6"(kptr),
                    "r"(biasptr) // %14
                    : "cc", "memory", "q0", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11");
    #endif // __aarch64__
            }
        }
    });

    remain_outch_start += nn_outch << 2;

    otter::parallel_for(remain_outch_start, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            float* outptr0 = output_a[p].data();

            const float bias0 = bias ? bias[p] : 0.f;

            int i = 0;
    #if __aarch64__
            for (; i + 11 < size; i += 12) {
                float* tmpptr = tmp_a[i / 12].data();
                const float* kptr = (const float*)kernel_a[p / 8 + (p % 8) / 4 + p % 4].data();

                int nn = inch * maxk; // inch always > 0

                asm volatile(
                    "dup    v8.4s, %w8              \n"
                    "dup    v9.4s, %w8              \n"
                    "dup    v10.4s, %w8             \n"
                    "eor    v5.16b, v5.16b, v5.16b  \n"
                    "eor    v6.16b, v6.16b, v6.16b  \n"
                    "eor    v7.16b, v7.16b, v7.16b  \n"

                    "0:                             \n"

                    "prfm   pldl1keep, [%2, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                    "prfm   pldl1keep, [%3, #128]   \n"
                    "ld1    {v4.4s}, [%3], #16      \n"

                    "subs   %w0, %w0, #1            \n"

                    "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                    "fmla   v9.4s, v1.4s, v4.s[0]   \n"
                    "fmla   v10.4s, v2.4s, v4.s[0]  \n"

                    "prfm   pldl1keep, [%2, #512]   \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%2], #64 \n"

                    "fmla   v5.4s, v3.4s, v4.s[1]   \n"
                    "fmla   v6.4s, v12.4s, v4.s[1]  \n"
                    "fmla   v7.4s, v13.4s, v4.s[1]  \n"

                    "prfm   pldl1keep, [%2, #512]   \n"
                    "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%2], #64 \n"

                    "fmla   v8.4s, v14.4s, v4.s[2]  \n"
                    "fmla   v9.4s, v15.4s, v4.s[2]  \n"
                    "fmla   v10.4s, v16.4s, v4.s[2] \n"

                    "fmla   v5.4s, v17.4s, v4.s[3]  \n"
                    "fmla   v6.4s, v18.4s, v4.s[3]  \n"
                    "fmla   v7.4s, v19.4s, v4.s[3]  \n"

                    "bne    0b                      \n"

                    "fadd   v8.4s, v8.4s, v5.4s     \n"
                    "fadd   v9.4s, v9.4s, v6.4s     \n"
                    "fadd   v10.4s, v10.4s, v7.4s   \n"

                    "st1    {v8.4s, v9.4s, v10.4s}, [%1], #48 \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(tmpptr),  // %2
                    "=r"(kptr)     // %3
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(tmpptr),
                    "3"(kptr),
                    "r"(bias0) // %8
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19");
            }
    #endif // __aarch64__
            for (; i + 7 < size; i += 8) {
    #if __aarch64__
                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8].data();
                const float* kptr = (const float*)kernel_a[p / 8 + (p % 8) / 4 + p % 4].data();
    #else
                float* tmpptr = tmp_a[i / 8].data();
                const float* kptr = (const float*)kernel_a[p / 4 + p % 4].data();
    #endif

                int nn = inch * maxk; // inch always > 0

    #if __aarch64__
                asm volatile(
                    "dup    v8.4s, %w8              \n"
                    "dup    v9.4s, %w8              \n"
                    "eor    v10.16b, v10.16b, v10.16b \n"
                    "eor    v11.16b, v11.16b, v11.16b \n"

                    "0:                             \n"

                    "prfm   pldl1keep, [%2, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                    "prfm   pldl1keep, [%3, #128]   \n"
                    "ld1    {v4.4s}, [%3], #16      \n"

                    "subs   %w0, %w0, #1            \n"

                    "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                    "fmla   v9.4s, v1.4s, v4.s[0]   \n"
                    "fmla   v10.4s, v2.4s, v4.s[1]  \n"
                    "fmla   v11.4s, v3.4s, v4.s[1]  \n"

                    "prfm   pldl1keep, [%2, #512]   \n"
                    "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%2], #64 \n"

                    "fmla   v8.4s, v12.4s, v4.s[2]  \n"
                    "fmla   v9.4s, v13.4s, v4.s[2]  \n"
                    "fmla   v10.4s, v14.4s, v4.s[3] \n"
                    "fmla   v11.4s, v15.4s, v4.s[3] \n"

                    "bne    0b                      \n"

                    "fadd   v8.4s, v8.4s, v10.4s    \n"
                    "fadd   v9.4s, v9.4s, v11.4s    \n"

                    "st1    {v8.4s, v9.4s}, [%1], #32 \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(tmpptr),  // %2
                    "=r"(kptr)     // %3
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(tmpptr),
                    "3"(kptr),
                    "r"(bias0) // %8
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
    #else  // __aarch64__
                asm volatile(
                    "vdup.f32   q8, %8          \n"
                    "vdup.f32   q9, %8          \n"
                    "veor       q10, q10        \n"
                    "veor       q11, q11        \n"

                    "0:                         \n"

                    "pld        [%2, #512]      \n"
                    "vldm       %2!, {d0-d7}    \n"

                    "pld        [%3, #128]      \n"
                    "vld1.f32   {d8-d9}, [%3]!  \n"

                    "vmla.f32   q8, q0, d8[0]   \n"
                    "vmla.f32   q9, q1, d8[0]   \n"
                    "vmla.f32   q10, q2, d8[1]  \n"
                    "vmla.f32   q11, q3, d8[1]  \n"

                    "pld        [%2, #512]      \n"
                    "vldm       %2!, {d24-d31}  \n"

                    "subs       %0, %0, #1      \n"

                    "vmla.f32   q8, q12, d9[0]  \n"
                    "vmla.f32   q9, q13, d9[0]  \n"
                    "vmla.f32   q10, q14, d9[1] \n"
                    "vmla.f32   q11, q15, d9[1] \n"

                    "bne        0b              \n"

                    "vadd.f32   q8, q8, q10     \n"
                    "vadd.f32   q9, q9, q11     \n"

                    "vst1.f32   {d16-d19}, [%1 :128]! \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(tmpptr),  // %2
                    "=r"(kptr)     // %3
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(tmpptr),
                    "3"(kptr),
                    "r"(bias0) // %8
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
    #endif // __aarch64__
            }
            for (; i + 3 < size; i += 4)
            {
    #if __aarch64__
                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4].data();
                const float* kptr = (const float*)kernel_a[p / 8 + (p % 8) / 4 + p % 4].data();
    #else
                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();
                const float* kptr = (const float*)kernel_a[p / 4 + p % 4].data();
    #endif

                int nn = inch * maxk; // inch always > 0

    #if __aarch64__
                asm volatile(
                    "dup    v8.4s, %w8              \n"
                    "eor    v9.16b, v9.16b, v9.16b  \n"
                    "eor    v10.16b, v10.16b, v10.16b \n"
                    "eor    v11.16b, v11.16b, v11.16b \n"

                    "0:                             \n"

                    "prfm   pldl1keep, [%2, #512]   \n"
                    "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                    "prfm   pldl1keep, [%3, #128]   \n"
                    "ld1    {v4.4s}, [%3], #16      \n"

                    "subs   %w0, %w0, #1            \n"

                    "fmla   v8.4s, v0.4s, v4.s[0]   \n"
                    "fmla   v9.4s, v1.4s, v4.s[1]   \n"
                    "fmla   v10.4s, v2.4s, v4.s[2]  \n"
                    "fmla   v11.4s, v3.4s, v4.s[3]  \n"

                    "bne    0b                      \n"

                    "fadd   v8.4s, v8.4s, v9.4s     \n"
                    "fadd   v10.4s, v10.4s, v11.4s  \n"
                    "fadd   v8.4s, v8.4s, v10.4s    \n"

                    "st1    {v8.4s}, [%1], #16      \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(tmpptr),  // %2
                    "=r"(kptr)     // %3
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(tmpptr),
                    "3"(kptr),
                    "r"(bias0) // %8
                    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v8", "v9", "v10", "v11");
    #else  // __aarch64__
                asm volatile(
                    "vdup.f32   q8, %8          \n"
                    "veor       q9, q9          \n"
                    "veor       q10, q10        \n"
                    "veor       q11, q11        \n"

                    "0:                         \n"

                    "pld        [%2, #512]      \n"
                    "vldm       %2!, {d0-d7}    \n"

                    "pld        [%3, #128]      \n"
                    "vld1.f32   {d8-d9}, [%3]!  \n"

                    "subs       %0, %0, #1      \n"

                    "vmla.f32   q8, q0, d8[0]   \n"
                    "vmla.f32   q9, q1, d8[1]   \n"
                    "vmla.f32   q10, q2, d9[0]  \n"
                    "vmla.f32   q11, q3, d9[1]  \n"

                    "bne        0b              \n"

                    "vadd.f32   q8, q8, q9      \n"
                    "vadd.f32   q10, q10, q11   \n"
                    "vadd.f32   q8, q8, q10     \n"

                    "vst1.f32   {d16-d17}, [%1]! \n"

                    : "=r"(nn),      // %0
                    "=r"(outptr0), // %1
                    "=r"(tmpptr),  // %2
                    "=r"(kptr)     // %3
                    : "0"(nn),
                    "1"(outptr0),
                    "2"(tmpptr),
                    "3"(kptr),
                    "r"(bias0) // %8
                    : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q8", "q9", "q10", "q11");
    #endif // __aarch64__
            }
            for (; i < size; i++)
            {
    #if __aarch64__
                float* tmpptr = tmp_a[i / 12 + (i % 12) / 8 + (i % 12 % 8) / 4 + i % 12 % 4].data();
                const float* kptr = (const float*)kernel_a[p / 8 + (p % 8) / 4 + p % 4].data();
    #else
                float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();
                const float* kptr = (const float*)kernel_a[p / 4 + p % 4].data();
    #endif

                int nn = inch * maxk; // inch always > 0

                float32x4_t _sum0 = vdupq_n_f32(0.f);

                for (int q = 0; q < nn; q++)
                {
                    float32x4_t _r0 = vld1q_f32(tmpptr);

                    float32x4_t _k0 = vld1q_f32(kptr);

                    _sum0 = vmlaq_f32(_sum0, _r0, _k0);

                    kptr += 4;
                    tmpptr += 4;
                }

    #if __aarch64__
                float sum0 = vaddvq_f32(_sum0);
    #else
                float32x2_t _ss = vadd_f32(vget_low_f32(_sum0), vget_high_f32(_sum0));
                float32x2_t _ss2 = vpadd_f32(_ss, _ss);
                float sum0 = vget_lane_f32(_ss2, 0);
    #endif

                outptr0[0] = bias0 + sum0;

                outptr0++;
            }
        }
    });
}

void im2col_sgemm_conv2d_pack1to4_impl_neon(const Tensor& im2col, Tensor& output_, const Tensor& kernel, const Tensor& _bias) {
    const int size = im2col.size(2);
    const int maxk = im2col.size(1);
    const int inch = im2col.size(0);

    const int outch = output_.size(1);

    const float* bias = (_bias.defined()) ? _bias.data_ptr<float>() : nullptr;
    
    auto output_a = output_.accessor<float, 4, 4>()[0];
    auto im2col_a = im2col.accessor<float, 3>();
    auto kernel_a = kernel.accessor<float, 3>();
    
    // permute
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
                        vst1q_f32(tmpptr, vld1q_f32(img0));
                        vst1q_f32(tmpptr + 4, vld1q_f32(img0 + 4));
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
                    const float* img0 = (const float*)im2col_a[q].data() + i;

                    for (int k = 0; k < maxk; k++)
                    {
                        vst1q_f32(tmpptr, vld1q_f32(img0));
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

                for (int q = 0; q < inch; q++) {
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

    int remain_outch_start = 0;

#if __aarch64__
    int nn_outch = outch >> 1;
    remain_outch_start = nn_outch << 1;

    otter::parallel_for(0, nn_outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto pp : otter::irange(begin, end)) {
            int p = pp * 2;

            float* outptr0 = output_a[p + 0].data();
            float* outptr1 = output_a[p + 1].data();

            const float zeros[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + p * 4 : zeros;

            int i = 0;
            for (; i + 7 < size; i += 8)
            {
                const float* tmpptr = tmp_a[i / 8].data();
                const float* kptr0 = kernel_a[p / 2].data();

                int nn = inch * maxk; // inch always > 0

                float32x4_t _sum00 = vld1q_f32(biasptr);
                float32x4_t _sum01 = vld1q_f32(biasptr);
                float32x4_t _sum02 = vld1q_f32(biasptr);
                float32x4_t _sum03 = vld1q_f32(biasptr);
                float32x4_t _sum04 = vld1q_f32(biasptr);
                float32x4_t _sum05 = vld1q_f32(biasptr);
                float32x4_t _sum06 = vld1q_f32(biasptr);
                float32x4_t _sum07 = vld1q_f32(biasptr);
                float32x4_t _sum10 = vld1q_f32(biasptr + 4);
                float32x4_t _sum11 = vld1q_f32(biasptr + 4);
                float32x4_t _sum12 = vld1q_f32(biasptr + 4);
                float32x4_t _sum13 = vld1q_f32(biasptr + 4);
                float32x4_t _sum14 = vld1q_f32(biasptr + 4);
                float32x4_t _sum15 = vld1q_f32(biasptr + 4);
                float32x4_t _sum16 = vld1q_f32(biasptr + 4);
                float32x4_t _sum17 = vld1q_f32(biasptr + 4);

                for (int j = 0; j < nn; j++)
                {
                    float32x4_t _val0 = vld1q_f32(tmpptr);
                    float32x4_t _val1 = vld1q_f32(tmpptr + 4);
                    float32x4_t _w0 = vld1q_f32(kptr0);
                    float32x4_t _w1 = vld1q_f32(kptr0 + 4);

                    _sum00 = vmlaq_laneq_f32(_sum00, _w0, _val0, 0);
                    _sum01 = vmlaq_laneq_f32(_sum01, _w0, _val0, 1);
                    _sum02 = vmlaq_laneq_f32(_sum02, _w0, _val0, 2);
                    _sum03 = vmlaq_laneq_f32(_sum03, _w0, _val0, 3);
                    _sum04 = vmlaq_laneq_f32(_sum04, _w0, _val1, 0);
                    _sum05 = vmlaq_laneq_f32(_sum05, _w0, _val1, 1);
                    _sum06 = vmlaq_laneq_f32(_sum06, _w0, _val1, 2);
                    _sum07 = vmlaq_laneq_f32(_sum07, _w0, _val1, 3);
                    _sum10 = vmlaq_laneq_f32(_sum10, _w1, _val0, 0);
                    _sum11 = vmlaq_laneq_f32(_sum11, _w1, _val0, 1);
                    _sum12 = vmlaq_laneq_f32(_sum12, _w1, _val0, 2);
                    _sum13 = vmlaq_laneq_f32(_sum13, _w1, _val0, 3);
                    _sum14 = vmlaq_laneq_f32(_sum14, _w1, _val1, 0);
                    _sum15 = vmlaq_laneq_f32(_sum15, _w1, _val1, 1);
                    _sum16 = vmlaq_laneq_f32(_sum16, _w1, _val1, 2);
                    _sum17 = vmlaq_laneq_f32(_sum17, _w1, _val1, 3);

                    tmpptr += 8;
                    kptr0 += 8;
                }

                vst1q_f32(outptr0, _sum00);
                vst1q_f32(outptr0 + 4, _sum01);
                vst1q_f32(outptr0 + 8, _sum02);
                vst1q_f32(outptr0 + 12, _sum03);
                vst1q_f32(outptr0 + 16, _sum04);
                vst1q_f32(outptr0 + 20, _sum05);
                vst1q_f32(outptr0 + 24, _sum06);
                vst1q_f32(outptr0 + 28, _sum07);
                vst1q_f32(outptr1, _sum10);
                vst1q_f32(outptr1 + 4, _sum11);
                vst1q_f32(outptr1 + 8, _sum12);
                vst1q_f32(outptr1 + 12, _sum13);
                vst1q_f32(outptr1 + 16, _sum14);
                vst1q_f32(outptr1 + 20, _sum15);
                vst1q_f32(outptr1 + 24, _sum16);
                vst1q_f32(outptr1 + 28, _sum17);
                outptr0 += 32;
                outptr1 += 32;
            }
            for (; i + 3 < size; i += 4)
            {
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();
                const float* kptr0 = kernel_a[p / 2].data();

                int nn = inch * maxk; // inch always > 0

                float32x4_t _sum00 = vld1q_f32(biasptr);
                float32x4_t _sum01 = vld1q_f32(biasptr);
                float32x4_t _sum02 = vld1q_f32(biasptr);
                float32x4_t _sum03 = vld1q_f32(biasptr);
                float32x4_t _sum10 = vld1q_f32(biasptr + 4);
                float32x4_t _sum11 = vld1q_f32(biasptr + 4);
                float32x4_t _sum12 = vld1q_f32(biasptr + 4);
                float32x4_t _sum13 = vld1q_f32(biasptr + 4);

                for (int j = 0; j < nn; j++)
                {
                    float32x4_t _val = vld1q_f32(tmpptr);
                    float32x4_t _w0 = vld1q_f32(kptr0);
                    float32x4_t _w1 = vld1q_f32(kptr0 + 4);

                    _sum00 = vmlaq_laneq_f32(_sum00, _w0, _val, 0);
                    _sum01 = vmlaq_laneq_f32(_sum01, _w0, _val, 1);
                    _sum02 = vmlaq_laneq_f32(_sum02, _w0, _val, 2);
                    _sum03 = vmlaq_laneq_f32(_sum03, _w0, _val, 3);
                    _sum10 = vmlaq_laneq_f32(_sum10, _w1, _val, 0);
                    _sum11 = vmlaq_laneq_f32(_sum11, _w1, _val, 1);
                    _sum12 = vmlaq_laneq_f32(_sum12, _w1, _val, 2);
                    _sum13 = vmlaq_laneq_f32(_sum13, _w1, _val, 3);

                    tmpptr += 4;
                    kptr0 += 8;
                }

                vst1q_f32(outptr0, _sum00);
                vst1q_f32(outptr0 + 4, _sum01);
                vst1q_f32(outptr0 + 8, _sum02);
                vst1q_f32(outptr0 + 12, _sum03);
                vst1q_f32(outptr1, _sum10);
                vst1q_f32(outptr1 + 4, _sum11);
                vst1q_f32(outptr1 + 8, _sum12);
                vst1q_f32(outptr1 + 12, _sum13);
                outptr0 += 16;
                outptr1 += 16;
            }
            for (; i < size; i++)
            {
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();
                const float* kptr0 = kernel_a[p / 2].data();

                int nn = inch * maxk; // inch always > 0

                float32x4_t _sum0 = vld1q_f32(biasptr);
                float32x4_t _sum1 = vld1q_f32(biasptr + 4);

                for (int j = 0; j < nn; j++)
                {
                    float32x4_t _val = vdupq_n_f32(tmpptr[0]);
                    float32x4_t _w0 = vld1q_f32(kptr0);
                    float32x4_t _w1 = vld1q_f32(kptr0 + 4);
                    _sum0 = vmlaq_f32(_sum0, _val, _w0);
                    _sum1 = vmlaq_f32(_sum1, _val, _w1);

                    tmpptr += 1;
                    kptr0 += 8;
                }

                vst1q_f32(outptr0, _sum0);
                vst1q_f32(outptr1, _sum1);
                outptr0 += 4;
                outptr1 += 4;
            }
        }
    });
#endif // __aarch64__

    otter::parallel_for(remain_outch_start, outch, 0, [&](int64_t begin, int64_t end) {
        for (const auto p : otter::irange(begin, end)) {
            float* outptr0 = output_a[p].data();

            const float zeros[4] = {0.f, 0.f, 0.f, 0.f};
            const float* biasptr = bias ? bias + p * 4 : zeros;

            int i = 0;
            for (; i + 7 < size; i += 8) {
                const float* tmpptr = tmp_a[i / 8].data();
    #if __aarch64__
                const float* kptr0 = kernel_a[p / 2 + p % 2].data();
    #else
                const float* kptr0 = kernel_a[p].data();
    #endif

                int nn = inch * maxk; // inch always > 0

                float32x4_t _sum0 = vld1q_f32(biasptr);
                float32x4_t _sum1 = vld1q_f32(biasptr);
                float32x4_t _sum2 = vld1q_f32(biasptr);
                float32x4_t _sum3 = vld1q_f32(biasptr);
                float32x4_t _sum4 = vld1q_f32(biasptr);
                float32x4_t _sum5 = vld1q_f32(biasptr);
                float32x4_t _sum6 = vld1q_f32(biasptr);
                float32x4_t _sum7 = vld1q_f32(biasptr);

                for (int j = 0; j < nn; j++)
                {
                    float32x4_t _val0 = vld1q_f32(tmpptr);
                    float32x4_t _val1 = vld1q_f32(tmpptr + 4);
                    float32x4_t _w0 = vld1q_f32(kptr0);

    #if __aarch64__
                    _sum0 = vmlaq_laneq_f32(_sum0, _w0, _val0, 0);
                    _sum1 = vmlaq_laneq_f32(_sum1, _w0, _val0, 1);
                    _sum2 = vmlaq_laneq_f32(_sum2, _w0, _val0, 2);
                    _sum3 = vmlaq_laneq_f32(_sum3, _w0, _val0, 3);
                    _sum4 = vmlaq_laneq_f32(_sum4, _w0, _val1, 0);
                    _sum5 = vmlaq_laneq_f32(_sum5, _w0, _val1, 1);
                    _sum6 = vmlaq_laneq_f32(_sum6, _w0, _val1, 2);
                    _sum7 = vmlaq_laneq_f32(_sum7, _w0, _val1, 3);
    #else
                    _sum0 = vmlaq_lane_f32(_sum0, _w0, vget_low_f32(_val0), 0);
                    _sum1 = vmlaq_lane_f32(_sum1, _w0, vget_low_f32(_val0), 1);
                    _sum2 = vmlaq_lane_f32(_sum2, _w0, vget_high_f32(_val0), 0);
                    _sum3 = vmlaq_lane_f32(_sum3, _w0, vget_high_f32(_val0), 1);
                    _sum4 = vmlaq_lane_f32(_sum4, _w0, vget_low_f32(_val1), 0);
                    _sum5 = vmlaq_lane_f32(_sum5, _w0, vget_low_f32(_val1), 1);
                    _sum6 = vmlaq_lane_f32(_sum6, _w0, vget_high_f32(_val1), 0);
                    _sum7 = vmlaq_lane_f32(_sum7, _w0, vget_high_f32(_val1), 1);
    #endif

                    tmpptr += 8;
                    kptr0 += 4;
                }

                vst1q_f32(outptr0, _sum0);
                vst1q_f32(outptr0 + 4, _sum1);
                vst1q_f32(outptr0 + 8, _sum2);
                vst1q_f32(outptr0 + 12, _sum3);
                vst1q_f32(outptr0 + 16, _sum4);
                vst1q_f32(outptr0 + 20, _sum5);
                vst1q_f32(outptr0 + 24, _sum6);
                vst1q_f32(outptr0 + 28, _sum7);
                outptr0 += 32;
            }
            for (; i + 3 < size; i += 4)
            {
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4].data();
    #if __aarch64__
                const float* kptr0 = kernel_a[p / 2 + p % 2].data();
    #else
                const float* kptr0 = kernel_a[p].data();
    #endif

                int nn = inch * maxk; // inch always > 0

                float32x4_t _sum0 = vld1q_f32(biasptr);
                float32x4_t _sum1 = vld1q_f32(biasptr);
                float32x4_t _sum2 = vld1q_f32(biasptr);
                float32x4_t _sum3 = vld1q_f32(biasptr);

                for (int j = 0; j < nn; j++)
                {
                    float32x4_t _val = vld1q_f32(tmpptr);
                    float32x4_t _w0 = vld1q_f32(kptr0);

    #if __aarch64__
                    _sum0 = vmlaq_laneq_f32(_sum0, _w0, _val, 0);
                    _sum1 = vmlaq_laneq_f32(_sum1, _w0, _val, 1);
                    _sum2 = vmlaq_laneq_f32(_sum2, _w0, _val, 2);
                    _sum3 = vmlaq_laneq_f32(_sum3, _w0, _val, 3);
    #else
                    _sum0 = vmlaq_lane_f32(_sum0, _w0, vget_low_f32(_val), 0);
                    _sum1 = vmlaq_lane_f32(_sum1, _w0, vget_low_f32(_val), 1);
                    _sum2 = vmlaq_lane_f32(_sum2, _w0, vget_high_f32(_val), 0);
                    _sum3 = vmlaq_lane_f32(_sum3, _w0, vget_high_f32(_val), 1);
    #endif

                    tmpptr += 4;
                    kptr0 += 4;
                }

                vst1q_f32(outptr0, _sum0);
                vst1q_f32(outptr0 + 4, _sum1);
                vst1q_f32(outptr0 + 8, _sum2);
                vst1q_f32(outptr0 + 12, _sum3);
                outptr0 += 16;
            }
            for (; i < size; i++)
            {
                const float* tmpptr = tmp_a[i / 8 + (i % 8) / 4 + i % 4].data();
    #if __aarch64__
                const float* kptr0 = kernel_a[p / 2 + p % 2].data();
    #else
                const float* kptr0 = kernel_a[p].data();
    #endif

                int nn = inch * maxk; // inch always > 0

                float32x4_t _sum = vld1q_f32(biasptr);

                for (int j = 0; j < nn; j++)
                {
                    float32x4_t _val = vdupq_n_f32(tmpptr[0]);
                    float32x4_t _w0 = vld1q_f32(kptr0);
                    _sum = vmlaq_f32(_sum, _val, _w0);

                    tmpptr += 1;
                    kptr0 += 4;
                }

                vst1q_f32(outptr0, _sum);
                outptr0 += 4;
            }
        }
    });
}

void convolution_im2col_sgemm_transform_kernel_pack4_neon(const Tensor& kernel_, Tensor& kernel_tf, int inch, int outch, int kernel_w, int kernel_h) {
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 4b-4a-maxk-inch/4a-outch/4b
    Tensor kernel = kernel_.view({outch, inch, maxk});;
#if __aarch64__
    kernel_tf = otter::empty({outch / 8 + (outch % 8) / 4, inch / 4, 32 * maxk}, otter::ScalarType::Float);
#else
    kernel_tf = otter::empty({outch / 4, inch / 4, 16 * maxk}, otter::ScalarType::Float);
#endif
    
    auto kernel_a = kernel.accessor<float, 3>();
    auto kernel_tf_a = kernel_tf.accessor<float, 3>();

    int q = 0;
#if __aarch64__
    for (; q + 7 < outch; q += 8) {
        const auto k0 = kernel_a[q + 0];
        const auto k1 = kernel_a[q + 1];
        const auto k2 = kernel_a[q + 2];
        const auto k3 = kernel_a[q + 3];
        const auto k4 = kernel_a[q + 4];
        const auto k5 = kernel_a[q + 5];
        const auto k6 = kernel_a[q + 6];
        const auto k7 = kernel_a[q + 7];

        float* g00 = kernel_tf_a[q / 8].data();

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

            const float* k40 = k4[p + 0].data();
            const float* k41 = k4[p + 1].data();
            const float* k42 = k4[p + 2].data();
            const float* k43 = k4[p + 3].data();

            const float* k50 = k5[p + 0].data();
            const float* k51 = k5[p + 1].data();
            const float* k52 = k5[p + 2].data();
            const float* k53 = k5[p + 3].data();

            const float* k60 = k6[p + 0].data();
            const float* k61 = k6[p + 1].data();
            const float* k62 = k6[p + 2].data();
            const float* k63 = k6[p + 3].data();

            const float* k70 = k7[p + 0].data();
            const float* k71 = k7[p + 1].data();
            const float* k72 = k7[p + 2].data();
            const float* k73 = k7[p + 3].data();

            for (int k = 0; k < maxk; k++) {
                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];
                g00[4] = k40[k];
                g00[5] = k50[k];
                g00[6] = k60[k];
                g00[7] = k70[k];

                g00[8] = k01[k];
                g00[9] = k11[k];
                g00[10] = k21[k];
                g00[11] = k31[k];
                g00[12] = k41[k];
                g00[13] = k51[k];
                g00[14] = k61[k];
                g00[15] = k71[k];

                g00[16] = k02[k];
                g00[17] = k12[k];
                g00[18] = k22[k];
                g00[19] = k32[k];
                g00[20] = k42[k];
                g00[21] = k52[k];
                g00[22] = k62[k];
                g00[23] = k72[k];

                g00[24] = k03[k];
                g00[25] = k13[k];
                g00[26] = k23[k];
                g00[27] = k33[k];
                g00[28] = k43[k];
                g00[29] = k53[k];
                g00[30] = k63[k];
                g00[31] = k73[k];

                g00 += 32;
            }
        }
    }
#endif // __aarch64__
    for (; q + 3 < outch; q += 4) {
        const auto k0 = kernel_a[q + 0];
        const auto k1 = kernel_a[q + 1];
        const auto k2 = kernel_a[q + 2];
        const auto k3 = kernel_a[q + 3];

#if __aarch64__
        float* g00 = kernel_tf_a[q / 8 + (q % 8) / 4].data();
#else
        float* g00 = kernel_tf_a[q / 4].data();
#endif

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

void convolution_im2col_sgemm_transform_kernel_pack4to1_neon(const Tensor& kernel_, Tensor& kernel_tf, int inch, int outch, int kernel_w, int kernel_h) {
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = pb-pa-maxk-inch/pa-outch/pb
    Tensor kernel = kernel_.view({outch, inch, maxk});
#if __aarch64__
    kernel_tf = otter::empty({outch / 8 + (outch % 8) / 4 + outch % 4, inch / 4, 32 * maxk}, otter::ScalarType::Float);
#else
    kernel_tf = otter::empty({outch / 4 + outch % 4, inch / 4, 16 * maxk}, otter::ScalarType::Float);
#endif
    
    auto kernel_a = kernel.accessor<float, 3>();
    auto kernel_tf_a = kernel_tf.accessor<float, 3>();

    int q = 0;
#if __aarch64__
    for (; q + 7 < outch; q += 8) {
        float* g00 = kernel_tf_a[q / 8].data();

        for (int p = 0; p + 3 < inch; p += 4) {
            for (int k = 0; k < maxk; k++) {
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 8; j++) {
                        const float* k00 = kernel_a[q + j][p + i].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
    }
#endif // __aarch64__
    for (; q + 3 < outch; q += 4) {
#if __aarch64__
        float* g00 = kernel_tf_a[q / 8 + (q % 8) / 4].data();
#else
        float* g00 = kernel_tf_a[q / 4].data();
#endif

        for (int p = 0; p + 3 < inch; p += 4) {
            for (int k = 0; k < maxk; k++) {
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        const float* k00 = kernel_a[q + j][p + i].data();

                        g00[0] = k00[k];

                        g00++;
                    }
                }
            }
        }
    }
    for (; q < outch; q++) {
        const auto k0 = kernel_a[q];

#if __aarch64__
        float* g00 = kernel_tf_a[q / 8 + (q % 8) / 4 + q % 4].data();
#else
        float* g00 = kernel_tf_a[q / 4 + q % 4].data();
#endif

        for (int p = 0; p + 3 < inch; p += 4) {
            for (int k = 0; k < maxk; k++) {
                for (int j = 0; j < 4; j++) {
                    const float* k00 = k0[p + j].data();

                    g00[0] = k00[k];

                    g00++;
                }
            }
        }
    }
}

void convolution_im2col_sgemm_transform_kernel_pack1to4_neon(const Tensor& kernel_, Tensor& kernel_tf, int inch, int outch, int kernel_w, int kernel_h) {
    const int maxk = kernel_w * kernel_h;

    // interleave
    // src = maxk-inch-outch
    // dst = 4b-4a-maxk-inch/4a-outch/4b
    Tensor kernel = kernel_.view({outch, inch, maxk});
#if __aarch64__
    kernel_tf = otter::empty({outch / 8 + (outch % 8) / 4, inch, 8 * maxk}, otter::ScalarType::Float);
#else
    kernel_tf = otter::empty({outch / 4, inch, 4 * maxk}, otter::ScalarType::Float);
#endif
    
    auto kernel_a = kernel.accessor<float, 3>();
    auto kernel_tf_a = kernel_tf.accessor<float, 3>();

    int q = 0;
#if __aarch64__
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

        float* g00 = kernel_tf_a[q / 8].data();

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
#endif // __aarch64__
    for (; q + 3 < outch; q += 4)
    {
        const auto k0 = kernel_a[q + 0];
        const auto k1 = kernel_a[q + 1];
        const auto k2 = kernel_a[q + 2];
        const auto k3 = kernel_a[q + 3];

#if __aarch64__
        float* g00 = kernel_tf_a[q / 8 + (q % 8) / 4].data();
#else
        float* g00 = kernel_tf_a[q / 4].data();
#endif

        for (int p = 0; p < inch; p++)
        {
            const float* k00 = k0[p].data();
            const float* k10 = k1[p].data();
            const float* k20 = k2[p].data();
            const float* k30 = k3[p].data();

            for (int k = 0; k < maxk; k++)
            {
                g00[0] = k00[k];
                g00[1] = k10[k];
                g00[2] = k20[k];
                g00[3] = k30[k];

                g00 += 4;
            }
        }
    }
}

Tensor& sgemm_conv2d_pack4_neon_out(
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
    
    int w = self.size(3);
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
        convolution_im2col_sgemm_transform_kernel_pack4_neon(weight, kernel_tf, inch * 4, outch * 4, kernel_w, kernel_h);
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
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

                for (int u = 0; u < kernel_h; u++) {
                    for (int v = 0; v < kernel_w; v++) {
                        const float* sptr = img[dilation_h * u].data() + dilation_w * v * 4;

                        for (int i = 0; i < outh; i++) {
                            int j = 0;
                            for (; j + 3 < outw; j += 4) {
                                float32x4_t _val0 = vld1q_f32(sptr);
                                float32x4_t _val1 = vld1q_f32(sptr + stride_w * 4);
                                float32x4_t _val2 = vld1q_f32(sptr + stride_w * 8);
                                float32x4_t _val3 = vld1q_f32(sptr + stride_w * 12);
                                vst1q_f32(ptr, _val0);
                                vst1q_f32(ptr + 4, _val1);
                                vst1q_f32(ptr + 8, _val2);
                                vst1q_f32(ptr + 12, _val3);

                                sptr += stride_w * 16;
                                ptr += 16;
                            }
                            for (; j + 1 < outw; j += 2) {
                                float32x4_t _val0 = vld1q_f32(sptr);
                                float32x4_t _val1 = vld1q_f32(sptr + stride_w * 4);
                                vst1q_f32(ptr, _val0);
                                vst1q_f32(ptr + 4, _val1);

                                sptr += stride_w * 8;
                                ptr += 8;
                            }
                            for (; j < outw; j++) {
                                float32x4_t _val = vld1q_f32(sptr);
                                vst1q_f32(ptr, _val);

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
    
    im2col_sgemm_conv2d_pack4_impl_neon(im2col, output, kernel_tf, bias);
    
    return output;
}
    
Tensor sgemm_conv2d_pack4_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    Tensor output = otter::empty({}, otter::ScalarType::Float4);
    sgemm_conv2d_pack4_neon_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
    
    return output;
}

Tensor& sgemm_conv2d_pack4to1_neon_out(
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
    
    int w = self.size(3);
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
        convolution_im2col_sgemm_transform_kernel_pack4to1_neon(weight, kernel_tf, inch * 4, outch, kernel_w, kernel_h);
    
    Tensor input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    Tensor im2col = otter::empty({inch, maxk, size}, ScalarType::Float4);
    
    auto input_a = input.accessor<float, 3, 4>();
    auto im2col_a = im2col.accessor<float, 3, 4>();
    
    {
        const int gap = (w * stride_h - outw * stride_w) * 4;

        otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
            for (const auto p : otter::irange(begin, end)) {
                const auto img = input_a[p];
                float* ptr = im2col_a[p].data();

                for (int u = 0; u < kernel_h; u++) {
                    for (int v = 0; v < kernel_w; v++) {
                        const float* sptr = img[dilation_h * u].data() + dilation_w * v * 4;

                        for (int i = 0; i < outh; i++) {
                            int j = 0;
                            for (; j + 3 < outw; j += 4) {
                                float32x4_t _val0 = vld1q_f32(sptr);
                                float32x4_t _val1 = vld1q_f32(sptr + stride_w * 4);
                                float32x4_t _val2 = vld1q_f32(sptr + stride_w * 8);
                                float32x4_t _val3 = vld1q_f32(sptr + stride_w * 12);
                                vst1q_f32(ptr, _val0);
                                vst1q_f32(ptr + 4, _val1);
                                vst1q_f32(ptr + 8, _val2);
                                vst1q_f32(ptr + 12, _val3);

                                sptr += stride_w * 16;
                                ptr += 16;
                            }
                            for (; j + 1 < outw; j += 2) {
                                float32x4_t _val0 = vld1q_f32(sptr);
                                float32x4_t _val1 = vld1q_f32(sptr + stride_w * 4);
                                vst1q_f32(ptr, _val0);
                                vst1q_f32(ptr + 4, _val1);

                                sptr += stride_w * 8;
                                ptr += 8;
                            }
                            for (; j < outw; j++) {
                                float32x4_t _val = vld1q_f32(sptr);
                                vst1q_f32(ptr, _val);

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
    
    im2col_sgemm_conv2d_pack4to1_impl_neon(im2col, output, kernel_tf, bias);
    
    return output;
}
    
Tensor sgemm_conv2d_pack4to1_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    Tensor output = otter::empty({}, otter::ScalarType::Float);
    sgemm_conv2d_pack4to1_neon_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
    
    return output;
}

Tensor& sgemm_conv2d_pack1to4_neon_out(
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
    
    int w = self.size(3);
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
        convolution_im2col_sgemm_transform_kernel_pack1to4_neon(weight, kernel_tf, inch, outch * 4, kernel_w, kernel_h);
    
    Tensor im2col = otter::im2col_cpu(self, kernel_size, stride, padding, {1, 1}).view({inch, maxk, size});
    
    im2col_sgemm_conv2d_pack1to4_impl_neon(im2col, output, kernel_tf, bias);
    
    return output;
    
    return output;
}
    
Tensor sgemm_conv2d_pack1to4_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
    
    Tensor output = otter::empty({}, otter::ScalarType::Float4);
    sgemm_conv2d_pack1to4_neon_out(self, weight, weight_o, bias, kernel_size, stride, padding, dilation, output);
    
    return output;
}
    
Tensor conv2d_1x1s1_sgemm_pack4_neon_out(
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
        convolution_im2col_sgemm_transform_kernel_pack4_neon(weight, kernel_tf, inch * 4, outch * 4, 1, 1);
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    int h = input.size(1);
    const int size = w * h;
    
    Tensor im2col = input.view({-1, 1, size});
    
    im2col_sgemm_conv2d_pack4_impl_neon(im2col, output, kernel_tf, bias);
    
    return output;
}

Tensor conv2d_1x1s1_sgemm_pack4_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
               
    auto output = otter::empty({}, otter::ScalarType::Float4);
    
    return conv2d_1x1s1_sgemm_pack4_neon_out(self, weight, weight_o, bias, padding, output);
}

Tensor conv2d_1x1s1_sgemm_pack4to1_neon_out(
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
        convolution_im2col_sgemm_transform_kernel_pack4_neon(weight, kernel_tf, inch * 4, outch, 1, 1);
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    int h = input.size(1);
    const int size = w * h;
    
    Tensor im2col = input.view({-1, 1, size});
    
    im2col_sgemm_conv2d_pack4to1_impl_neon(im2col, output, kernel_tf, bias);
    
    return output;
}

Tensor conv2d_1x1s1_sgemm_pack4to1_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, otter::ScalarType::Float);
    
    return conv2d_1x1s1_sgemm_pack4to1_neon_out(self, weight, weight_o, bias, padding, output);
}

Tensor conv2d_1x1s1_sgemm_pack1to4_neon_out(
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
        convolution_im2col_sgemm_transform_kernel_pack4_neon(weight, kernel_tf, inch, outch * 4, 1, 1);
    
    auto input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0)[0];
    
    int w = input.size(2);
    int h = input.size(1);
    const int size = w * h;
    
    Tensor im2col = input.view({-1, 1, size});
    
    im2col_sgemm_conv2d_pack1to4_impl_neon(im2col, output, kernel_tf, bias);
    
    return output;
}

Tensor conv2d_1x1s1_sgemm_pack1to4_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& weight_o,
    const Tensor& bias,
    IntArrayRef padding) {
    
    auto output = otter::empty({}, otter::ScalarType::Float4);
    
    return conv2d_1x1s1_sgemm_pack1to4_neon_out(self, weight, weight_o, bias, padding, output);
}

#endif  // __ARM_NEON

}   // end namespace otter
