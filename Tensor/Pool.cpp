//
//  Pool.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/1.
//

#include "TensorFunction.hpp"
#include "Pool.hpp"
#include "TensorFactory.hpp"
#include "Padding.hpp"
#include "Dispatch.hpp"
#include "Parallel.hpp"
#include "VecIntrinsic.hpp"

namespace otter {

DEFINE_DISPATCH(max_pool2d_stub);

DEFINE_META_FUNCTION(max_pool2d_with_indices) (const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
    OTTER_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2, "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
    const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
    const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);
    
    OTTER_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2, "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
    const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
    const int dW = stride.empty() ? kW : stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);
    
    OTTER_CHECK(padding.size() == 1 || padding.size() == 2, "max_pool2d: padding must be either be a single int, or a tuple of two ints");
    const int padH = safe_downcast<int, int64_t>(padding[0]);
    const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);
    
    OTTER_CHECK(dilation.size() == 1 || dilation.size() == 2, "max_pool2d: dilation must be either a single int, or a tuple of two ints");
    const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
    const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);
    
    const auto memory_format = input.suggest_memory_format();
    
    if (memory_format == MemoryFormat::ChannelsLast) {
        OTTER_CHECK(input.dim() == 4, "non-empty 4D (batch mode) tensor expected for input with channels_last layout");
    } else if (memory_format == MemoryFormat::Contiguous) {
        OTTER_CHECK((input.dim() == 3 || input.dim() == 4), "non-empty 3D or 4D (batch mode) tensor expected for input");
    } else {
        OTTER_CHECK(false, "Unsupport memory format. Supports only ChannelsLast, Contiguous");
    }
    
    const int64_t nbatch = input.dim() == 4 ? input.size(-4) : 1;
    const int64_t nInputPlane = input.size(-3);
    const int64_t inputHeight = input.size(-2);
    const int64_t inputWidth = input.size(-1);

    const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
    const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);
    
    pool2d_shape_check(
                       input,
                       kH, kW, dH, dW, padH, padW, dilationH, dilationW,
                       nInputPlane,
                       inputHeight, inputWidth,
                       outputHeight, outputWidth, memory_format);
    
    if (input.dim() == 3) {
        set_output(0, {nInputPlane, outputHeight, outputWidth}, {}, input.options().memory_format(memory_format));
        /* indices will contain the locations for each output point */
        set_output(1, {nInputPlane, outputHeight, outputWidth}, {}, input.options().memory_format(memory_format).dtype(ScalarType::Long));
    } else {
        set_output(0, {nbatch, nInputPlane, outputHeight, outputWidth}, {}, input.options().memory_format(memory_format));
        /* indices will contain the locations for each output point */
        set_output(1, {nbatch, nInputPlane, outputHeight, outputWidth}, {}, input.options().memory_format(memory_format).dtype(ScalarType::Long));
    }
}

DEFINE_IMPL_FUNCTION(max_pool2d_with_indices_out_cpu) (const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool /*ceil_mode*/, const Tensor& output, const Tensor& indices) {
    
    const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
    const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

    const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
    const int dW = stride.empty() ? kW : stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

    const int padH = safe_downcast<int, int64_t>(padding[0]);
    const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

    const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
    const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);
    
    max_pool2d_stub(Device::CPU, output, indices, input, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
}

#if defined(__ARM_NEON__)
Tensor max_pool2d_3x3s2_neon(const Tensor& self, IntArrayRef padding) {
    Tensor input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0);
    
    int64_t w = input.size(3);
    int64_t h = input.size(2);

    int64_t outw = (w - 3) / 2 + 1;
    int64_t outh = (h - 3) / 2 + 1;
    
    int64_t inch = input.size(1);
    
    Tensor output = otter::empty({1, inch, outh, outw}, otter::ScalarType::Float);
    
    const int64_t tailstep = w - 2 * outw + w;
    
    auto input_a = input.accessor<float, 4>()[0];
    auto output_a = output.accessor<float, 4>()[0];

    otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
        for (const auto q : otter::irange(begin, end)) {
            const float* img0 = input_a[q].data();
            float* outptr = output_a[q].data();

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;

            for (int i = 0; i < outh; i++)
            {
    #if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw - (nn << 2);
    #else
                int remain = outw;
    #endif // __ARM_NEON

    #if __ARM_NEON
    #if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "prfm       pldl1keep, [%1, #256]       \n"
                        "ld2        {v0.4s, v1.4s}, [%1], #32   \n"
                        "prfm       pldl1keep, [%2, #256]       \n"
                        "ld2        {v2.4s, v3.4s}, [%2], #32   \n"
                        "prfm       pldl1keep, [%3, #256]       \n"
                        "ld2        {v4.4s, v5.4s}, [%3], #32   \n"
                        "0:                                     \n"

                        "prfm       pldl1keep, [%1, #256]       \n"
                        "ld2        {v6.4s, v7.4s}, [%1], #32   \n"

                        "fmax       v12.4s, v0.4s, v1.4s        \n"
                        "fmax       v13.4s, v2.4s, v3.4s        \n"

                        "prfm       pldl1keep, [%2, #256]       \n"
                        "ld2        {v8.4s, v9.4s}, [%2], #32   \n"

                        "fmax       v14.4s, v4.4s, v5.4s        \n"
                        "ext        v0.16b, v0.16b, v6.16b, #4  \n"

                        "prfm       pldl1keep, [%3, #256]       \n"
                        "ld2        {v10.4s, v11.4s}, [%3], #32 \n"

                        "ext        v2.16b,  v2.16b, v8.16b, #4 \n"

                        "fmax       v12.4s, v12.4s, v0.4s       \n"
                        "ext        v4.16b, v4.16b, v10.16b, #4 \n"

                        "fmax       v13.4s, v13.4s, v2.4s       \n"
                        "fmax       v14.4s, v14.4s, v4.4s       \n"
                        "fmax       v12.4s, v12.4s, v13.4s      \n"

                        "orr        v0.16b, v6.16b, v6.16b      \n"
                        "orr        v1.16b, v7.16b, v7.16b      \n"
                        "fmax       v12.4s, v12.4s, v14.4s      \n"

                        "orr        v2.16b, v8.16b, v8.16b      \n"
                        "orr        v3.16b, v9.16b, v9.16b      \n"
                        "orr        v4.16b, v10.16b, v10.16b    \n"
                        "orr        v5.16b, v11.16b, v11.16b    \n"

                        "subs       %w0, %w0, #1                \n"
                        "st1        {v12.4s}, [%4], #16         \n"
                        "bne        0b                          \n"
                        "sub        %1, %1, #32                 \n"
                        "sub        %2, %2, #32                 \n"
                        "sub        %3, %3, #32                 \n"
                        : "=r"(nn),    // %0
                        "=r"(r0),    // %1
                        "=r"(r1),    // %2
                        "=r"(r2),    // %3
                        "=r"(outptr) // %4
                        : "0"(nn),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(outptr)
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14");
                }
    #else
                if (nn > 0)
                {
                    asm volatile(
                        "pld        [%1, #256]          \n"
                        "vld2.f32   {d0-d3}, [%1]!      \n" // q0 = 0 2 4 6  q1 = 1 3 5 7
                        "pld        [%2, #256]          \n"
                        "vld2.f32   {d4-d7}, [%2]!      \n"
                        "pld        [%3, #256]          \n"
                        "vld2.f32   {d8-d11}, [%3]!     \n"
                        "0:                             \n"
                        "pld        [%1, #256]          \n"
                        "vld2.f32   {d12-d15}, [%1]!    \n" // q6 = 8 10 12 14  q7 = 9 11 13 15

                        "vmax.f32   q12, q0, q1         \n"
                        "vmax.f32   q13, q2, q3         \n"

                        "pld        [%2, #256]          \n"
                        "vld2.f32   {d16-d19}, [%2]!    \n"

                        "vmax.f32   q14, q4, q5         \n"
                        "vext.32    q0, q0, q6, #1      \n"

                        "pld        [%3, #256]          \n"
                        "vld2.f32   {d20-d23}, [%3]!    \n"

                        "vext.32    q2, q2, q8, #1      \n"

                        "vmax.f32   q12, q12, q0        \n"
                        "vext.32    q4, q4, q10, #1     \n"

                        "vmax.f32   q13, q13, q2        \n"
                        "vmax.f32   q14, q14, q4        \n"
                        "vmax.f32   q12, q12, q13       \n"

                        "vorr       q0, q6, q6          \n"
                        "vorr       q1, q7, q7          \n"
                        "vmax.f32   q12, q12, q14       \n"

                        "vorr       q2, q8, q8          \n"
                        "vorr       q3, q9, q9          \n"
                        "vorr       q4, q10, q10        \n"
                        "vorr       q5, q11, q11        \n"

                        "subs       %0, #1              \n"
                        "vst1.f32   {d24-d25}, [%4]!    \n"
                        "bne        0b                  \n"
                        "sub        %1, #32             \n"
                        "sub        %2, #32             \n"
                        "sub        %3, #32             \n"
                        : "=r"(nn),    // %0
                        "=r"(r0),    // %1
                        "=r"(r1),    // %2
                        "=r"(r2),    // %3
                        "=r"(outptr) // %4
                        : "0"(nn),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(outptr)
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14");
                }
    #endif // __aarch64__
    #endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    float max0 = std::max(std::max(r0[0], r0[1]), r0[2]);
                    float max1 = std::max(std::max(r1[0], r1[1]), r1[2]);
                    float max2 = std::max(std::max(r2[0], r2[1]), r2[2]);

                    *outptr = std::max(std::max(max0, max1), max2);

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr++;
                }

                r0 += tailstep; //1 + w;
                r1 += tailstep; //1 + w;
                r2 += tailstep; //1 + w;
            }
        }
    });
    
    return output;
}

Tensor max_pool2d_3x3s2_pack4_neon(const Tensor& self, IntArrayRef padding) {
    Tensor input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0);
    
    int64_t w = input.size(3);
    int64_t h = input.size(2);

    int64_t outw = (w - 3) / 2 + 1;
    int64_t outh = (h - 3) / 2 + 1;
    
    int64_t inch = input.size(1);
    
    Tensor output = otter::empty({1, inch, outh, outw}, otter::ScalarType::Float4);
    
    const int64_t tailstep = (w - 2 * outw + w) * 4;
    
    auto input_a = input.accessor<float, 4, 4>()[0];
    auto output_a = output.accessor<float, 4, 4>()[0];
    
    otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
        for (const auto q : otter::irange(begin, end)) {
            const auto img0 = input_a[q];
            float* outptr = output_a[q].data();

            const float* r0 = img0[0].data();
            const float* r1 = img0[1].data();
            const float* r2 = img0[2].data();

            for (int i = 0; i < outh; i++)
            {
                int j = 0;

                for (; j + 3 < outw; j += 4)
                {
        #if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"

                        "fmax   v16.4s, v0.4s, v1.4s    \n"
                        "fmax   v17.4s, v2.4s, v3.4s    \n"

                        "prfm   pldl1keep, [%1, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n"

                        "fmax   v18.4s, v4.4s, v5.4s    \n"
                        "fmax   v19.4s, v6.4s, v7.4s    \n"

                        "ld1    {v8.4s}, [%1]           \n"

                        "fmax   v20.4s, v16.4s, v2.4s   \n"
                        "fmax   v21.4s, v17.4s, v4.4s   \n"

                        "prfm   pldl1keep, [%2, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%2], #64 \n"

                        "fmax   v22.4s, v18.4s, v6.4s   \n"
                        "fmax   v23.4s, v19.4s, v8.4s   \n"

                        "prfm   pldl1keep, [%2, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2], #64 \n"

                        "fmax   v16.4s, v0.4s, v1.4s    \n"
                        "fmax   v17.4s, v2.4s, v3.4s    \n"

                        "fmax   v18.4s, v4.4s, v5.4s    \n"
                        "fmax   v19.4s, v6.4s, v7.4s    \n"

                        "ld1    {v8.4s}, [%2]           \n"

                        "fmax   v24.4s, v16.4s, v2.4s   \n"
                        "fmax   v25.4s, v17.4s, v4.4s   \n"

                        "prfm   pldl1keep, [%3, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%3], #64 \n"

                        "fmax   v26.4s, v18.4s, v6.4s   \n"
                        "fmax   v27.4s, v19.4s, v8.4s   \n"

                        "prfm   pldl1keep, [%3, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%3], #64 \n"

                        "fmax   v16.4s, v0.4s, v1.4s    \n"
                        "fmax   v17.4s, v2.4s, v3.4s    \n"

                        "fmax   v18.4s, v4.4s, v5.4s    \n"
                        "fmax   v19.4s, v6.4s, v7.4s    \n"

                        "ld1    {v8.4s}, [%3]           \n"

                        "fmax   v28.4s, v16.4s, v2.4s   \n"
                        "fmax   v29.4s, v17.4s, v4.4s   \n"
                        "fmax   v30.4s, v18.4s, v6.4s   \n"
                        "fmax   v31.4s, v19.4s, v8.4s   \n"

                        "fmax   v20.4s, v20.4s, v24.4s  \n"
                        "fmax   v21.4s, v21.4s, v25.4s  \n"
                        "fmax   v22.4s, v22.4s, v26.4s  \n"
                        "fmax   v23.4s, v23.4s, v27.4s  \n"

                        "fmax   v20.4s, v20.4s, v28.4s  \n"
                        "fmax   v21.4s, v21.4s, v29.4s  \n"
                        "fmax   v22.4s, v22.4s, v30.4s  \n"
                        "fmax   v23.4s, v23.4s, v31.4s  \n"

                        "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"

                        : "=r"(outptr), // %0
                        "=r"(r0),     // %1
                        "=r"(r1),     // %2
                        "=r"(r2)      // %3
                        : "0"(outptr),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
        #else  // __aarch64__
                    asm volatile(
                        "pld        [%1, #512]      \n"
                        "vldm       %1!, {d0-d7}    \n"

                        "pld        [%2, #512]      \n"
                        "vldm       %2!, {d8-d15}   \n"

                        "vmax.f32   q0, q0, q4      \n"
                        "vmax.f32   q1, q1, q5      \n"

                        "pld        [%3, #512]      \n"
                        "vldm       %3!, {d16-d23}  \n"

                        "vmax.f32   q2, q2, q6      \n"
                        "vmax.f32   q3, q3, q7      \n"

                        "vmax.f32   q0, q0, q8      \n"
                        "vmax.f32   q1, q1, q9      \n"

                        "pld        [%1, #512]      \n"
                        "vldm       %1!, {d8-d15}   \n"

                        "vmax.f32   q2, q2, q10     \n"
                        "vmax.f32   q3, q3, q11     \n"

                        "pld        [%2, #512]      \n"
                        "vldm       %2!, {d16-d23}  \n"

                        "vmax.f32   q4, q4, q8      \n"
                        "vmax.f32   q5, q5, q9      \n"

                        "pld        [%3, #512]      \n"
                        "vldm       %3!, {d24-d31}  \n"

                        "vmax.f32   q6, q6, q10     \n"
                        "vmax.f32   q7, q7, q11     \n"

                        "vmax.f32   q4, q4, q12     \n"
                        "vmax.f32   q5, q5, q13     \n"

                        "vld1.f32   {d24-d25}, [%1 :128] \n"
                        "vld1.f32   {d26-d27}, [%2 :128] \n"

                        "vmax.f32   q6, q6, q14     \n"
                        "vmax.f32   q7, q7, q15     \n"

                        "vld1.f32   {d28-d29}, [%3 :128] \n"

                        "vmax.f32   q8, q12, q13    \n"
                        "vmax.f32   q8, q8, q14     \n"

                        "vmax.f32   q12, q0, q1     \n"
                        "vmax.f32   q13, q2, q3     \n"
                        "vmax.f32   q14, q4, q5     \n"
                        "vmax.f32   q15, q6, q7     \n"

                        "vmax.f32   q12, q12, q2    \n"
                        "vmax.f32   q13, q13, q4    \n"
                        "vmax.f32   q14, q14, q6    \n"
                        "vmax.f32   q15, q15, q8    \n"

                        "vstm       %0!, {d24-d31}  \n"

                        : "=r"(outptr), // %0
                        "=r"(r0),     // %1
                        "=r"(r1),     // %2
                        "=r"(r2)      // %3
                        : "0"(outptr),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
        #endif // __aarch64__
                }
                for (; j + 1 < outw; j += 2)
                {
        #if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"

                        "prfm   pldl1keep, [%2, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%2], #64 \n"

                        "fmax   v16.4s, v0.4s, v4.4s    \n"
                        "fmax   v17.4s, v1.4s, v5.4s    \n"

                        "prfm   pldl1keep, [%3, #512]   \n"
                        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%3], #64 \n"

                        "fmax   v18.4s, v2.4s, v6.4s    \n"
                        "fmax   v19.4s, v3.4s, v7.4s    \n"

                        "ld1    {v0.4s}, [%1]           \n"

                        "fmax   v16.4s, v16.4s, v20.4s  \n"
                        "fmax   v17.4s, v17.4s, v21.4s  \n"

                        "ld1    {v1.4s}, [%2]           \n"

                        "fmax   v18.4s, v18.4s, v22.4s  \n"
                        "fmax   v19.4s, v19.4s, v23.4s  \n"

                        "ld1    {v2.4s}, [%3]           \n"

                        "fmax   v3.4s, v0.4s, v1.4s     \n"

                        "fmax   v20.4s, v16.4s, v17.4s  \n"
                        "fmax   v21.4s, v18.4s, v19.4s  \n"

                        "fmax   v3.4s, v3.4s, v2.4s     \n"

                        "fmax   v20.4s, v20.4s, v18.4s  \n"
                        "fmax   v21.4s, v21.4s, v3.4s   \n"

                        "st1    {v20.4s, v21.4s}, [%0], #32 \n"

                        : "=r"(outptr), // %0
                        "=r"(r0),     // %1
                        "=r"(r1),     // %2
                        "=r"(r2)      // %3
                        : "0"(outptr),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
        #else  // __aarch64__
                    asm volatile(
                        "pld        [%1, #512]      \n"
                        "vldm       %1!, {d0-d7}    \n"

                        "pld        [%2, #512]      \n"
                        "vldm       %2!, {d8-d15}   \n"

                        "vmax.f32   q12, q0, q4     \n"
                        "vmax.f32   q13, q1, q5     \n"

                        "pld        [%3, #512]      \n"
                        "vldm       %3!, {d16-d23}  \n"

                        "vmax.f32   q14, q2, q6     \n"
                        "vmax.f32   q15, q3, q7     \n"

                        "vld1.f32   {d0-d1}, [%1 :128] \n"

                        "vmax.f32   q12, q12, q8    \n"
                        "vmax.f32   q13, q13, q9    \n"

                        "vld1.f32   {d2-d3}, [%2 :128] \n"

                        "vmax.f32   q14, q14, q10   \n"
                        "vmax.f32   q15, q15, q11   \n"

                        "vld1.f32   {d4-d5}, [%3 :128] \n"

                        "vmax.f32   q3, q0, q1      \n"

                        "vmax.f32   q4, q12, q13    \n"
                        "vmax.f32   q5, q14, q15    \n"

                        "vmax.f32   q3, q3, q2      \n"

                        "vmax.f32   q4, q4, q14     \n"
                        "vmax.f32   q5, q5, q3      \n"

                        "vst1.f32   {d8-d11}, [%0 :128]! \n"

                        : "=r"(outptr), // %0
                        "=r"(r0),     // %1
                        "=r"(r1),     // %2
                        "=r"(r2)      // %3
                        : "0"(outptr),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
        #endif // __aarch64__
                }
                for (; j < outw; j++)
                {
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r01 = vld1q_f32(r0 + 4);
                    float32x4_t _r02 = vld1q_f32(r0 + 8);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r11 = vld1q_f32(r1 + 4);
                    float32x4_t _r12 = vld1q_f32(r1 + 8);
                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r21 = vld1q_f32(r2 + 4);
                    float32x4_t _r22 = vld1q_f32(r2 + 8);

                    float32x4_t _max0 = vmaxq_f32(vmaxq_f32(_r00, _r01), _r02);
                    float32x4_t _max1 = vmaxq_f32(vmaxq_f32(_r10, _r11), _r12);
                    float32x4_t _max2 = vmaxq_f32(vmaxq_f32(_r20, _r21), _r22);

                    float32x4_t _max = vmaxq_f32(vmaxq_f32(_max0, _max1), _max2);

                    vst1q_f32(outptr, _max);

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    outptr += 4;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }
        }
    });
    
    return output;
}

Tensor max_pool2d_2x2s2_pack4_neon(const Tensor& self, IntArrayRef padding) {
    Tensor input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0);
    
    int64_t w = input.size(3);
    int64_t h = input.size(2);

    int64_t outw = (w - 2) / 2 + 1;
    int64_t outh = (h - 2) / 2 + 1;
    
    int64_t inch = input.size(1);
    
    Tensor output = otter::empty({1, inch, outh, outw}, otter::ScalarType::Float4);
    
    const int64_t tailstep = (w - 2 * outw + w) * 4;
    
    auto input_a = input.accessor<float, 4, 4>()[0];
    auto output_a = output.accessor<float, 4, 4>()[0];
    
    otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
        for (const auto q : otter::irange(begin, end)) {
            const auto img0 = input_a[q];
            float* outptr = output_a[q].data();

            const float* r0 = img0[0].data();
            const float* r1 = img0[1].data();

            for (int i = 0; i < outh; i++)
            {
                int j = 0;

                for (; j + 3 < outw; j += 4)
                {
        #if __aarch64__
                    asm volatile(
                        "prfm   pldl1keep, [%1, #512]   \n"
                        "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"

                        "fmax   v0.4s, v0.4s, v1.4s     \n"
                        "fmax   v2.4s, v2.4s, v3.4s     \n"

                        "prfm   pldl1keep, [%1, #512]   \n"
                        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s}, [%1], #64 \n"

                        "fmax   v4.4s, v4.4s, v5.4s     \n"
                        "fmax   v6.4s, v6.4s, v7.4s     \n"

                        "prfm   pldl1keep, [%2, #512]   \n"
                        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%2], #64 \n"

                        "fmax   v16.4s, v16.4s, v17.4s  \n"
                        "fmax   v18.4s, v18.4s, v19.4s  \n"

                        "prfm   pldl1keep, [%2, #512]   \n"
                        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%2], #64 \n"

                        "fmax   v20.4s, v20.4s, v21.4s  \n"
                        "fmax   v22.4s, v22.4s, v23.4s  \n"

                        "fmax   v0.4s, v0.4s, v16.4s    \n"
                        "fmax   v1.4s, v2.4s, v18.4s    \n"
                        "fmax   v2.4s, v4.4s, v20.4s    \n"
                        "fmax   v3.4s, v6.4s, v22.4s    \n"

                        "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"

                        : "=r"(outptr), // %0
                        "=r"(r0),     // %1
                        "=r"(r1)      // %2
                        : "0"(outptr),
                        "1"(r0),
                        "2"(r1)
                        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
        #else  // __aarch64__
                    asm volatile(
                        "pld        [%1, #512]      \n"
                        "vldm       %1!, {d0-d7}    \n"

                        "vmax.f32   q0, q0, q1      \n"
                        "vmax.f32   q2, q2, q3      \n"

                        "pld        [%1, #512]      \n"
                        "vldm       %1!, {d8-d15}   \n"

                        "vmax.f32   q4, q4, q5      \n"
                        "vmax.f32   q6, q6, q7      \n"

                        "pld        [%2, #512]      \n"
                        "vldm       %2!, {d16-d23}  \n"

                        "vmax.f32   q8, q8, q9      \n"
                        "vmax.f32   q10, q10, q11   \n"

                        "pld        [%2, #512]      \n"
                        "vldm       %2!, {d24-d31}  \n"

                        "vmax.f32   q12, q12, q13   \n"
                        "vmax.f32   q14, q14, q15   \n"

                        "vmax.f32   q0, q0, q8      \n"
                        "vmax.f32   q1, q2, q10     \n"
                        "vmax.f32   q2, q4, q12     \n"
                        "vmax.f32   q3, q6, q14     \n"

                        "vstm       %0!, {d0-d7}    \n"

                        : "=r"(outptr), // %0
                        "=r"(r0),     // %1
                        "=r"(r1)      // %2
                        : "0"(outptr),
                        "1"(r0),
                        "2"(r1)
                        : "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
        #endif // __aarch64__
                }
                for (; j < outw; j++)
                {
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r01 = vld1q_f32(r0 + 4);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r11 = vld1q_f32(r1 + 4);

                    float32x4_t _max0 = vmaxq_f32(_r00, _r01);
                    float32x4_t _max1 = vmaxq_f32(_r10, _r11);
                    float32x4_t _max = vmaxq_f32(_max0, _max1);

                    vst1q_f32(outptr, _max);

                    r0 += 8;
                    r1 += 8;
                    outptr += 4;
                }

                r0 += tailstep;
                r1 += tailstep;
            }
        }
    });
    
    return output;
}
#endif

#if __SSE2__
Tensor max_pool2d_2x2s2_pack4_x86(const Tensor& self, IntArrayRef padding) {
    Tensor input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0);
    
    int64_t w = input.size(3);
    int64_t h = input.size(2);

    int64_t outw = (w - 2) / 2 + 1;
    int64_t outh = (h - 2) / 2 + 1;
    
    int64_t inch = input.size(1);
    
    Tensor output = otter::empty({1, inch, outh, outw}, otter::ScalarType::Float4);

    const int64_t tailstep = (w - 2 * outw + w) * 4;
    
    auto input_a = input.accessor<float, 4, 4>()[0];
    auto output_a = output.accessor<float, 4, 4>()[0];

    otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
        for (const auto q : otter::irange(begin, end)) {
            const auto img0 = input_a[q];
            float* outptr = output_a[q].data();

            const float* r0 = img0[0].data();
            const float* r1 = img0[1].data();

            for (int i = 0; i < outh; i++) {
                int j = 0;

                for (; j < outw; j++) {
                    __m128 _r00 = _mm_loadu_ps(r0);
                    __m128 _r01 = _mm_loadu_ps(r0 + 4);
                    __m128 _r10 = _mm_loadu_ps(r1);
                    __m128 _r11 = _mm_loadu_ps(r1 + 4);

                    __m128 _max0 = _mm_max_ps(_r00, _r01);
                    __m128 _max1 = _mm_max_ps(_r10, _r11);
                    __m128 _max = _mm_max_ps(_max0, _max1);

                    _mm_storeu_ps(outptr, _max);

                    r0 += 8;
                    r1 += 8;
                    outptr += 4;
                }

                r0 += tailstep;
                r1 += tailstep;
            }
        }
    });
    
    return output;
}

Tensor max_pool2d_3x3s2_pack4_x86(const Tensor& self, IntArrayRef padding) {
    Tensor input = otter::constant_pad(self, {padding[1], padding[1], padding[0], padding[0]}, 0);
    
    int64_t w = input.size(3);
    int64_t h = input.size(2);

    int64_t outw = (w - 3) / 2 + 1;
    int64_t outh = (h - 3) / 2 + 1;
    
    int64_t inch = input.size(1);
    
    Tensor output = otter::empty({1, inch, outh, outw}, otter::ScalarType::Float4);
    
    const int64_t tailstep = (w - 2 * outw + w) * 4;
    
    auto input_a = input.accessor<float, 4, 4>()[0];
    auto output_a = output.accessor<float, 4, 4>()[0];

    otter::parallel_for(0, inch, 0, [&](int64_t begin, int64_t end) {
        for (const auto q : otter::irange(begin, end)) {
            const auto img0 = input_a[q];
            float* outptr = output_a[q].data();

            const float* r0 = img0[0].data();
            const float* r1 = img0[1].data();
            const float* r2 = img0[2].data();
            for (int i = 0; i < outh; i++) {
                int j = 0;
                for (; j + 1 < outw; j += 2) {
                    __m128 _r00 = _mm_loadu_ps(r0);
                    __m128 _r01 = _mm_loadu_ps(r0 + 4);
                    __m128 _r02 = _mm_loadu_ps(r0 + 8);
                    __m128 _r10 = _mm_loadu_ps(r1);
                    __m128 _r11 = _mm_loadu_ps(r1 + 4);
                    __m128 _r12 = _mm_loadu_ps(r1 + 8);
                    __m128 _r20 = _mm_loadu_ps(r2);
                    __m128 _r21 = _mm_loadu_ps(r2 + 4);
                    __m128 _r22 = _mm_loadu_ps(r2 + 8);

                    __m128 _max00 = _mm_max_ps(_r00, _r01);
                    _max00 = _mm_max_ps(_max00, _r02);
                    _max00 = _mm_max_ps(_max00, _r10);
                    _max00 = _mm_max_ps(_max00, _r11);
                    __m128 _max01 = _mm_max_ps(_r12, _r20);
                    _max01 = _mm_max_ps(_max01, _r21);
                    _max01 = _mm_max_ps(_max01, _r22);

                    __m128 _r03 = _mm_loadu_ps(r0 + 12);
                    __m128 _r04 = _mm_loadu_ps(r0 + 16);
                    __m128 _r13 = _mm_loadu_ps(r1 + 12);
                    __m128 _r14 = _mm_loadu_ps(r1 + 16);
                    __m128 _r23 = _mm_loadu_ps(r2 + 12);
                    __m128 _r24 = _mm_loadu_ps(r2 + 16);

                    _mm_storeu_ps(outptr, _mm_max_ps(_max00, _max01));

                    __m128 _max10 = _mm_max_ps(_r03, _r04);
                    _max10 = _mm_max_ps(_max10, _r02);
                    _max10 = _mm_max_ps(_max10, _r13);
                    _max10 = _mm_max_ps(_max10, _r14);
                    __m128 _max11 = _mm_max_ps(_r12, _r23);
                    _max10 = _mm_max_ps(_max10, _r24);
                    _max10 = _mm_max_ps(_max10, _r22);

                    _mm_storeu_ps(outptr + 4, _mm_max_ps(_max10, _max11));

                    r0 += 16;
                    r1 += 16;
                    r2 += 16;
                    outptr += 8;
                }

                for (; j < outw; j++) {
                    __m128 _r00 = _mm_loadu_ps(r0);
                    __m128 _r01 = _mm_loadu_ps(r0 + 4);
                    __m128 _r02 = _mm_loadu_ps(r0 + 8);
                    __m128 _r10 = _mm_loadu_ps(r1);
                    __m128 _r11 = _mm_loadu_ps(r1 + 4);
                    __m128 _r12 = _mm_loadu_ps(r1 + 8);
                    __m128 _r20 = _mm_loadu_ps(r2);
                    __m128 _r21 = _mm_loadu_ps(r2 + 4);
                    __m128 _r22 = _mm_loadu_ps(r2 + 8);

                    __m128 _max0 = _mm_max_ps(_r00, _r01);
                    _max0 = _mm_max_ps(_max0, _r02);
                    _max0 = _mm_max_ps(_max0, _r10);
                    _max0 = _mm_max_ps(_max0, _r11);
                    __m128 _max1 = _mm_max_ps(_r12, _r20);
                    _max1 = _mm_max_ps(_max1, _r21);
                    _max1 = _mm_max_ps(_max1, _r22);

                    _mm_storeu_ps(outptr, _mm_max_ps(_max0, _max1));

                    r0 += 8;
                    r1 += 8;
                    r2 += 8;
                    outptr += 4;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }
        }
    });
    
    return output;
}
#endif

Tensor max_pool2d(const Tensor& self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
    Tensor output;
    
#if __ARM_NEON__
    if (self.elempack() != 1) {
        if (!ceil_mode && dilation[0] == 1 && dilation[1] == 1 && kernel_size[0] == 3 && kernel_size[1] == 3 && stride[0] == 2 && stride[1] == 2 && self.scalar_type() == otter::ScalarType::Float4) {
            return max_pool2d_3x3s2_pack4_neon(self, padding);
        } else if (!ceil_mode && dilation[0] == 1 && dilation[1] == 1 && kernel_size[0] == 3 && kernel_size[1] == 3 && stride[0] == 2 && stride[1] == 2 && self.scalar_type() == otter::ScalarType::Float4) {
            return max_pool2d_2x2s2_pack4_neon(self, padding);
        }
        return max_pool2d(self.packing(1), kernel_size, stride, padding, dilation, ceil_mode);
    } else {
        if (!ceil_mode && dilation[0] == 1 && dilation[1] == 1 && kernel_size[0] == 3 && kernel_size[1] == 3 && stride[0] == 2 && stride[1] == 2 && self.scalar_type() == otter::ScalarType::Float) {
            return max_pool2d_3x3s2_neon(self, padding);
        }
        
        auto output_and_indices = otter::native::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
        output = std::get<0>(output_and_indices);
        
        return output;
    }
#elif __SSE2__
    if (self.elempack() != 1) {
        if (!ceil_mode && dilation[0] == 1 && dilation[1] == 1 && kernel_size[0] == 3 && kernel_size[1] == 3 && stride[0] == 2 && stride[1] == 2 && self.scalar_type() == otter::ScalarType::Float4) {
            return max_pool2d_3x3s2_pack4_x86(self, padding);
        } else if (!ceil_mode && dilation[0] == 1 && dilation[1] == 1 && kernel_size[0] == 3 && kernel_size[1] == 3 && stride[0] == 2 && stride[1] == 2 && self.scalar_type() == otter::ScalarType::Float4) {
            return max_pool2d_2x2s2_pack4_x86(self, padding);
        }
        return max_pool2d(self.packing(1), kernel_size, stride, padding, dilation, ceil_mode);
    } else {
        auto output_and_indices = otter::native::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
        output = std::get<0>(output_and_indices);
        
        return output;
    }
#else
    auto output_and_indices = otter::native::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
    output = std::get<0>(output_and_indices);
#endif
    
    return output;
}

}   // end namespace otter
