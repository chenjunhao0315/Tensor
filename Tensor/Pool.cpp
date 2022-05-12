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
    
    Tensor output = otter::empty({1, input.size(1), outh, outw}, otter::ScalarType::Float);
    
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
#endif

Tensor max_pool2d(const Tensor& self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode) {
    Tensor output;
    
#if defined(__ARM_NEON__)
    if (!ceil_mode && dilation[0] == 1 && dilation[1] == 1 && kernel_size[0] == 3 && kernel_size[1] == 3 && stride[0] == 2 && stride[1] == 2 && self.scalar_type() == otter::ScalarType::Float) {
        return max_pool2d_3x3s2_neon(self, padding);
    } else {
        auto output_and_indices = otter::native::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
        output = std::get<0>(output_and_indices);
    }
#else
    auto output_and_indices = otter::native::max_pool2d_with_indices(self, kernel_size, stride, padding, dilation, ceil_mode);
    output = std::get<0>(output_and_indices);
#endif
    
    return output;
}

}   // end namespace otter
