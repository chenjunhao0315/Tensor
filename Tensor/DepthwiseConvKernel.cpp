//
//  DepthwiseConvKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#include "Tensor.hpp"
#include "Macro.hpp"
#include "DepthwiseConvKernel.hpp"
#include "TensorFactory.hpp"
#include "Parallel.hpp"
#include "ConvolutionUtils.hpp"
#include "Padding.hpp"

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

namespace otter {

struct Arguments final {
    // Input layer dimensions
    int64_t batch;
    int64_t in_rows;
    int64_t in_cols;
    int64_t stride;
    int64_t pad_rows;
    int64_t pad_cols;

    // Output layer dimensions
    int64_t out_rows;
    int64_t out_cols;
    int64_t out_channels;
};

#ifdef __ARM_NEON__

inline void winograd_f2k3_input_transform_inplace__neon(
    float32x4_t* const d0,
    float32x4_t* const d1,
    float32x4_t* const d2,
    float32x4_t* const d3) {
    
    const float32x4_t wd0 = *d0 - *d2;
    const float32x4_t wd1 = *d1 + *d2;
    const float32x4_t wd2 = -*d1 + *d2;
    const float32x4_t wd3 = *d1 - *d3;
    *d0 = wd0;
    *d1 = wd1;
    *d2 = wd2;
    *d3 = wd3;
}

inline void winograd_f2k3_output_transform_inplace__neon(
    float32x4_t* const m0,
    float32x4_t* const m1,
    const float32x4_t* const m2,
    const float32x4_t* const m3) {
    
    *m0 = *m0 + *m1 + *m2;
    *m1 = *m1 - *m2 - *m3;
}

inline float32x4_t vmuladdq_f32(const float32x4_t c, const float32x4_t a, const float32x4_t b) {
#if defined(__aarch64__)
    return vfmaq_f32(c, a, b);
#else
    return vmlaq_f32(c, a, b);
#endif
}

inline float32x4_t vmulsubq_f32(const float32x4_t c, const float32x4_t a, const float32x4_t b) {
#if defined(__aarch64__)
    return vfmsq_f32(c, a, b);
#else
    return vmlsq_f32(c, a, b);
#endif
}

inline void winograd_f2k3_kernel_transform__neon(
    const float32x4_t g0,
    const float32x4_t g1,
    const float32x4_t g2,
    float32x4_t* const transform0,
    float32x4_t* const transform1,
    float32x4_t* const transform2,
    float32x4_t* const transform3) {
    
    const float32x4_t const_half = vdupq_n_f32(0.5f);
    float32x4_t half_g0_plus_g2 = const_half * (g0 + g2);
    *transform0 = g0;
    *transform1 = vmuladdq_f32(half_g0_plus_g2, const_half, g1);
    *transform2 = vmulsubq_f32(half_g0_plus_g2, const_half, g1);
    *transform3 = g2;
}

inline float32x4x4_t v4f_transpose4x4__neon(const float32x4x4_t m) {
    float32x4x4_t ret;
    vst4q_f32((float*)(&ret), m);
    return ret;
}

void convolution_depthwise3x3_winograd_impl(
    const Arguments& args,
    const float* const input,
    const float* const kernel,
    const float* const bias,
    float* const output) {
    
    const float32x4_t vbias = vsetq_lane_f32(*bias, vdupq_n_f32(0.0), 1);
    float32x4x4_t kernel_tile;

    {
        const float32x4_t g0 = vld1q_f32(kernel);
        const float32x4_t g1 = vld1q_f32(kernel + 3);
        // g2[3] is junk
        const float32x4_t g2 = vextq_f32(vld1q_f32(kernel + 5), vld1q_f32(kernel + 5), 1);
        float32x4x4_t w;
        winograd_f2k3_kernel_transform__neon(g0, g1, g2, &w.val[0], &w.val[1], &w.val[2], &w.val[3]);
        w = v4f_transpose4x4__neon(w);

        winograd_f2k3_kernel_transform__neon(
            w.val[0],
            w.val[1],
            w.val[2],
            &kernel_tile.val[0],
            &kernel_tile.val[1],
            &kernel_tile.val[2],
            &kernel_tile.val[3]);
    }

#define TILE                                                  \
  winograd_f2k3_input_transform_inplace__neon(                \
      &input_tile.val[0],                                     \
      &input_tile.val[1],                                     \
      &input_tile.val[2],                                     \
      &input_tile.val[3]);                                    \
  input_tile = v4f_transpose4x4__neon(input_tile);            \
  winograd_f2k3_input_transform_inplace__neon(                \
      &input_tile.val[0],                                     \
      &input_tile.val[1],                                     \
      &input_tile.val[2],                                     \
      &input_tile.val[3]);                                    \
                                                              \
  for (const auto row : otter::irange(4)) {                   \
    input_tile.val[row] =                                     \
        vmulq_f32(input_tile.val[row], kernel_tile.val[row]); \
  }                                                           \
                                                              \
  input_tile.val[1] = input_tile.val[1] + vbias;              \
  winograd_f2k3_output_transform_inplace__neon(               \
      &input_tile.val[0],                                     \
      &input_tile.val[1],                                     \
      &input_tile.val[2],                                     \
      &input_tile.val[3]);                                    \
  input_tile = v4f_transpose4x4__neon(input_tile);            \
  winograd_f2k3_output_transform_inplace__neon(               \
      &input_tile.val[0],                                     \
      &input_tile.val[1],                                     \
      &input_tile.val[2],                                     \
      &input_tile.val[3])

  // Non-padded regime.

  // Iterate over non-padded output tiles.
  // TODO: avoid spilling W by breaking out the non-padded vs padded case.
    for (int64_t oth = 0; oth < (args.out_rows + 1) / 2; ++oth) {
        for (int64_t otw = 0; otw < (args.out_cols + 1) / 2; ++otw) {
            // load input tile for [oth, otw];
            int64_t ih = oth * 2 - args.pad_rows;
            int64_t iw = otw * 2 - args.pad_cols;
            // fast-path, all accesses in-bounds
            if (OTTER_LIKELY(ih >= 0 && iw >= 0 && ih + 3 < args.in_rows &&
                  iw + 3 < args.in_cols && 2 * oth + 1 < args.out_rows &&
                  2 * otw + 1 < args.out_cols
            )) {
                float32x4x4_t input_tile;
                for (const auto row : otter::irange(4)) {
                    input_tile.val[row] = vld1q_f32(input + (ih + row) * args.in_cols + iw);
                }

        TILE;

                for (const auto row : otter::irange(2)) {
                    vst1_f32(output + (oth * 2 + row) * args.out_cols + otw * 2, vget_low_f32(input_tile.val[row]));
                }
            } else {
                float block[4][4];
                for (const auto row : otter::irange(4)) {
                    for (const auto col : otter::irange(4)) {
                        if (ih + row >= 0 && iw + col >= 0 && ih + row < args.in_rows &&
                            iw + col < args.in_cols) {
                            block[row][col] = input[(ih + row) * args.in_cols + iw + col];
                        } else {
                            block[row][col] = 0.0;
                        }
                    }
                }

                float32x4x4_t input_tile;
                for (const auto row : otter::irange(4)) {
                    input_tile.val[row] = vld1q_f32(&block[row][0]);
                }

                TILE;

                float oblock[2][2];
                for (const auto row : otter::irange(2)) {
                    vst1_f32(&oblock[row][0], vget_low_f32(input_tile.val[row]));
                }
                for (const auto row : otter::irange(2)) {
                    for (const auto col : otter::irange(2)) {
                        if (2 * oth + row < args.out_rows &&
                            2 * otw + col < args.out_cols) {
                            output[(2 * oth + row) * args.out_cols + 2 * otw + col] =
                            oblock[row][col];
                        }
                    }
                }
            }
        }
    }
}

#else

void convolution_depthwise3x3_winograd_impl(
    const Arguments&,
    const float* const,
    const float* const,
    const float* const,
    float* const) {
}

#endif /* __ARM_NEON__ */

Tensor _convolution_depthwise3x3_winograd(
    const Tensor & input,
    const Tensor & kernel,
    const Tensor & bias_potentially_undefined,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const int64_t groups)
{
    const IntArrayRef input_sizes = input.sizes();
    const IntArrayRef kernel_sizes = kernel.sizes();

    Tensor output = otter::empty(calculate_conv_output_size(input_sizes, kernel_sizes, stride, padding), input.options());

    const IntArrayRef output_sizes = output.sizes();

    const Arguments args {
        input_sizes[0],     // Input N
        input_sizes[2],     // Input H
        input_sizes[3],     // Input W
        stride[0],          // Stride
        padding[0],         // Padding Rows
        padding[1],         // Padding Columns
        output_sizes[2],    // Output H
        output_sizes[3],    // Output W
        output_sizes[1],    // Output C
    };

    const int64_t input_hxw = args.in_rows * args.in_cols;
    const int64_t output_hxw = args.out_rows * args.out_cols;

    const Tensor bias = bias_potentially_undefined.defined() ?
                        bias_potentially_undefined :
                        otter::zeros({kernel_sizes[0]}, input.options());

    otter::parallel_for(0, args.batch * args.out_channels, 0, [&](int64_t start, int64_t end) {
        for (const auto k : otter::irange(start, end)) {
            const int64_t g = k % args.out_channels;
            const int64_t i = k / (args.out_channels / groups);
            convolution_depthwise3x3_winograd_impl(
                args,
                input.data_ptr<float>() + i * input_hxw,
                kernel.data_ptr<float>() + g * 3 * 3,
                bias.data_ptr<float>() + g,
                output.data_ptr<float>() + k * output_hxw);
        }
    });

  return output;
}

REGISTER_DISPATCH(convolution_depthwise3x3_winograd_stub, &_convolution_depthwise3x3_winograd);

Tensor& depthwise_conv2d_3x3s2_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias_,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    int64_t w = self.size(3);

    int64_t outw = output.size(3);
    int64_t outh = output.size(2);

    const int64_t group = self.size(1);

    const int64_t tailstep = w - 2 * outw + w;

    const float* kernel = weight.data_ptr<float>();
    const float* bias = (bias_.defined()) ? bias_.data_ptr<float>() : nullptr;
    
    auto input_a = self.accessor<float, 4>()[0];
    auto output_a = output.accessor<float, 4>()[0];

    otter::parallel_for(0, group, 0, [&](int64_t begin, int64_t end) {
        for (const auto g : otter::irange(begin, end)) {
            auto out = output_a[g];

            const float bias0 = bias ? bias[g] : 0.f;

            const float* kernel0 = kernel + g * 9;

            float* outptr = out.data();

            const float* img0 = input_a[g].data();

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;

    #if __ARM_NEON
            float32x4_t _k012x = vld1q_f32(kernel0);
            float32x4_t _k345x = vld1q_f32(kernel0 + 3);
            float32x4_t _k678x = vld1q_f32(kernel0 + 6);

            _k012x = vsetq_lane_f32(0.f, _k012x, 3);
            _k345x = vsetq_lane_f32(0.f, _k345x, 3);
            _k678x = vsetq_lane_f32(0.f, _k678x, 3);

            float32x4_t _bias0 = vdupq_n_f32(bias0);
    #else
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;
    #endif // __ARM_NEON

            int i = 0;

            for (; i < outh; i++)
            {
    #if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
    #else
                int remain = outw;
    #endif // __ARM_NEON

    #if __ARM_NEON
    #if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld2        {v2.4s, v3.4s}, [%2], #32      \n"

                        "and        v11.16b, %13.16b, %13.16b      \n" // v11 = _bias0

                        "0:                                        \n"
                        "fmul       v0.4s,  v2.4s, %10.s[0]        \n"
                        "fmul       v10.4s, v3.4s, %10.s[1]        \n"

                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld2        {v8.4s, v9.4s}, [%2]           \n"
                        "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                        "fmla       v11.4s, v1.4s, %10.s[2]        \n"

                        "prfm       pldl1keep, [%3, #256]          \n"
                        "ld2        {v2.4s, v3.4s}, [%3], #32      \n"

                        "fmla       v0.4s,  v2.4s, %11.s[0]        \n"
                        "fmla       v10.4s, v3.4s, %11.s[1]        \n"

                        "prfm       pldl1keep, [%3, #256]          \n"
                        "ld2        {v8.4s, v9.4s}, [%3]           \n"
                        "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                        "fmla       v11.4s, v1.4s, %11.s[2]        \n"

                        "prfm       pldl1keep, [%4, #256]          \n"
                        "ld2        {v2.4s, v3.4s}, [%4], #32      \n"

                        "fmla       v0.4s,  v2.4s, %12.s[0]        \n"
                        "fmla       v10.4s, v3.4s, %12.s[1]        \n"

                        "prfm       pldl1keep, [%4, #256]          \n"
                        "ld2        {v8.4s, v9.4s}, [%4]           \n"
                        "ext        v1.16b, v2.16b, v8.16b, #4     \n"

                        "fmla       v11.4s, v1.4s, %12.s[2]        \n"

                        "prfm       pldl1keep, [%2, #256]          \n"
                        "ld2        {v2.4s, v3.4s}, [%2], #32      \n"

                        "fadd       v0.4s, v0.4s, v10.4s           \n"
                        "fadd       v0.4s, v0.4s, v11.4s           \n"

                        "and        v11.16b, %13.16b, %13.16b      \n" // v11 = _bias0

                        "subs       %w0, %w0, #1                   \n"
                        "st1        {v0.4s}, [%1], #16             \n"
                        "bne        0b                             \n"
                        "sub        %2, %2, #32                    \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2)      // %4
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(_k012x), // %10
                        "w"(_k345x), // %11
                        "w"(_k678x), // %12
                        "w"(_bias0)  // %13
                        : "cc", "memory", "v0", "v1", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }
    #else
                if (nn > 0)
                {
                    asm volatile(
                        "pld        [%2, #256]          \n"
                        "vld2.f32   {d4-d7}, [%2]!      \n"

                        "vand       q11, %q13, %q13     \n"

                        "0:                             \n"
                        "vmul.f32   q0, q2, %e10[0]     \n"
                        "vmul.f32   q10, q3, %e10[1]    \n"

                        "pld        [%2, #128]          \n"
                        "vld2.f32   {d16-d17}, [%2]     \n"
                        "vext.32    q1, q2, q8, #1      \n"

                        "vmla.f32   q11, q1, %f10[0]    \n"

                        "pld        [%3, #256]          \n"
                        "vld2.f32   {d4-d7}, [%3]!      \n"

                        "vmla.f32   q0, q2, %e11[0]     \n"
                        "vmla.f32   q10, q3, %e11[1]    \n"

                        "pld        [%3, #128]          \n"
                        "vld2.f32   {d16-d17}, [%3]     \n"
                        "vext.32    q1, q2, q8, #1      \n"

                        "vmla.f32   q11, q1, %f11[0]    \n"

                        "pld        [%4, #256]          \n"
                        "vld2.f32   {d4-d7}, [%4]!      \n"

                        "vmla.f32   q0, q2, %e12[0]     \n"
                        "vmla.f32   q10, q3, %e12[1]    \n"

                        "pld        [%4, #128]          \n"
                        "vld2.f32   {d16-d17}, [%4]     \n"
                        "vext.32    q1, q2, q8, #1      \n"

                        "vmla.f32   q11, q1, %f12[0]    \n"

                        "pld        [%2, #256]          \n"
                        "vld2.f32   {d4-d7}, [%2]!      \n"

                        "vadd.f32   q0, q0, q10         \n"
                        "vadd.f32   q0, q0, q11         \n"

                        "vand       q11, %q13, %q13     \n"

                        "subs       %0, #1              \n"
                        "vst1.f32   {d0-d1}, [%1]!      \n"
                        "bne        0b                  \n"
                        "sub        %2, #32             \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2)      // %4
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "w"(_k012x), // %10
                        "w"(_k345x), // %11
                        "w"(_k678x), // %12
                        "w"(_bias0)  // %13
                        : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
    #endif // __aarch64__
    #endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
    #if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);

                    float32x4_t _sum = vmulq_f32(_r00, _k012x);
                    _sum = vmlaq_f32(_sum, _r10, _k345x);
                    _sum = vmlaq_f32(_sum, _r20, _k678x);

                    _sum = vsetq_lane_f32(bias0, _sum, 3);
    #if __aarch64__
                    *outptr = vaddvq_f32(_sum);
    #else
                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    _ss = vpadd_f32(_ss, _ss);

                    *outptr = vget_lane_f32(_ss, 0);
    #endif // __aarch64__
    #else
                    float sum = bias0;
                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];

                    *outptr = sum;
    #endif // __ARM_NEON

                    r0 += 2;
                    r1 += 2;
                    r2 += 2;
                    outptr++;
                }

                r0 += tailstep;
                r1 += tailstep;
                r2 += tailstep;
            }
        }
    });
    
    return output;
}

Tensor depthwise_conv2d_3x3s2_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    auto output = otter::empty(output_size, self.options());
    auto input = otter::constant_pad(self, {padding[0], padding[1], padding[0], padding[1]}, 0);

    return depthwise_conv2d_3x3s2_neon_out(input, weight, bias, stride, padding, output);
}

Tensor& depthwise_conv2d_5x5s1_neon_out(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias_,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
    
    int64_t w = self.size(3);

    int64_t outw = output.size(3);
    int64_t outh = output.size(2);

    const int64_t group = self.size(1);

    const float* kernel = weight.data_ptr<float>();
    const float* bias = (bias_.defined()) ? bias_.data_ptr<float>() : nullptr;
    
    auto input_a = self.accessor<float, 4>()[0];
    auto output_a = output.accessor<float, 4>()[0];

    otter::parallel_for(0, group, 0, [&](int64_t begin, int64_t end) {
        for (const auto g : otter::irange(begin, end))
        {
            auto out = output_a[g];

            const float bias0 = bias ? bias[g] : 0.f;

            const float* kernel0 = kernel + g * 25;

            float* outptr = out.data();
            float* outptr2 = outptr + outw;

            const float* img0 = input_a[g].data();

            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;
            const float* r3 = img0 + w * 3;
            const float* r4 = img0 + w * 4;
            const float* r5 = img0 + w * 5;

            const float* k0 = kernel0;
            const float* k1 = kernel0 + 5;
            const float* k2 = kernel0 + 10;
            const float* k3 = kernel0 + 15;
            const float* k4 = kernel0 + 20;

    #if __ARM_NEON
            float32x4_t _k0123 = vld1q_f32(kernel0);
            float32x4_t _k4567 = vld1q_f32(kernel0 + 4);
            float32x4_t _k891011 = vld1q_f32(kernel0 + 8);
            float32x4_t _k12131415 = vld1q_f32(kernel0 + 12);
            float32x4_t _k16171819 = vld1q_f32(kernel0 + 16);
            float32x4_t _k20212223 = vld1q_f32(kernel0 + 20);
            float32x4_t _k24242424 = vdupq_n_f32(kernel0[24]);

            float32x4_t _bias0 = vdupq_n_f32(bias0);
    #endif // __ARM_NEON

            int i = 0;

            for (; i + 1 < outh; i += 2)
            {
    #if __ARM_NEON
    #if __aarch64__
                int nn = outw >> 3;
                int remain = outw & 7;
    #else
                int nn = outw >> 2;
                int remain = outw & 3;
    #endif // __aarch64__
    #else
                int remain = outw;
    #endif // __ARM_NEON

    #if __ARM_NEON
    #if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        // r1
                        "prfm   pldl1keep, [%4, #384]           \n"
                        "ld1    {v16.4s, v17.4s, v18.4s}, [%4]  \n" // v16 v17 v18 = r10 r14 r18

                        "mov    v8.16b, %25.16b                 \n" // v8 = _bias0
                        "mov    v9.16b, %25.16b                 \n" // v9 = _bias0

                        "0:                                     \n"

                        "mov    v10.16b, %25.16b                \n" // v10 = _bias0
                        "mov    v11.16b, %25.16b                \n" // v11 = _bias0

                        "fmla   v8.4s, v16.4s, %19.s[1]         \n"
                        "fmla   v10.4s, v16.4s, %18.s[0]        \n"

                        "ext    v19.16b, v16.16b, v17.16b, #4   \n" // r11

                        "fmla   v9.4s, v17.4s, %19.s[1]         \n"
                        "fmla   v11.4s, v17.4s, %18.s[0]        \n"

                        "ext    v20.16b, v17.16b, v18.16b, #4   \n" // r15

                        "fmla   v8.4s, v17.4s, %20.s[1]         \n"
                        "fmla   v10.4s, v17.4s, %19.s[0]        \n"

                        "ext    v21.16b, v16.16b, v17.16b, #8   \n" // r12

                        "fmla   v9.4s, v18.4s, %20.s[1]         \n"
                        "fmla   v11.4s, v18.4s, %19.s[0]        \n"

                        "ext    v22.16b, v17.16b, v18.16b, #8   \n" // r16

                        "fmla   v8.4s, v19.4s, %19.s[2]         \n"
                        "fmla   v10.4s, v19.4s, %18.s[1]        \n"

                        "ext    v19.16b, v16.16b, v17.16b, #12  \n" // r13

                        "fmla   v9.4s, v20.4s, %19.s[2]         \n"
                        "fmla   v11.4s, v20.4s, %18.s[1]        \n"

                        "ext    v20.16b, v17.16b, v18.16b, #12  \n" // r17

                        "fmla   v8.4s, v21.4s, %19.s[3]         \n"
                        "fmla   v10.4s, v21.4s, %18.s[2]        \n"

                        "add    %4, %4, #32                     \n"

                        "fmla   v9.4s, v22.4s, %19.s[3]         \n"
                        "fmla   v11.4s, v22.4s, %18.s[2]        \n"

                        // r2
                        "prfm   pldl1keep, [%5, #384]           \n"
                        "ld1    {v12.4s, v13.4s, v14.4s}, [%5]  \n" // v12 v13 v14 = r20 r24 r28

                        "fmla   v8.4s, v19.4s, %20.s[0]         \n"
                        "fmla   v10.4s, v19.4s, %18.s[3]        \n"
                        "fmla   v9.4s, v20.4s, %20.s[0]         \n"
                        "fmla   v11.4s, v20.4s, %18.s[3]        \n"

                        "add    %5, %5, #32                     \n"

                        "fmla   v8.4s, v12.4s, %20.s[2]         \n"
                        "fmla   v10.4s, v12.4s, %19.s[1]        \n"

                        "ext    v21.16b, v12.16b, v13.16b, #4   \n" // r21

                        "fmla   v9.4s, v13.4s, %20.s[2]         \n"
                        "fmla   v11.4s, v13.4s, %19.s[1]        \n"

                        "ext    v22.16b, v13.16b, v14.16b, #4   \n" // r25

                        "fmla   v8.4s, v13.4s, %21.s[2]         \n"
                        "fmla   v10.4s, v13.4s, %20.s[1]        \n"

                        "ext    v19.16b, v12.16b, v13.16b, #8   \n" // r22

                        "fmla   v9.4s, v14.4s, %21.s[2]         \n"
                        "fmla   v11.4s, v14.4s, %20.s[1]        \n"

                        "ext    v20.16b, v13.16b, v14.16b, #8   \n" // r26

                        "fmla   v8.4s, v21.4s, %20.s[3]         \n"
                        "fmla   v10.4s, v21.4s, %19.s[2]        \n"

                        "ext    v21.16b, v12.16b, v13.16b, #12  \n" // r23

                        "fmla   v9.4s, v22.4s, %20.s[3]         \n"
                        "fmla   v11.4s, v22.4s, %19.s[2]        \n"

                        "ext    v22.16b, v13.16b, v14.16b, #12  \n" // r27

                        "fmla   v8.4s, v19.4s, %21.s[0]         \n"
                        "fmla   v10.4s, v19.4s, %19.s[3]        \n"
                        "fmla   v9.4s, v20.4s, %21.s[0]         \n"
                        "fmla   v11.4s, v20.4s, %19.s[3]        \n"

                        // r3
                        "prfm   pldl1keep, [%6, #384]           \n"
                        "ld1    {v16.4s, v17.4s, v18.4s}, [%6]  \n" // v16 v17 v18 = r30 r34 r38

                        "fmla   v8.4s, v21.4s, %21.s[1]         \n"
                        "fmla   v10.4s, v21.4s, %20.s[0]        \n"
                        "fmla   v9.4s, v22.4s, %21.s[1]         \n"
                        "fmla   v11.4s, v22.4s, %20.s[0]        \n"

                        "add    %6, %6, #32                     \n"

                        "fmla   v8.4s, v16.4s, %21.s[3]         \n"
                        "fmla   v10.4s, v16.4s, %20.s[2]        \n"

                        "ext    v19.16b, v16.16b, v17.16b, #4   \n" // r31

                        "fmla   v9.4s, v17.4s, %21.s[3]         \n"
                        "fmla   v11.4s, v17.4s, %20.s[2]        \n"

                        "ext    v20.16b, v17.16b, v18.16b, #4   \n" // r35

                        "fmla   v8.4s, v17.4s, %22.s[3]         \n"
                        "fmla   v10.4s, v17.4s, %21.s[2]        \n"

                        "ext    v21.16b, v16.16b, v17.16b, #8   \n" // r32

                        "fmla   v9.4s, v18.4s, %22.s[3]         \n"
                        "fmla   v11.4s, v18.4s, %21.s[2]        \n"

                        "ext    v22.16b, v17.16b, v18.16b, #8   \n" // r36

                        "fmla   v8.4s, v19.4s, %22.s[0]         \n"
                        "fmla   v10.4s, v19.4s, %20.s[3]        \n"

                        "ext    v19.16b, v16.16b, v17.16b, #12  \n" // r33

                        "fmla   v9.4s, v20.4s, %22.s[0]         \n"
                        "fmla   v11.4s, v20.4s, %20.s[3]        \n"

                        "ext    v20.16b, v17.16b, v18.16b, #12  \n" // r37

                        "fmla   v8.4s, v21.4s, %22.s[1]         \n"
                        "fmla   v10.4s, v21.4s, %21.s[0]        \n"
                        "fmla   v9.4s, v22.4s, %22.s[1]         \n"
                        "fmla   v11.4s, v22.4s, %21.s[0]        \n"

                        // r4
                        "prfm   pldl1keep, [%7, #384]           \n"
                        "ld1    {v12.4s, v13.4s, v14.4s}, [%7]  \n" // v12 v13 v14 = r40 r44 r48

                        "fmla   v8.4s, v19.4s, %22.s[2]         \n"
                        "fmla   v10.4s, v19.4s, %21.s[1]        \n"

                        "add    %7, %7, #32                     \n"

                        "fmla   v9.4s, v20.4s, %22.s[2]         \n"
                        "fmla   v11.4s, v20.4s, %21.s[1]        \n"

                        "ext    v21.16b, v12.16b, v13.16b, #4   \n" // r41

                        "fmla   v8.4s, v12.4s, %23.s[0]         \n"
                        "fmla   v10.4s, v12.4s, %21.s[3]        \n"

                        "ext    v22.16b, v13.16b, v14.16b, #4   \n" // r45

                        "fmla   v9.4s, v13.4s, %23.s[0]         \n"
                        "fmla   v11.4s, v13.4s, %21.s[3]        \n"

                        "ext    v19.16b, v12.16b, v13.16b, #8   \n" // r42

                        "fmla   v8.4s, v13.4s, %24.s[0]         \n"
                        "fmla   v10.4s, v13.4s, %22.s[3]        \n"

                        "ext    v20.16b, v13.16b, v14.16b, #8   \n" // r46

                        "fmla   v9.4s, v14.4s, %24.s[0]         \n"
                        "fmla   v11.4s, v14.4s, %22.s[3]        \n"

                        // r0 and r5
                        "prfm   pldl1keep, [%3, #384]           \n"
                        "ld1    {v16.4s, v17.4s, v18.4s}, [%3]  \n" // v16 v17 v18 = r00 r04 r08

                        "fmla   v8.4s, v21.4s, %23.s[1]         \n"
                        "fmla   v10.4s, v21.4s, %22.s[0]        \n"

                        "ext    v21.16b, v12.16b, v13.16b, #12  \n" // r43

                        "fmla   v9.4s, v22.4s, %23.s[1]         \n"
                        "fmla   v11.4s, v22.4s, %22.s[0]        \n"

                        "ext    v22.16b, v13.16b, v14.16b, #12  \n" // r47

                        "fmla   v8.4s, v19.4s, %23.s[2]         \n"
                        "fmla   v10.4s, v19.4s, %22.s[1]        \n"

                        "prfm   pldl1keep, [%8, #384]           \n"
                        "ld1    {v12.4s, v13.4s, v14.4s}, [%8]  \n" // v12 v13 v14 = r50 r54 r58

                        "fmla   v9.4s, v20.4s, %23.s[2]         \n"
                        "fmla   v11.4s, v20.4s, %22.s[1]        \n"

                        "ext    v19.16b, v16.16b, v17.16b, #4   \n" // r01

                        "fmla   v8.4s, v21.4s, %23.s[3]         \n"
                        "fmla   v10.4s, v21.4s, %22.s[2]        \n"

                        "ext    v23.16b, v12.16b, v13.16b, #4   \n" // r51

                        "fmla   v9.4s, v22.4s, %23.s[3]         \n"
                        "fmla   v11.4s, v22.4s, %22.s[2]        \n"

                        "ext    v20.16b, v17.16b, v18.16b, #4   \n" // r05

                        "fmla   v8.4s, v16.4s, %18.s[0]         \n"
                        "fmla   v10.4s, v12.4s, %23.s[0]        \n"

                        "ext    v24.16b, v13.16b, v14.16b, #4   \n" // r55

                        "fmla   v9.4s, v17.4s, %18.s[0]         \n"
                        "fmla   v11.4s, v13.4s, %23.s[0]        \n"

                        "ext    v21.16b, v16.16b, v17.16b, #8   \n" // r02

                        "fmla   v8.4s, v17.4s, %19.s[0]         \n"
                        "fmla   v10.4s, v13.4s, %24.s[0]        \n"

                        "ext    v25.16b, v12.16b, v13.16b, #8   \n" // r52

                        "fmla   v9.4s, v18.4s, %19.s[0]         \n"
                        "fmla   v11.4s, v14.4s, %24.s[0]        \n"

                        "ext    v22.16b, v17.16b, v18.16b, #8   \n" // r06

                        "fmla   v8.4s, v19.4s, %18.s[1]         \n"
                        "fmla   v10.4s, v23.4s, %23.s[1]        \n"

                        "ext    v26.16b, v13.16b, v14.16b, #8   \n" // r56

                        "fmla   v9.4s, v20.4s, %18.s[1]         \n"
                        "fmla   v11.4s, v24.4s, %23.s[1]        \n"

                        "ext    v19.16b, v16.16b, v17.16b, #12  \n" // r03

                        "fmla   v8.4s, v21.4s, %18.s[2]         \n"
                        "fmla   v10.4s, v25.4s, %23.s[2]        \n"

                        "ext    v23.16b, v12.16b, v13.16b, #12  \n" // r53

                        "fmla   v9.4s, v22.4s, %18.s[2]         \n"
                        "fmla   v11.4s, v26.4s, %23.s[2]        \n"

                        "ext    v20.16b, v17.16b, v18.16b, #12  \n" // r07

                        "fmla   v8.4s, v19.4s, %18.s[3]         \n"
                        "fmla   v10.4s, v23.4s, %23.s[3]        \n"

                        "ext    v24.16b, v13.16b, v14.16b, #12  \n" // r57

                        "fmla   v9.4s, v20.4s, %18.s[3]         \n"

                        "add    %3, %3, #32                     \n"

                        "fmla   v11.4s, v24.4s, %23.s[3]        \n"

                        "add    %8, %8, #32                     \n"

                        // r1
                        "prfm   pldl1keep, [%4, #384]           \n"
                        "ld1    {v16.4s, v17.4s, v18.4s}, [%4]  \n" // v16 v17 v18 = r10 r14 r18

                        "subs   %w0, %w0, #1                    \n"

                        "st1    {v8.4s, v9.4s}, [%1], #32       \n"

                        "mov    v8.16b, %25.16b                 \n" // v8 = _bias0
                        "mov    v9.16b, %25.16b                 \n" // v9 = _bias0

                        "st1    {v10.4s, v11.4s}, [%2], #32     \n"

                        "bne    0b                              \n"
                        : "=r"(nn),      // %0
                        "=r"(outptr),  // %1
                        "=r"(outptr2), // %2
                        "=r"(r0),      // %3
                        "=r"(r1),      // %4
                        "=r"(r2),      // %5
                        "=r"(r3),      // %6
                        "=r"(r4),      // %7
                        "=r"(r5)       // %8
                        : "0"(nn),
                        "1"(outptr),
                        "2"(outptr2),
                        "3"(r0),
                        "4"(r1),
                        "5"(r2),
                        "6"(r3),
                        "7"(r4),
                        "8"(r5),
                        "w"(_k0123),     // %18
                        "w"(_k4567),     // %19
                        "w"(_k891011),   // %20
                        "w"(_k12131415), // %21
                        "w"(_k16171819), // %22
                        "w"(_k20212223), // %23
                        "w"(_k24242424), // %24
                        "w"(_bias0)      // %25
                        : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26");
                }

                if (remain >= 4)
                {
                    remain -= 4;
                    asm volatile(
                        // r1
                        "prfm   pldl1keep, [%3, #256]           \n"
                        "ld1    {v12.4s, v13.4s}, [%3]          \n" // v12 v13 = r10 r14

                        "mov    v8.16b, %23.16b                 \n" // v8 = _bias0
                        "mov    v9.16b, %23.16b                 \n" // v9 = _bias0

                        "fmul   v10.4s, v12.4s, %17.s[1]        \n"
                        "fmul   v11.4s, v12.4s, %16.s[0]        \n"

                        "ext    v21.16b, v12.16b, v13.16b, #4   \n" // r11

                        "fmla   v8.4s, v13.4s, %18.s[1]         \n"
                        "fmla   v9.4s, v13.4s, %17.s[0]         \n"

                        "ext    v22.16b, v12.16b, v13.16b, #8   \n" // r12

                        "fmla   v10.4s, v21.4s, %17.s[2]        \n"
                        "fmla   v11.4s, v21.4s, %16.s[1]        \n"

                        "ext    v23.16b, v12.16b, v13.16b, #12  \n" // r13

                        "fmla   v8.4s, v22.4s, %17.s[3]         \n"
                        "fmla   v9.4s, v22.4s, %16.s[2]         \n"

                        // r2
                        "prfm   pldl1keep, [%4, #256]           \n"
                        "ld1    {v16.4s, v17.4s}, [%4]          \n" // v16 v17 = r20 r24

                        "fmla   v10.4s, v23.4s, %18.s[0]        \n"
                        "fmla   v11.4s, v23.4s, %16.s[3]        \n"

                        "add    %4, %4, #16                     \n"

                        "fmla   v8.4s, v16.4s, %18.s[2]         \n"
                        "fmla   v9.4s, v16.4s, %17.s[1]         \n"

                        "ext    v18.16b, v16.16b, v17.16b, #4   \n" // r21

                        "fmla   v10.4s, v17.4s, %19.s[2]        \n"
                        "fmla   v11.4s, v17.4s, %18.s[1]        \n"

                        "ext    v19.16b, v16.16b, v17.16b, #8   \n" // r22

                        "fmla   v8.4s, v18.4s, %18.s[3]         \n"
                        "fmla   v9.4s, v18.4s, %17.s[2]         \n"

                        "ext    v20.16b, v16.16b, v17.16b, #12  \n" // r23

                        "fmla   v10.4s, v19.4s, %19.s[0]        \n"
                        "fmla   v11.4s, v19.4s, %17.s[3]        \n"

                        // r3
                        "prfm   pldl1keep, [%5, #256]           \n"
                        "ld1    {v12.4s, v13.4s}, [%5]          \n" // v12 v13 = r30 r34

                        "fmla   v8.4s, v20.4s, %19.s[1]         \n"
                        "fmla   v9.4s, v20.4s, %18.s[0]         \n"

                        "add    %5, %5, #16                     \n"

                        "fmla   v10.4s, v12.4s, %19.s[3]        \n"
                        "fmla   v11.4s, v12.4s, %18.s[2]        \n"

                        "ext    v21.16b, v12.16b, v13.16b, #4   \n" // r31

                        "fmla   v8.4s, v13.4s, %20.s[3]         \n"
                        "fmla   v9.4s, v13.4s, %19.s[2]         \n"

                        "ext    v22.16b, v12.16b, v13.16b, #8   \n" // r32

                        "fmla   v10.4s, v21.4s, %20.s[0]        \n"
                        "fmla   v11.4s, v21.4s, %18.s[3]        \n"

                        "ext    v23.16b, v12.16b, v13.16b, #12  \n" // r33

                        "fmla   v8.4s, v22.4s, %20.s[1]         \n"
                        "fmla   v9.4s, v22.4s, %19.s[0]         \n"

                        // r4
                        "prfm   pldl1keep, [%6, #256]           \n"
                        "ld1    {v16.4s, v17.4s}, [%6]          \n" // v16 v17 = r40 r44

                        "fmla   v10.4s, v23.4s, %20.s[2]        \n"
                        "fmla   v11.4s, v23.4s, %19.s[1]        \n"

                        "add    %6, %6, #16                     \n"

                        "fmla   v8.4s, v16.4s, %21.s[0]         \n"
                        "fmla   v9.4s, v16.4s, %19.s[3]         \n"

                        "ext    v18.16b, v16.16b, v17.16b, #4   \n" // r41

                        "fmla   v10.4s, v17.4s, %22.s[0]        \n"
                        "fmla   v11.4s, v17.4s, %20.s[3]        \n"

                        "ext    v19.16b, v16.16b, v17.16b, #8   \n" // r42

                        "fmla   v8.4s, v18.4s, %21.s[1]         \n"
                        "fmla   v9.4s, v18.4s, %20.s[0]         \n"

                        "ext    v20.16b, v16.16b, v17.16b, #12  \n" // r43

                        "fmla   v10.4s, v19.4s, %21.s[2]        \n"
                        "fmla   v11.4s, v19.4s, %20.s[1]        \n"

                        // r0
                        "prfm   pldl1keep, [%2, #256]           \n"
                        "ld1    {v16.4s, v17.4s}, [%2]          \n" // v16 v17 = r00 r04

                        "fmla   v8.4s, v20.4s, %21.s[3]         \n"
                        "fmla   v9.4s, v20.4s, %20.s[2]         \n"

                        // r5
                        "prfm   pldl1keep, [%7, #256]           \n"
                        "ld1    {v12.4s, v13.4s}, [%7]          \n" // v12 v13 = r50 r54

                        "fmla   v10.4s, v16.4s, %16.s[0]        \n"
                        "fmla   v11.4s, v12.4s, %21.s[0]        \n"

                        "ext    v18.16b, v16.16b, v17.16b, #4   \n" // r01

                        "fmla   v8.4s, v17.4s, %17.s[0]         \n"

                        "ext    v21.16b, v12.16b, v13.16b, #4   \n" // r51

                        "fmla   v9.4s, v13.4s, %22.s[0]         \n"

                        "ext    v19.16b, v16.16b, v17.16b, #8   \n" // r02

                        "fmla   v10.4s, v18.4s, %16.s[1]        \n"

                        "ext    v22.16b, v12.16b, v13.16b, #8   \n" // r52

                        "fmla   v11.4s, v21.4s, %21.s[1]        \n"

                        "ext    v20.16b, v16.16b, v17.16b, #12  \n" // r03

                        "fmla   v8.4s, v19.4s, %16.s[2]         \n"

                        "ext    v23.16b, v12.16b, v13.16b, #12  \n" // r53

                        "fmla   v9.4s, v22.4s, %21.s[2]         \n"

                        "add    %3, %3, #16                     \n"

                        "fmla   v10.4s, v20.4s, %16.s[3]        \n"
                        "fmla   v11.4s, v23.4s, %21.s[3]        \n"

                        "add    %2, %2, #16                     \n"

                        "fadd   v8.4s, v8.4s, v10.4s            \n"
                        "fadd   v9.4s, v9.4s, v11.4s            \n"

                        "add    %7, %7, #16                     \n"

                        "st1    {v8.4s}, [%0], #16              \n"
                        "st1    {v9.4s}, [%1], #16              \n"

                        : "=r"(outptr),  // %0
                        "=r"(outptr2), // %1
                        "=r"(r0),      // %2
                        "=r"(r1),      // %3
                        "=r"(r2),      // %4
                        "=r"(r3),      // %5
                        "=r"(r4),      // %6
                        "=r"(r5)       // %7
                        : "0"(outptr),
                        "1"(outptr2),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "6"(r4),
                        "7"(r5),
                        "w"(_k0123),     // %16
                        "w"(_k4567),     // %17
                        "w"(_k891011),   // %18
                        "w"(_k12131415), // %19
                        "w"(_k16171819), // %20
                        "w"(_k20212223), // %21
                        "w"(_k24242424), // %22
                        "w"(_bias0)      // %23
                        : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
                }
    #else
                if (nn > 0)
                {
                    asm volatile(
                        // r1
                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d28-d31}, [%4]     \n" // q14 q15 = r10 r14

                        "vmov       q8, %q25            \n" // q8 = _bias0

                        "0:                             \n"

                        "vmov       q9, %q25            \n" // q9 = _bias0

                        "vmla.f32   q8, q14, %e19[1]    \n"
                        "vmla.f32   q9, q14, %e18[0]    \n"

                        "vext.32    q12, q14, q15, #1   \n" // r11

                        "vmla.f32   q8, q15, %e20[1]    \n"
                        "vmla.f32   q9, q15, %e19[0]    \n"

                        "vext.32    q13, q14, q15, #2   \n" // r12

                        "vmla.f32   q8, q12, %f19[0]    \n"
                        "vmla.f32   q9, q12, %e18[1]    \n"

                        "vext.32    q12, q14, q15, #3   \n" // r13

                        "vmla.f32   q8, q13, %f19[1]    \n"
                        "vmla.f32   q9, q13, %f18[0]    \n"

                        // r2
                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d20-d23}, [%5]     \n" // q10 q11 = r20 r24

                        "vmla.f32   q8, q12, %e20[0]    \n"
                        "vmla.f32   q9, q12, %f18[1]    \n"

                        "add        %5, #16             \n"

                        "vmla.f32   q8, q10, %f20[0]    \n"
                        "vmla.f32   q9, q10, %e19[1]    \n"

                        "vext.32    q12, q10, q11, #1   \n" // r21

                        "vmla.f32   q8, q11, %f21[0]    \n"
                        "vmla.f32   q9, q11, %e20[1]    \n"

                        "vext.32    q13, q10, q11, #2   \n" // r22

                        "vmla.f32   q8, q12, %f20[1]    \n"
                        "vmla.f32   q9, q12, %f19[0]    \n"

                        "vext.32    q12, q10, q11, #3   \n" // r23

                        "vmla.f32   q8, q13, %e21[0]    \n"
                        "vmla.f32   q9, q13, %f19[1]    \n"

                        // r3
                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d28-d31}, [%6]     \n" // q14 q15 = r30 r34

                        "vmla.f32   q8, q12, %e21[1]    \n"
                        "vmla.f32   q9, q12, %e20[0]    \n"

                        "add        %6, #16             \n"

                        "vmla.f32   q8, q14, %f21[1]    \n"
                        "vmla.f32   q9, q14, %f20[0]    \n"

                        "vext.32    q12, q14, q15, #1   \n" // r31

                        "vmla.f32   q8, q15, %f22[1]    \n"
                        "vmla.f32   q9, q15, %f21[0]    \n"

                        "vext.32    q13, q14, q15, #2   \n" // r32

                        "vmla.f32   q8, q12, %e22[0]    \n"
                        "vmla.f32   q9, q12, %f20[1]    \n"

                        "vext.32    q12, q14, q15, #3   \n" // r33

                        "vmla.f32   q8, q13, %e22[1]    \n"
                        "vmla.f32   q9, q13, %e21[0]    \n"

                        // r4
                        "pld        [%7, #256]          \n"
                        "vld1.f32   {d20-d23}, [%7]     \n" // q10 q11 = r40 r44

                        "vmla.f32   q8, q12, %f22[0]    \n"
                        "vmla.f32   q9, q12, %e21[1]    \n"

                        "add        %7, #16             \n"

                        "vmla.f32   q8, q10, %e23[0]    \n"
                        "vmla.f32   q9, q10, %f21[1]    \n"

                        "vext.32    q12, q10, q11, #1   \n" // r41

                        "vmla.f32   q8, q11, %e24[0]    \n"
                        "vmla.f32   q9, q11, %f22[1]    \n"

                        "vext.32    q13, q10, q11, #2   \n" // r42

                        "vmla.f32   q8, q12, %e23[1]    \n"
                        "vmla.f32   q9, q12, %e22[0]    \n"

                        "vext.32    q12, q10, q11, #3   \n" // r43

                        "vmla.f32   q8, q13, %f23[0]    \n"
                        "vmla.f32   q9, q13, %e22[1]    \n"

                        // r0 and r5
                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d20-d23}, [%3]     \n" // q10 q11 = r00 r04

                        "vmla.f32   q8, q12, %f23[1]    \n"
                        "vmla.f32   q9, q12, %f22[0]    \n"

                        // r5
                        "pld        [%8, #256]          \n"
                        "vld1.f32   {d28-d31}, [%8]     \n" // q14 q15 = r50 r54

                        "vmla.f32   q8, q10, %e18[0]    \n"
                        "vmla.f32   q9, q14, %e23[0]    \n"

                        "vext.32    q12, q10, q11, #1   \n" // r01

                        "vmla.f32   q8, q11, %e19[0]    \n"
                        "vmla.f32   q9, q15, %e24[0]    \n"

                        "vext.32    q13, q14, q15, #1   \n" // r51

                        "vmla.f32   q8, q12, %e18[1]    \n"

                        "vext.32    q12, q10, q11, #2   \n" // r02

                        "vmla.f32   q9, q13, %e23[1]    \n"

                        "vext.32    q13, q14, q15, #2   \n" // r52

                        "vmla.f32   q8, q12, %f18[0]    \n"

                        "vext.32    q12, q10, q11, #3   \n" // r03

                        "vmla.f32   q9, q13, %f23[0]    \n"

                        "vext.32    q13, q14, q15, #3   \n" // r33

                        "vmla.f32   q8, q12, %f18[1]    \n"

                        "add        %3, #16             \n"

                        "vmla.f32   q9, q13, %f23[1]    \n"

                        "add        %4, #16             \n"

                        // r1
                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d28-d31}, [%4]     \n" // q14 q15 = r10 r14

                        "add        %8, #16             \n"

                        "vst1.f32   {d16-d17}, [%1]!    \n"

                        "vmov       q8, %q25            \n" // q8 = _bias0

                        "subs       %0, #1              \n"

                        "vst1.f32   {d18-d19}, [%2]!    \n"

                        "bne        0b                  \n"
                        : "=r"(nn),      // %0
                        "=r"(outptr),  // %1
                        "=r"(outptr2), // %2
                        "=r"(r0),      // %3
                        "=r"(r1),      // %4
                        "=r"(r2),      // %5
                        "=r"(r3),      // %6
                        "=r"(r4),      // %7
                        "=r"(r5)       // %8
                        : "0"(nn),
                        "1"(outptr),
                        "2"(outptr2),
                        "3"(r0),
                        "4"(r1),
                        "5"(r2),
                        "6"(r3),
                        "7"(r4),
                        "8"(r5),
                        "w"(_k0123),     // %18
                        "w"(_k4567),     // %19
                        "w"(_k891011),   // %20
                        "w"(_k12131415), // %21
                        "w"(_k16171819), // %22
                        "w"(_k20212223), // %23
                        "w"(_k24242424), // %24
                        "w"(_bias0)      // %25
                        : "cc", "memory", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
    #endif // __aarch64__
    #endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
                    float sum = bias0;
                    float sum2 = bias0;
    #if __ARM_NEON
                    // TODO neon assembly optimize
                    float32x4_t _r1 = vld1q_f32(r1);
                    float32x4_t _k1 = vld1q_f32(k1);
                    float32x4_t _sum = vmulq_f32(_r1, _k1);
                    float32x4_t _sum2 = vmulq_f32(_r1, _k0123);

                    float32x4_t _r2 = vld1q_f32(r2);
                    float32x4_t _k2 = vld1q_f32(k2);
                    _sum = vmlaq_f32(_sum, _r2, _k2);
                    _sum2 = vmlaq_f32(_sum2, _r2, _k1);

                    float32x4_t _r3 = vld1q_f32(r3);
                    float32x4_t _k3 = vld1q_f32(k3);
                    _sum = vmlaq_f32(_sum, _r3, _k3);
                    _sum2 = vmlaq_f32(_sum2, _r3, _k2);

                    float32x4_t _r4 = vld1q_f32(r4);
                    _sum = vmlaq_f32(_sum, _r4, _k20212223);
                    _sum2 = vmlaq_f32(_sum2, _r4, _k3);

                    float32x4_t _r0 = vld1q_f32(r0);
                    _sum = vmlaq_f32(_sum, _r0, _k0123);
                    float32x4_t _r5 = vld1q_f32(r5);
                    _sum2 = vmlaq_f32(_sum2, _r5, _k20212223);

                    float32x4_t _k_t4 = {};

                    _k_t4 = vsetq_lane_f32(k0[4], _k_t4, 0);
                    _k_t4 = vsetq_lane_f32(k1[4], _k_t4, 1);
                    _k_t4 = vsetq_lane_f32(k2[4], _k_t4, 2);
                    _k_t4 = vsetq_lane_f32(k3[4], _k_t4, 3);

                    float32x4_t _r_t4 = {};

                    _r_t4 = vsetq_lane_f32(r0[4], _r_t4, 0);
                    _r_t4 = vsetq_lane_f32(r1[4], _r_t4, 1);
                    _r_t4 = vsetq_lane_f32(r2[4], _r_t4, 2);
                    _r_t4 = vsetq_lane_f32(r3[4], _r_t4, 3);
                    _sum = vmlaq_f32(_sum, _r_t4, _k_t4);

                    sum += r4[4] * k4[4];

                    _r_t4 = vextq_f32(_r_t4, _r_t4, 1);
                    _r_t4 = vsetq_lane_f32(r4[4], _r_t4, 3);
                    _sum2 = vmlaq_f32(_sum2, _r_t4, _k_t4);

                    sum2 += r5[4] * k4[4];

                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    float32x2_t _ss2 = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
                    float32x2_t _ss_ss2 = vpadd_f32(_ss, _ss2);

                    sum += vget_lane_f32(_ss_ss2, 0);
                    sum2 += vget_lane_f32(_ss_ss2, 1);
    #else
                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r0[3] * k0[3];
                    sum += r0[4] * k0[4];

                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r1[3] * k1[3];
                    sum += r1[4] * k1[4];

                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];
                    sum += r2[3] * k2[3];
                    sum += r2[4] * k2[4];

                    sum += r3[0] * k3[0];
                    sum += r3[1] * k3[1];
                    sum += r3[2] * k3[2];
                    sum += r3[3] * k3[3];
                    sum += r3[4] * k3[4];

                    sum += r4[0] * k4[0];
                    sum += r4[1] * k4[1];
                    sum += r4[2] * k4[2];
                    sum += r4[3] * k4[3];
                    sum += r4[4] * k4[4];

                    sum2 += r1[0] * k0[0];
                    sum2 += r1[1] * k0[1];
                    sum2 += r1[2] * k0[2];
                    sum2 += r1[3] * k0[3];
                    sum2 += r1[4] * k0[4];

                    sum2 += r2[0] * k1[0];
                    sum2 += r2[1] * k1[1];
                    sum2 += r2[2] * k1[2];
                    sum2 += r2[3] * k1[3];
                    sum2 += r2[4] * k1[4];

                    sum2 += r3[0] * k2[0];
                    sum2 += r3[1] * k2[1];
                    sum2 += r3[2] * k2[2];
                    sum2 += r3[3] * k2[3];
                    sum2 += r3[4] * k2[4];

                    sum2 += r4[0] * k3[0];
                    sum2 += r4[1] * k3[1];
                    sum2 += r4[2] * k3[2];
                    sum2 += r4[3] * k3[3];
                    sum2 += r4[4] * k3[4];

                    sum2 += r5[0] * k4[0];
                    sum2 += r5[1] * k4[1];
                    sum2 += r5[2] * k4[2];
                    sum2 += r5[3] * k4[3];
                    sum2 += r5[4] * k4[4];
    #endif // __ARM_NEON
                    *outptr = sum;
                    *outptr2 = sum2;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    r4++;
                    r5++;
                    outptr++;
                    outptr2++;
                }

                r0 += 4 + w;
                r1 += 4 + w;
                r2 += 4 + w;
                r3 += 4 + w;
                r4 += 4 + w;
                r5 += 4 + w;

                outptr += outw;
                outptr2 += outw;
            }

            for (; i < outh; i++)
            {
    #if __ARM_NEON
    #if __aarch64__
                int nn = outw >> 3;
                int remain = outw & 7;
    #else
                int nn = outw >> 2;
                int remain = outw & 3;
    #endif // __aarch64__
    #else
                int remain = outw;
    #endif // __ARM_NEON

    #if __ARM_NEON
    #if __aarch64__
                if (nn > 0)
                {
                    asm volatile(
                        // v10 v11
                        // r0
                        "prfm   pldl1keep, [%2, #384]           \n"
                        "ld1    {v16.4s, v17.4s, v18.4s}, [%2]  \n" // v16 v17 v18 = r00 r04 r08

                        "mov    v8.16b, %21.16b                 \n" // v8 = _bias0
                        "mov    v9.16b, %21.16b                 \n" // v9 = _bias0

                        "0:                                     \n"

                        "fmul   v10.4s, v16.4s, %14.s[0]         \n"

                        "ext    v19.16b, v16.16b, v17.16b, #4   \n" // r01

                        "fmul   v11.4s, v17.4s, %14.s[0]         \n"

                        "ext    v20.16b, v17.16b, v18.16b, #4   \n" // r05

                        "fmla   v8.4s, v17.4s, %15.s[0]         \n"

                        "ext    v21.16b, v16.16b, v17.16b, #8   \n" // r02

                        "fmla   v9.4s, v18.4s, %15.s[0]         \n"

                        "ext    v22.16b, v17.16b, v18.16b, #8   \n" // r06

                        "fmla   v10.4s, v19.4s, %14.s[1]         \n"

                        "ext    v19.16b, v16.16b, v17.16b, #12  \n" // r03

                        "fmla   v11.4s, v20.4s, %14.s[1]         \n"

                        "ext    v20.16b, v17.16b, v18.16b, #12  \n" // r07

                        "fmla   v8.4s, v21.4s, %14.s[2]         \n"
                        "fmla   v9.4s, v22.4s, %14.s[2]         \n"

                        // r1
                        "prfm   pldl1keep, [%3, #384]           \n"
                        "ld1    {v12.4s, v13.4s, v14.4s}, [%3]  \n" // v12 v13 v14 = r10 r14 r18

                        "fmla   v10.4s, v19.4s, %14.s[3]         \n"
                        "fmla   v11.4s, v20.4s, %14.s[3]         \n"

                        "fmla   v8.4s, v12.4s, %15.s[1]         \n"

                        "ext    v19.16b, v12.16b, v13.16b, #4   \n" // r11

                        "fmla   v9.4s, v13.4s, %15.s[1]         \n"

                        "ext    v20.16b, v13.16b, v14.16b, #4   \n" // r15

                        "fmla   v10.4s, v13.4s, %16.s[1]         \n"

                        "ext    v21.16b, v12.16b, v13.16b, #8   \n" // r12

                        "fmla   v11.4s, v14.4s, %16.s[1]         \n"

                        "ext    v22.16b, v13.16b, v14.16b, #8   \n" // r16

                        "fmla   v8.4s, v19.4s, %15.s[2]         \n"

                        "ext    v19.16b, v12.16b, v13.16b, #12  \n" // r13

                        "fmla   v9.4s, v20.4s, %15.s[2]         \n"

                        "ext    v20.16b, v13.16b, v14.16b, #12  \n" // r17

                        "fmla   v10.4s, v21.4s, %15.s[3]         \n"
                        "fmla   v11.4s, v22.4s, %15.s[3]         \n"

                        // r2
                        "prfm   pldl1keep, [%4, #384]           \n"
                        "ld1    {v16.4s, v17.4s, v18.4s}, [%4]  \n" // v16 v17 v18 = r20 r24 r28

                        "fmla   v8.4s, v19.4s, %16.s[0]         \n"
                        "fmla   v9.4s, v20.4s, %16.s[0]         \n"

                        "fmla   v10.4s, v16.4s, %16.s[2]         \n"

                        "ext    v19.16b, v16.16b, v17.16b, #4   \n" // r21

                        "fmla   v11.4s, v17.4s, %16.s[2]         \n"

                        "ext    v20.16b, v17.16b, v18.16b, #4   \n" // r25

                        "fmla   v8.4s, v17.4s, %17.s[2]         \n"

                        "ext    v21.16b, v16.16b, v17.16b, #8   \n" // r22

                        "fmla   v9.4s, v18.4s, %17.s[2]         \n"

                        "ext    v22.16b, v17.16b, v18.16b, #8   \n" // r26

                        "fmla   v10.4s, v19.4s, %16.s[3]         \n"

                        "ext    v19.16b, v16.16b, v17.16b, #12  \n" // r23

                        "fmla   v11.4s, v20.4s, %16.s[3]         \n"

                        "ext    v20.16b, v17.16b, v18.16b, #12  \n" // r27

                        "fmla   v8.4s, v21.4s, %17.s[0]         \n"
                        "fmla   v9.4s, v22.4s, %17.s[0]         \n"

                        // r3
                        "prfm   pldl1keep, [%5, #384]           \n"
                        "ld1    {v12.4s, v13.4s, v14.4s}, [%5]  \n" // v12 v13 v14 = r30 r34 r38

                        "fmla   v10.4s, v19.4s, %17.s[1]         \n"
                        "fmla   v11.4s, v20.4s, %17.s[1]         \n"

                        "fmla   v8.4s, v12.4s, %17.s[3]         \n"

                        "ext    v19.16b, v12.16b, v13.16b, #4   \n" // r11

                        "fmla   v9.4s, v13.4s, %17.s[3]         \n"

                        "ext    v20.16b, v13.16b, v14.16b, #4   \n" // r15

                        "fmla   v10.4s, v13.4s, %18.s[3]         \n"

                        "ext    v21.16b, v12.16b, v13.16b, #8   \n" // r12

                        "fmla   v11.4s, v14.4s, %18.s[3]         \n"

                        "ext    v22.16b, v13.16b, v14.16b, #8   \n" // r16

                        "fmla   v8.4s, v19.4s, %18.s[0]         \n"

                        "ext    v19.16b, v12.16b, v13.16b, #12  \n" // r13

                        "fmla   v9.4s, v20.4s, %18.s[0]         \n"

                        "ext    v20.16b, v13.16b, v14.16b, #12  \n" // r17

                        "fmla   v10.4s, v21.4s, %18.s[1]         \n"
                        "fmla   v11.4s, v22.4s, %18.s[1]         \n"

                        // r4
                        "prfm   pldl1keep, [%6, #384]           \n"
                        "ld1    {v16.4s, v17.4s, v18.4s}, [%6]  \n" // v16 v17 v18 = r40 r44 r48

                        "fmla   v8.4s, v19.4s, %18.s[2]         \n"
                        "fmla   v9.4s, v20.4s, %18.s[2]         \n"

                        "fmla   v10.4s, v16.4s, %19.s[0]         \n"

                        "ext    v19.16b, v16.16b, v17.16b, #4   \n" // r41

                        "fmla   v11.4s, v17.4s, %19.s[0]         \n"

                        "ext    v20.16b, v17.16b, v18.16b, #4   \n" // r45

                        "fmla   v8.4s, v17.4s, %20.s[0]         \n"

                        "ext    v21.16b, v16.16b, v17.16b, #8   \n" // r42

                        "fmla   v9.4s, v18.4s, %20.s[0]         \n"

                        "ext    v22.16b, v17.16b, v18.16b, #8   \n" // r46

                        "fmla   v10.4s, v19.4s, %19.s[1]         \n"

                        "ext    v19.16b, v16.16b, v17.16b, #12  \n" // r43

                        "fmla   v11.4s, v20.4s, %19.s[1]         \n"

                        "ext    v20.16b, v17.16b, v18.16b, #12  \n" // r47

                        "fmla   v8.4s, v21.4s, %19.s[2]         \n"

                        "add    %2, %2, #32                     \n"

                        "fmla   v9.4s, v22.4s, %19.s[2]         \n"

                        "add    %3, %3, #32                     \n"

                        "fmla   v10.4s, v19.4s, %19.s[3]         \n"

                        "add    %4, %4, #32                     \n"

                        "fmla   v11.4s, v20.4s, %19.s[3]         \n"

                        // r0
                        "prfm   pldl1keep, [%2, #384]           \n"
                        "ld1    {v16.4s, v17.4s, v18.4s}, [%2]  \n" // v16 v17 v18 = r00 r04 r08

                        "add    %5, %5, #32                     \n"

                        "fadd   v10.4s, v8.4s, v10.4s           \n"

                        "add    %6, %6, #32                     \n"

                        "fadd   v11.4s, v9.4s, v11.4s           \n"

                        "mov    v8.16b, %21.16b                 \n" // v8 = _bias0
                        "mov    v9.16b, %21.16b                 \n" // v9 = _bias0

                        "subs   %w0, %w0, #1                    \n"

                        "st1    {v10.4s, v11.4s}, [%1], #32     \n"

                        "bne    0b                              \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2),     // %4
                        "=r"(r3),     // %5
                        "=r"(r4)      // %6
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "6"(r4),
                        "w"(_k0123),     // %14
                        "w"(_k4567),     // %15
                        "w"(_k891011),   // %16
                        "w"(_k12131415), // %17
                        "w"(_k16171819), // %18
                        "w"(_k20212223), // %19
                        "w"(_k24242424), // %20
                        "w"(_bias0)      // %21
                        : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v16", "v17", "v18", "v19", "v20", "v21", "v22");
                }

                if (remain >= 4)
                {
                    remain -= 4;
                    asm volatile(
                        // r0
                        "prfm   pldl1keep, [%1, #256]           \n"
                        "ld1    {v16.4s, v17.4s}, [%1]          \n" // v16 v17 = r00 r04

                        "mov    v8.16b, %19.16b                 \n" // v8 = _bias0

                        "add    %1, %1, #16                     \n"

                        "fmul   v9.4s, v16.4s, %12.s[0]         \n"

                        "ext    v18.16b, v16.16b, v17.16b, #4   \n" // r01

                        "fmla   v8.4s, v17.4s, %13.s[0]         \n"

                        "ext    v19.16b, v16.16b, v17.16b, #8   \n" // r02

                        "fmla   v9.4s, v18.4s, %12.s[1]         \n"

                        "ext    v20.16b, v16.16b, v17.16b, #12  \n" // r03

                        "fmla   v8.4s, v19.4s, %12.s[2]         \n"

                        // r1
                        "prfm   pldl1keep, [%2, #256]           \n"
                        "ld1    {v10.4s, v11.4s}, [%2]          \n" // v10 v11 = r10 r14

                        "fmla   v9.4s, v20.4s, %12.s[3]         \n"

                        "add    %2, %2, #16                     \n"

                        "fmla   v8.4s, v10.4s, %13.s[1]         \n"

                        "ext    v12.16b, v10.16b, v11.16b, #4   \n" // r11

                        "fmla   v9.4s, v11.4s, %14.s[1]         \n"

                        "ext    v13.16b, v10.16b, v11.16b, #8   \n" // r12

                        "fmla   v8.4s, v12.4s, %13.s[2]         \n"

                        "ext    v14.16b, v10.16b, v11.16b, #12  \n" // r13

                        "fmla   v9.4s, v13.4s, %13.s[3]         \n"

                        // r2
                        "prfm   pldl1keep, [%3, #256]           \n"
                        "ld1    {v16.4s, v17.4s}, [%3]          \n" // v16 v17 = r20 r24

                        "fmla   v8.4s, v14.4s, %14.s[0]         \n"

                        "add    %3, %3, #16                     \n"

                        "fmla   v9.4s, v16.4s, %14.s[2]         \n"

                        "ext    v18.16b, v16.16b, v17.16b, #4   \n" // r21

                        "fmla   v8.4s, v17.4s, %15.s[2]         \n"

                        "ext    v19.16b, v16.16b, v17.16b, #8   \n" // r22

                        "fmla   v9.4s, v18.4s, %14.s[3]         \n"

                        "ext    v20.16b, v16.16b, v17.16b, #12  \n" // r23

                        "fmla   v8.4s, v19.4s, %15.s[0]         \n"

                        // r3
                        "prfm   pldl1keep, [%4, #256]           \n"
                        "ld1    {v10.4s, v11.4s}, [%4]          \n" // v10 v11 = r30 r34

                        "fmla   v9.4s, v20.4s, %15.s[1]         \n"

                        "add    %4, %4, #16                     \n"

                        "fmla   v8.4s, v10.4s, %15.s[3]         \n"

                        "ext    v12.16b, v10.16b, v11.16b, #4   \n" // r31

                        "fmla   v9.4s, v11.4s, %16.s[3]         \n"

                        "ext    v13.16b, v10.16b, v11.16b, #8   \n" // r32

                        "fmla   v8.4s, v12.4s, %16.s[0]         \n"

                        "ext    v14.16b, v10.16b, v11.16b, #12  \n" // r33

                        "fmla   v9.4s, v13.4s, %16.s[1]         \n"

                        // r4
                        "prfm   pldl1keep, [%5, #256]           \n"
                        "ld1    {v16.4s, v17.4s}, [%5]          \n" // v16 v17 = r40 r44

                        "fmla   v8.4s, v14.4s, %16.s[2]         \n"

                        "add    %5, %5, #16                     \n"

                        "fmla   v9.4s, v16.4s, %17.s[0]         \n"

                        "ext    v18.16b, v16.16b, v17.16b, #4   \n" // r41

                        "fmla   v8.4s, v17.4s, %18.s[0]         \n"

                        "ext    v19.16b, v16.16b, v17.16b, #8   \n" // r42

                        "fmla   v9.4s, v18.4s, %17.s[1]         \n"

                        "ext    v20.16b, v16.16b, v17.16b, #12  \n" // r43

                        "fmla   v8.4s, v19.4s, %17.s[2]         \n"

                        "fmla   v9.4s, v20.4s, %17.s[3]         \n"

                        "fadd   v8.4s, v8.4s, v9.4s             \n"

                        "st1    {v8.4s}, [%0], #16              \n"

                        : "=r"(outptr), // %0
                        "=r"(r0),     // %1
                        "=r"(r1),     // %2
                        "=r"(r2),     // %3
                        "=r"(r3),     // %4
                        "=r"(r4)      // %5
                        : "0"(outptr),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "w"(_k0123),     // %12
                        "w"(_k4567),     // %13
                        "w"(_k891011),   // %14
                        "w"(_k12131415), // %15
                        "w"(_k16171819), // %16
                        "w"(_k20212223), // %17
                        "w"(_k24242424), // %18
                        "w"(_bias0)      // %19
                        : "cc", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v16", "v17", "v18", "v19", "v20");
                }
    #else
                if (nn > 0)
                {
                    asm volatile(
                        // r0
                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d20-d23}, [%2]     \n" // q10 q11 = r00 r04

                        "vmov       q8, %q21            \n" // q8 = _bias0

                        "0:                             \n"

                        "vmul.f32   q9, q10, %e14[0]    \n"

                        "vext.32    q12, q10, q11, #1   \n" // r01

                        "vmla.f32   q8, q11, %e15[0]    \n"

                        "vext.32    q13, q10, q11, #2   \n" // r02

                        "vmla.f32   q9, q12, %e14[1]    \n"

                        "vext.32    q12, q10, q11, #3   \n" // r03

                        "vmla.f32   q8, q13, %f14[0]    \n"

                        // r1
                        "pld        [%3, #256]          \n"
                        "vld1.f32   {d28-d31}, [%3]     \n" // q14 q15 = r10 r14

                        "vmla.f32   q9, q12, %f14[1]    \n"

                        "add        %3, #16             \n"

                        "vmla.f32   q8, q14, %e15[1]    \n"

                        "vext.32    q12, q14, q15, #1   \n" // r11

                        "vmla.f32   q9, q15, %e16[1]    \n"

                        "vext.32    q13, q14, q15, #2   \n" // r12

                        "vmla.f32   q8, q12, %f15[0]    \n"

                        "vext.32    q12, q14, q15, #3   \n" // r13

                        "vmla.f32   q9, q13, %f15[1]    \n"

                        // r2
                        "pld        [%4, #256]          \n"
                        "vld1.f32   {d20-d23}, [%4]     \n" // q10 q11 = r20 r24

                        "vmla.f32   q8, q12, %e16[0]    \n"

                        "add        %4, #16             \n"

                        "vmla.f32   q9, q10, %f16[0]    \n"

                        "vext.32    q12, q10, q11, #1   \n" // r21

                        "vmla.f32   q8, q11, %f17[0]    \n"

                        "vext.32    q13, q10, q11, #2   \n" // r22

                        "vmla.f32   q9, q12, %f16[1]    \n"

                        "vext.32    q12, q10, q11, #3   \n" // r23

                        "vmla.f32   q8, q13, %e17[0]    \n"

                        // r3
                        "pld        [%5, #256]          \n"
                        "vld1.f32   {d28-d31}, [%5]     \n" // q14 q15 = r30 r34

                        "vmla.f32   q9, q12, %e17[1]    \n"

                        "add        %5, #16             \n"

                        "vmla.f32   q8, q14, %f17[1]    \n"

                        "vext.32    q12, q14, q15, #1   \n" // r31

                        "vmla.f32   q9, q15, %f18[1]    \n"

                        "vext.32    q13, q14, q15, #2   \n" // r32

                        "vmla.f32   q8, q12, %e18[0]    \n"

                        "vext.32    q12, q14, q15, #3   \n" // r33

                        "vmla.f32   q9, q13, %e18[1]    \n"

                        // r4
                        "pld        [%6, #256]          \n"
                        "vld1.f32   {d20-d23}, [%6]     \n" // q10 q11 = r40 r44

                        "vmla.f32   q8, q12, %f18[0]    \n"

                        "add        %6, #16             \n"

                        "vmla.f32   q9, q10, %e19[0]    \n"

                        "vext.32    q12, q10, q11, #1   \n" // r41

                        "vmla.f32   q8, q11, %e20[0]    \n"

                        "vext.32    q13, q10, q11, #2   \n" // r42

                        "vmla.f32   q9, q12, %e19[1]    \n"

                        "vext.32    q12, q10, q11, #3   \n" // r43

                        "vmla.f32   q8, q13, %f19[0]    \n"

                        "add        %2, #16             \n"

                        "vmla.f32   q9, q12, %f19[1]    \n"

                        // r0
                        "pld        [%2, #256]          \n"
                        "vld1.f32   {d20-d23}, [%2]     \n" // q10 q11 = r00 r04

                        "vadd.f32   q9, q9, q8          \n"

                        "vmov       q8, %q21            \n" // q8 = _bias0

                        "subs       %0, #1              \n"

                        "vst1.f32   {d18-d19}, [%1]!    \n"

                        "bne        0b                  \n"
                        : "=r"(nn),     // %0
                        "=r"(outptr), // %1
                        "=r"(r0),     // %2
                        "=r"(r1),     // %3
                        "=r"(r2),     // %4
                        "=r"(r3),     // %5
                        "=r"(r4)      // %6
                        : "0"(nn),
                        "1"(outptr),
                        "2"(r0),
                        "3"(r1),
                        "4"(r2),
                        "5"(r3),
                        "6"(r4),
                        "w"(_k0123),     // %14
                        "w"(_k4567),     // %15
                        "w"(_k891011),   // %16
                        "w"(_k12131415), // %17
                        "w"(_k16171819), // %18
                        "w"(_k20212223), // %19
                        "w"(_k24242424), // %20
                        "w"(_bias0)      // %21
                        : "cc", "memory", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
                }
    #endif // __aarch64__
    #endif // __ARM_NEON
                for (; remain > 0; remain--)
                {
    #if __ARM_NEON
    #if __aarch64__
                    // TODO neon assembly optimize
                    float sum = bias0;

                    float32x4_t _r0 = vld1q_f32(r0);
                    float32x4_t _sum = vmulq_f32(_r0, _k0123);

                    float32x4_t _r1 = vld1q_f32(r1);
                    _sum = vmlaq_f32(_sum, _r1, vld1q_f32(k1));

                    float32x4_t _r2 = vld1q_f32(r2);
                    _sum = vmlaq_f32(_sum, _r2, vld1q_f32(k2));

                    float32x4_t _r3 = vld1q_f32(r3);
                    _sum = vmlaq_f32(_sum, _r3, vld1q_f32(k3));

                    float32x4_t _r4 = vld1q_f32(r4);
                    _sum = vmlaq_f32(_sum, _r4, _k20212223);

                    float32x4_t _k_t4 = {};

                    _k_t4 = vsetq_lane_f32(k0[4], _k_t4, 0);
                    _k_t4 = vsetq_lane_f32(k1[4], _k_t4, 1);
                    _k_t4 = vsetq_lane_f32(k2[4], _k_t4, 2);
                    _k_t4 = vsetq_lane_f32(k3[4], _k_t4, 3);

                    float32x4_t _r_t4 = {};

                    _r_t4 = vsetq_lane_f32(r0[4], _r_t4, 0);
                    _r_t4 = vsetq_lane_f32(r1[4], _r_t4, 1);
                    _r_t4 = vsetq_lane_f32(r2[4], _r_t4, 2);
                    _r_t4 = vsetq_lane_f32(r3[4], _r_t4, 3);
                    _sum = vmlaq_f32(_sum, _r_t4, _k_t4);

                    sum += r4[4] * k4[4];

                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    _ss = vpadd_f32(_ss, _ss);

                    sum += vget_lane_f32(_ss, 0);

                    *outptr = sum;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    r4++;
                    outptr++;
    #else
                    // TODO neon assembly optimize
                    asm volatile(
                        "veor       q14, q14            \n"
                        "vext.32    q14, %q19, q14, #3  \n" // q14 = bias0 0 0 0

                        "vld1.f32   {d16-d17}, [%1]     \n" // q8 = r00 r01 r02 r03

                        "vld1.f32   {d18-d19}, [%2]     \n" // q9 = r10 r11 r12 r13(X)
                        "add        r4, %1, #16         \n"
                        "vld1.f32   {d19[1]}, [r4]      \n"
                        "vext.32    q9, q9, q9, #3      \n" // q9 = r04 r10 r11 r12

                        "vmla.f32   q14, q8, %q12       \n"

                        "add        r4, %2, #12         \n"
                        "vld1.f32   {d20}, [r4]         \n" // d20 = r13 r14
                        "vld1.f32   {d21}, [%3]         \n" // d21 = r20 r21

                        "vmla.f32   q14, q9, %q13       \n"

                        "add        r4, %3, #8          \n"
                        "vld1.f32   {d22-d23}, [r4]     \n" // q11 = r22 r23 r24 X
                        "vld1.f32   {d23[1]}, [%4]      \n" // q11 = r22 r23 r24 r30

                        "vmla.f32   q14, q10, %q14      \n"

                        "add        r4, %4, #4          \n"
                        "vld1.f32   {d24-d25}, [r4]     \n" // q12 = r31 r32 r33 r34

                        "vmla.f32   q14, q11, %q15      \n"

                        "vld1.f32   {d26-d27}, [%5]     \n" // q13 = r40 r41 r42 r43

                        "vmla.f32   q14, q12, %q16      \n"

                        "veor       d30, d30            \n"
                        "add        r4, %5, #16         \n"
                        "vld1.f32   {d30[0]}, [r4]      \n" // d30 = r44 0

                        "vmla.f32   q14, q13, %q17      \n"

                        "vmla.f32   d28, d30, %e18      \n"

                        "add        %1, #4              \n"

                        // h-sum
                        "vadd.f32   d28, d28, d29       \n"

                        "add        %2, #4              \n"
                        "add        %3, #4              \n"

                        "vpadd.f32  d28, d28, d28       \n"

                        "add        %4, #4              \n"
                        "add        %5, #4              \n"

                        "vst1.f32   {d28[0]}, [%0]!     \n"

                        : "=r"(outptr), // %0
                        "=r"(r0),     // %1
                        "=r"(r1),     // %2
                        "=r"(r2),     // %3
                        "=r"(r3),     // %4
                        "=r"(r4)      // %5
                        : "0"(outptr),
                        "1"(r0),
                        "2"(r1),
                        "3"(r2),
                        "4"(r3),
                        "5"(r4),
                        "w"(_k0123),     // %12
                        "w"(_k4567),     // %13
                        "w"(_k891011),   // %14
                        "w"(_k12131415), // %15
                        "w"(_k16171819), // %16
                        "w"(_k20212223), // %17
                        "w"(_k24242424), // %18
                        "w"(_bias0)      // %19
                        : "cc", "memory", "r4", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
    #endif // __aarch64__
    #else
                    float sum = bias0;

                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r0[3] * k0[3];
                    sum += r0[4] * k0[4];

                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r1[3] * k1[3];
                    sum += r1[4] * k1[4];

                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];
                    sum += r2[3] * k2[3];
                    sum += r2[4] * k2[4];

                    sum += r3[0] * k3[0];
                    sum += r3[1] * k3[1];
                    sum += r3[2] * k3[2];
                    sum += r3[3] * k3[3];
                    sum += r3[4] * k3[4];

                    sum += r4[0] * k4[0];
                    sum += r4[1] * k4[1];
                    sum += r4[2] * k4[2];
                    sum += r4[3] * k4[3];
                    sum += r4[4] * k4[4];

                    *outptr = sum;

                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    r4++;
                    outptr++;
    #endif
                }

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                r4 += 4;
            }
        }
    });
    
    return output;
}

Tensor depthwise_conv2d_5x5s1_neon(
    const Tensor& self,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding) {
    
    auto output_size = otter::calculate_conv_output_size(self.sizes(), weight.sizes(), stride, padding);
    auto output = otter::empty(output_size, self.options());
    auto input = otter::constant_pad(self, {padding[0], padding[1], padding[0], padding[1]}, 0);

    return depthwise_conv2d_5x5s1_neon_out(input, weight, bias, stride, padding, output);
}

}   // end namespace otter
