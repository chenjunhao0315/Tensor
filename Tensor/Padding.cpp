//
//  Padding.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/2.
//

#include "Padding.hpp"
#include "Tensor.hpp"
#include "TensorFactory.hpp"
#include "VecIntrinsic.hpp"
#include "Parallel.hpp"

namespace otter {

Tensor constant_pad_int8_packed(const Tensor& self, IntArrayRef pad, const Scalar& value);
void constant_pad_int8_packed_x86(const Tensor& self, Tensor& dst, IntArrayRef pad, const Scalar& value);
void constant_pad_int8_packed_neon(const Tensor& self, Tensor& dst, IntArrayRef pad, const Scalar& value);

Tensor constant_pad_float_packed(const Tensor& self, IntArrayRef pad, const Scalar& value);
void constant_pad_float_packed_x86(const Tensor& self, Tensor& dst, IntArrayRef pad, const Scalar& value);
void constant_pad_float_packed_neon(const Tensor& self, Tensor& dst, IntArrayRef pad, const Scalar& value);

Tensor constant_pad(const Tensor& self, IntArrayRef pad, const Scalar& value) {
    OTTER_CHECK(pad.size() % 2 == 0, "Length of pad must be even but instead it equals ", pad.size());
    
    if (self.elempack() != 1) {
        if (self.dim() == 4)
            OTTER_CHECK(self.size(0) == 1, "Only accept batchsize = 1");
        if (self.scalar_type() == ScalarType::Float4 || self.scalar_type() == ScalarType::Float8) {
            return constant_pad_float_packed(self, pad, value);
        } else if (self.scalar_type() == ScalarType::Byte4 || self.scalar_type() == ScalarType::Byte8) {
            return constant_pad_int8_packed(self, pad, value);
        } else {
            OTTER_CHECK(false, "Unsupport padding format!");
        }
    }
    
    auto input_sizes = self.sizes();
    auto l_inp = self.dim();

    auto l_pad = pad.size() / 2;
    auto l_diff = l_inp - l_pad;
    OTTER_CHECK(l_inp >= (int64_t)l_pad, "Length of pad should be no more than twice the number of "
                "dimensions of the input. Pad length is ", pad.size(), "while the input has ",
                l_inp, "dimensions.");

    std::vector<int64_t> new_shape;
    
    bool all_pads_non_positive = true;

    auto c_input = self;
    for (const auto i : otter::irange(l_diff, l_inp)) {
        auto pad_idx = 2 * (l_inp - i - 1);
        if (pad[pad_idx] < 0) {
            c_input = c_input.narrow(i, -pad[pad_idx], c_input.size(i) + pad[pad_idx]);
        } else if (pad[pad_idx] != 0) {
            all_pads_non_positive = false;
        }
        if (pad[pad_idx + 1] < 0) {
            c_input = c_input.narrow(i, 0, c_input.size(i) + pad[pad_idx + 1]);
        } else if (pad[pad_idx + 1] != 0) {
            all_pads_non_positive = false;
        }
    }

    // if none of the pads are positive we can optimize and just return the result
    // of calling .narrow() on the input
    if (all_pads_non_positive) {
        return c_input.clone();
    }
    
    for (size_t i = 0; i < (size_t)l_diff; i ++) {
        new_shape.emplace_back(input_sizes[i]);
    }

    for (const auto i : otter::irange((size_t)l_pad)) {
        auto pad_idx = pad.size() - ((i + 1) * 2);
        auto new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1];
        OTTER_CHECK(new_dim > 0, "The input size ", input_sizes[l_diff + i], ", plus negative padding ",
                    pad[pad_idx], " and ", pad[pad_idx + 1], " resulted in a negative output size, "
                    "which is invalid. Check dimension ", l_diff + i, " of your input.");
        new_shape.emplace_back(new_dim);
    }

    Tensor output;
    const auto memory_format = self.suggest_memory_format();
    output = otter::empty(new_shape, self.options().memory_format(memory_format));
    output.fill_(value);

    auto c_output = output;
    for (const auto i : otter::irange(l_diff, l_inp)) {
        auto pad_idx = 2 * (l_inp - i - 1);
        if (pad[pad_idx] > 0) {
            c_output = c_output.narrow(i, pad[pad_idx], c_output.size(i) - pad[pad_idx]);
        }
        if (pad[pad_idx + 1] > 0) {
            c_output = c_output.narrow(i, 0, c_output.size(i) - pad[pad_idx + 1]);
        }
    }
    c_output.copy_(c_input);
    return output;
}

Tensor constant_pad_float_packed(const Tensor& self, IntArrayRef pad, const Scalar& value) {
    int64_t elempack = self.elempack();
    
    // Now we suppose that the first axis of Tensor is batchsize
    bool batch = self.dim() == 4;
    Tensor input = (self.dim() == 4) ? self.squeeze(0) : self;
    
    Tensor out;
#if __SSE2__
    (void)elempack;
    constant_pad_float_packed_x86(input, out, pad, value);
    
    return (batch) ? out.unsqueeze(0) : out;
#elif __ARM_NEON__
    (void)elempack;
    constant_pad_float_packed_neon(input, out, pad, value);
    
    return (batch) ? out.unsqueeze(0) : out;
#else
    out = input;
    if (elempack != 1) {
        out = input.packing(1);
    }
    
    out = constant_pad(out, pad, value);

    return (batch) ? out.unsqueeze(0) : out;
#endif
}

#if __SSE2__
static void padding_constant_pack4_sse(const Tensor& src, Tensor& dst, int top, int bottom, int left, int right, __m128 v) {
    const float* ptr = (const float*)src.raw_data();
    float* outptr = (float*)dst.raw_data();
    int top_size = top * dst.size(1);
    int bottom_size = bottom * dst.size(1);

    // fill top
    for (int y = 0; y < top_size; y++) {
        _mm_store_ps(outptr, v);
        outptr += 4;
    }
    // fill center
    for (int y = 0; y < src.size(0); y++) {
        for (int x = 0; x < left; x++) {
            _mm_store_ps(outptr, v);
            outptr += 4;
        }
        for (int x = 0; x < src.size(1); x++) {
            _mm_store_ps(outptr, _mm_load_ps(ptr));
            ptr += 4;
            outptr += 4;
        }
        for (int x = 0; x < right; x++) {
            _mm_store_ps(outptr, v);
            outptr += 4;
        }
    }
    // fill top
    for (int y = 0; y < bottom_size; y++) {
        _mm_store_ps(outptr, v);
        outptr += 4;
    }
}
#if __AVX__
static void padding_constant_pack8_avx(const Tensor& src, Tensor& dst, int top, int bottom, int left, int right, __m256 v) {
    const float* ptr = (const float*)src.raw_data();
    float* outptr = (float*)dst.raw_data();
    int top_size = top * dst.size(1);
    int bottom_size = bottom * dst.size(1);

    // fill top
    for (int y = 0; y < top_size; y++) {
        _mm256_store_ps(outptr, v);
        outptr += 8;
    }
    // fill center
    for (int y = 0; y < src.size(0); y++) {
        for (int x = 0; x < left; x++) {
            _mm256_store_ps(outptr, v);
            outptr += 8;
        }
        for (int x = 0; x < src.size(1); x++) {
            _mm256_store_ps(outptr, _mm256_load_ps(ptr));
            ptr += 8;
            outptr += 8;
        }
        for (int x = 0; x < right; x++) {
            _mm256_store_ps(outptr, v);
            outptr += 8;
        }
    }
    // fill top
    for (int y = 0; y < bottom_size; y++) {
        _mm256_store_ps(outptr, v);
        outptr += 8;
    }
}
#endif // __AVX__
#endif // __SSE2__

#if __SSE2__
void constant_pad_float_packed_x86(const Tensor& self, Tensor& dst, IntArrayRef pad, const Scalar& value) {
    int64_t left   = (pad.size() > 0) ? pad[0] : 0;
    int64_t right  = (pad.size() > 1) ? pad[1] : 0;
    int64_t top    = (pad.size() > 2) ? pad[2] : 0;
    int64_t bottom = (pad.size() > 3) ? pad[3] : 0;
    int64_t front  = (pad.size() > 4) ? pad[4] : 0;
    int64_t behind = (pad.size() > 5) ? pad[5] : 0;
    
    if (left == 0 && right == 0 && top == 0 && bottom == 0 && front == 0 && behind == 0) {
        dst = self;
        
        return;
    }
    
    int64_t dims = self.dim();
    int64_t elempack = self.elempack();
    
#if __SSE2__
#if __AVX__
    if (elempack == 8) {
        if (dims == 1) {
            int w = self.size(0);
            
            int outw = w * elempack + left + right;

            int out_elempack = outw % 8 == 0 ? 8 : outw % 4 == 0 ? 4 : 1;
            ScalarType out_dtype = outw % 8 == 0 ? ScalarType::Float8 : outw % 4 == 0 ? ScalarType::Float4 : ScalarType::Float;

            if (left % 8 == 0 && out_elempack == 8) {
                dst = otter::empty({outw / out_elempack}, out_dtype);

                __m256 pad_value = _mm256_set1_ps(value.toFloat());
                padding_constant_pack8_avx(self, dst, 0, 0, left / 8, right / 8, pad_value);

                return;
            }
        }

        if (dims == 2) {
            int w = self.size(1);
            int h = self.size(0);
            
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

            int out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
            ScalarType out_dtype = outh % 8 == 0 ? ScalarType::Float8 : outh % 4 == 0 ? ScalarType::Float4 : ScalarType::Float;

            if (top % 8 == 0 && out_elempack == 8) {
                dst = otter::empty({outh / out_elempack, outw}, out_dtype);

                __m256 pad_value = _mm256_set1_ps(value.toFloat());
                padding_constant_pack8_avx(self, dst, top / 8, bottom / 8, left, right, pad_value);

                return;
            }
        }

        if (dims == 3) {
            int w = self.size(2);
            int h = self.size(1);
            int channels = self.size(0);
            int type = 0;
            
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

            int out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
            ScalarType out_dtype = outc % 8 == 0 ? ScalarType::Float8 : outc % 4 == 0 ? ScalarType::Float4 : ScalarType::Float;

            if (front % 8 == 0 && out_elempack == 8 && !(outc != channels * elempack && type != 0)) {
                dst = otter::empty({outc / out_elempack, outh, outw}, out_dtype);

                int front_ = front / elempack;
                otter::parallel_for(0, outc / out_elempack, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        auto borderm = dst[q];

                        __m256 pad_value = _mm256_set1_ps(value.toFloat());
                        //Channel padding
                        if ((q - front_) < 0 || (q - front_) >= channels) {
                            borderm.fill_(pad_value);
                        } else {
                            const auto m = self[q - front_];
                            padding_constant_pack8_avx(m, borderm, top, bottom, left, right, pad_value);
                        }
                    }
                });

                return;
            }
        }
    }
#endif // __AVX__

    if (elempack == 4) {
        if (dims == 1) {
            int w = self.size(0);
            
            int outw = w * elempack + left + right;

#if false
            int out_elempack = outw % 8 == 0 ? 8 : outw % 4 == 0 ? 4 : 1;
            ScalarType out_dtype = outw % 8 == 0 ? ScalarType::Float8 : outw % 4 == 0 ? ScalarType::Float4 : ScalarType::Float;
#else
            int out_elempack = outw % 4 == 0 ? 4 : 1;
            ScalarType out_dtype = outw % 4 == 0 ? ScalarType::Float4 : ScalarType::Float;
#endif

            if (left % 4 == 0 && out_elempack == 4) {
                dst = otter::empty({1, outw / out_elempack}, out_dtype);

                __m128 pad_value = _mm_set1_ps(value.toFloat());
                padding_constant_pack4_sse(self.view({1, -1}), dst, 0, 0, left / 4, right / 4, pad_value);

                dst.squeeze_(0);
                
                return;
            }
        }

        if (dims == 2) {
            int w = self.size(1);
            int h = self.size(0);
            
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

#if false
            int out_elempack = outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4 : 1;
            ScalarType out_dtype = outh % 8 == 0 ? ScalarType::Float8 : outh % 4 == 0 ? ScalarType::Float4 : ScalarType::Float;
#else
            int out_elempack = outh % 4 == 0 ? 4 : 1;
            ScalarType out_dtype = outh % 4 == 0 ? ScalarType::Float4 : ScalarType::Float;
#endif
            if (top % 4 == 0 && out_elempack == 4) {
                dst = otter::empty({outh / out_elempack, outw}, out_dtype);

                __m128 pad_value = _mm_set1_ps(value.toFloat());
                padding_constant_pack4_sse(self, dst, top / 4, bottom / 4, left, right, pad_value);

                return;
            }
        }

        if (dims == 3) {
            int w = self.size(2);
            int h = self.size(1);
            int channels = self.size(0);
            int type = 0;
            
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

#if false
            int out_elempack = outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
            ScalarType out_dtype = outc % 8 == 0 ? ScalarType::Float8 : outc % 4 == 0 ? ScalarType::Float4 : ScalarType::Float;
#else
            int out_elempack = outc % 4 == 0 ? 4 : 1;
            ScalarType out_dtype = outc % 4 == 0 ? ScalarType::Float4 : ScalarType::Float;
#endif

            if (front % 4 == 0 && out_elempack == 4 && !(outc != channels * elempack && type != 0)) {
                dst = otter::empty({outc / out_elempack, outh, outw}, out_dtype);

                int front_ = front / elempack;
                otter::parallel_for(0, outc / out_elempack, 0, [&](int64_t begin, int64_t end) {
                    for (int q = 0; q < outc / out_elempack; q++) {
                        auto borderm = dst[q];

                        __m128 pad_value = _mm_set1_ps(value.toFloat());
                        //Channel padding
                        if ((q - front_) < 0 || (q - front_) >= channels) {
                            borderm.fill_(pad_value);
                        } else {
                            const auto m = self[q - front_];
                            padding_constant_pack4_sse(m, borderm, top, bottom, left, right, pad_value);
                        }
                    }
                });

                return;
            }
        }
    }
    
    dst = self;
    if (elempack != 1) {
        dst = self.packing(1);
    }

    dst = constant_pad(dst, pad, value);
#endif // __SSE2__
}
#else
void constant_pad_float_packed_x86(const Tensor& /*self*/, Tensor& /*dst*/, IntArrayRef /*pad*/, const Scalar& /*value*/) {}
#endif

#if __ARM_NEON__
static void padding_constant_pack4_neon(const Tensor& src, Tensor& dst, int top, int bottom, int left, int right, float32x4_t v) {
    const float* ptr = (const float*)src.raw_data();
    float* outptr = (float*)dst.raw_data();

    int w = src.size(1);
    int h = src.size(0);

    int top_size = top * dst.size(1);
    int bottom_size = bottom * dst.size(1);

#if __aarch64__
    asm volatile(
        "mov    v0.16b, %10.16b         \n"
        "mov    v1.16b, %10.16b         \n"
        "mov    v2.16b, %10.16b         \n"
        "mov    v3.16b, %10.16b         \n"

        // fill top
        "lsr    w4, %w8, #3             \n" // w4 = nn = top_size >> 3
        "cmp    w4, #0                  \n"
        "beq    1f                      \n"

        "0:                             \n"
        "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
        "subs   w4, w4, #1              \n"
        "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
        "bne    0b                      \n"

        "1:                             \n"

        // fill top remain
        "and    w4, %w8, #7             \n" // w4 = remain = top_size & 7

        "cmp    w4, #4                  \n" // w4 >= 4
        "blt    2f                      \n"
        "sub    w4, w4, #4              \n"
        "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
        "2:                             \n"

        "cmp    w4, #2                  \n" // w4 >= 2
        "blt    3f                      \n"
        "sub    w4, w4, #2              \n"
        "st1    {v0.4s, v1.4s}, [%0], #32 \n"
        "3:                             \n"

        "cmp    w4, #0                  \n" // w4 > 0
        "beq    4f                      \n"
        "st1    {v0.4s}, [%0], #16      \n"
        "4:                             \n"

        // fill center h loop
        "cmp    %w5, #0                 \n"
        "beq    15f                     \n"
        "5:                             \n"

        // fill left
        "mov    w4, %w6                 \n" // w4 = left
        "cmp    w4, #0                  \n"
        "beq    7f                      \n"

        "6:                             \n"
        "st1    {v0.4s}, [%0], #16      \n"
        "subs   w4, w4, #1              \n"
        "bne    6b                      \n"

        "7:                             \n"

        // fill middle
        "lsr    w4, %w4, #3             \n" // w4 = nn = w >> 3
        "cmp    w4, #0                  \n"
        "beq    9f                      \n"

        "8:                             \n"
        "prfm   pldl1keep, [%1, #512]   \n"
        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
        "prfm   pldl1keep, [%1, #512]   \n"
        "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%1], #64 \n"
        "subs   w4, w4, #1              \n"
        "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
        "st1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%0], #64 \n"
        "bne    8b                      \n"

        "9:                             \n"

        "and    w4, %w4, #7             \n" // w4 = remain = w & 7

        "cmp    w4, #4                  \n" // w4 >= 4
        "blt    10f                     \n"
        "prfm   pldl1keep, [%1, #512]   \n"
        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%1], #64 \n"
        "sub    w4, w4, #4              \n"
        "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%0], #64 \n"
        "10:                            \n"

        "cmp    w4, #2                  \n" // w4 >= 2
        "blt    11f                     \n"
        "prfm   pldl1keep, [%1, #256]   \n"
        "ld1    {v16.4s, v17.4s}, [%1], #32 \n"
        "sub    w4, w4, #2              \n"
        "st1    {v16.4s, v17.4s}, [%0], #32 \n"
        "11:                            \n"

        "cmp    w4, #0                  \n" // w4 > 0
        "beq    12f                     \n"
        "prfm   pldl1keep, [%1, #128]   \n"
        "ld1    {v16.4s}, [%1], #16     \n"
        "st1    {v16.4s}, [%0], #16     \n"
        "12:                            \n"

        // fill right
        "mov    w4, %w7                 \n" // w4 = right
        "cmp    w4, #0                  \n"
        "beq    14f                     \n"

        "13:                            \n"
        "subs   w4, w4, #1              \n"
        "st1    {v0.4s}, [%0], #16      \n"
        "bne    13b                     \n"
        "14:                            \n"

        "subs   %w5, %w5, #1            \n"
        "bne    5b                      \n"

        "15:                            \n"

        // fill bottom
        "lsr    w4, %w9, #3             \n" // w4 = nn = bottom_size >> 3
        "cmp    w4, #0                  \n"
        "beq    17f                     \n"

        "16:                            \n"
        "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
        "subs   w4, w4, #1              \n"
        "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
        "bne    16b                     \n"
        "17:                            \n"

        // fill bottom remain
        "and    w4, %w9, #7             \n" // w4 = remain = bottom_size & 7

        "cmp    w4, #4                  \n" // w4 >= 4
        "blt    18f                     \n"
        "sub    w4, w4, #4              \n"
        "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0], #64 \n"
        "18:                            \n"

        "cmp    w4, #2                  \n" // w4 >= 2
        "blt    19f                     \n"
        "sub    w4, w4, #2              \n"
        "st1    {v0.4s, v1.4s}, [%0], #32 \n"
        "19:                            \n"

        "cmp    w4, #0                  \n" // w4 > 0
        "beq    20f                     \n"
        "st1    {v0.4s}, [%0], #16      \n"
        "20:                            \n"

        : "=r"(outptr), // %0
        "=r"(ptr)     // %1
        : "0"(outptr),
        "1"(ptr),
        "r"(w),           // %4
        "r"(h),           // %5
        "r"(left),        // %6
        "r"(right),       // %7
        "r"(top_size),    // %8
        "r"(bottom_size), // %9
        "w"(v)            // %10
        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
#else  // __aarch64__
    asm volatile(
        "vmov       q0, %q10            \n"
        "vmov       q1, %q10            \n"
        "vmov       q2, %q10            \n"
        "vmov       q3, %q10            \n"

        // fill top
        "lsr        r4, %8, #3          \n" // r4 = nn = top_size >> 3
        "cmp        r4, #0              \n"
        "beq        1f                  \n"

        "0:                             \n"
        "vstm       %0!, {d0-d7}        \n"
        "subs       r4, r4, #1          \n"
        "vstm       %0!, {d0-d7}        \n"
        "bne        0b                  \n"

        "1:                             \n"

        // fill top remain
        "and        r4, %8, #7          \n" // r4 = remain = top_size & 7

        "cmp        r4, #4              \n" // r4 >= 4
        "blt        2f                  \n"
        "sub        r4, r4, #4          \n"
        "vstm       %0!, {d0-d7}        \n"
        "2:                             \n"

        "cmp        r4, #2              \n" // r4 >= 2
        "blt        3f                  \n"
        "sub        r4, r4, #2          \n"
        "vst1.f32   {d0-d3}, [%0 :128]! \n"
        "3:                             \n"

        "cmp        r4, #0              \n" // r4 > 0
        "beq        4f                  \n"
        "vst1.f32   {d0-d1}, [%0 :128]! \n"
        "4:                             \n"

        // fill center h loop
        "cmp        %5, #0              \n"
        "beq        15f                 \n"
        "5:                             \n"

        // fill left
        "mov        r4, %6              \n" // r4 = left
        "cmp        r4, #0              \n"
        "beq        7f                  \n"

        "6:                             \n"
        "vst1.f32   {d0-d1}, [%0 :128]! \n"
        "subs       r4, r4, #1          \n"
        "bne        6b                  \n"

        "7:                             \n"

        // fill middle
        "lsr        r4, %4, #3          \n" // r4 = nn = w >> 3
        "cmp        r4, #0              \n"
        "beq        9f                  \n"

        "8:                             \n"
        "pld        [%1, #512]          \n"
        "vldm       %1!, {d16-d23}      \n"
        "pld        [%1, #512]          \n"
        "vldm       %1!, {d24-d31}      \n"
        "subs       r4, r4, #1          \n"
        "vstm       %0!, {d16-d23}      \n"
        "vstm       %0!, {d24-d31}      \n"
        "bne        8b                  \n"

        "9:                             \n"

        "and        r4, %4, #7          \n" // r4 = remain = w & 7

        "cmp        r4, #4              \n" // r4 >= 4
        "blt        10f                 \n"
        "pld        [%1, #512]          \n"
        "vldm       %1!, {d16-d23}      \n"
        "sub        r4, r4, #4          \n"
        "vstm       %0!, {d16-d23}      \n"
        "10:                            \n"

        "cmp        r4, #2              \n" // r4 >= 2
        "blt        11f                 \n"
        "pld        [%1, #256]          \n"
        "vld1.f32   {d16-d19}, [%1 :128]! \n"
        "sub        r4, r4, #2          \n"
        "vst1.f32   {d16-d19}, [%0 :128]! \n"
        "11:                            \n"

        "cmp        r4, #0              \n" // r4 > 0
        "beq        12f                 \n"
        "pld        [%1, #128]          \n"
        "vld1.f32   {d16-d17}, [%1 :128]! \n"
        "vst1.f32   {d16-d17}, [%0 :128]! \n"
        "12:                            \n"

        // fill right
        "mov        r4, %7              \n" // r4 = right
        "cmp        r4, #0              \n"
        "beq        14f                 \n"

        "13:                            \n"
        "subs       r4, r4, #1          \n"
        "vst1.f32   {d0-d1}, [%0 :128]! \n"
        "bne        13b                 \n"
        "14:                            \n"

        "subs       %5, %5, #1          \n"
        "bne        5b                  \n"

        "15:                            \n"

        // fill bottom
        "lsr        r4, %9, #3          \n" // r4 = nn = bottom_size >> 3
        "cmp        r4, #0              \n"
        "beq        17f                 \n"

        "16:                            \n"
        "vstm       %0!, {d0-d7}        \n"
        "subs       r4, r4, #1          \n"
        "vstm       %0!, {d0-d7}        \n"
        "bne        16b                 \n"
        "17:                            \n"

        // fill bottom remain
        "and        r4, %9, #7          \n" // r4 = remain = bottom_size & 7

        "cmp        r4, #4              \n" // r4 >= 4
        "blt        18f                 \n"
        "sub        r4, r4, #4          \n"
        "vstm       %0!, {d0-d7}        \n"
        "18:                            \n"

        "cmp        r4, #2              \n" // r4 >= 2
        "blt        19f                 \n"
        "sub        r4, r4, #2          \n"
        "vst1.f32   {d0-d3}, [%0 :128]! \n"
        "19:                            \n"

        "cmp        r4, #0              \n" // r4 > 0
        "beq        20f                 \n"
        "vst1.f32   {d0-d1}, [%0 :128]! \n"
        "20:                            \n"

        : "=r"(outptr), // %0
        "=r"(ptr)     // %1
        : "0"(outptr),
        "1"(ptr),
        "r"(w),           // %4
        "r"(h),           // %5
        "r"(left),        // %6
        "r"(right),       // %7
        "r"(top_size),    // %8
        "r"(bottom_size), // %9
        "w"(v)            // %10
        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11", "q12", "q13", "q14", "q15");
#endif // __aarch64__
}
#endif

#if __ARM_NEON__
void constant_pad_float_packed_neon(const Tensor& self, Tensor& dst, IntArrayRef pad, const Scalar& value) {
    int64_t left   = (pad.size() > 0) ? pad[0] : 0;
    int64_t right  = (pad.size() > 1) ? pad[1] : 0;
    int64_t top    = (pad.size() > 2) ? pad[2] : 0;
    int64_t bottom = (pad.size() > 3) ? pad[3] : 0;
    int64_t front  = (pad.size() > 4) ? pad[4] : 0;
    int64_t behind = (pad.size() > 5) ? pad[5] : 0;
    
    if (left == 0 && right == 0 && top == 0 && bottom == 0 && front == 0 && behind == 0) {
        dst = self;
        
        return;
    }
    
    int64_t dims = self.dim();
    int64_t elempack = self.elempack();
    
#if __ARM_NEON
    if (elempack == 4) {
        if (dims == 1) {
            int w = self.size(0);
            
            int outw = w * elempack + left + right;

            int out_elempack = outw % 4 == 0 ? 4 : 1;
            ScalarType out_dtype = outw % 4 == 0 ? ScalarType::Float4 : ScalarType::Float;

            if (left % 4 == 0 && out_elempack == 4) {
                dst = otter::empty({1, outw / out_elempack}, out_dtype);

                float32x4_t pad_value = vdupq_n_f32(value.toFloat());
                padding_constant_pack4_neon(self.view({1, -1}), dst, 0, 0, left / 4, right / 4, pad_value);

                dst.squeeze_(0);
                
                return;
            }
        }

        if (dims == 2) {
            int w = self.size(1);
            int h = self.size(0);
            
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

            int out_elempack = outh % 4 == 0 ? 4 : 1;
            ScalarType out_dtype = outh % 4 == 0 ? ScalarType::Float4 : ScalarType::Float;

            if (top % 4 == 0 && out_elempack == 4) {
                dst = otter::empty({outh / out_elempack, outw}, out_dtype);

                float32x4_t pad_value = vdupq_n_f32(value.toFloat());
                padding_constant_pack4_neon(self, dst, top / 4, bottom / 4, left, right, pad_value);

                return;
            }
        }

        if (dims == 3) {
            int w = self.size(2);
            int h = self.size(1);
            int channels = self.size(0);
            int type = 0;
            
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

            int out_elempack = outc % 4 == 0 ? 4 : 1;
            ScalarType out_dtype = outc % 4 == 0 ? ScalarType::Float4 : ScalarType::Float;

            if (front % 4 == 0 && out_elempack == 4 && !(outc != channels * elempack && type != 0)) {
                dst = otter::empty({outc / out_elempack, outh, outw}, out_dtype);

                int front_ = front / elempack;
                otter::parallel_for(0, outc / out_elempack, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        auto borderm = dst[q];

                        float32x4_t pad_value = vdupq_n_f32(value.toFloat());
                        //Channel padding
                        if ((q - front_) < 0 || (q - front_) >= channels) {
                            borderm.fill_(pad_value);
                        } else {
                            const auto m = self[q - front_];
                            padding_constant_pack4_neon(m, borderm, top, bottom, left, right, pad_value);
                        }
                    }
                });

                return 0;
            }
        }
    }
#endif // __ARM_NEON
    
    dst = self;
    if (elempack != 1) {
        dst = self.packing(1);
    }

    dst = constant_pad(dst, pad, value);
}
#else
void constant_pad_float_packed_neon(const Tensor& /*self*/, Tensor& /*dst*/, IntArrayRef /*pad*/, const Scalar& /*value*/) {}
#endif

Tensor constant_pad_int8_packed(const Tensor& self, IntArrayRef pad, const Scalar& value) {
    int64_t elempack = self.elempack();
    
    // Now we suppose that the first axis of Tensor is batchsize
    bool batch = self.dim() == 4;
    Tensor input = (self.dim() == 4) ? self.squeeze(0) : self;
    
    Tensor out;
#if __SSE2__
    (void)elempack;
    constant_pad_int8_packed_x86(input, out, pad, value);
    
    return (batch) ? out.unsqueeze(0) : out;
#elif __ARM_NEON__
    (void)elempack;
    
    constant_pad_int8_packed_neon(input, out, pad, value);
    
    return (batch) ? out.unsqueeze(0) : out;
#else
    out = input;
    if (elempack != 1) {
        out = input.packing(1);
    }
    out = constant_pad(out, pad, value);
    
    return (batch) ? out.unsqueeze(0) : out;
#endif
}

#if __SSE2__
static void padding_constant_pack8_int8_sse(const Tensor& src, Tensor& dst, int top, int bottom, int left, int right, int64_t _v) {
    const int64_t* ptr = (const int64_t*)src.raw_data();
    int64_t* outptr = (int64_t*)dst.raw_data();

    // fill top
    for (int y = 0; y < top; y++) {
        for (int x = 0; x < dst.size(1); x++) {
            *outptr++ = _v;
        }
    }
    // fill center
    for (int y = 0; y < src.size(0); y++) {
        for (int x = 0; x < left; x++) {
            *outptr++ = _v;
        }
        for (int x = 0; x < src.size(1); x++) {
            *outptr++ = *ptr++;
        }
        for (int x = 0; x < right; x++) {
            *outptr++ = _v;
        }
    }
    // fill bottom
    for (int y = 0; y < bottom; y++) {
        for (int x = 0; x < dst.size(1); x++) {
            *outptr++ = _v;
        }
    }
}
#endif

#if __SSE2__
void constant_pad_int8_packed_x86(const Tensor& self, Tensor& dst, IntArrayRef pad, const Scalar& value) {
    int64_t left   = (pad.size() > 0) ? pad[0] : 0;
    int64_t right  = (pad.size() > 1) ? pad[1] : 0;
    int64_t top    = (pad.size() > 2) ? pad[2] : 0;
    int64_t bottom = (pad.size() > 3) ? pad[3] : 0;
    int64_t front  = (pad.size() > 4) ? pad[4] : 0;
    int64_t behind = (pad.size() > 5) ? pad[5] : 0;
    
    if (left == 0 && right == 0 && top == 0 && bottom == 0 && front == 0 && behind == 0) {
        dst = self;
        
        return;
    }
    
    int64_t dims = self.dim();
    int64_t elempack = self.elempack();
    
#if __SSE2__
    if (elempack == 8) {
        if (dims == 1) {
            int w = self.size(0);
            
            int outw = w * elempack + left + right;

            int out_elempack = outw % 8 == 0 ? 8 : 1;
            ScalarType out_dtype = outw % 8 == 0 ? ScalarType::Byte8 : ScalarType::Byte;

            if (left % 8 == 0 && out_elempack == 8) {
                dst = otter::empty({1, outw / out_elempack}, out_dtype);

                int64_t v8 = (int64_t)value.toLong();
                int64_t pad_value = v8 | (v8 << 8) | (v8 << 16) | (v8 << 24) | (v8 << 32) | (v8 << 40) | (v8 << 48) | (v8 << 56);
                padding_constant_pack8_int8_sse(self.view({1, -1}), dst, 0, 0, left / 8, right / 8, pad_value);

                dst.squeeze_(0);
                
                return;
            }
        }

        if (dims == 2) {
            int w = self.size(1);
            int h = self.size(0);
            
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

            int out_elempack = outh % 8 == 0 ? 8 : 1;
            ScalarType out_dtype = outh % 8 == 0 ? ScalarType::Byte8 : ScalarType::Byte;

            if (top % 8 == 0 && out_elempack == 8) {
                dst = otter::empty({outh / out_elempack, outw}, out_dtype);

                int64_t v8 = (int64_t)value.toLong();
                int64_t pad_value = v8 | (v8 << 8) | (v8 << 16) | (v8 << 24) | (v8 << 32) | (v8 << 40) | (v8 << 48) | (v8 << 56);
                padding_constant_pack8_int8_sse(self, dst, top / 8, bottom / 8, left, right, pad_value);

                return;
            }
        }

        if (dims == 3) {
            int w = self.size(2);
            int h = self.size(1);
            int channels = self.size(0);
            int type = 0;
            
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

            int out_elempack = outc % 8 == 0 ? 8 : 1;
            ScalarType out_dtype = outc % 8 == 0 ? ScalarType::Byte8 : ScalarType::Byte;

            if (front % 8 == 0 && out_elempack == 8 && !(outc != channels * elempack && type != 0)) {
                dst = otter::empty({outc / out_elempack, outh, outw}, out_dtype);

                int front_ = front / elempack;
                otter::parallel_for(0, outc / out_elempack, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        auto borderm = dst[q];

                        int64_t v8 = (int64_t)value.toLong();
                        int64_t pad_value = v8 | (v8 << 8) | (v8 << 16) | (v8 << 24) | (v8 << 32) | (v8 << 40) | (v8 << 48) | (v8 << 56);

                        //Channel padding
                        if ((q - front_) < 0 || (q - front_) >= channels) {
                            borderm.type_fill_<int64_t>(pad_value);
                        } else {
                            const auto m = self[q - front_];
                            padding_constant_pack8_int8_sse(m, borderm, top, bottom, left, right, pad_value);
                        }
                    }
                });

                return;
            }
        }
    }
#endif // __SSE2__
}
#else
void constant_pad_int8_packed_x86(const Tensor& /*self*/, Tensor& /*dst*/, IntArrayRef /*pad*/, const Scalar& /*value*/) {}
#endif

#if __ARM_NEON__
static void padding_constant_pack8_int8_neon(const Tensor& src, Tensor& dst, int top, int bottom, int left, int right, int8x8_t v) {
    const signed char* ptr = (const signed char*)src.raw_data();
    signed char* outptr = (signed char*)dst.raw_data();

    int w = src.size(1);
    int h = src.size(0);

    int top_size = top * dst.size(1);
    int bottom_size = bottom * dst.size(1);

#if __aarch64__
    asm volatile(
        "mov    v0.8b, %10.8b           \n"
        "mov    v0.d[1], v0.d[0]        \n"
        "mov    v1.16b, v0.16b          \n"
        "mov    v2.16b, v0.16b          \n"
        "mov    v3.16b, v0.16b          \n"

        // fill top
        "lsr    w4, %w8, #3             \n" // w4 = nn = top_size >> 3
        "cmp    w4, #0                  \n"
        "beq    1f                      \n"

        "0:                             \n"
        "st1    {v0.16b, v1.16b, v2.16b, v3.16b}, [%0], #64 \n"
        "subs   w4, w4, #1              \n"
        "bne    0b                      \n"

        "1:                             \n"

        // fill top remain
        "and    w4, %w8, #7             \n" // w4 = remain = top_size & 7

        "cmp    w4, #4                  \n" // w4 >= 4
        "blt    2f                      \n"
        "sub    w4, w4, #4              \n"
        "st1    {v0.16b, v1.16b}, [%0], #32 \n"
        "2:                             \n"

        "cmp    w4, #2                  \n" // w4 >= 2
        "blt    3f                      \n"
        "sub    w4, w4, #2              \n"
        "st1    {v0.16b}, [%0], #16     \n"
        "3:                             \n"

        "cmp    w4, #0                  \n" // w4 > 0
        "beq    4f                      \n"
        "st1    {v0.8b}, [%0], #8       \n"
        "4:                             \n"

        // fill center h loop
        "cmp    %w5, #0                 \n"
        "beq    15f                     \n"
        "5:                             \n"

        // fill left
        "mov    w4, %w6                 \n" // w4 = left
        "cmp    w4, #0                  \n"
        "beq    7f                      \n"

        "6:                             \n"
        "st1    {v0.8b}, [%0], #8       \n"
        "subs   w4, w4, #1              \n"
        "bne    6b                      \n"

        "7:                             \n"

        // fill middle
        "lsr    w4, %w4, #3             \n" // w4 = nn = w >> 3
        "cmp    w4, #0                  \n"
        "beq    9f                      \n"

        "8:                             \n"
        "prfm   pldl1keep, [%1, #512]   \n"
        "ld1    {v16.16b, v17.16b, v18.16b, v19.16b}, [%1], #64 \n"
        "subs   w4, w4, #1              \n"
        "st1    {v16.16b, v17.16b, v18.16b, v19.16b}, [%0], #64 \n"
        "bne    8b                      \n"

        "9:                             \n"

        "and    w4, %w4, #7             \n" // w4 = remain = w & 7

        "cmp    w4, #4                  \n" // w4 >= 4
        "blt    10f                     \n"
        "prfm   pldl1keep, [%1, #256]   \n"
        "ld1    {v16.16b, v17.16b}, [%1], #32 \n"
        "sub    w4, w4, #4              \n"
        "st1    {v16.16b, v17.16b}, [%0], #32 \n"
        "10:                            \n"

        "cmp    w4, #2                  \n" // w4 >= 2
        "blt    11f                     \n"
        "prfm   pldl1keep, [%1, #128]   \n"
        "ld1    {v16.16b}, [%1], #16    \n"
        "sub    w4, w4, #2              \n"
        "st1    {v16.16b}, [%0], #16    \n"
        "11:                            \n"

        "cmp    w4, #0                  \n" // w4 > 0
        "beq    12f                     \n"
        "prfm   pldl1keep, [%1, #64]    \n"
        "ld1    {v16.8b}, [%1], #8      \n"
        "st1    {v16.8b}, [%0], #8      \n"
        "12:                            \n"

        // fill right
        "mov    w4, %w7                 \n" // w4 = right
        "cmp    w4, #0                  \n"
        "beq    14f                     \n"

        "13:                            \n"
        "subs   w4, w4, #1              \n"
        "st1    {v0.8b}, [%0], #8       \n"
        "bne    13b                     \n"
        "14:                            \n"

        "subs   %w5, %w5, #1            \n"
        "bne    5b                      \n"

        "15:                            \n"

        // fill bottom
        "lsr    w4, %w9, #3             \n" // w4 = nn = bottom_size >> 3
        "cmp    w4, #0                  \n"
        "beq    17f                     \n"

        "16:                            \n"
        "st1    {v0.16b, v1.16b, v2.16b, v3.16b}, [%0], #64 \n"
        "subs   w4, w4, #1              \n"
        "bne    16b                     \n"
        "17:                            \n"

        // fill bottom remain
        "and    w4, %w9, #7             \n" // w4 = remain = bottom_size & 7

        "cmp    w4, #4                  \n" // w4 >= 4
        "blt    18f                     \n"
        "sub    w4, w4, #4              \n"
        "st1    {v0.16b, v1.16b}, [%0], #32 \n"
        "18:                            \n"

        "cmp    w4, #2                  \n" // w4 >= 2
        "blt    19f                     \n"
        "sub    w4, w4, #2              \n"
        "st1    {v0.16b}, [%0], #16     \n"
        "19:                            \n"

        "cmp    w4, #0                  \n" // w4 > 0
        "beq    20f                     \n"
        "st1    {v0.8b}, [%0], #8       \n"
        "20:                            \n"

        : "=r"(outptr), // %0
        "=r"(ptr)     // %1
        : "0"(outptr),
        "1"(ptr),
        "r"(w),           // %4
        "r"(h),           // %5
        "r"(left),        // %6
        "r"(right),       // %7
        "r"(top_size),    // %8
        "r"(bottom_size), // %9
        "w"(v)            // %10
        : "cc", "memory", "x4", "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19");
#else  // __aarch64__
    asm volatile(
        "vmov       d0, %P10            \n"
        "vmov       d1, d0              \n"
        "vmov       q1, q0              \n"
        "vmov       q2, q0              \n"
        "vmov       q3, q0              \n"

        // fill top
        "lsr        r4, %8, #3          \n" // r4 = nn = top_size >> 3
        "cmp        r4, #0              \n"
        "beq        1f                  \n"

        "0:                             \n"
        "vstm       %0!, {d0-d7}        \n"
        "subs       r4, r4, #1          \n"
        "bne        0b                  \n"

        "1:                             \n"

        // fill top remain
        "and        r4, %8, #7          \n" // r4 = remain = top_size & 7

        "cmp        r4, #4              \n" // r4 >= 4
        "blt        2f                  \n"
        "sub        r4, r4, #4          \n"
        "vst1.s8    {d0-d3}, [%0 :128]! \n"
        "2:                             \n"

        "cmp        r4, #2              \n" // r4 >= 2
        "blt        3f                  \n"
        "sub        r4, r4, #2          \n"
        "vst1.s8    {d0-d1}, [%0 :128]! \n"
        "3:                             \n"

        "cmp        r4, #0              \n" // r4 > 0
        "beq        4f                  \n"
        "vst1.s8    {d0}, [%0 :64]!     \n"
        "4:                             \n"

        // fill center h loop
        "cmp        %5, #0              \n"
        "beq        15f                 \n"
        "5:                             \n"

        // fill left
        "mov        r4, %6              \n" // r4 = left
        "cmp        r4, #0              \n"
        "beq        7f                  \n"

        "6:                             \n"
        "vst1.s8    {d0}, [%0 :64]!     \n"
        "subs       r4, r4, #1          \n"
        "bne        6b                  \n"

        "7:                             \n"

        // fill middle
        "lsr        r4, %4, #3          \n" // r4 = nn = w >> 3
        "cmp        r4, #0              \n"
        "beq        9f                  \n"

        "8:                             \n"
        "pld        [%1, #512]          \n"
        "vldm       %1!, {d16-d23}      \n"
        "subs       r4, r4, #1          \n"
        "vstm       %0!, {d16-d23}      \n"
        "bne        8b                  \n"

        "9:                             \n"

        "and        r4, %4, #7          \n" // r4 = remain = w & 7

        "cmp        r4, #4              \n" // r4 >= 4
        "blt        10f                 \n"
        "pld        [%1, #256]          \n"
        "vld1.s8    {d16-d19}, [%1 :64]! \n"
        "sub        r4, r4, #4          \n"
        "vst1.s8    {d16-d19}, [%0 :64]! \n"
        "10:                            \n"

        "cmp        r4, #2              \n" // r4 >= 2
        "blt        11f                 \n"
        "pld        [%1, #128]          \n"
        "vld1.s8    {d16-d17}, [%1 :64]! \n"
        "sub        r4, r4, #2          \n"
        "vst1.s8    {d16-d17}, [%0 :64]! \n"
        "11:                            \n"

        "cmp        r4, #0              \n" // r4 > 0
        "beq        12f                 \n"
        "pld        [%1, #64]           \n"
        "vld1.s8    {d16}, [%1 :64]!    \n"
        "vst1.s8    {d16}, [%0 :64]!    \n"
        "12:                            \n"

        // fill right
        "mov        r4, %7              \n" // r4 = right
        "cmp        r4, #0              \n"
        "beq        14f                 \n"

        "13:                            \n"
        "subs       r4, r4, #1          \n"
        "vst1.s8    {d0}, [%0 :64]!     \n"
        "bne        13b                 \n"
        "14:                            \n"

        "subs       %5, %5, #1          \n"
        "bne        5b                  \n"

        "15:                            \n"

        // fill bottom
        "lsr        r4, %9, #3          \n" // r4 = nn = bottom_size >> 3
        "cmp        r4, #0              \n"
        "beq        17f                 \n"

        "16:                            \n"
        "vstm       %0!, {d0-d7}        \n"
        "subs       r4, r4, #1          \n"
        "bne        16b                 \n"
        "17:                            \n"

        // fill bottom remain
        "and        r4, %9, #7          \n" // r4 = remain = bottom_size & 7

        "cmp        r4, #4              \n" // r4 >= 4
        "blt        18f                 \n"
        "sub        r4, r4, #4          \n"
        "vst1.s8    {d0-d3}, [%0 :64]!  \n"
        "18:                            \n"

        "cmp        r4, #2              \n" // r4 >= 2
        "blt        19f                 \n"
        "sub        r4, r4, #2          \n"
        "vst1.s8    {d0-d1}, [%0 :64]!  \n"
        "19:                            \n"

        "cmp        r4, #0              \n" // r4 > 0
        "beq        20f                 \n"
        "vst1.s8    {d0}, [%0 :64]!     \n"
        "20:                            \n"

        : "=r"(outptr), // %0
        "=r"(ptr)     // %1
        : "0"(outptr),
        "1"(ptr),
        "r"(w),           // %4
        "r"(h),           // %5
        "r"(left),        // %6
        "r"(right),       // %7
        "r"(top_size),    // %8
        "r"(bottom_size), // %9
        "w"(v)            // %10
        : "cc", "memory", "r4", "q0", "q1", "q2", "q3", "q8", "q9", "q10", "q11");
#endif // __aarch64__
}
#endif

#if __ARM_NEON__
void constant_pad_int8_packed_neon(const Tensor& self, Tensor& dst, IntArrayRef pad, const Scalar& value) {
    int64_t left   = (pad.size() > 0) ? pad[0] : 0;
    int64_t right  = (pad.size() > 1) ? pad[1] : 0;
    int64_t top    = (pad.size() > 2) ? pad[2] : 0;
    int64_t bottom = (pad.size() > 3) ? pad[3] : 0;
    int64_t front  = (pad.size() > 4) ? pad[4] : 0;
    int64_t behind = (pad.size() > 5) ? pad[5] : 0;
    
    int64_t elemapck = self.elempack();
    
    if (left == 0 && right == 0 && top == 0 && bottom == 0 && front == 0 && behind == 0) {
        dst = self;
        
        return;
    }
    
    int64_t dims = self.dim();
    int64_t elempack = self.elempack();
    
#if __ARM_NEON
    if (elempack == 8) {
        if (dims == 1) {
            int w = self.size(0);
            
            int outw = w * elempack + left + right;

            int out_elempack = outw % 8 == 0 ? 8 : 1;
            ScalarType out_dtype = outw % 8 == 0 ? ScalarType::Byte8 : ScalarType::Byte;

            if (left % 8 == 0 && out_elempack == 8) {
                dst = otter::empty({1, outw / out_elempack}, out_dtype);

                int8x8_t pad_value = vdup_n_s8((signed char)value.toByte());
                padding_constant_pack8_int8_neon(self.view({1, -1}), dst, 0, 0, left / 8, right / 8, pad_value);

                dst.squeeze_(0);
                
                return;
            }
        }

        if (dims == 2) {
            int w = self.size(1);
            int h = self.size(0);
            
            int outw = w + left + right;
            int outh = h * elempack + top + bottom;

            int out_elempack = outh % 8 == 0 ? 8 : 1;
            ScalarType out_dtype = outh % 8 == 0 ? ScalarType::Byte8 : ScalarType::Byte;

            if (top % 8 == 0 && out_elempack == 8) {
                dst = otter::empty({outh / out_elempack, outw}, out_dtype);

                int8x8_t pad_value = vdup_n_s8((signed char)value.toByte());
                padding_constant_pack8_int8_neon(self, dst, top / 8, bottom / 8, left, right, pad_value);

                return;
            }
        }

        if (dims == 3) {
            int w = self.size(2);
            int h = self.size(1);
            int channel = self.size(0);
            int type = 0;
            
            int outw = w + left + right;
            int outh = h + top + bottom;
            int outc = channels * elempack + front + behind;

            int out_elempack = outc % 8 == 0 ? 8 : 1;
            ScalarType out_dtype = outc % 8 == 0 ? ScalarType::Byte8 : ScalarType::Byte;

            if (front % 8 == 0 && out_elempack == 8 && !(outc != channels * elempack && type != 0)) {
                dst = otter::empty({outc / out_elempack, outh, outw}, out_dtype);
                
                int front_ = front / elempack;
                otter::parallel_for(0, outc / out_elempack, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        auto borderm = dst[q];

                        int8x8_t pad_value = vdup_n_s8((signed char)value.toByte());

                        //Channel padding
                        if ((q - front_) < 0 || (q - front_) >= channels) {
                            borderm.type_fill_<int8x8_t>(pad_value);
                        } else {
                            const auto m = self[q - front_];
                            padding_constant_pack8_int8_neon(m, borderm, top, bottom, left, right, pad_value);
                        }
                    }
                });

                return;
            }
        }
    }
#endif // __ARM_NEON
    
    dst = self;
    if (elempack != 1) {
        dst = self.packing(1);
    }

    dst = constant_pad(dst, pad, value);
}
#else
void constant_pad_int8_packed_neon(const Tensor& /*self*/, Tensor& /*dst*/, IntArrayRef /*pad*/, const Scalar& /*value*/) {}
#endif

}   // end namespace otter
