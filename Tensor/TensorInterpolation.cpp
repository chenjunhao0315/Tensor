//
//  TensorInterpolation.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/12.
//

#include "TensorInterpolation.hpp"
#include "Tensor.hpp"
#include "TensorFactory.hpp"
#include "Parallel.hpp"
#include "VecIntrinsic.hpp"

namespace otter {

static void linear_coeffs(int w, int outw, int* xofs, float* alpha, int align_corner) {
    double scale = (double)w / outw;
    if (align_corner) {
        scale = (double)(w - 1) / (outw - 1);
    }

    for (int dx = 0; dx < outw; dx++) {
        float fx = (float)((dx + 0.5) * scale - 0.5);
        if (align_corner) {
            fx = static_cast<float>(dx * scale);
        }

        int sx = static_cast<int>(floor(fx));
        fx -= sx;

        if (sx < 0) {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= w - 1) {
            sx = w - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        alpha[dx * 2] = 1.f - fx;
        alpha[dx * 2 + 1] = fx;
    }
}

#if __SSE2__
static void resize_bilinear_image_pack4(const Tensor& src, Tensor& dst, float* alpha, int* xofs, float* beta, int* yofs) {
    int w = dst.size(1);
    int h = dst.size(0);
    
    auto src_a = src.accessor<float, 2, 4>();
    auto dst_a = dst.accessor<float, 2, 4>();

    // loop body
    Tensor rowsbuf0 = otter::empty({w}, otter::ScalarType::Float4);
    Tensor rowsbuf1 = otter::empty({w}, otter::ScalarType::Float4);
    float* rows0 = (float*)rowsbuf0.raw_data();
    float* rows1 = (float*)rowsbuf1.raw_data();

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const float* S1 = src_a[sy + 1].data();

            const float* alphap = alpha;
            float* rows1p = rows1;
            int dx = 0;
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const float* S1p = S1 + sx;

                __m128 _a0 = _mm_set1_ps(alphap[0]);
                __m128 _a1 = _mm_set1_ps(alphap[1]);

                __m128 _S10 = _mm_load_ps(S1p);
                __m128 _S11 = _mm_load_ps(S1p + 4);
                __m128 _rows1 = _mm_mul_ps(_S10, _a0);
                _rows1 = _mm_comp_fmadd_ps(_S11, _a1, _rows1);
                _mm_store_ps(rows1p + dx * 4, _rows1);

                alphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const float* S0 = src_a[sy].data();
            const float* S1 = src_a[sy + 1].data();

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            int dx = 0;
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                __m128 _a0 = _mm_set1_ps(alphap[0]);
                __m128 _a1 = _mm_set1_ps(alphap[1]);

                __m128 _S00 = _mm_load_ps(S0p);
                __m128 _S01 = _mm_load_ps(S0p + 4);
                __m128 _S10 = _mm_load_ps(S1p);
                __m128 _S11 = _mm_load_ps(S1p + 4);
                __m128 _rows0 = _mm_mul_ps(_S00, _a0);
                __m128 _rows1 = _mm_mul_ps(_S10, _a0);
                _rows0 = _mm_comp_fmadd_ps(_S01, _a1, _rows0);
                _rows1 = _mm_comp_fmadd_ps(_S11, _a1, _rows1);
                _mm_store_ps(rows0p + dx * 4, _rows0);
                _mm_store_ps(rows1p + dx * 4, _rows1);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        __m128 _b0 = _mm_set1_ps(beta[0]);
        __m128 _b1 = _mm_set1_ps(beta[1]);

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst_a[dy].data();

        for (int dx = 0; dx < w; dx++)
        {
            __m128 _rows0 = _mm_load_ps(rows0p);
            __m128 _rows1 = _mm_load_ps(rows1p);
            __m128 _D = _mm_mul_ps(_rows0, _b0);
            _D = _mm_comp_fmadd_ps(_rows1, _b1, _D);
            _mm_store_ps(Dp, _D);

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
        }

        beta += 2;
    }
}

static void resize_bilinear_image(const Tensor& src, Tensor& dst, float* alpha, int* xofs, float* beta, int* yofs) {
    int w = dst.size(1);
    int h = dst.size(0);
    
    auto src_a = src.accessor<float, 2>();
    auto dst_a = dst.accessor<float, 2>();

    // loop body
    Tensor rowsbuf0 = otter::empty({w}, otter::ScalarType::Float);
    Tensor rowsbuf1 = otter::empty({w}, otter::ScalarType::Float);
    float* rows0 = rowsbuf0.data_ptr<float>();
    float* rows1 = rowsbuf1.data_ptr<float>();

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const float* S1 = src_a[sy + 1].data();

            const float* alphap = alpha;
            float* rows1p = rows1;
            int dx = 0;
            for (; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

                alphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const float* S0 = src_a[sy].data();
            const float* S1 = src_a[sy + 1].data();

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            int dx = 0;
            for (; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows0p[dx] = S0p[0] * a0 + S0p[1] * a1;
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst_a[dy].data();

        int dx = 0;
#if __SSE2__
#if __AVX__
        __m256 _b0_256 = _mm256_set1_ps(b0);
        __m256 _b1_256 = _mm256_set1_ps(b1);
        for (; dx + 7 < w; dx += 8)
        {
            __m256 _rows0 = _mm256_loadu_ps(rows0p);
            __m256 _rows1 = _mm256_loadu_ps(rows1p);
            __m256 _D = _mm256_mul_ps(_rows0, _b0_256);
            _D = _mm256_comp_fmadd_ps(_rows1, _b1_256, _D);
            _mm256_storeu_ps(Dp, _D);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }
#endif // __AVX__
        __m128 _b0_128 = _mm_set1_ps(b0);
        __m128 _b1_128 = _mm_set1_ps(b1);
        for (; dx + 3 < w; dx += 4)
        {
            __m128 _rows0 = _mm_loadu_ps(rows0p);
            __m128 _rows1 = _mm_loadu_ps(rows1p);
            __m128 _D = _mm_mul_ps(_rows0, _b0_128);
            _D = _mm_comp_fmadd_ps(_rows1, _b1_128, _D);
            _mm_storeu_ps(Dp, _D);

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
        }
#endif // __SSE2__
        for (; dx < w; dx++)
        {
            *Dp++ = *rows0p++ * b0 + *rows1p++ * b1;
        }

        beta += 2;
    }
}

Tensor interpolate_packed_x86(const Tensor& input, IntArrayRef size, InterpolateMode mode, bool align_corners) {
    
    Tensor output;
    
    int dims = input.dim();
    int outw = size[1];
    int outh = size[0];
    
    int elempack = input.elempack();
    
    if (dims == 1) {
        int w = input.size(0);
        
        output = otter::empty({w, outh, outw}, input.scalar_type());
        
        const float* in = (const float*)input.raw_data();
        
        if (elempack == 4) {
            otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    Tensor output_c = output[q];
                    __m128 _v = _mm_load_ps(in + q * 4);
                    output_c.fill_(_v);
                }
            });
            
            return output;
        }
        
        otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
            for (const auto q : otter::irange(begin, end)) {
                Tensor output_c = output[q];
                const float v = in[q];
                output_c.fill_(v);
            }
        });
        
        return output;
    }
    
    if (dims == 2) {
        int h = input.size(0);
        int w = input.size(1);
        
        if (size[1] == w) {
            output = input;
            
            return output;
        }
        
        output = otter::empty({h, outw}, input.scalar_type());
        
        if (elempack == 4) {
            auto input_a = input.accessor<float, 2, 4>();
            auto output_a = output.accessor<float, 2, 4>();
            
            if (mode == InterpolateMode::NEAREST) {
                const float ws =  w / (float)outw;

                otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                    for (const auto y : otter::irange(begin, end)) {
                        const float* ptr = input_a[y].data();
                        float* outptr = output_a[y].data();
                        for (int x = 0; x < outw; x++) {
                            int in_x = std::min((int)(x * ws), (w - 1));

                            __m128 _p = _mm_load_ps(ptr + in_x * 4);
                            _mm_store_ps(outptr, _p);

                            outptr += 4;
                        }
                    }
                });
                
                return output;
            }
            
            if (mode == InterpolateMode::BILINEAR) {
                int* buf = new int[outw + outw * 2];

                int* xofs = buf;
                float* alpha = (float*)(buf + outw);

                linear_coeffs(w, outw, xofs, alpha, align_corners);

                otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                    for (const auto y : otter::irange(begin, end)) {
                        const float* ptr = input_a[y].data();
                        float* outptr = output_a[y].data();
                        const float* alphap = alpha;

                        for (int x = 0; x < outw; x++) {
                            int sx = xofs[x] * 4;
                            const float* Sp = ptr + sx;

                            __m128 _a0 = _mm_set1_ps(alphap[0]);
                            __m128 _a1 = _mm_set1_ps(alphap[1]);

                            __m128 _S0 = _mm_load_ps(Sp);
                            __m128 _S1 = _mm_load_ps(Sp + 4);
                            __m128 _p = _mm_mul_ps(_S0, _a0);
                            _p = _mm_comp_fmadd_ps(_S1, _a1, _p);
                            _mm_store_ps(outptr, _p);

                            alphap += 2;
                            outptr += 4;
                        }
                    }
                });

                delete[] buf;
                
                return output;
            }
        }
        
        auto input_a = input.accessor<float, 2>();
        auto output_a = output.accessor<float, 2>();
        
        if (mode == InterpolateMode::NEAREST) {
            const float ws =  w / (float)outw;

            otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                for (const auto y : otter::irange(begin, end)) {
                    const float* ptr = input_a[y].data();
                    float* outptr = output_a[y].data();
                    for (int x = 0; x < outw; x++) {
                        int in_x = std::min((int)(x * ws), (w - 1));
                        *outptr++ = ptr[in_x];
                    }
                }
            });
            
            return output;
        }

        if (mode == InterpolateMode::BILINEAR) {
            int* buf = new int[outw + outw * 2];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            linear_coeffs(w, outw, xofs, alpha, align_corners);

            otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                for (const auto y : otter::irange(begin, end)) {
                    const float* ptr = input_a[y].data();
                    float* outptr = output_a[y].data();
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++) {
                        int sx = xofs[x];
                        const float* Sp = ptr + sx;
                        float a0 = alphap[0];
                        float a1 = alphap[1];
                        *outptr++ = Sp[0] * a0 + Sp[1] * a1;
                        alphap += 2;
                    }
                }
            });

            delete[] buf;
            
            return output;
        }
    }
    
    if (dims == 3) {
        int w = input.size(2);
        int h = input.size(1);
        int channels = input.size(0);
        
        if (w == outw && h == outh) {
            output = input;
            
            return output;
        }
        
        output = otter::empty({channels, outh, outw}, input.scalar_type());
        
        if (elempack == 4) {
            auto input_a = input.accessor<float, 3, 4>();
            auto output_a = output.accessor<float, 3, 4>();
            
            if (mode == InterpolateMode::NEAREST) {
                const float hs =  h / (float)outh;
                const float ws = w / (float)outw;

                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const auto src = input_a[q];
                        auto dst = output_a[q];

                        for (int y = 0; y < outh; y++) {
                            int in_y = std::min((int)(y * hs), (h - 1));

                            const float* ptr = src[in_y].data();
                            float* outptr = dst[y].data();
                            for (int x = 0; x < outw; x++)
                            {
                                int in_x = std::min((int)(x * ws), (w - 1));

                                __m128 _p = _mm_load_ps(ptr + in_x * 4);
                                _mm_store_ps(outptr, _p);

                                outptr += 4;
                            }
                        }
                    }
                });
                
                return output;
            }

            if (mode == InterpolateMode::BILINEAR) {
                int* buf = new int[outw + outh + outw * 2 + outh * 2];

                int* xofs = buf;        //new int[outw];
                int* yofs = buf + outw; //new int[outh];

                float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
                float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

                linear_coeffs(w, outw, xofs, alpha, align_corners);
                linear_coeffs(h, outh, yofs, beta, align_corners);

                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const Tensor src = input[q];
                        Tensor dst = output[q];

                        resize_bilinear_image_pack4(src, dst, alpha, xofs, beta, yofs);
                    }
                });

                delete[] buf;
                
                return output;
            }
        }
        
        auto input_a = input.accessor<float, 3>();
        auto output_a = output.accessor<float, 3>();
        
        if (mode == InterpolateMode::NEAREST) {
            const float hs = h / (float)outh;
            const float ws = w / (float)outw;

            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const auto src = input_a[q];
                    auto dst = output_a[q];

                    for (int y = 0; y < outh; y++) {
                        int in_y = std::min((int)(y * hs), (h - 1));

                        const float* ptr = src[in_y].data();
                        float* outptr = dst[y].data();
                        for (int x = 0; x < outw; x++)
                        {
                            int in_x = std::min((int)(x * ws), (w - 1));
                            *outptr++ = ptr[in_x];
                        }
                    }
                }
            });
            
            return output;
        }

        if (mode == InterpolateMode::BILINEAR) {
            int* buf = new int[outw + outh + outw * 2 + outh * 2];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
            float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

            linear_coeffs(w, outw, xofs, alpha, align_corners);
            linear_coeffs(h, outh, yofs, beta, align_corners);

            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const Tensor src = input[q];
                    Tensor dst = output[q];

                    resize_bilinear_image(src, dst, alpha, xofs, beta, yofs);
                }
            });

            delete[] buf;
            
            return output;
        }
    }
    
    return output;
}
#endif  // __SSE2__

#if __ARM_NEON__
static void resize_bilinear_image(const Tensor& src, Tensor& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.size(1);
    int h = dst.size(0);
    
    auto src_a = src.accessor<float, 2>();
    auto dst_a = dst.accessor<float, 2>();

    // loop body
    Tensor rowsbuf0 = otter::empty({w}, otter::ScalarType::Float);
    Tensor rowsbuf1 = otter::empty({w}, otter::ScalarType::Float);
    float* rows0 = rowsbuf0.data_ptr<float>();
    float* rows1 = rowsbuf1.data_ptr<float>();

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const float* S1 = src_a[sy + 1].data();

            const float* alphap = alpha;
            float* rows1p = rows1;
            int dx = 0;
#if __ARM_NEON
            for (; dx + 1 < w; dx += 2)
            {
                int sx = xofs[dx];
                int sxn = xofs[dx + 1];
                const float* S1p = S1 + sx;
                const float* S1np = S1 + sxn;

                float32x4_t _a = vld1q_f32(alphap);
                float32x2_t _S1 = vld1_f32(S1p);
                float32x2_t _S1n = vld1_f32(S1np);

                float32x4_t _S1S1n = vcombine_f32(_S1, _S1n);
                float32x4_t _ms1 = vmulq_f32(_S1S1n, _a);
                float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));

                vst1_f32(rows1p + dx, _rows1);

                alphap += 4;
            }
#endif // __ARM_NEON
            for (; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

                alphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const float* S0 = src_a[sy + 0].data();
            const float* S1 = src_a[sy + 1].data();

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            int dx = 0;
#if __ARM_NEON
            for (; dx + 1 < w; dx += 2)
            {
                int sx = xofs[dx];
                int sxn = xofs[dx + 1];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;
                const float* S0np = S0 + sxn;
                const float* S1np = S1 + sxn;

                float32x4_t _a = vld1q_f32(alphap);
                float32x2_t _S0 = vld1_f32(S0p);
                float32x2_t _S1 = vld1_f32(S1p);
                float32x2_t _S0n = vld1_f32(S0np);
                float32x2_t _S1n = vld1_f32(S1np);

                float32x4_t _S0S0n = vcombine_f32(_S0, _S0n);
                float32x4_t _S1S1n = vcombine_f32(_S1, _S1n);
                float32x4_t _ms0 = vmulq_f32(_S0S0n, _a);
                float32x4_t _ms1 = vmulq_f32(_S1S1n, _a);
                float32x2_t _rows0 = vpadd_f32(vget_low_f32(_ms0), vget_high_f32(_ms0));
                float32x2_t _rows1 = vpadd_f32(vget_low_f32(_ms1), vget_high_f32(_ms1));

                vst1_f32(rows0p + dx, _rows0);
                vst1_f32(rows1p + dx, _rows1);

                alphap += 4;
            }
#endif // __ARM_NEON
            for (; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows0p[dx] = S0p[0] * a0 + S0p[1] * a1;
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst_a[dy].data();

#if __ARM_NEON
        int nn = w >> 3;
#else
        int nn = 0;
#endif
        int remain = w - (nn << 3);

#if __ARM_NEON
        float32x4_t _b0 = vdupq_n_f32(b0);
        float32x4_t _b1 = vdupq_n_f32(b1);
        for (; nn > 0; nn--)
        {
            float32x4_t _rows0 = vld1q_f32(rows0p);
            float32x4_t _rows1 = vld1q_f32(rows1p);

            float32x4_t _D = vmulq_f32(_rows0, _b0);
            _D = vmlaq_f32(_D, _rows1, _b1);

            vst1q_f32(Dp, _D);

            float32x4_t _rows0n = vld1q_f32(rows0p + 4);
            float32x4_t _rows1n = vld1q_f32(rows1p + 4);

            float32x4_t _Dn = vmulq_f32(_rows0n, _b0);
            _Dn = vmlaq_f32(_Dn, _rows1n, _b1);

            vst1q_f32(Dp + 4, _Dn);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }
#endif // __ARM_NEON
        for (; remain; --remain)
        {
            //             D[x] = rows0[x]*b0 + rows1[x]*b1;
            *Dp++ = *rows0p++ * b0 + *rows1p++ * b1;
        }

        beta += 2;
    }
}

static void resize_bilinear_image_pack4(const Tensor& src, Tensor& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.size(1);
    int h = dst.size(0);
    
    auto src_a = src.accessor<float, 2, 4>();
    auto dst_a = dst.accessor<float, 2, 4>();

    // loop body
    Tensor rowsbuf0 = otter::empty({w}, otter::ScalarType::Float4);
    Tensor rowsbuf1 = otter::empty({w}, otter::ScalarType::Float4);
    float* rows0 = (float*)rowsbuf0.raw_data();
    float* rows1 = (float*)rowsbuf1.raw_data();

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const float* S1 = src_a[sy + 1].data();

            const float* alphap = alpha;
            float* rows1p = rows1;
            int dx = 0;
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const float* S1p = S1 + sx;

                float32x2_t _a01 = vld1_f32(alphap);

                float32x4_t _S10 = vld1q_f32(S1p);
                float32x4_t _S11 = vld1q_f32(S1p + 4);
                float32x4_t _rows1 = vmulq_lane_f32(_S10, _a01, 0);
                _rows1 = vmlaq_lane_f32(_rows1, _S11, _a01, 1);
                vst1q_f32(rows1p + dx * 4, _rows1);

                alphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const float* S0 = src_a[sy + 0].data();
            const float* S1 = src_a[sy + 1].data();

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            int dx = 0;
            for (; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                float32x2_t _a01 = vld1_f32(alphap);

                float32x4_t _S00 = vld1q_f32(S0p);
                float32x4_t _S01 = vld1q_f32(S0p + 4);
                float32x4_t _S10 = vld1q_f32(S1p);
                float32x4_t _S11 = vld1q_f32(S1p + 4);
                float32x4_t _rows0 = vmulq_lane_f32(_S00, _a01, 0);
                float32x4_t _rows1 = vmulq_lane_f32(_S10, _a01, 0);
                _rows0 = vmlaq_lane_f32(_rows0, _S01, _a01, 1);
                _rows1 = vmlaq_lane_f32(_rows1, _S11, _a01, 1);
                vst1q_f32(rows0p + dx * 4, _rows0);
                vst1q_f32(rows1p + dx * 4, _rows1);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        float32x2_t _b01 = vld1_f32(beta);

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst_a[dy].data();

        for (int dx = 0; dx < w; dx++)
        {
            float32x4_t _rows0 = vld1q_f32(rows0p);
            float32x4_t _rows1 = vld1q_f32(rows1p);
            float32x4_t _D = vmulq_lane_f32(_rows0, _b01, 0);
            _D = vmlaq_lane_f32(_D, _rows1, _b01, 1);
            vst1q_f32(Dp, _D);

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
        }

        beta += 2;
    }
}

Tensor interpolate_packed_neon(const Tensor& input, IntArrayRef size, InterpolateMode mode, bool align_corners) {
    
    Tensor output;
    
    int dims = input.dim();
    int outw = size[1];
    int outh = size[0];
    
    int elempack = input.elempack();
    
    if (dims == 1) {
        int w = input.size(0);
        
        output = otter::empty({w, outh, outw}, input.scalar_type());
        
        const float* in = (const float*)input.raw_data();
        
        if (elempack == 4) {
            otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    Tensor output_c = output[q];
                    float32x4_t _v = vld1q_f32((const float*)in + q * 4);
                    output_c.fill_(_v);
                }
            });
            
            return output;
        }
        
        otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
            for (const auto q : otter::irange(begin, end)) {
                Tensor output_c = output[q];
                const float v = in[q];
                output_c.fill_(v);
            }
        });
        
        return output;
    }
    
    if (dims == 2) {
        int h = input.size(0);
        int w = input.size(1);
        
        if (size[1] == w) {
            output = input;
            
            return output;
        }
        
        output = otter::empty({h, outw}, input.scalar_type());
        
        if (elempack == 4) {
            auto input_a = input.accessor<float, 2, 4>();
            auto output_a = output.accessor<float, 2, 4>();
            
            if (mode == InterpolateMode::NEAREST) {
                const float ws =  w / (float)outw;

                otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                    for (const auto y : otter::irange(begin, end)) {
                        const float* ptr = input_a[y].data();
                        float* outptr = output_a[y].data();
                        for (int x = 0; x < outw; x++) {
                            int in_x = std::min((int)(x * ws), (w - 1));

                            float32x4_t _p = vld1q_f32(ptr + in_x * 4);
                            vst1q_f32(outptr, _p);

                            outptr += 4;
                        }
                    }
                });
                
                return output;
            }
            
            if (mode == InterpolateMode::BILINEAR) {
                int* buf = new int[outw + outw * 2];

                int* xofs = buf;
                float* alpha = (float*)(buf + outw);

                linear_coeffs(w, outw, xofs, alpha, align_corners);

                otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                    for (const auto y : otter::irange(begin, end)) {
                        const float* ptr = input_a[y].data();
                        float* outptr = output_a[y].data();
                        const float* alphap = alpha;

                        for (int x = 0; x < outw; x++) {
                            int sx = xofs[x] * 4;
                            const float* Sp = ptr + sx;

                            float32x2_t _a01 = vld1_f32(alphap);

                            float32x4_t _S0 = vld1q_f32(Sp);
                            float32x4_t _S1 = vld1q_f32(Sp + 4);
                            float32x4_t _p = vmulq_lane_f32(_S0, _a01, 0);
                            _p = vmlaq_lane_f32(_p, _S1, _a01, 1);
                            vst1q_f32(outptr, _p);

                            alphap += 2;
                            outptr += 4;
                        }
                    }
                });

                delete[] buf;
                
                return output;
            }
        }
        
        auto input_a = input.accessor<float, 2>();
        auto output_a = output.accessor<float, 2>();
        
        if (mode == InterpolateMode::NEAREST) {
            const float ws =  w / (float)outw;

            otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                for (const auto y : otter::irange(begin, end)) {
                    const float* ptr = input_a[y].data();
                    float* outptr = output_a[y].data();
                    for (int x = 0; x < outw; x++) {
                        int in_x = std::min((int)(x * ws), (w - 1));
                        *outptr++ = ptr[in_x];
                    }
                }
            });
            
            return output;
        }

        if (mode == InterpolateMode::BILINEAR) {
            int* buf = new int[outw + outw * 2];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            linear_coeffs(w, outw, xofs, alpha, align_corners);

            otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                for (const auto y : otter::irange(begin, end)) {
                    const float* ptr = input_a[y].data();
                    float* outptr = output_a[y].data();
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++) {
                        int sx = xofs[x];
                        const float* Sp = ptr + sx;
                        float a0 = alphap[0];
                        float a1 = alphap[1];
                        *outptr++ = Sp[0] * a0 + Sp[1] * a1;
                        alphap += 2;
                    }
                }
            });

            delete[] buf;
            
            return output;
        }
    }
    
    if (dims == 3) {
        int w = input.size(2);
        int h = input.size(1);
        int channels = input.size(0);
        
        if (w == outw && h == outh) {
            output = input;
            
            return output;
        }
        
        output = otter::empty({channels, outh, outw}, input.scalar_type());
        
        if (elempack == 4) {
            auto input_a = input.accessor<float, 3, 4>();
            auto output_a = output.accessor<float, 3, 4>();
            
            if (mode == InterpolateMode::NEAREST) {
                const float hs =  h / (float)outh;
                const float ws = w / (float)outw;

                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const auto src = input_a[q];
                        auto dst = output_a[q];

                        for (int y = 0; y < outh; y++) {
                            int in_y = std::min((int)(y * hs), (h - 1));

                            const float* ptr = src[in_y].data();
                            float* outptr = dst[y].data();
                            for (int x = 0; x < outw; x++)
                            {
                                int in_x = std::min((int)(x * ws), (w - 1));

                                float32x4_t _p = vld1q_f32(ptr + in_x * 4);
                                vst1q_f32(outptr, _p);

                                outptr += 4;
                            }
                        }
                    }
                });
                
                return output;
            }

            if (mode == InterpolateMode::BILINEAR) {
                int* buf = new int[outw + outh + outw * 2 + outh * 2];

                int* xofs = buf;        //new int[outw];
                int* yofs = buf + outw; //new int[outh];

                float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
                float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

                linear_coeffs(w, outw, xofs, alpha, align_corners);
                linear_coeffs(h, outh, yofs, beta, align_corners);

                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const Tensor src = input[q];
                        Tensor dst = output[q];

                        resize_bilinear_image_pack4(src, dst, alpha, xofs, beta, yofs);
                    }
                });

                delete[] buf;
                
                return output;
            }
        }
        
        auto input_a = input.accessor<float, 3>();
        auto output_a = output.accessor<float, 3>();
        
        if (mode == InterpolateMode::NEAREST) {
            const float hs = h / (float)outh;
            const float ws = w / (float)outw;

            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const auto src = input_a[q];
                    auto dst = output_a[q];

                    for (int y = 0; y < outh; y++) {
                        int in_y = std::min((int)(y * hs), (h - 1));

                        const float* ptr = src[in_y].data();
                        float* outptr = dst[y].data();
                        for (int x = 0; x < outw; x++)
                        {
                            int in_x = std::min((int)(x * ws), (w - 1));
                            *outptr++ = ptr[in_x];
                        }
                    }
                }
            });
            
            return output;
        }

        if (mode == InterpolateMode::BILINEAR) {
            int* buf = new int[outw + outh + outw * 2 + outh * 2];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
            float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

            linear_coeffs(w, outw, xofs, alpha, align_corners);
            linear_coeffs(h, outh, yofs, beta, align_corners);

            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const Tensor src = input[q];
                    Tensor dst = output[q];

                    resize_bilinear_image(src, dst, alpha, xofs, beta, yofs);
                }
            });

            delete[] buf;
            
            return output;
        }
    }
    
    return output;
}
#endif  // __ARM_NEON__

Tensor interpolate_packed(const Tensor& input, IntArrayRef size, InterpolateMode mode, bool align_corners) {
#if __SSE2__
    return interpolate_packed_x86(input, size, mode, align_corners);
#elif __ARM_NEON__
    return interpolate_packed_neon(input, size, mode, align_corners);
#else
    int elempack = input.elempack();
    Tensor input_unpacked = input.packing(1);
    Interpolate(input_unpacked, size, {0, 0}, mode);
    return input_unpacked.packing(elempack);
#endif
}

Tensor Interpolate(const Tensor& input, IntArrayRef size, ArrayRef<double> scale_factor, InterpolateMode mode, bool align_corners) {
    
    if (scale_factor.empty()) {
        scale_factor = {0, 0};
    } else if (size.empty()) {
        size = {static_cast<long long>(input.size(2) * scale_factor[0]), static_cast<long long>(input.size(3) * scale_factor[1])};
    } else if (scale_factor.empty() && size.empty()) {
        OTTER_CHECK(false, "Invalid interpolation");
    }
    
    if (input.elempack() != 1) {
        // assume that the fist axis is batchsize
        if (input.dim() == 4 && input.size(0) == 1) {
            return interpolate_packed(input.squeeze(0), size, mode, align_corners).unsqueeze_(0);
        }
        return interpolate_packed(input, size, mode, align_corners);
    }
    
    switch (mode) {
        case InterpolateMode::NEAREST:
            return otter::native::upsample_nearest2d(input, size, scale_factor[0], scale_factor[1]);
            break;
        case InterpolateMode::BILINEAR:
            return otter::native::upsample_bilinear2d(input, size, align_corners, scale_factor[0], scale_factor[1]);
            break;
    }
    OTTER_CHECK(false, "Unsupport interpolation mode");
    return Tensor();
}

}
