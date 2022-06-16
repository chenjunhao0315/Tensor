//
//  TensorPacking.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/16.
//

#include "TensorPacking.hpp"
#include "TensorFactory.hpp"
#include "Parallel.hpp"

#include "VecIntrinsic.hpp"

namespace otter {

void convertPackingNative(const Tensor& src, Tensor& dst, int out_elempack);

void convertPackingNeon(const Tensor& src, Tensor& dst, int out_elempack);

void convertPackingX86(const Tensor& src, Tensor& dst, int out_elempack);

void check_convert_packing(const Tensor& src, int elempack) {
    OTTER_CHECK(src.scalar_type() == ScalarType::Float || src.scalar_type() == ScalarType::Float4 || src.scalar_type() == ScalarType::Float8 || src.scalar_type() == ScalarType::Byte || src.scalar_type() == ScalarType::Byte4 || src.scalar_type() == ScalarType::Byte8, "Only support Float and Byte!");
    OTTER_CHECK(elempack == 1 || elempack == 4 || elempack == 8, "Only support elempack = 1, 4, 8 but get ", elempack);
    OTTER_CHECK(src.dim() <= 4 && src.dim() != 1, "Only support 1 < dim <= 4 but get ", src.dim());
}

ScalarType get_update_scalarType(const ScalarType& src, int out_elempack) {
    if (src == ScalarType::Float) {
        if (out_elempack == 4) return ScalarType::Float4;
        else if (out_elempack == 8) return ScalarType::Float8;
    } else if (src == ScalarType::Byte) {
        if (out_elempack == 4) return ScalarType::Byte4;
        else if (out_elempack == 8) return ScalarType::Byte8;
    } else if (src == ScalarType::Float4) {
        if (out_elempack == 1) return ScalarType::Float;
        else if (out_elempack == 8) return ScalarType::Float8;
    } else if (src == ScalarType::Float8) {
        if (out_elempack == 1) return ScalarType::Float;
        else if (out_elempack == 4) return ScalarType::Float4;
    } else if (src == ScalarType::Byte4) {
        if (out_elempack == 1) return ScalarType::Byte;
        else if (out_elempack == 8) return ScalarType::Byte8;
    } else if (src == ScalarType::Byte8) {
        if (out_elempack == 1) return ScalarType::Byte;
        else if (out_elempack == 4) return ScalarType::Byte4;
    }
    return src;
}

void convertPacking(const Tensor& src, Tensor& dst, int out_elempack) {
    int elempack = src.elempack();
    
    if (elempack == out_elempack) {
        dst = src;
        
        return;
    }
    
    check_convert_packing(src, out_elempack);
    
#if __SSE2__
    convertPackingX86(src.contiguous(), dst, out_elempack);
#elif __ARM_NEON__
    convertPackingNeon(src.contiguous(), dst, out_elempack);
#else
    convertPackingNative(src.contiguous(), dst, out_elempack);
#endif
}

void convertPackingNeon(const Tensor& src, Tensor& dst, int out_elempack) {
    int64_t elempack = src.elempack();
    int64_t dim = src.dim();
        
    ScalarType out_dtype = get_update_scalarType(src.scalar_type(), out_elempack);
    
    if (out_dtype == ScalarType::Byte || out_dtype == ScalarType::Byte4 || out_dtype == ScalarType::Byte8) {
        convertPackingNative(src, dst, out_elempack);
        
        return;
    }
    
    bool pack1to4 = elempack == 1 && out_elempack == 4;
    bool pack4to1 = elempack == 4 && out_elempack == 1;
    
    if (!pack1to4 && !pack4to1) {
        convertPackingNative(src, dst, out_elempack);
        
        return;
    }
    
    if (dim == 1) {
        int64_t w = src.size(0);
        int64_t outw = w * elempack / out_elempack;
        
        dst = src;
        dst.unsafeGetTensorNucleus()->force_set_sizes_and_dtype({outw}, out_dtype);
        
        return;
    }
    
    if (dim == 2) {
        int64_t w = src.size(1);
        int64_t h = src.size(0);
        
        int64_t outh = h * elempack / out_elempack;

        dst = otter::empty({outh, w}, out_dtype);

        if (pack1to4) {
            otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src[i * 4 + 0].raw_data();
                    const float* r1 = (const float*)src[i * 4 + 1].raw_data();
                    const float* r2 = (const float*)src[i * 4 + 2].raw_data();
                    const float* r3 = (const float*)src[i * 4 + 3].raw_data();

                    float* outptr = (float*)dst[i].raw_data();

                    int j = 0;
    #if __ARM_NEON
                    for (; j + 3 < w; j += 4) {
                        float32x4x4_t _p;
                        _p.val[0] = vld1q_f32(r0);
                        _p.val[1] = vld1q_f32(r1);
                        _p.val[2] = vld1q_f32(r2);
                        _p.val[3] = vld1q_f32(r3);
                        vst4q_f32(outptr, _p);

                        r0 += 4;
                        r1 += 4;
                        r2 += 4;
                        r3 += 4;
                        outptr += 16;
                    }
    #endif
                    for (; j < w; j++)
                    {
                        outptr[0] = *r0++;
                        outptr[1] = *r1++;
                        outptr[2] = *r2++;
                        outptr[3] = *r3++;

                        outptr += 4;
                    }
                }
            });
        } else if (pack4to1) {
            otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end))
                {
                    const float* r0 = (const float*)src[i].raw_data();

                    float* outptr0 = (float*)dst[i * 4 + 0].raw_data();
                    float* outptr1 = (float*)dst[i * 4 + 1].raw_data();
                    float* outptr2 = (float*)dst[i * 4 + 2].raw_data();
                    float* outptr3 = (float*)dst[i * 4 + 3].raw_data();

                    int j = 0;
    #if __ARM_NEON
                    for (; j + 3 < w; j += 4)
                    {
                        float32x4x4_t _p = vld4q_f32(r0);
                        vst1q_f32(outptr0, _p.val[0]);
                        vst1q_f32(outptr1, _p.val[1]);
                        vst1q_f32(outptr2, _p.val[2]);
                        vst1q_f32(outptr3, _p.val[3]);

                        r0 += 16;
                        outptr0 += 4;
                        outptr1 += 4;
                        outptr2 += 4;
                        outptr3 += 4;
                    }
    #endif
                    for (; j < w; j++)
                    {
                        *outptr0++ = r0[0];
                        *outptr1++ = r0[1];
                        *outptr2++ = r0[2];
                        *outptr3++ = r0[3];

                        r0 += 4;
                    }
                }
            });
        }
        
        return;
    }
    
    if (dim == 3) {
        int64_t channels = src.size(0);
        int64_t h = src.size(1);
        int64_t w = src.size(2);
        
        int64_t size = w * h;
        int64_t outc = channels * elempack / out_elempack;
        
        dst = otter::empty({outc, h, w}, out_dtype);
        
        if (pack1to4) {
            otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src[q * 4 + 0].raw_data();
                    const float* r1 = (const float*)src[q * 4 + 1].raw_data();
                    const float* r2 = (const float*)src[q * 4 + 2].raw_data();
                    const float* r3 = (const float*)src[q * 4 + 3].raw_data();

                    float* outptr = (float*)dst[q].raw_data();

                    int i = 0;
    #if __ARM_NEON
                    for (; i + 3 < size; i += 4)
                    {
                        float32x4x4_t _p;
                        _p.val[0] = vld1q_f32(r0);
                        _p.val[1] = vld1q_f32(r1);
                        _p.val[2] = vld1q_f32(r2);
                        _p.val[3] = vld1q_f32(r3);
                        vst4q_f32(outptr, _p);

                        r0 += 4;
                        r1 += 4;
                        r2 += 4;
                        r3 += 4;
                        outptr += 16;
                    }
    #endif
                    for (; i < size; i++)
                    {
                        outptr[0] = *r0++;
                        outptr[1] = *r1++;
                        outptr[2] = *r2++;
                        outptr[3] = *r3++;

                        outptr += 4;
                    }
                }
            });
        } else if (pack4to1) {
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src[q].raw_data();

                    float* outptr0 = (float*)dst[q * 4 + 0].raw_data();
                    float* outptr1 = (float*)dst[q * 4 + 1].raw_data();
                    float* outptr2 = (float*)dst[q * 4 + 2].raw_data();
                    float* outptr3 = (float*)dst[q * 4 + 3].raw_data();

                    int i = 0;
    #if __ARM_NEON
                    for (; i + 3 < size; i += 4)
                    {
                        float32x4x4_t _p = vld4q_f32(r0);
                        vst1q_f32(outptr0, _p.val[0]);
                        vst1q_f32(outptr1, _p.val[1]);
                        vst1q_f32(outptr2, _p.val[2]);
                        vst1q_f32(outptr3, _p.val[3]);

                        r0 += 16;
                        outptr0 += 4;
                        outptr1 += 4;
                        outptr2 += 4;
                        outptr3 += 4;
                    }
    #endif
                    for (; i < size; i++)
                    {
                        *outptr0++ = r0[0];
                        *outptr1++ = r0[1];
                        *outptr2++ = r0[2];
                        *outptr3++ = r0[3];

                        r0 += 4;
                    }
                }
            });
        }
        
        return;
    }
    
    if (dim == 4) {
        int64_t batchsize = src.size(0);
        int64_t channels = src.size(1);
        int64_t h = src.size(2);
        int64_t w = src.size(3);
        
        int64_t size = w * h;
        int64_t outc = channels * elempack / out_elempack;
        
        dst = otter::empty({batchsize, outc, h, w}, out_dtype);
        
        if (pack1to4) {
            for (const auto b : otter::irange(0, batchsize)) {
                otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* r0 = (const float*)src[b][q * 4 + 0].raw_data();
                        const float* r1 = (const float*)src[b][q * 4 + 1].raw_data();
                        const float* r2 = (const float*)src[b][q * 4 + 2].raw_data();
                        const float* r3 = (const float*)src[b][q * 4 + 3].raw_data();

                        float* outptr = (float*)dst[b][q].raw_data();

                        int i = 0;
        #if __ARM_NEON
                        for (; i + 3 < size; i += 4)
                        {
                            float32x4x4_t _p;
                            _p.val[0] = vld1q_f32(r0);
                            _p.val[1] = vld1q_f32(r1);
                            _p.val[2] = vld1q_f32(r2);
                            _p.val[3] = vld1q_f32(r3);
                            vst4q_f32(outptr, _p);

                            r0 += 4;
                            r1 += 4;
                            r2 += 4;
                            r3 += 4;
                            outptr += 16;
                        }
        #endif
                        for (; i < size; i++)
                        {
                            outptr[0] = *r0++;
                            outptr[1] = *r1++;
                            outptr[2] = *r2++;
                            outptr[3] = *r3++;

                            outptr += 4;
                        }
                    }
                });
            }
        } else if (pack4to1) {
            for (const auto b : otter::irange(0, batchsize)) {
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* r0 = (const float*)src[b][q].raw_data();

                        float* outptr0 = (float*)dst[b][q * 4 + 0].raw_data();
                        float* outptr1 = (float*)dst[b][q * 4 + 1].raw_data();
                        float* outptr2 = (float*)dst[b][q * 4 + 2].raw_data();
                        float* outptr3 = (float*)dst[b][q * 4 + 3].raw_data();

                        int i = 0;
        #if __ARM_NEON
                        for (; i + 3 < size; i += 4)
                        {
                            float32x4x4_t _p = vld4q_f32(r0);
                            vst1q_f32(outptr0, _p.val[0]);
                            vst1q_f32(outptr1, _p.val[1]);
                            vst1q_f32(outptr2, _p.val[2]);
                            vst1q_f32(outptr3, _p.val[3]);

                            r0 += 16;
                            outptr0 += 4;
                            outptr1 += 4;
                            outptr2 += 4;
                            outptr3 += 4;
                        }
        #endif
                        for (; i < size; i++)
                        {
                            *outptr0++ = r0[0];
                            *outptr1++ = r0[1];
                            *outptr2++ = r0[2];
                            *outptr3++ = r0[3];

                            r0 += 4;
                        }
                    }
                });
            }
        }
        
        return;
    }
}

void convertPackingX86(const Tensor& src, Tensor& dst, int out_elempack) {
    int64_t elempack = src.elempack();
    int64_t dim = src.dim();
        
    ScalarType out_dtype = get_update_scalarType(src.scalar_type(), out_elempack);
    
    if (out_dtype == ScalarType::Byte || out_dtype == ScalarType::Byte4 || out_dtype == ScalarType::Byte8) {
        convertPackingNative(src, dst, out_elempack);
        
        return;
    }
    
    bool pack1to4 = elempack == 1 && out_elempack == 4;
    bool pack4to1 = elempack == 4 && out_elempack == 1;
    bool pack1to8 = elempack == 1 && out_elempack == 8;
    bool pack8to1 = elempack == 8 && out_elempack == 1;
    bool pack4to8 = elempack == 4 && out_elempack == 8;
    bool pack8to4 = elempack == 8 && out_elempack == 4;
    
    if (!pack1to4 && !pack4to1 && !pack1to8 && !pack8to1 && !pack4to8 && !pack8to4) {
        convertPackingNative(src, dst, out_elempack);
        
        return;
    }
    
    if (dim == 1) {
        int64_t w = src.size(0);
        int64_t outw = w * elempack / out_elempack;
        
        dst = src;
        dst.unsafeGetTensorNucleus()->force_set_sizes_and_dtype({outw}, out_dtype);
        
        return;
    }
    
    if (dim == 2) {
        int64_t w = src.size(1);
        int64_t h = src.size(0);
        
        int64_t outh = h * elempack / out_elempack;
        
        dst = otter::empty({outh, w}, out_dtype);
        
        if (pack1to4) {
            otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src[i * 4 + 0].raw_data();
                    const float* r1 = (const float*)src[i * 4 + 1].raw_data();
                    const float* r2 = (const float*)src[i * 4 + 2].raw_data();
                    const float* r3 = (const float*)src[i * 4 + 3].raw_data();

                    float* outptr = (float*)dst[i].raw_data();

                    int j = 0;
    #if __SSE2__
                    for (; j + 3 < w; j += 4) {
                        // transpose 4x4
                        __m128 _r0 = _mm_loadu_ps(r0);
                        __m128 _r1 = _mm_loadu_ps(r1);
                        __m128 _r2 = _mm_loadu_ps(r2);
                        __m128 _r3 = _mm_loadu_ps(r3);

                        _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

                        _mm_store_ps(outptr, _r0);
                        _mm_store_ps(outptr + 4, _r1);
                        _mm_store_ps(outptr + 4 * 2, _r2);
                        _mm_store_ps(outptr + 4 * 3, _r3);

                        r0 += 4;
                        r1 += 4;
                        r2 += 4;
                        r3 += 4;
                        outptr += 16;
                    }
    #endif // __SSE2__
                    for (; j < w; j++) {
                        outptr[0] = *r0++;
                        outptr[1] = *r1++;
                        outptr[2] = *r2++;
                        outptr[3] = *r3++;

                        outptr += 4;
                    }
                }
            });
        } else if (pack4to1) {
            otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src[i].raw_data();

                    float* outptr0 = (float*)dst[i * 4 + 0].raw_data();
                    float* outptr1 = (float*)dst[i * 4 + 1].raw_data();
                    float* outptr2 = (float*)dst[i * 4 + 2].raw_data();
                    float* outptr3 = (float*)dst[i * 4 + 3].raw_data();

                    int j = 0;
    #if __SSE2__
                    for (; j + 3 < w; j += 4)
                    {
                        // transpose 4x4
                        __m128 _r0 = _mm_load_ps(r0);
                        __m128 _r1 = _mm_load_ps(r0 + 4);
                        __m128 _r2 = _mm_load_ps(r0 + 4 * 2);
                        __m128 _r3 = _mm_load_ps(r0 + 4 * 3);

                        _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

                        _mm_storeu_ps(outptr0, _r0);
                        _mm_storeu_ps(outptr1, _r1);
                        _mm_storeu_ps(outptr2, _r2);
                        _mm_storeu_ps(outptr3, _r3);

                        r0 += 16;
                        outptr0 += 4;
                        outptr1 += 4;
                        outptr2 += 4;
                        outptr3 += 4;
                    }
    #endif // __SSE2__
                    for (; j < w; j++)
                    {
                        *outptr0++ = r0[0];
                        *outptr1++ = r0[1];
                        *outptr2++ = r0[2];
                        *outptr3++ = r0[3];

                        r0 += 4;
                    }
                }
            });
        } else if (pack1to8) {
            otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src[i * 8 + 0].raw_data();
                    const float* r1 = (const float*)src[i * 8 + 1].raw_data();
                    const float* r2 = (const float*)src[i * 8 + 2].raw_data();
                    const float* r3 = (const float*)src[i * 8 + 3].raw_data();
                    const float* r4 = (const float*)src[i * 8 + 4].raw_data();
                    const float* r5 = (const float*)src[i * 8 + 5].raw_data();
                    const float* r6 = (const float*)src[i * 8 + 6].raw_data();
                    const float* r7 = (const float*)src[i * 8 + 7].raw_data();

                    float* outptr = (float*)dst[i].raw_data();

                    int j = 0;
    #if __AVX__
                    for (; j + 7 < w; j += 8) {
                        __m256 _row0 = _mm256_loadu_ps(r0);
                        __m256 _row1 = _mm256_loadu_ps(r1);
                        __m256 _row2 = _mm256_loadu_ps(r2);
                        __m256 _row3 = _mm256_loadu_ps(r3);
                        __m256 _row4 = _mm256_loadu_ps(r4);
                        __m256 _row5 = _mm256_loadu_ps(r5);
                        __m256 _row6 = _mm256_loadu_ps(r6);
                        __m256 _row7 = _mm256_loadu_ps(r7);
                        transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
                        _mm256_storeu_ps(outptr, _row0);
                        _mm256_storeu_ps(outptr + 8, _row1);
                        _mm256_storeu_ps(outptr + 16, _row2);
                        _mm256_storeu_ps(outptr + 24, _row3);
                        _mm256_storeu_ps(outptr + 32, _row4);
                        _mm256_storeu_ps(outptr + 40, _row5);
                        _mm256_storeu_ps(outptr + 48, _row6);
                        _mm256_storeu_ps(outptr + 56, _row7);
                        r0 += 8;
                        r1 += 8;
                        r2 += 8;
                        r3 += 8;
                        r4 += 8;
                        r5 += 8;
                        r6 += 8;
                        r7 += 8;
                        outptr += 64;
                    }
    #endif // __AVX__
                    for (; j < w; j++) {
                        outptr[0] = *r0++;
                        outptr[1] = *r1++;
                        outptr[2] = *r2++;
                        outptr[3] = *r3++;
                        outptr[4] = *r4++;
                        outptr[5] = *r5++;
                        outptr[6] = *r6++;
                        outptr[7] = *r7++;

                        outptr += 8;
                    }
                }
            });
        } else if (pack8to1) {
            otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src[i].raw_data();

                    float* outptr0 = (float*)dst[i * 8 + 0].raw_data();
                    float* outptr1 = (float*)dst[i * 8 + 1].raw_data();
                    float* outptr2 = (float*)dst[i * 8 + 2].raw_data();
                    float* outptr3 = (float*)dst[i * 8 + 3].raw_data();
                    float* outptr4 = (float*)dst[i * 8 + 4].raw_data();
                    float* outptr5 = (float*)dst[i * 8 + 5].raw_data();
                    float* outptr6 = (float*)dst[i * 8 + 6].raw_data();
                    float* outptr7 = (float*)dst[i * 8 + 7].raw_data();

                    int j = 0;
    #if __AVX__
                    for (; j + 7 < w; j += 8) {
                        __m256 _row0 = _mm256_loadu_ps(r0);
                        __m256 _row1 = _mm256_loadu_ps(r0 + 8);
                        __m256 _row2 = _mm256_loadu_ps(r0 + 16);
                        __m256 _row3 = _mm256_loadu_ps(r0 + 24);
                        __m256 _row4 = _mm256_loadu_ps(r0 + 32);
                        __m256 _row5 = _mm256_loadu_ps(r0 + 40);
                        __m256 _row6 = _mm256_loadu_ps(r0 + 48);
                        __m256 _row7 = _mm256_loadu_ps(r0 + 56);
                        transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
                        _mm256_storeu_ps(outptr0, _row0);
                        _mm256_storeu_ps(outptr1, _row1);
                        _mm256_storeu_ps(outptr2, _row2);
                        _mm256_storeu_ps(outptr3, _row3);
                        _mm256_storeu_ps(outptr4, _row4);
                        _mm256_storeu_ps(outptr5, _row5);
                        _mm256_storeu_ps(outptr6, _row6);
                        _mm256_storeu_ps(outptr7, _row7);

                        r0 += 64;
                        outptr0 += 8;
                        outptr1 += 8;
                        outptr2 += 8;
                        outptr3 += 8;
                        outptr4 += 8;
                        outptr5 += 8;
                        outptr6 += 8;
                        outptr7 += 8;
                    }
    #endif // __AVX__
                    for (; j < w; j++) {
                        *outptr0++ = r0[0];
                        *outptr1++ = r0[1];
                        *outptr2++ = r0[2];
                        *outptr3++ = r0[3];
                        *outptr4++ = r0[4];
                        *outptr5++ = r0[5];
                        *outptr6++ = r0[6];
                        *outptr7++ = r0[7];

                        r0 += 8;
                    }
                }
            });
        } else if (pack4to8) {
            otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src[i * 2 + 0].raw_data();
                    const float* r1 = (const float*)src[i * 2 + 1].raw_data();

                    float* outptr = (float*)dst[i].raw_data();

                    for (int j = 0; j < w; j++) {
                        outptr[0] = r0[0];
                        outptr[1] = r0[1];
                        outptr[2] = r0[2];
                        outptr[3] = r0[3];
                        outptr[4] = r1[0];
                        outptr[5] = r1[1];
                        outptr[6] = r1[2];
                        outptr[7] = r1[3];

                        r0 += 4;
                        r1 += 4;
                        outptr += 8;
                    }
                }
            });
        } else if (pack8to4) {
            otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                for (int i = 0; i < h; i++)
                {
                    const float* r0 = (const float*)src[i].raw_data();

                    float* outptr0 = (float*)dst[i * 2 + 0].raw_data();
                    float* outptr1 = (float*)dst[i * 2 + 1].raw_data();

                    for (int j = 0; j < w; j++)
                    {
                        outptr0[0] = r0[0];
                        outptr0[1] = r0[1];
                        outptr0[2] = r0[2];
                        outptr0[3] = r0[3];
                        outptr1[0] = r0[4];
                        outptr1[1] = r0[5];
                        outptr1[2] = r0[6];
                        outptr1[3] = r0[7];

                        r0 += 8;
                        outptr0 += 4;
                        outptr1 += 4;
                    }
                }
            });
        }
        
        return;
    }
    
    if (dim == 3) {
        int64_t channels = src.size(0);
        int64_t h = src.size(1);
        int64_t w = src.size(2);
        
        int64_t size = w * h;
        int64_t outc = (channels * elempack + out_elempack - 1) / out_elempack;
        
        dst = otter::empty({outc, h, w}, out_dtype);
        
        if (pack1to4) {
            otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src[q * 4 + 0].raw_data();
                    const float* r1 = (const float*)src[q * 4 + 1].raw_data();
                    const float* r2 = (const float*)src[q * 4 + 2].raw_data();
                    const float* r3 = (const float*)src[q * 4 + 3].raw_data();

                    float* outptr = (float*)dst[q].raw_data();

                    int i = 0;
    #if __SSE2__
                    for (; i + 3 < size; i += 4) {
                        // transpose 4x4
                        __m128 _r0 = _mm_loadu_ps(r0);
                        __m128 _r1 = _mm_loadu_ps(r1);
                        __m128 _r2 = _mm_loadu_ps(r2);
                        __m128 _r3 = _mm_loadu_ps(r3);

                        _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

                        _mm_store_ps(outptr, _r0);
                        _mm_store_ps(outptr + 4, _r1);
                        _mm_store_ps(outptr + 4 * 2, _r2);
                        _mm_store_ps(outptr + 4 * 3, _r3);

                        r0 += 4;
                        r1 += 4;
                        r2 += 4;
                        r3 += 4;
                        outptr += 16;
                    }
    #endif // __SSE2__
                    for (; i < size; i++) {
                        outptr[0] = *r0++;
                        outptr[1] = *r1++;
                        outptr[2] = *r2++;
                        outptr[3] = *r3++;

                        outptr += 4;
                    }
                }
            });
        } else if (pack4to1) {
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src[q].raw_data();

                    float* outptr0 = (float*)dst[q * 4 + 0].raw_data();
                    float* outptr1 = (float*)dst[q * 4 + 1].raw_data();
                    float* outptr2 = (float*)dst[q * 4 + 2].raw_data();
                    float* outptr3 = (float*)dst[q * 4 + 3].raw_data();

                    int i = 0;
    #if __SSE2__
                    for (; i + 3 < size; i += 4) {
                        // transpose 4x4
                        __m128 _r0 = _mm_load_ps(r0);
                        __m128 _r1 = _mm_load_ps(r0 + 4);
                        __m128 _r2 = _mm_load_ps(r0 + 4 * 2);
                        __m128 _r3 = _mm_load_ps(r0 + 4 * 3);

                        _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

                        _mm_storeu_ps(outptr0, _r0);
                        _mm_storeu_ps(outptr1, _r1);
                        _mm_storeu_ps(outptr2, _r2);
                        _mm_storeu_ps(outptr3, _r3);

                        r0 += 16;
                        outptr0 += 4;
                        outptr1 += 4;
                        outptr2 += 4;
                        outptr3 += 4;
                    }
    #endif // __SSE2__
                    for (; i < size; i++)
                    {
                        *outptr0++ = r0[0];
                        *outptr1++ = r0[1];
                        *outptr2++ = r0[2];
                        *outptr3++ = r0[3];

                        r0 += 4;
                    }
                }
            });
        } else if (pack1to8) {
            otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end))
                {
                    const float* r0 = (const float*)src[q * 8 + 0].raw_data();
                    const float* r1 = (const float*)src[q * 8 + 1].raw_data();
                    const float* r2 = (const float*)src[q * 8 + 2].raw_data();
                    const float* r3 = (const float*)src[q * 8 + 3].raw_data();
                    const float* r4 = (const float*)src[q * 8 + 4].raw_data();
                    const float* r5 = (const float*)src[q * 8 + 5].raw_data();
                    const float* r6 = (const float*)src[q * 8 + 6].raw_data();
                    const float* r7 = (const float*)src[q * 8 + 7].raw_data();

                    float* outptr = (float*)dst[q].raw_data();

                    int i = 0;
    #if __AVX__
                    for (; i + 7 < size; i += 8)
                    {
                        __m256 _row0 = _mm256_loadu_ps(r0);
                        __m256 _row1 = _mm256_loadu_ps(r1);
                        __m256 _row2 = _mm256_loadu_ps(r2);
                        __m256 _row3 = _mm256_loadu_ps(r3);
                        __m256 _row4 = _mm256_loadu_ps(r4);
                        __m256 _row5 = _mm256_loadu_ps(r5);
                        __m256 _row6 = _mm256_loadu_ps(r6);
                        __m256 _row7 = _mm256_loadu_ps(r7);
                        transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
                        _mm256_storeu_ps(outptr, _row0);
                        _mm256_storeu_ps(outptr + 8, _row1);
                        _mm256_storeu_ps(outptr + 16, _row2);
                        _mm256_storeu_ps(outptr + 24, _row3);
                        _mm256_storeu_ps(outptr + 32, _row4);
                        _mm256_storeu_ps(outptr + 40, _row5);
                        _mm256_storeu_ps(outptr + 48, _row6);
                        _mm256_storeu_ps(outptr + 56, _row7);
                        r0 += 8;
                        r1 += 8;
                        r2 += 8;
                        r3 += 8;
                        r4 += 8;
                        r5 += 8;
                        r6 += 8;
                        r7 += 8;
                        outptr += 64;
                    }
    #endif // __AVX__
                    for (; i < size; i++)
                    {
                        outptr[0] = *r0++;
                        outptr[1] = *r1++;
                        outptr[2] = *r2++;
                        outptr[3] = *r3++;
                        outptr[4] = *r4++;
                        outptr[5] = *r5++;
                        outptr[6] = *r6++;
                        outptr[7] = *r7++;

                        outptr += 8;
                    }
                }
            });
        } else if (pack8to1) {
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end))
                {
                    const float* r0 = (const float*)src[q].raw_data();

                    float* outptr0 = (float*)dst[q * 8 + 0].raw_data();
                    float* outptr1 = (float*)dst[q * 8 + 1].raw_data();
                    float* outptr2 = (float*)dst[q * 8 + 2].raw_data();
                    float* outptr3 = (float*)dst[q * 8 + 3].raw_data();
                    float* outptr4 = (float*)dst[q * 8 + 4].raw_data();
                    float* outptr5 = (float*)dst[q * 8 + 5].raw_data();
                    float* outptr6 = (float*)dst[q * 8 + 6].raw_data();
                    float* outptr7 = (float*)dst[q * 8 + 7].raw_data();

                    int i = 0;
    #if __AVX__
                    for (; i + 7 < size; i += 8)
                    {
                        __m256 _row0 = _mm256_loadu_ps(r0);
                        __m256 _row1 = _mm256_loadu_ps(r0 + 8);
                        __m256 _row2 = _mm256_loadu_ps(r0 + 16);
                        __m256 _row3 = _mm256_loadu_ps(r0 + 24);
                        __m256 _row4 = _mm256_loadu_ps(r0 + 32);
                        __m256 _row5 = _mm256_loadu_ps(r0 + 40);
                        __m256 _row6 = _mm256_loadu_ps(r0 + 48);
                        __m256 _row7 = _mm256_loadu_ps(r0 + 56);
                        transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
                        _mm256_storeu_ps(outptr0, _row0);
                        _mm256_storeu_ps(outptr1, _row1);
                        _mm256_storeu_ps(outptr2, _row2);
                        _mm256_storeu_ps(outptr3, _row3);
                        _mm256_storeu_ps(outptr4, _row4);
                        _mm256_storeu_ps(outptr5, _row5);
                        _mm256_storeu_ps(outptr6, _row6);
                        _mm256_storeu_ps(outptr7, _row7);

                        r0 += 64;
                        outptr0 += 8;
                        outptr1 += 8;
                        outptr2 += 8;
                        outptr3 += 8;
                        outptr4 += 8;
                        outptr5 += 8;
                        outptr6 += 8;
                        outptr7 += 8;
                    }
    #endif // __AVX__
                    for (; i < size; i++) {
                        *outptr0++ = r0[0];
                        *outptr1++ = r0[1];
                        *outptr2++ = r0[2];
                        *outptr3++ = r0[3];
                        *outptr4++ = r0[4];
                        *outptr5++ = r0[5];
                        *outptr6++ = r0[6];
                        *outptr7++ = r0[7];

                        r0 += 8;
                    }
                }
            });
        } else if (pack4to8) {
            otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src[q * 2 + 0].raw_data();
                    const float* r1 = (const float*)src[q * 2 + 1].raw_data();

                    float* outptr = (float*)dst[q].raw_data();

                    for (int i = 0; i < size; i++) {
                        outptr[0] = r0[0];
                        outptr[1] = r0[1];
                        outptr[2] = r0[2];
                        outptr[3] = r0[3];
                        outptr[4] = r1[0];
                        outptr[5] = r1[1];
                        outptr[6] = r1[2];
                        outptr[7] = r1[3];

                        r0 += 4;
                        r1 += 4;
                        outptr += 8;
                    }
                }
            });
        } else if (pack8to4) {
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src[q].raw_data();

                    float* outptr0 = (float*)dst[q * 2 + 0].raw_data();
                    float* outptr1 = (float*)dst[q * 2 + 1].raw_data();

                    for (int i = 0; i < size; i++) {
                        outptr0[0] = r0[0];
                        outptr0[1] = r0[1];
                        outptr0[2] = r0[2];
                        outptr0[3] = r0[3];
                        outptr1[0] = r0[4];
                        outptr1[1] = r0[5];
                        outptr1[2] = r0[6];
                        outptr1[3] = r0[7];

                        r0 += 8;
                        outptr0 += 4;
                        outptr1 += 4;
                    }
                }
            });
        }
        
        return;
    }
    
    if (dim == 4) {
        int64_t batchsize = src.size(0);
        int64_t channels = src.size(1);
        int64_t h = src.size(2);
        int64_t w = src.size(3);
        
        int64_t size = w * h;
        int64_t outc = (channels * elempack + out_elempack - 1) / out_elempack;
        
        dst = otter::empty({batchsize, outc, h, w}, out_dtype);
        
        if (pack1to4) {
            for (const auto b : otter::irange(0, batchsize)) {
                otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* r0 = (const float*)src[b][q * 4 + 0].raw_data();
                        const float* r1 = (const float*)src[b][q * 4 + 1].raw_data();
                        const float* r2 = (const float*)src[b][q * 4 + 2].raw_data();
                        const float* r3 = (const float*)src[b][q * 4 + 3].raw_data();

                        float* outptr = (float*)dst[b][q].raw_data();

                        int i = 0;
        #if __SSE2__
                        for (; i + 3 < size; i += 4) {
                            // transpose 4x4
                            __m128 _r0 = _mm_loadu_ps(r0);
                            __m128 _r1 = _mm_loadu_ps(r1);
                            __m128 _r2 = _mm_loadu_ps(r2);
                            __m128 _r3 = _mm_loadu_ps(r3);

                            _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

                            _mm_store_ps(outptr, _r0);
                            _mm_store_ps(outptr + 4, _r1);
                            _mm_store_ps(outptr + 4 * 2, _r2);
                            _mm_store_ps(outptr + 4 * 3, _r3);

                            r0 += 4;
                            r1 += 4;
                            r2 += 4;
                            r3 += 4;
                            outptr += 16;
                        }
        #endif // __SSE2__
                        for (; i < size; i++) {
                            outptr[0] = *r0++;
                            outptr[1] = *r1++;
                            outptr[2] = *r2++;
                            outptr[3] = *r3++;

                            outptr += 4;
                        }
                    }
                });
            }
        } else if (pack4to1) {
            for (const auto b : otter::irange(0, batchsize)) {
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* r0 = (const float*)src[b][q].raw_data();

                        float* outptr0 = (float*)dst[b][q * 4 + 0].raw_data();
                        float* outptr1 = (float*)dst[b][q * 4 + 1].raw_data();
                        float* outptr2 = (float*)dst[b][q * 4 + 2].raw_data();
                        float* outptr3 = (float*)dst[b][q * 4 + 3].raw_data();

                        int i = 0;
        #if __SSE2__
                        for (; i + 3 < size; i += 4) {
                            // transpose 4x4
                            __m128 _r0 = _mm_load_ps(r0);
                            __m128 _r1 = _mm_load_ps(r0 + 4);
                            __m128 _r2 = _mm_load_ps(r0 + 4 * 2);
                            __m128 _r3 = _mm_load_ps(r0 + 4 * 3);

                            _MM_TRANSPOSE4_PS(_r0, _r1, _r2, _r3);

                            _mm_storeu_ps(outptr0, _r0);
                            _mm_storeu_ps(outptr1, _r1);
                            _mm_storeu_ps(outptr2, _r2);
                            _mm_storeu_ps(outptr3, _r3);

                            r0 += 16;
                            outptr0 += 4;
                            outptr1 += 4;
                            outptr2 += 4;
                            outptr3 += 4;
                        }
        #endif // __SSE2__
                        for (; i < size; i++)
                        {
                            *outptr0++ = r0[0];
                            *outptr1++ = r0[1];
                            *outptr2++ = r0[2];
                            *outptr3++ = r0[3];

                            r0 += 4;
                        }
                    }
                });
            }
        } else if (pack1to8) {
            for (const auto b : otter::irange(0, batchsize)) {
                otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end))
                    {
                        const float* r0 = (const float*)src[b][q * 8 + 0].raw_data();
                        const float* r1 = (const float*)src[b][q * 8 + 1].raw_data();
                        const float* r2 = (const float*)src[b][q * 8 + 2].raw_data();
                        const float* r3 = (const float*)src[b][q * 8 + 3].raw_data();
                        const float* r4 = (const float*)src[b][q * 8 + 4].raw_data();
                        const float* r5 = (const float*)src[b][q * 8 + 5].raw_data();
                        const float* r6 = (const float*)src[b][q * 8 + 6].raw_data();
                        const float* r7 = (const float*)src[b][q * 8 + 7].raw_data();

                        float* outptr = (float*)dst[b][q].raw_data();

                        int i = 0;
        #if __AVX__
                        for (; i + 7 < size; i += 8)
                        {
                            __m256 _row0 = _mm256_loadu_ps(r0);
                            __m256 _row1 = _mm256_loadu_ps(r1);
                            __m256 _row2 = _mm256_loadu_ps(r2);
                            __m256 _row3 = _mm256_loadu_ps(r3);
                            __m256 _row4 = _mm256_loadu_ps(r4);
                            __m256 _row5 = _mm256_loadu_ps(r5);
                            __m256 _row6 = _mm256_loadu_ps(r6);
                            __m256 _row7 = _mm256_loadu_ps(r7);
                            transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
                            _mm256_storeu_ps(outptr, _row0);
                            _mm256_storeu_ps(outptr + 8, _row1);
                            _mm256_storeu_ps(outptr + 16, _row2);
                            _mm256_storeu_ps(outptr + 24, _row3);
                            _mm256_storeu_ps(outptr + 32, _row4);
                            _mm256_storeu_ps(outptr + 40, _row5);
                            _mm256_storeu_ps(outptr + 48, _row6);
                            _mm256_storeu_ps(outptr + 56, _row7);
                            r0 += 8;
                            r1 += 8;
                            r2 += 8;
                            r3 += 8;
                            r4 += 8;
                            r5 += 8;
                            r6 += 8;
                            r7 += 8;
                            outptr += 64;
                        }
        #endif // __AVX__
                        for (; i < size; i++)
                        {
                            outptr[0] = *r0++;
                            outptr[1] = *r1++;
                            outptr[2] = *r2++;
                            outptr[3] = *r3++;
                            outptr[4] = *r4++;
                            outptr[5] = *r5++;
                            outptr[6] = *r6++;
                            outptr[7] = *r7++;

                            outptr += 8;
                        }
                    }
                });
            }
        } else if (pack8to1) {
            for (const auto b : otter::irange(0, batchsize)) {
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end))
                    {
                        const float* r0 = (const float*)src[b][q].raw_data();

                        float* outptr0 = (float*)dst[b][q * 8 + 0].raw_data();
                        float* outptr1 = (float*)dst[b][q * 8 + 1].raw_data();
                        float* outptr2 = (float*)dst[b][q * 8 + 2].raw_data();
                        float* outptr3 = (float*)dst[b][q * 8 + 3].raw_data();
                        float* outptr4 = (float*)dst[b][q * 8 + 4].raw_data();
                        float* outptr5 = (float*)dst[b][q * 8 + 5].raw_data();
                        float* outptr6 = (float*)dst[b][q * 8 + 6].raw_data();
                        float* outptr7 = (float*)dst[b][q * 8 + 7].raw_data();

                        int i = 0;
        #if __AVX__
                        for (; i + 7 < size; i += 8)
                        {
                            __m256 _row0 = _mm256_loadu_ps(r0);
                            __m256 _row1 = _mm256_loadu_ps(r0 + 8);
                            __m256 _row2 = _mm256_loadu_ps(r0 + 16);
                            __m256 _row3 = _mm256_loadu_ps(r0 + 24);
                            __m256 _row4 = _mm256_loadu_ps(r0 + 32);
                            __m256 _row5 = _mm256_loadu_ps(r0 + 40);
                            __m256 _row6 = _mm256_loadu_ps(r0 + 48);
                            __m256 _row7 = _mm256_loadu_ps(r0 + 56);
                            transpose8_ps(_row0, _row1, _row2, _row3, _row4, _row5, _row6, _row7);
                            _mm256_storeu_ps(outptr0, _row0);
                            _mm256_storeu_ps(outptr1, _row1);
                            _mm256_storeu_ps(outptr2, _row2);
                            _mm256_storeu_ps(outptr3, _row3);
                            _mm256_storeu_ps(outptr4, _row4);
                            _mm256_storeu_ps(outptr5, _row5);
                            _mm256_storeu_ps(outptr6, _row6);
                            _mm256_storeu_ps(outptr7, _row7);

                            r0 += 64;
                            outptr0 += 8;
                            outptr1 += 8;
                            outptr2 += 8;
                            outptr3 += 8;
                            outptr4 += 8;
                            outptr5 += 8;
                            outptr6 += 8;
                            outptr7 += 8;
                        }
        #endif // __AVX__
                        for (; i < size; i++) {
                            *outptr0++ = r0[0];
                            *outptr1++ = r0[1];
                            *outptr2++ = r0[2];
                            *outptr3++ = r0[3];
                            *outptr4++ = r0[4];
                            *outptr5++ = r0[5];
                            *outptr6++ = r0[6];
                            *outptr7++ = r0[7];

                            r0 += 8;
                        }
                    }
                });
            }
        } else if (pack4to8) {
            for (const auto b : otter::irange(0, batchsize)) {
                otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* r0 = (const float*)src[b][q * 2 + 0].raw_data();
                        const float* r1 = (const float*)src[b][q * 2 + 1].raw_data();

                        float* outptr = (float*)dst[b][q].raw_data();

                        for (int i = 0; i < size; i++) {
                            outptr[0] = r0[0];
                            outptr[1] = r0[1];
                            outptr[2] = r0[2];
                            outptr[3] = r0[3];
                            outptr[4] = r1[0];
                            outptr[5] = r1[1];
                            outptr[6] = r1[2];
                            outptr[7] = r1[3];

                            r0 += 4;
                            r1 += 4;
                            outptr += 8;
                        }
                    }
                });
            }
        } else if (pack8to4) {
            for (const auto b : otter::irange(0, batchsize)) {
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* r0 = (const float*)src[b][q].raw_data();

                        float* outptr0 = (float*)dst[b][q * 2 + 0].raw_data();
                        float* outptr1 = (float*)dst[b][q * 2 + 1].raw_data();

                        for (int i = 0; i < size; i++) {
                            outptr0[0] = r0[0];
                            outptr0[1] = r0[1];
                            outptr0[2] = r0[2];
                            outptr0[3] = r0[3];
                            outptr1[0] = r0[4];
                            outptr1[1] = r0[5];
                            outptr1[2] = r0[6];
                            outptr1[3] = r0[7];

                            r0 += 8;
                            outptr0 += 4;
                            outptr1 += 4;
                        }
                    }
                });
            }
            
            return;
        }
        
        return;
    }
    
}

void convertPackingNative(const Tensor& src, Tensor& dst, int out_elempack) {
    int64_t elemsize = src.itemsize();
    int64_t elempack = src.elempack();
    int64_t dim = src.dim();
    
    ScalarType out_dtype = get_update_scalarType(src.scalar_type(), out_elempack);
    
    if (dim == 1) {
        int64_t w = src.size(0);
        
        if (out_elempack == 1) {
            dst = src;
            dst.unsafeGetTensorNucleus()->force_set_sizes_and_dtype({w * elempack}, out_dtype);
            return;
        }
        
        int64_t out_w = (w * elempack + out_elempack - 1) / out_elempack;
        dst = empty({out_w}, out_dtype);
        
        memcpy(dst.raw_data(), src.raw_data(), w * elemsize);
        
        return;
    }
    
    if (dim == 2) {
        int64_t w = src.size(1);
        int64_t h = src.size(0);
        
        int64_t outh = (h * elempack + out_elempack - 1) / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;
        size_t lane_size = out_elemsize / out_elempack;
        
        dst = otter::empty({outh, w}, out_dtype);
        
        otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
            for (const auto i : otter::irange(begin, end)) {
                unsigned char* outptr = (unsigned char*)dst.raw_data() + (size_t)i * w * out_elemsize;

                for (int j = 0; j < w; j++) {
                    unsigned char* out_elem_ptr = outptr + j * out_elemsize;

                    for (int k = 0; k < out_elempack; k++) {
                        int srcy = (i * out_elempack + k) / elempack;
                        if (srcy >= h)
                            break;

                        int srck = (i * out_elempack + k) % elempack;

                        const unsigned char* ptr = (const unsigned char*)src.raw_data() + (size_t)srcy * w * elemsize;
                        const unsigned char* elem_ptr = ptr + j * elemsize;

                        memcpy(out_elem_ptr + k * lane_size, elem_ptr + srck * lane_size, lane_size);
                    }
                }
            }
        });
        
        return;
    }
    
    if (dim == 3) {
        int64_t channels = src.size(0);
        int64_t h = src.size(1);
        int64_t w = src.size(2);
        
        int64_t outc = (channels * elempack + out_elempack - 1) / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;
        size_t lane_size = out_elemsize / out_elempack;
        
        dst = otter::empty({outc, h, w}, out_dtype);
        
        otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
            for (const auto q : otter::irange(begin, end)) {
                auto out = dst[q];

                for (int i = 0; i < h; i++) {
                    unsigned char* outptr = (unsigned char*)out.raw_data() + (size_t)i * w * out_elemsize;

                    for (int j = 0; j < w; j++) {
                        unsigned char* out_elem_ptr = outptr + j * out_elemsize;

                        for (int k = 0; k < out_elempack; k++) {
                            int srcq = (q * out_elempack + k) / elempack;
                            if (srcq >= channels)
                                break;

                            int srck = (q * out_elempack + k) % elempack;

                            const auto m = src[srcq];
                            const unsigned char* ptr = (const unsigned char*)m.raw_data() + (size_t)i * w * elemsize;
                            const unsigned char* elem_ptr = ptr + j * elemsize;

                            memcpy(out_elem_ptr + k * lane_size, elem_ptr + srck * lane_size, lane_size);
                        }
                    }
                }
            }
        });
        
        return;
    }
    
    if (dim == 4) {
        int64_t channels = src.size(1);
        int64_t h = src.size(2);
        int64_t w = src.size(3);
        
        int64_t outc = (channels * elempack + out_elempack - 1) / out_elempack;
        size_t out_elemsize = elemsize / elempack * out_elempack;
        size_t lane_size = out_elemsize / out_elempack;
        
        dst = otter::empty({src.size(0), outc, h, w}, out_dtype);
        
        for (const auto b : otter::irange(0, src.size(0))) {
            otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    auto out = dst[b][q];

                    for (int i = 0; i < h; i++) {
                        unsigned char* outptr = (unsigned char*)out.raw_data() + (size_t)i * w * out_elemsize;

                        for (int j = 0; j < w; j++) {
                            unsigned char* out_elem_ptr = outptr + j * out_elemsize;

                            for (int k = 0; k < out_elempack; k++) {
                                int srcq = (q * out_elempack + k) / elempack;
                                if (srcq >= channels)
                                    break;

                                int srck = (q * out_elempack + k) % elempack;

                                const auto m = src[b][srcq];
                                const unsigned char* ptr = (const unsigned char*)m.raw_data() + (size_t)i * w * elemsize;
                                const unsigned char* elem_ptr = ptr + j * elemsize;

                                memcpy(out_elem_ptr + k * lane_size, elem_ptr + srck * lane_size, lane_size);
                            }
                        }
                    }
                }
            });
        }
        
        return;
    }
}

}   // end namespace otter
