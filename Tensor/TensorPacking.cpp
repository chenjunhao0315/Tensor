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

ScalarType get_update_scalarType(const ScalarType& src, int out_elempack) {
    constexpr auto sp1 = ScalarType::Byte;
    constexpr auto sp4 = ScalarType::Byte4;
    constexpr auto sp8 = ScalarType::Byte8;
    constexpr auto ip1 = ScalarType::Int;
    constexpr auto ip4 = ScalarType::Int4;
    constexpr auto ip8 = ScalarType::Int8;
    constexpr auto fp1 = ScalarType::Float;
    constexpr auto fp4 = ScalarType::Float4;
    constexpr auto fp8 = ScalarType::Float8;
    
    constexpr auto iu1 = ScalarType::Char;
    constexpr auto iu2 = ScalarType::Short;
    constexpr auto iu8 = ScalarType::Long;
    constexpr auto fu8 = ScalarType::Double;
    constexpr auto bu1 = ScalarType::Bool;

    static constexpr ScalarType _promoteTypesLookup[static_cast<int>(
        ScalarType::NumOptions)][static_cast<int>(ScalarType::NumOptions)] = {
        /*       sp1  iu1  iu2  ip1  iu8  fp1  fu8  bu1  sp4  ip4  fp4  sp8  ip8  fp8 */
        /* 0 */ {sp1, iu1, iu2, ip1, iu8, fp1, fu8, bu1, sp4, ip4, fp4, sp8, ip8, fp8},
        /* 1 */ {sp1, iu1, iu2, ip1, iu8, fp1, fu8, bu1, sp1, ip1, fp1, sp1, ip1, fp1},
        /* 2 */ {sp1, iu1, iu2, ip1, iu8, fp1, fu8, bu1, sp4, ip4, fp4, sp8, ip8, fp8},
        /* 3 */ {sp1, iu1, iu2, ip1, iu8, fp1, fu8, bu1, sp4, ip4, fp4, sp8, ip8, fp8},
        /* 4 */ {sp4, iu1, iu2, ip4, iu8, fp4, fu8, bu1, sp4, ip4, fp4, sp4, ip4, fp4},
        /* 5 */ {sp1, iu1, iu2, ip1, iu8, fp1, fu8, bu1, sp4, ip4, fp4, sp8, ip8, fp8},
        /* 6 */ {sp1, iu1, iu2, ip1, iu8, fp1, fu8, bu1, sp4, ip4, fp4, sp8, ip8, fp8},
        /* 7 */ {sp1, iu1, iu2, ip1, iu8, fp1, fu8, bu1, sp4, ip4, fp4, sp8, ip8, fp8},
        /* 8 */ {sp8, iu1, iu2, ip8, iu8, fp8, fu8, bu1, sp8, ip8, fp8, sp8, ip8, fp8},
    };
    return _promoteTypesLookup[static_cast<int>(out_elempack)][static_cast<int>(src)];
}

int64_t get_elempack_from_type(const ScalarType& src) {
    static constexpr int64_t _elempackLookup[static_cast<int>(
        ScalarType::NumOptions)] =
    /*       sp1  iu1  iu2  ip1  iu8  fp1  fu8  bu1  sp4  ip4  fp4  sp8  ip8  fp8 */
    /* 0 */ {  1,   1,   1,   1,   1,   1,   1,   1,   4,   4,   4,   8,   8,   8};
    return _elempackLookup[static_cast<int>(src)];
}

void check_convert_packing(const Tensor& src, int elempack) {
    auto dtype = src.scalar_type();
    OTTER_CHECK(dtype == ScalarType::Float || dtype == ScalarType::Float4 || dtype == ScalarType::Float8 || dtype == ScalarType::Byte || dtype == ScalarType::Byte4 || dtype == ScalarType::Byte8 || dtype == ScalarType::Int || dtype == ScalarType::Int4 || dtype == ScalarType::Int8, "Only support Float and Byte!");
    OTTER_CHECK(elempack == 1 || elempack == 4 || elempack == 8, "Only support elempack = 1, 4, 8 but get ", elempack);
    OTTER_CHECK(src.dim() <= 4, "Only support dim <= 4 but get ", src.dim());
}

void convertPacking(const Tensor& src, Tensor& dst, int out_elempack) {
    int elempack = src.elempack();
    
    if (elempack == out_elempack) {
        dst = src;
        
        return;
    }
    
    check_convert_packing(src, out_elempack);
    
#if __SSE2__
    return convertPackingX86(src.contiguous(), dst, out_elempack);
#elif __ARM_NEON__
    return convertPackingNeon(src.contiguous(), dst, out_elempack);
#endif
    return convertPackingNative(src.contiguous(), dst, out_elempack);

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
        
        dst = src.clone();
        dst.unsafeGetTensorNucleus()->force_set_sizes_and_dtype({outw}, out_dtype);
        
        return;
    }
    
    if (dim == 2) {
        int64_t w = src.size(1);
        int64_t h = src.size(0);
        
        int64_t outh = h * elempack / out_elempack;

        dst = otter::empty({outh, w}, out_dtype);

        if (pack1to4) {
            auto src_a = src.accessor<float, 2>();
            auto dst_a = dst.accessor<float, 2, 4>();
            
            otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src_a[i * 4 + 0].data();
                    const float* r1 = (const float*)src_a[i * 4 + 1].data();
                    const float* r2 = (const float*)src_a[i * 4 + 2].data();
                    const float* r3 = (const float*)src_a[i * 4 + 3].data();

                    float* outptr = (float*)dst_a[i].data();

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
            auto src_a = src.accessor<float, 2, 4>();
            auto dst_a = dst.accessor<float, 2>();
            
            otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src_a[i].data();

                    float* outptr0 = (float*)dst_a[i * 4 + 0].data();
                    float* outptr1 = (float*)dst_a[i * 4 + 1].data();
                    float* outptr2 = (float*)dst_a[i * 4 + 2].data();
                    float* outptr3 = (float*)dst_a[i * 4 + 3].data();

                    int j = 0;
    #if __ARM_NEON
                    for (; j + 3 < w; j += 4) {
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
                    for (; j < w; j++) {
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
            auto src_a = src.accessor<float, 3>();
            auto dst_a = dst.accessor<float, 3, 4>();
            
            otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src_a[q * 4 + 0].data();
                    const float* r1 = (const float*)src_a[q * 4 + 1].data();
                    const float* r2 = (const float*)src_a[q * 4 + 2].data();
                    const float* r3 = (const float*)src_a[q * 4 + 3].data();

                    float* outptr = (float*)dst_a[q].data();

                    int i = 0;
    #if __ARM_NEON
                    for (; i + 3 < size; i += 4) {
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
            auto src_a = src.accessor<float, 3, 4>();
            auto dst_a = dst.accessor<float, 3>();
            
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src_a[q].data();

                    float* outptr0 = (float*)dst_a[q * 4 + 0].data();
                    float* outptr1 = (float*)dst_a[q * 4 + 1].data();
                    float* outptr2 = (float*)dst_a[q * 4 + 2].data();
                    float* outptr3 = (float*)dst_a[q * 4 + 3].data();

                    int i = 0;
    #if __ARM_NEON
                    for (; i + 3 < size; i += 4) {
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
                    for (; i < size; i++) {
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
            auto src_a = src.accessor<float, 4>();
            auto dst_a = dst.accessor<float, 4, 4>();
            
            for (const auto b : otter::irange(0, batchsize)) {
                otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* r0 = (const float*)src_a[b][q * 4 + 0].data();
                        const float* r1 = (const float*)src_a[b][q * 4 + 1].data();
                        const float* r2 = (const float*)src_a[b][q * 4 + 2].data();
                        const float* r3 = (const float*)src_a[b][q * 4 + 3].data();

                        float* outptr = (float*)dst_a[b][q].data();

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
            auto src_a = src.accessor<float, 4, 4>();
            auto dst_a = dst.accessor<float, 4>();
            
            for (const auto b : otter::irange(0, batchsize)) {
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* r0 = (const float*)src_a[b][q].data();

                        float* outptr0 = (float*)dst_a[b][q * 4 + 0].data();
                        float* outptr1 = (float*)dst_a[b][q * 4 + 1].data();
                        float* outptr2 = (float*)dst_a[b][q * 4 + 2].data();
                        float* outptr3 = (float*)dst_a[b][q * 4 + 3].data();

                        int i = 0;
        #if __ARM_NEON
                        for (; i + 3 < size; i += 4) {
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
                        for (; i < size; i++) {
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
    
    if (out_dtype == ScalarType::Byte || out_dtype == ScalarType::Byte4 || out_dtype == ScalarType::Byte8 || out_dtype == ScalarType::Int || out_dtype == ScalarType::Int4 || out_dtype == ScalarType::Int8) {
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
        
        dst = src.clone();
        dst.unsafeGetTensorNucleus()->force_set_sizes_and_dtype({outw}, out_dtype);
        
        return;
    }
    
    if (dim == 2) {
        int64_t w = src.size(1);
        int64_t h = src.size(0);
        
        int64_t outh = h * elempack / out_elempack;
        
        dst = otter::empty({outh, w}, out_dtype);
        
        if (pack1to4) {
            auto src_a = src.accessor<float, 2>();
            auto dst_a = dst.accessor<float, 2, 4>();
            
            otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src_a[i * 4 + 0].data();
                    const float* r1 = (const float*)src_a[i * 4 + 1].data();
                    const float* r2 = (const float*)src_a[i * 4 + 2].data();
                    const float* r3 = (const float*)src_a[i * 4 + 3].data();

                    float* outptr = (float*)dst_a[i].data();

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
            auto src_a = src.accessor<float, 2, 4>();
            auto dst_a = dst.accessor<float, 2>();
            
            otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src_a[i].data();

                    float* outptr0 = (float*)dst_a[i * 4 + 0].data();
                    float* outptr1 = (float*)dst_a[i * 4 + 1].data();
                    float* outptr2 = (float*)dst_a[i * 4 + 2].data();
                    float* outptr3 = (float*)dst_a[i * 4 + 3].data();

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
            auto src_a = src.accessor<float, 2>();
            auto dst_a = dst.accessor<float, 2, 8>();
            
            otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src_a[i * 8 + 0].data();
                    const float* r1 = (const float*)src_a[i * 8 + 1].data();
                    const float* r2 = (const float*)src_a[i * 8 + 2].data();
                    const float* r3 = (const float*)src_a[i * 8 + 3].data();
                    const float* r4 = (const float*)src_a[i * 8 + 4].data();
                    const float* r5 = (const float*)src_a[i * 8 + 5].data();
                    const float* r6 = (const float*)src_a[i * 8 + 6].data();
                    const float* r7 = (const float*)src_a[i * 8 + 7].data();

                    float* outptr = (float*)dst_a[i].data();

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
            auto src_a = src.accessor<float, 2, 8>();
            auto dst_a = dst.accessor<float, 2>();
            
            otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src_a[i].data();

                    float* outptr0 = (float*)dst_a[i * 8 + 0].data();
                    float* outptr1 = (float*)dst_a[i * 8 + 1].data();
                    float* outptr2 = (float*)dst_a[i * 8 + 2].data();
                    float* outptr3 = (float*)dst_a[i * 8 + 3].data();
                    float* outptr4 = (float*)dst_a[i * 8 + 4].data();
                    float* outptr5 = (float*)dst_a[i * 8 + 5].data();
                    float* outptr6 = (float*)dst_a[i * 8 + 6].data();
                    float* outptr7 = (float*)dst_a[i * 8 + 7].data();

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
            auto src_a = src.accessor<float, 2, 4>();
            auto dst_a = dst.accessor<float, 2, 8>();
            
            otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src_a[i * 2 + 0].data();
                    const float* r1 = (const float*)src_a[i * 2 + 1].data();

                    float* outptr = (float*)dst_a[i].data();

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
            auto src_a = src.accessor<float, 2, 8>();
            auto dst_a = dst.accessor<float, 2, 4>();
            
            otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end))
                {
                    const float* r0 = (const float*)src_a[i].data();

                    float* outptr0 = (float*)dst_a[i * 2 + 0].data();
                    float* outptr1 = (float*)dst_a[i * 2 + 1].data();

                    for (int j = 0; j < w; j++) {
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
            auto src_a = src.accessor<float, 3>();
            auto dst_a = dst.accessor<float, 3, 4>();
            
            otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src_a[q * 4 + 0].data();
                    const float* r1 = (const float*)src_a[q * 4 + 1].data();
                    const float* r2 = (const float*)src_a[q * 4 + 2].data();
                    const float* r3 = (const float*)src_a[q * 4 + 3].data();

                    float* outptr = (float*)dst_a[q].data();

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
            auto src_a = src.accessor<float, 3, 4>();
            auto dst_a = dst.accessor<float, 3>();
            
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src_a[q].data();

                    float* outptr0 = (float*)dst_a[q * 4 + 0].data();
                    float* outptr1 = (float*)dst_a[q * 4 + 1].data();
                    float* outptr2 = (float*)dst_a[q * 4 + 2].data();
                    float* outptr3 = (float*)dst_a[q * 4 + 3].data();

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
                    for (; i < size; i++) {
                        *outptr0++ = r0[0];
                        *outptr1++ = r0[1];
                        *outptr2++ = r0[2];
                        *outptr3++ = r0[3];

                        r0 += 4;
                    }
                }
            });
        } else if (pack1to8) {
            auto src_a = src.accessor<float, 3>();
            auto dst_a = dst.accessor<float, 3, 8>();
            
            otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end))
                {
                    const float* r0 = (const float*)src_a[q * 8 + 0].data();
                    const float* r1 = (const float*)src_a[q * 8 + 1].data();
                    const float* r2 = (const float*)src_a[q * 8 + 2].data();
                    const float* r3 = (const float*)src_a[q * 8 + 3].data();
                    const float* r4 = (const float*)src_a[q * 8 + 4].data();
                    const float* r5 = (const float*)src_a[q * 8 + 5].data();
                    const float* r6 = (const float*)src_a[q * 8 + 6].data();
                    const float* r7 = (const float*)src_a[q * 8 + 7].data();

                    float* outptr = (float*)dst_a[q].data();

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
            auto src_a = src.accessor<float, 3, 8>();
            auto dst_a = dst.accessor<float, 3>();
            
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end))
                {
                    const float* r0 = (const float*)src_a[q].data();

                    float* outptr0 = (float*)dst_a[q * 8 + 0].data();
                    float* outptr1 = (float*)dst_a[q * 8 + 1].data();
                    float* outptr2 = (float*)dst_a[q * 8 + 2].data();
                    float* outptr3 = (float*)dst_a[q * 8 + 3].data();
                    float* outptr4 = (float*)dst_a[q * 8 + 4].data();
                    float* outptr5 = (float*)dst_a[q * 8 + 5].data();
                    float* outptr6 = (float*)dst_a[q * 8 + 6].data();
                    float* outptr7 = (float*)dst_a[q * 8 + 7].data();

                    int i = 0;
    #if __AVX__
                    for (; i + 7 < size; i += 8) {
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
            auto src_a = src.accessor<float, 3, 4>();
            auto dst_a = dst.accessor<float, 3, 8>();
            
            otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src_a[q * 2 + 0].data();
                    const float* r1 = (const float*)src_a[q * 2 + 1].data();

                    float* outptr = (float*)dst_a[q].data();

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
            auto src_a = src.accessor<float, 3, 8>();
            auto dst_a = dst.accessor<float, 3, 4>();
            
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* r0 = (const float*)src_a[q].data();

                    float* outptr0 = (float*)dst_a[q * 2 + 0].data();
                    float* outptr1 = (float*)dst_a[q * 2 + 1].data();

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
            auto src_a = src.accessor<float, 4>();
            auto dst_a = dst.accessor<float, 4, 4>();
            
            for (const auto b : otter::irange(0, batchsize)) {
                otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* r0 = (const float*)src_a[b][q * 4 + 0].data();
                        const float* r1 = (const float*)src_a[b][q * 4 + 1].data();
                        const float* r2 = (const float*)src_a[b][q * 4 + 2].data();
                        const float* r3 = (const float*)src_a[b][q * 4 + 3].data();

                        float* outptr = (float*)dst_a[b][q].data();

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
            auto src_a = src.accessor<float, 4, 4>();
            auto dst_a = dst.accessor<float, 4>();
            
            for (const auto b : otter::irange(0, batchsize)) {
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* r0 = (const float*)src_a[b][q].data();

                        float* outptr0 = (float*)dst_a[b][q * 4 + 0].data();
                        float* outptr1 = (float*)dst_a[b][q * 4 + 1].data();
                        float* outptr2 = (float*)dst_a[b][q * 4 + 2].data();
                        float* outptr3 = (float*)dst_a[b][q * 4 + 3].data();

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
                        for (; i < size; i++) {
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
            auto src_a = src.accessor<float, 4>();
            auto dst_a = dst.accessor<float, 4, 8>();
            
            for (const auto b : otter::irange(0, batchsize)) {
                otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end))
                    {
                        const float* r0 = (const float*)src_a[b][q * 8 + 0].data();
                        const float* r1 = (const float*)src_a[b][q * 8 + 1].data();
                        const float* r2 = (const float*)src_a[b][q * 8 + 2].data();
                        const float* r3 = (const float*)src_a[b][q * 8 + 3].data();
                        const float* r4 = (const float*)src_a[b][q * 8 + 4].data();
                        const float* r5 = (const float*)src_a[b][q * 8 + 5].data();
                        const float* r6 = (const float*)src_a[b][q * 8 + 6].data();
                        const float* r7 = (const float*)src_a[b][q * 8 + 7].data();

                        float* outptr = (float*)dst_a[b][q].data();

                        int i = 0;
        #if __AVX__
                        for (; i + 7 < size; i += 8) {
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
                        for (; i < size; i++) {
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
            auto src_a = src.accessor<float, 4, 8>();
            auto dst_a = dst.accessor<float, 4>();
            
            for (const auto b : otter::irange(0, batchsize)) {
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end))
                    {
                        const float* r0 = (const float*)src_a[b][q].data();

                        float* outptr0 = (float*)dst_a[b][q * 8 + 0].data();
                        float* outptr1 = (float*)dst_a[b][q * 8 + 1].data();
                        float* outptr2 = (float*)dst_a[b][q * 8 + 2].data();
                        float* outptr3 = (float*)dst_a[b][q * 8 + 3].data();
                        float* outptr4 = (float*)dst_a[b][q * 8 + 4].data();
                        float* outptr5 = (float*)dst_a[b][q * 8 + 5].data();
                        float* outptr6 = (float*)dst_a[b][q * 8 + 6].data();
                        float* outptr7 = (float*)dst_a[b][q * 8 + 7].data();

                        int i = 0;
        #if __AVX__
                        for (; i + 7 < size; i += 8) {
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
            auto src_a = src.accessor<float, 4, 4>();
            auto dst_a = dst.accessor<float, 4, 8>();
            
            for (const auto b : otter::irange(0, batchsize)) {
                otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* r0 = (const float*)src_a[b][q * 2 + 0].data();
                        const float* r1 = (const float*)src_a[b][q * 2 + 1].data();

                        float* outptr = (float*)dst_a[b][q].data();

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
            auto src_a = src.accessor<float, 4, 8>();
            auto dst_a = dst.accessor<float, 4, 4>();
            
            for (const auto b : otter::irange(0, batchsize)) {
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const float* r0 = (const float*)src_a[b][q].data();

                        float* outptr0 = (float*)dst_a[b][q * 2 + 0].data();
                        float* outptr1 = (float*)dst_a[b][q * 2 + 1].data();

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
        unsigned char* src_ptr = (unsigned char*)src.raw_data();
        unsigned char* dst_ptr = (unsigned char*)dst.raw_data();
        
        otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
            for (const auto i : otter::irange(begin, end)) {
                unsigned char* outptr = dst_ptr + (size_t)i * w * out_elemsize;

                for (int j = 0; j < w; j++) {
                    unsigned char* out_elem_ptr = outptr + j * out_elemsize;

                    for (int k = 0; k < out_elempack; k++) {
                        int srcy = (i * out_elempack + k) / elempack;
                        if (srcy >= h)
                            break;

                        int srck = (i * out_elempack + k) % elempack;

                        const unsigned char* ptr = (const unsigned char*)src_ptr + (size_t)srcy * w * elemsize;
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
        
        auto src_ra = src.raw_accessor<unsigned char, 3>();
        auto dst_ra = dst.raw_accessor<unsigned char, 3>();
        
        otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
            for (const auto q : otter::irange(begin, end)) {
                auto out = dst_ra[q];

                for (int i = 0; i < h; i++) {
                    unsigned char* outptr = (unsigned char*)out.data() + (size_t)i * w * out_elemsize;

                    for (int j = 0; j < w; j++) {
                        unsigned char* out_elem_ptr = outptr + j * out_elemsize;

                        for (int k = 0; k < out_elempack; k++) {
                            int srcq = (q * out_elempack + k) / elempack;
                            if (srcq >= channels)
                                break;

                            int srck = (q * out_elempack + k) % elempack;

                            const auto m = src_ra[srcq];
                            const unsigned char* ptr = (const unsigned char*)m.data() + (size_t)i * w * elemsize;
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
        
        auto src_ra = src.raw_accessor<unsigned char, 4>();
        auto dst_ra = dst.raw_accessor<unsigned char, 4>();
        
        for (const auto b : otter::irange(0, src.size(0))) {
            otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    auto out = dst_ra[b][q];

                    for (int i = 0; i < h; i++) {
                        unsigned char* outptr = (unsigned char*)out.data() + (size_t)i * w * out_elemsize;

                        for (int j = 0; j < w; j++) {
                            unsigned char* out_elem_ptr = outptr + j * out_elemsize;

                            for (int k = 0; k < out_elempack; k++) {
                                int srcq = (q * out_elempack + k) / elempack;
                                if (srcq >= channels)
                                    break;

                                int srck = (q * out_elempack + k) % elempack;

                                const auto m = src_ra[b][srcq];
                                const unsigned char* ptr = (const unsigned char*)m.data() + (size_t)i * w * elemsize;
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
