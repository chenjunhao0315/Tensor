//
//  QuantizeX86.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/6.
//

#if __SSE2__

#include "QuantizeX86.hpp"
#include "TensorFactory.hpp"
#include "Parallel.hpp"
#include "TensorPacking.hpp"
#include "VecIntrinsic.hpp"

namespace otter {

static inline signed char float2int8(float v) {
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

Tensor quantize_to_int8_x86(const Tensor& src, const Tensor& scale_data, bool pack) {
    int dims = src.dim();
    int elempack = src.elempack();
    
    int scale_data_size = scale_data.size(0);
    auto scale_data_a = scale_data.accessor<float, 1>();
    const float* scale_data_ptr = (const float*)scale_data.data_ptr();
    
    Tensor dst;
    
    if (elempack == 8) {
        return quantize_to_int8_x86(src.packing(4), scale_data, pack);
    }
    
    if (elempack == 4) {
        
        if (dims == 1) {
            int w = src.size(0);
            int out_elempack = pack && w * elempack % 8 == 0 ? 8 : 1;
            int outw = w * elempack / out_elempack;
            
            dst = otter::empty({outw}, get_update_scalarType(otter::ScalarType::Byte, out_elempack));
            
            const float* src_ptr = (const float*)src.data_ptr();
            signed char* dst_ptr = (signed char*)dst.data_ptr();
            
            if (scale_data_size == 1) {
                const float scale = scale_data_a[0];
                
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        const float* ptr0 = (const float*)src_ptr + i * 4;
                        signed char* outptr = (signed char*)dst_ptr + i * 4;
                        
                        outptr[0] = float2int8(ptr0[0] * scale);
                        outptr[1] = float2int8(ptr0[1] * scale);
                        outptr[2] = float2int8(ptr0[2] * scale);
                        outptr[3] = float2int8(ptr0[3] * scale);
                    }
                });
            } else {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        const float* ptr0 = (const float*)src_ptr + i * 4;
                        signed char* outptr = (signed char*)dst_ptr + i * 4;
                        
                        outptr[0] = float2int8(ptr0[0] * scale_data_a[i * 4]);
                        outptr[1] = float2int8(ptr0[1] * scale_data_a[i * 4 + 1]);
                        outptr[2] = float2int8(ptr0[2] * scale_data_a[i * 4 + 2]);
                        outptr[3] = float2int8(ptr0[3] * scale_data_a[i * 4 + 3]);
                    }
                });
            }
        }
        
        if (dims == 2) {
            int w = src.size(1);
            int h = src.size(0);
            int out_elempack = pack && h * elempack % 8 == 0 ? 8 : 1;
            int outh = h * elempack / out_elempack;
            
            dst = otter::empty({outh, w}, get_update_scalarType(otter::ScalarType::Byte, out_elempack));
            
            auto src_a = src.accessor<float, 2, 4>();
            auto dst_ra = dst.raw_accessor<signed char, 2>();
            
            if (out_elempack == 8) {
                if (scale_data_size == 1) {
                    __m128 _scale = _mm_set1_ps(scale_data_a[0]);
                    
                    otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const float* ptr0 = src_a[i * 2].data();
                            const float* ptr1 = src_a[i * 2 + 1].data();
                            signed char* outptr = dst_ra[i].data();
                            
                            int j = 0;
                            for (; j + 1 < w; j += 2) {
                                __m128 _v0 = _mm_loadu_ps(ptr0);
                                __m128 _v1 = _mm_loadu_ps(ptr1);
                                __m128 _v2 = _mm_loadu_ps(ptr0 + 4);
                                __m128 _v3 = _mm_loadu_ps(ptr1 + 4);
                                _v0 = _mm_mul_ps(_v0, _scale);
                                _v1 = _mm_mul_ps(_v1, _scale);
                                _v2 = _mm_mul_ps(_v2, _scale);
                                _v3 = _mm_mul_ps(_v3, _scale);
                                __m128i _v = float2int8_sse(_v0, _v1, _v2, _v3);
                                _mm_storeu_si128((__m128i*)outptr, _v);
                                
                                ptr0 += 8;
                                ptr1 += 8;
                                outptr += 16;
                            }
                            for (; j < w; j++) {
                                __m128 _vlow = _mm_loadu_ps(ptr0);
                                __m128 _vhigh = _mm_loadu_ps(ptr1);
                                _vlow = _mm_mul_ps(_vlow, _scale);
                                _vhigh = _mm_mul_ps(_vhigh, _scale);
                                *(int64_t*)outptr = float2int8_sse(_vlow, _vhigh);
                                
                                ptr0 += 4;
                                ptr1 += 4;
                                outptr += 8;
                            }
                        }
                    });
                } else {
                    otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const float* ptr0 = src_a[i * 2].data();
                            const float* ptr1 = src_a[i * 2 + 1].data();
                            signed char* outptr = dst_ra[i].data();
                            
                            __m128 _scale0 = _mm_loadu_ps((const float*)scale_data_ptr + i * 8);
                            __m128 _scale1 = _mm_loadu_ps((const float*)scale_data_ptr + i * 8 + 4);
                            
                            int j = 0;
                            for (; j + 1 < w; j += 2) {
                                __m128 _v0 = _mm_loadu_ps(ptr0);
                                __m128 _v1 = _mm_loadu_ps(ptr1);
                                __m128 _v2 = _mm_loadu_ps(ptr0 + 4);
                                __m128 _v3 = _mm_loadu_ps(ptr1 + 4);
                                _v0 = _mm_mul_ps(_v0, _scale0);
                                _v1 = _mm_mul_ps(_v1, _scale1);
                                _v2 = _mm_mul_ps(_v2, _scale0);
                                _v3 = _mm_mul_ps(_v3, _scale1);
                                __m128i _v = float2int8_sse(_v0, _v1, _v2, _v3);
                                _mm_storeu_si128((__m128i*)outptr, _v);
                                
                                ptr0 += 8;
                                ptr1 += 8;
                                outptr += 16;
                            }
                            for (; j < w; j++) {
                                __m128 _vlow = _mm_loadu_ps(ptr0);
                                __m128 _vhigh = _mm_loadu_ps(ptr1);
                                _vlow = _mm_mul_ps(_vlow, _scale0);
                                _vhigh = _mm_mul_ps(_vhigh, _scale1);
                                *(int64_t*)outptr = float2int8_sse(_vlow, _vhigh);
                                
                                ptr0 += 4;
                                ptr1 += 4;
                                outptr += 8;
                            }
                        }
                    });
                }
            } else if (out_elempack == 1) {
                if (scale_data_size == 1) {
                    const float scale = scale_data_a[0];
                    
                    otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const float* ptr0 = src_a[i].data();
                            signed char* outptr0 = dst_ra[i * 4].data();
                            signed char* outptr1 = dst_ra[i * 4 + 1].data();
                            signed char* outptr2 = dst_ra[i * 4 + 2].data();
                            signed char* outptr3 = dst_ra[i * 4 + 3].data();
                            
                            for (int j = 0; j < w; j++) {
                                outptr0[0] = float2int8(ptr0[0] * scale);
                                outptr1[0] = float2int8(ptr0[1] * scale);
                                outptr2[0] = float2int8(ptr0[2] * scale);
                                outptr3[0] = float2int8(ptr0[3] * scale);
                                
                                ptr0 += 4;
                                outptr0 += 1;
                                outptr1 += 1;
                                outptr2 += 1;
                                outptr3 += 1;
                            }
                        }
                    });
                } else {
                    otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const float* ptr0 = src_a[i].data();
                            signed char* outptr0 = dst_ra[i * 4].data();
                            signed char* outptr1 = dst_ra[i * 4 + 1].data();
                            signed char* outptr2 = dst_ra[i * 4 + 2].data();
                            signed char* outptr3 = dst_ra[i * 4 + 3].data();
                            
                            const float s0 = scale_data_a[i * 4];
                            const float s1 = scale_data_a[i * 4 + 1];
                            const float s2 = scale_data_a[i * 4 + 2];
                            const float s3 = scale_data_a[i * 4 + 3];
                            
                            for (int j = 0; j < w; j++) {
                                outptr0[0] = float2int8(ptr0[0] * s0);
                                outptr1[0] = float2int8(ptr0[1] * s1);
                                outptr2[0] = float2int8(ptr0[2] * s2);
                                outptr3[0] = float2int8(ptr0[3] * s3);
                                
                                ptr0 += 4;
                                outptr0 += 1;
                                outptr1 += 1;
                                outptr2 += 1;
                                outptr3 += 1;
                            }
                        }
                    });
                }
            }
        }
        
        if (dims == 3) {
            int w = src.size(2);
            int h = src.size(1);
            int channels = src.size(0);
            int size = w * h;
            int out_elempack = pack && channels * elempack % 8 == 0 ? 8 : 1;
            int outc = channels * elempack / out_elempack;
            
            dst = otter::empty({outc, h, w}, get_update_scalarType(otter::ScalarType::Byte, out_elempack));
            
            auto src_a = src.accessor<float, 3, 4>();
            auto dst_ra = dst.raw_accessor<signed char, 3>();
            
            if (out_elempack == 8) {
                if (scale_data_size == 1) {
                    __m128 _scale = _mm_set1_ps(scale_data_a[0]);

                    otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                        for (const auto q : otter::irange(begin, end)) {
                            const float* ptr0 = src_a[q * 2].data();
                            const float* ptr1 = src_a[q * 2 + 1].data();
                            signed char* outptr = dst_ra[q].data();

                            int i = 0;
                            for (; i + 1 < size; i += 2) {
                                __m128 _v0 = _mm_loadu_ps(ptr0);
                                __m128 _v1 = _mm_loadu_ps(ptr1);
                                __m128 _v2 = _mm_loadu_ps(ptr0 + 4);
                                __m128 _v3 = _mm_loadu_ps(ptr1 + 4);
                                _v0 = _mm_mul_ps(_v0, _scale);
                                _v1 = _mm_mul_ps(_v1, _scale);
                                _v2 = _mm_mul_ps(_v2, _scale);
                                _v3 = _mm_mul_ps(_v3, _scale);
                                __m128i _v = float2int8_sse(_v0, _v1, _v2, _v3);
                                _mm_storeu_si128((__m128i*)outptr, _v);

                                ptr0 += 8;
                                ptr1 += 8;
                                outptr += 16;
                            }
                            for (; i < size; i++) {
                                __m128 _vlow = _mm_loadu_ps(ptr0);
                                __m128 _vhigh = _mm_loadu_ps(ptr1);
                                _vlow = _mm_mul_ps(_vlow, _scale);
                                _vhigh = _mm_mul_ps(_vhigh, _scale);
                                *(int64_t*)outptr = float2int8_sse(_vlow, _vhigh);

                                ptr0 += 4;
                                ptr1 += 4;
                                outptr += 8;
                            }
                        }
                    });
                } else {
                    otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                        for (const auto q : otter::irange(begin, end)) {
                            const float* ptr0 = src_a[q * 2].data();
                            const float* ptr1 = src_a[q * 2 + 1].data();
                            signed char* outptr = dst_ra[q].data();

                            __m128 _scale0 = _mm_loadu_ps((const float*)scale_data_ptr + q * 8);
                            __m128 _scale1 = _mm_loadu_ps((const float*)scale_data_ptr + q * 8 + 4);

                            int i = 0;
                            for (; i + 1 < size; i += 2) {
                                __m128 _v0 = _mm_loadu_ps(ptr0);
                                __m128 _v1 = _mm_loadu_ps(ptr1);
                                __m128 _v2 = _mm_loadu_ps(ptr0 + 4);
                                __m128 _v3 = _mm_loadu_ps(ptr1 + 4);
                                _v0 = _mm_mul_ps(_v0, _scale0);
                                _v1 = _mm_mul_ps(_v1, _scale1);
                                _v2 = _mm_mul_ps(_v2, _scale0);
                                _v3 = _mm_mul_ps(_v3, _scale1);
                                __m128i _v = float2int8_sse(_v0, _v1, _v2, _v3);
                                _mm_storeu_si128((__m128i*)outptr, _v);

                                ptr0 += 8;
                                ptr1 += 8;
                                outptr += 16;
                            }
                            for (; i < size; i++) {
                                __m128 _vlow = _mm_loadu_ps(ptr0);
                                __m128 _vhigh = _mm_loadu_ps(ptr1);
                                _vlow = _mm_mul_ps(_vlow, _scale0);
                                _vhigh = _mm_mul_ps(_vhigh, _scale1);
                                *(int64_t*)outptr = float2int8_sse(_vlow, _vhigh);

                                ptr0 += 4;
                                ptr1 += 4;
                                outptr += 8;
                            }
                        }
                    });
                }
            } else if (out_elempack == 1) {
                if (scale_data_size == 1) {
                    const float scale = scale_data_a[0];

                    otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                        for (const auto q : otter::irange(begin, end)) {
                            const float* ptr0 = src_a[q].data();
                            signed char* outptr0 = dst_ra[q * 4].data();
                            signed char* outptr1 = dst_ra[q * 4 + 1].data();
                            signed char* outptr2 = dst_ra[q * 4 + 2].data();
                            signed char* outptr3 = dst_ra[q * 4 + 3].data();

                            for (int i = 0; i < size; i++) {
                                outptr0[0] = float2int8(ptr0[0] * scale);
                                outptr1[0] = float2int8(ptr0[1] * scale);
                                outptr2[0] = float2int8(ptr0[2] * scale);
                                outptr3[0] = float2int8(ptr0[3] * scale);

                                ptr0 += 4;
                                outptr0 += 1;
                                outptr1 += 1;
                                outptr2 += 1;
                                outptr3 += 1;
                            }
                        }
                    });
                } else {
                    otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                        for (const auto q : otter::irange(begin, end)) {
                            const float* ptr0 = src_a[q].data();
                            signed char* outptr0 = dst_ra[q * 4].data();
                            signed char* outptr1 = dst_ra[q * 4 + 1].data();
                            signed char* outptr2 = dst_ra[q * 4 + 2].data();
                            signed char* outptr3 = dst_ra[q * 4 + 3].data();

                            const float s0 = scale_data_a[q * 4];
                            const float s1 = scale_data_a[q * 4 + 1];
                            const float s2 = scale_data_a[q * 4 + 2];
                            const float s3 = scale_data_a[q * 4 + 3];

                            for (int i = 0; i < size; i++) {
                                outptr0[0] = float2int8(ptr0[0] * s0);
                                outptr1[0] = float2int8(ptr0[1] * s1);
                                outptr2[0] = float2int8(ptr0[2] * s2);
                                outptr3[0] = float2int8(ptr0[3] * s3);

                                ptr0 += 4;
                                outptr0 += 1;
                                outptr1 += 1;
                                outptr2 += 1;
                                outptr3 += 1;
                            }
                        }
                    });
                }
            }
        }
        
        if (dims == 4) {
            int batchsize = src.size(0);
            int w = src.size(3);
            int h = src.size(2);
            int channels = src.size(1);
            int size = w * h;
            int out_elempack = pack && channels * elempack % 8 == 0 ? 8 : 1;
            int outc = channels * elempack / out_elempack;
            
            dst = otter::empty({batchsize, outc, h, w}, get_update_scalarType(otter::ScalarType::Byte, out_elempack));
            
            for (const auto b : otter::irange(0, batchsize)) {
                auto src_a = src.accessor<float, 4, 4>()[b];
                auto dst_ra = dst.raw_accessor<signed char, 4>()[b];
                
                if (out_elempack == 8) {
                    if (scale_data_size == 1) {
                        __m128 _scale = _mm_set1_ps(scale_data_a[0]);

                        otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                            for (const auto q : otter::irange(begin, end)) {
                                const float* ptr0 = src_a[q * 2].data();
                                const float* ptr1 = src_a[q * 2 + 1].data();
                                signed char* outptr = dst_ra[q].data();

                                int i = 0;
                                for (; i + 1 < size; i += 2) {
                                    __m128 _v0 = _mm_loadu_ps(ptr0);
                                    __m128 _v1 = _mm_loadu_ps(ptr1);
                                    __m128 _v2 = _mm_loadu_ps(ptr0 + 4);
                                    __m128 _v3 = _mm_loadu_ps(ptr1 + 4);
                                    _v0 = _mm_mul_ps(_v0, _scale);
                                    _v1 = _mm_mul_ps(_v1, _scale);
                                    _v2 = _mm_mul_ps(_v2, _scale);
                                    _v3 = _mm_mul_ps(_v3, _scale);
                                    __m128i _v = float2int8_sse(_v0, _v1, _v2, _v3);
                                    _mm_storeu_si128((__m128i*)outptr, _v);

                                    ptr0 += 8;
                                    ptr1 += 8;
                                    outptr += 16;
                                }
                                for (; i < size; i++) {
                                    __m128 _vlow = _mm_loadu_ps(ptr0);
                                    __m128 _vhigh = _mm_loadu_ps(ptr1);
                                    _vlow = _mm_mul_ps(_vlow, _scale);
                                    _vhigh = _mm_mul_ps(_vhigh, _scale);
                                    *(int64_t*)outptr = float2int8_sse(_vlow, _vhigh);

                                    ptr0 += 4;
                                    ptr1 += 4;
                                    outptr += 8;
                                }
                            }
                        });
                    } else {
                        otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                            for (const auto q : otter::irange(begin, end)) {
                                const float* ptr0 = src_a[q * 2].data();
                                const float* ptr1 = src_a[q * 2 + 1].data();
                                signed char* outptr = dst_ra[q].data();

                                __m128 _scale0 = _mm_loadu_ps((const float*)scale_data_ptr + q * 8);
                                __m128 _scale1 = _mm_loadu_ps((const float*)scale_data_ptr + q * 8 + 4);

                                int i = 0;
                                for (; i + 1 < size; i += 2) {
                                    __m128 _v0 = _mm_loadu_ps(ptr0);
                                    __m128 _v1 = _mm_loadu_ps(ptr1);
                                    __m128 _v2 = _mm_loadu_ps(ptr0 + 4);
                                    __m128 _v3 = _mm_loadu_ps(ptr1 + 4);
                                    _v0 = _mm_mul_ps(_v0, _scale0);
                                    _v1 = _mm_mul_ps(_v1, _scale1);
                                    _v2 = _mm_mul_ps(_v2, _scale0);
                                    _v3 = _mm_mul_ps(_v3, _scale1);
                                    __m128i _v = float2int8_sse(_v0, _v1, _v2, _v3);
                                    _mm_storeu_si128((__m128i*)outptr, _v);

                                    ptr0 += 8;
                                    ptr1 += 8;
                                    outptr += 16;
                                }
                                for (; i < size; i++) {
                                    __m128 _vlow = _mm_loadu_ps(ptr0);
                                    __m128 _vhigh = _mm_loadu_ps(ptr1);
                                    _vlow = _mm_mul_ps(_vlow, _scale0);
                                    _vhigh = _mm_mul_ps(_vhigh, _scale1);
                                    *(int64_t*)outptr = float2int8_sse(_vlow, _vhigh);

                                    ptr0 += 4;
                                    ptr1 += 4;
                                    outptr += 8;
                                }
                            }
                        });
                    }
                } else if (out_elempack == 1) {
                    if (scale_data_size == 1) {
                        const float scale = scale_data_a[0];

                        otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                            for (const auto q : otter::irange(begin, end)) {
                                const float* ptr0 = src_a[q].data();
                                signed char* outptr0 = dst_ra[q * 4].data();
                                signed char* outptr1 = dst_ra[q * 4 + 1].data();
                                signed char* outptr2 = dst_ra[q * 4 + 2].data();
                                signed char* outptr3 = dst_ra[q * 4 + 3].data();

                                for (int i = 0; i < size; i++) {
                                    outptr0[0] = float2int8(ptr0[0] * scale);
                                    outptr1[0] = float2int8(ptr0[1] * scale);
                                    outptr2[0] = float2int8(ptr0[2] * scale);
                                    outptr3[0] = float2int8(ptr0[3] * scale);

                                    ptr0 += 4;
                                    outptr0 += 1;
                                    outptr1 += 1;
                                    outptr2 += 1;
                                    outptr3 += 1;
                                }
                            }
                        });
                    } else {
                        otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                            for (const auto q : otter::irange(begin, end)) {
                                const float* ptr0 = src_a[q].data();
                                signed char* outptr0 = dst_ra[q * 4].data();
                                signed char* outptr1 = dst_ra[q * 4 + 1].data();
                                signed char* outptr2 = dst_ra[q * 4 + 2].data();
                                signed char* outptr3 = dst_ra[q * 4 + 3].data();

                                const float s0 = scale_data_a[q * 4];
                                const float s1 = scale_data_a[q * 4 + 1];
                                const float s2 = scale_data_a[q * 4 + 2];
                                const float s3 = scale_data_a[q * 4 + 3];

                                for (int i = 0; i < size; i++) {
                                    outptr0[0] = float2int8(ptr0[0] * s0);
                                    outptr1[0] = float2int8(ptr0[1] * s1);
                                    outptr2[0] = float2int8(ptr0[2] * s2);
                                    outptr3[0] = float2int8(ptr0[3] * s3);

                                    ptr0 += 4;
                                    outptr0 += 1;
                                    outptr1 += 1;
                                    outptr2 += 1;
                                    outptr3 += 1;
                                }
                            }
                        });
                    }
                }
            }
        }
    }
    
    return dst;
}

Tensor dequantize_from_int32_x86(const Tensor& src, const Tensor& scale_data, const Tensor& bias_data, bool pack) {
    int dims = src.dim();
    int elempack = src.elempack();
    
    if (elempack == 8) {
        Tensor tmp = dequantize_from_int32_x86(src.packing(4), scale_data, bias_data, pack);
        return tmp.packing(8);
    }
    
    Tensor dst;
    
    int scale_data_size = scale_data.size(0);
    auto scale_data_a = scale_data.accessor<float, 1>();
    const float* scale_data_ptr = (const float*)scale_data.data_ptr();
    
    int bias_data_size = bias_data.size(0);
    auto bias_data_a = bias_data.accessor<float, 1>();
    const float* bias_data_ptr = (const float*)bias_data.data_ptr();
    
    if (elempack == 4) {
        if (dims == 1) {
            OTTER_CHECK(false, "dequantize 1D unimplement");
        }
        
        if (dims == 2) {
            OTTER_CHECK(false, "dequantize 2D unimplement");
        }
        
        if (dims == 3) {
            int w = src.size(2);
            int h = src.size(1);
            int channels = src.size(0);
            int size = w * h;
            
            dst = otter::empty({channels, h, w}, otter::ScalarType::Float4);
            
            auto src_a = src.accessor<int, 3, 4>();
            auto dst_a = dst.accessor<float, 3, 4>();
            
            if (bias_data_size == 0) {
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const int* intptr = src_a[q].data();
                        float* ptr = dst_a[q].data();

                        __m128 _scale = scale_data_size == 1 ? _mm_set1_ps(scale_data_a[0]) : _mm_loadu_ps((const float*)scale_data_ptr + q * 4);

                        for (int i = 0; i < size; i++)
                        {
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_mul_ps(_v, _scale);
                            _mm_storeu_ps(ptr, _v);

                            intptr += 4;
                            ptr += 4;
                        }
                    }
                });
            } else {
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const int* intptr = src_a[q].data();
                        float* ptr = dst_a[q].data();

                        __m128 _scale = scale_data_size == 1 ? _mm_set1_ps(scale_data_a[0]) : _mm_loadu_ps((const float*)scale_data_ptr + q * 4);
                        __m128 _bias = bias_data_size == 1 ? _mm_set1_ps(bias_data_a[0]) : _mm_loadu_ps((const float*)bias_data_ptr + q * 4);

                        for (int i = 0; i < size; i++) {
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                            _mm_storeu_ps(ptr, _v);

                            intptr += 4;
                            ptr += 4;
                        }
                    }
                });
            }
        }
        
        if (dims == 4) {
            int batchsize = src.size(0);
            int w = src.size(3);
            int h = src.size(2);
            int channels = src.size(1);
            int size = w * h;
            
            dst = otter::empty({batchsize, channels, h, w}, otter::ScalarType::Float4);
            
            for (const auto b : otter::irange(0, batchsize)) {
                auto src_a = src.accessor<int, 4, 4>()[b];
                auto dst_a = dst.accessor<float, 4, 4>()[b];
                
                if (bias_data_size == 0) {
                    otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                        for (const auto q : otter::irange(begin, end)) {
                            const int* intptr = src_a[q].data();
                            float* ptr = dst_a[q].data();

                            __m128 _scale = scale_data_size == 1 ? _mm_set1_ps(scale_data_a[0]) : _mm_loadu_ps((const float*)scale_data_ptr + q * 4);

                            for (int i = 0; i < size; i++) {
                                __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                                _v = _mm_mul_ps(_v, _scale);
                                _mm_storeu_ps(ptr, _v);

                                intptr += 4;
                                ptr += 4;
                            }
                        }
                    });
                } else {
                    otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                        for (const auto q : otter::irange(begin, end)) {
                            const int* intptr = src_a[q].data();
                            float* ptr = dst_a[q].data();

                            __m128 _scale = scale_data_size == 1 ? _mm_set1_ps(scale_data_a[0]) : _mm_loadu_ps((const float*)scale_data_ptr + q * 4);
                            __m128 _bias = bias_data_size == 1 ? _mm_set1_ps(bias_data_a[0]) : _mm_loadu_ps((const float*)bias_data_ptr + q * 4);

                            for (int i = 0; i < size; i++) {
                                __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                                _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                                _mm_storeu_ps(ptr, _v);

                                intptr += 4;
                                ptr += 4;
                            }
                        }
                    });
                }
            }
        }
    }
    
    return dst;
}

}   // end namespace otter

#endif  // __SSE2__
