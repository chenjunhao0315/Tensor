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

static inline float activation_ss(float v, int activation_type, const Tensor& activation_params) {
    if (activation_type == 1) {
        v = fmax(v, 0.f);
    } else if (activation_type == 2) {
        float slope = activation_params.item().toFloat();
        v = v > 0.f ? v : v * slope;
    } else if (activation_type == 3) {
        float min = 0;
        float max = 6;
        if (v < min)
            v = min;
        if (v > max)
            v = max;
    } else if (activation_type == 4) {
        v = 1.f / (1.f + exp(-v));
    } else if (activation_type == 5) {
        v = v * tanh(log(exp(v) + 1.f));
    }

    return v;
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
        } else if (dims == 2) {
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
        } else if (dims == 3) {
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
        } else if (dims == 4) {
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
        
        return dst;
    }
    
    dst = otter::empty_like(src, otter::ScalarType::Byte);
    
    if (dims == 1) {
        int w = src.size(0);
        
        const float* ptr = src.data_ptr<float>();
        signed char* outptr = (signed char*)dst.data_ptr();
        
        if (scale_data_size == 1) {
            const float scale = scale_data_a[0];

            otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    outptr[i] = float2int8(ptr[i] * scale);
                }
            });
        } else {
            otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    outptr[i] = float2int8(ptr[i] * scale_data_a[i]);
                }
            });
        }
    } else if (dims == 2) {
        int w = src.size(1);
        int h = src.size(0);
        
        auto src_a = src.accessor<float, 2>();
        auto dst_a = dst.accessor<unsigned char, 2>();
        
        otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
            for (const auto i : otter::irange(begin, end)) {
                const float* ptr0 = src_a[i].data();
                signed char* outptr0 = (signed char*)dst_a[i].data();

                const float scale = scale_data_size == 1 ? scale_data_a[0] : scale_data_a[i];

                for (int j = 0; j < w; j++) {
                    *outptr0++ = float2int8(*ptr0++ * scale);
                }
            }
        });
    } else if (dims == 3) {
        int channels = src.size(0);
        int h = src.size(1);
        int w = src.size(2);
        int size = w * h;
        
        auto src_a = src.accessor<float, 3>();
        auto dst_a = dst.accessor<unsigned char, 3>();
        
        otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
            for (const auto q : otter::irange(begin, end)) {
                const float* ptr = src_a[q].data();
                signed char* outptr = (signed char*)dst_a[q].data();

                const float scale = scale_data_size == 1 ? scale_data_a[0] : scale_data_a[q];

                for (int i = 0; i < size; i++) {
                    outptr[i] = float2int8(ptr[i] * scale);
                }
            }
        });
    } else if (dims == 4) {
        int batchsize = src.size(0);
        int channels = src.size(1);
        int h = src.size(2);
        int w = src.size(3);
        int size = w * h;
        
        for (int b = 0; b < batchsize; ++b) {
            auto src_a = src.accessor<float, 4>()[b];
            auto dst_a = dst.accessor<unsigned char, 4>()[b];
            
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const float* ptr = src_a[q].data();
                    signed char* outptr = (signed char*)dst_a[q].data();

                    const float scale = scale_data_size == 1 ? scale_data_a[0] : scale_data_a[q];

                    for (int i = 0; i < size; i++) {
                        outptr[i] = float2int8(ptr[i] * scale);
                    }
                }
            });
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
            int w = src.size(0);
            
            dst = otter::empty({w}, otter::ScalarType::Float4);
            
            const int* srcptr = (const int*)src.data_ptr();
            float* dstptr = (float*)dst.data_ptr();
            
            if (scale_data_size == 1) {
                __m128 _scale = _mm_set1_ps(scale_data_a[0]);

                if (bias_data_size == 0) {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            float* ptr = (float*)dstptr + i * 4;

                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_mul_ps(_v, _scale);
                            _mm_storeu_ps(ptr, _v);
                        }
                    });
                } else if (bias_data_size == 1) {
                    __m128 _bias = _mm_set1_ps(bias_data_a[0]);

                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            float* ptr = (float*)dstptr + i * 4;

                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                            _mm_storeu_ps(ptr, _v);
                        }
                    });
                } else {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            float* ptr = (float*)dstptr + i * 4;

                            __m128 _bias = _mm_loadu_ps((const float*)bias_data_ptr + i * 4);
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                            _mm_storeu_ps(ptr, _v);
                        }
                    });
                }
            } else {
                if (bias_data_size == 0) {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            float* ptr = (float*)dstptr + i * 4;

                            __m128 _scale = _mm_loadu_ps((const float*)scale_data_ptr + i * 4);
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_mul_ps(_v, _scale);
                            _mm_storeu_ps(ptr, _v);
                        }
                    });
                } else if (bias_data_size == 1) {
                    __m128 _bias = _mm_set1_ps(bias_data_a[0]);

                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            float* ptr = (float*)dstptr + i * 4;

                            __m128 _scale = _mm_loadu_ps((const float*)scale_data_ptr + i * 4);
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                            _mm_storeu_ps(ptr, _v);
                        }
                    });
                } else {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            float* ptr = (float*)dstptr + i * 4;

                            __m128 _scale = _mm_loadu_ps((const float*)scale_data_ptr + i * 4);
                            __m128 _bias = _mm_loadu_ps((const float*)bias_data_ptr + i * 4);
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                            _mm_storeu_ps(ptr, _v);
                        }
                    });
                }
            }
        } else if (dims == 2) {
            int w = src.size(1);
            int h = src.size(0);
            
            auto src_a = src.accessor<int, 2, 4>();
            auto dst_a = dst.accessor<float, 2, 4>();
            
            dst = otter::empty({h, w}, otter::ScalarType::Float4);
            
            if (bias_data_size == 0) {
                otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        const int* intptr = src_a[i].data();
                        float* ptr = dst_a[i].data();

                        __m128 _scale = scale_data_size == 1 ? _mm_set1_ps(scale_data_a[0]) : _mm_loadu_ps((const float*)scale_data_ptr + i * 4);

                        for (int j = 0; j < w; j++) {
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_mul_ps(_v, _scale);
                            _mm_storeu_ps(ptr, _v);

                            intptr += 4;
                            ptr += 4;
                        }
                    }
                });
            } else {
                otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        const int* intptr = src_a[i].data();
                        float* ptr = dst_a[i].data();

                        __m128 _scale = scale_data_size == 1 ? _mm_set1_ps(scale_data_a[0]) : _mm_loadu_ps((const float*)scale_data_ptr + i * 4);
                        __m128 _bias = bias_data_size == 1 ? _mm_set1_ps(bias_data_a[0]) : _mm_loadu_ps((const float*)bias_data_ptr + i * 4);

                        for (int j = 0; j < w; j++) {
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                            _mm_storeu_ps(ptr, _v);

                            intptr += 4;
                            ptr += 4;
                        }
                    }
                });
            }
        } else if (dims == 3) {
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
        } else if (dims == 4) {
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
        
        return dst;
    }
    
    dst = otter::empty_like(src, otter::ScalarType::Float);
    
    if (dims == 1) {
        int w = src.size(0);
        
        const int* intptr = (const int*)src.data_ptr();
        float* ptr = (float*)dst.data_ptr();
        
        if (scale_data_size == 1) {
            const float scale = scale_data_a[0];

            if (bias_data_size == 0) {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        ptr[i] = intptr[i] * scale;
                    }
                });
            } else if (bias_data_size == 1) {
                const float bias = bias_data_a[0];

                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        ptr[i] = intptr[i] * scale + bias;
                    }
                });
            } else {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        ptr[i] = intptr[i] * scale + bias_data_a[i];
                    }
                });
            }
        } else {
            if (bias_data_size == 0) {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        ptr[i] = intptr[i] * scale_data_a[i];
                    }
                });
            } else if (bias_data_size == 1) {
                const float bias = bias_data_a[0];

                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        ptr[i] = intptr[i] * scale_data_a[i] + bias;
                    }
                });
            } else {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        ptr[i] = intptr[i] * scale_data_a[i] + bias_data_a[i];
                    }
                });
            }
        }
    } else if (dims == 2) {
        int w = src.size(1);
        int h = src.size(0);
        
        auto src_a = src.accessor<int, 2>();
        auto dst_a = dst.accessor<float, 2>();
        
        if (bias_data_size == 0) {
            otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    const int* intptr = src_a[i].data();
                    float* ptr = dst_a[i].data();

                    const float scale = scale_data_size == 1 ? scale_data_a[0] : scale_data_a[i];

                    int j = 0;
    #if __SSE2__
                    __m128 _scale = _mm_set1_ps(scale);
                    for (; j + 3 < w; j += 4)
                    {
                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_mul_ps(_v, _scale);
                        _mm_storeu_ps(ptr, _v);

                        intptr += 4;
                        ptr += 4;
                    }
    #endif // __SSE2__
                    for (; j < w; j++)
                    {
                        *ptr++ = *intptr++ * scale;
                    }
                }
            });
        } else {
            otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    const int* intptr = src_a[i].data();
                    float* ptr = dst_a[i].data();

                    const float scale = scale_data_size == 1 ? scale_data_a[0] : scale_data_a[i];
                    const float bias = bias_data_size == 1 ? bias_data_a[0] : bias_data_a[i];

                    int j = 0;
    #if __SSE2__
                    __m128 _scale = _mm_set1_ps(scale);
                    __m128 _bias = _mm_set1_ps(bias);
                    for (; j + 3 < w; j += 4)
                    {
                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                        _mm_storeu_ps(ptr, _v);

                        intptr += 4;
                        ptr += 4;
                    }
    #endif // __SSE2__
                    for (; j < w; j++)
                    {
                        *ptr++ = *intptr++ * scale + bias;
                    }
                }
            });
        }
    } else if (dims == 3) {
        int channels = src.size(0);
        int h = src.size(1);
        int w = src.size(2);
        int size = w * h;
        
        auto src_a = src.accessor<int, 3>();
        auto dst_a = dst.accessor<float, 3>();
        
        if (bias_data_size == 0) {
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const int* intptr = src_a[q].data();
                    float* ptr = dst_a[q].data();

                    const float scale = scale_data_size == 1 ? scale_data_a[0] : scale_data_a[q];

                    int i = 0;
#if __SSE2__
                    __m128 _scale = _mm_set1_ps(scale);
                    for (; i + 3 < size; i += 4) {
                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_mul_ps(_v, _scale);
                        _mm_storeu_ps(ptr, _v);

                        intptr += 4;
                        ptr += 4;
                    }
#endif  // __SSE2__
                    for (; i < size; i++) {
                        ptr[i] = intptr[i] * scale;
                    }
                }
            });
        } else {
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const int* intptr = src_a[q].data();
                    float* ptr = dst_a[q].data();

                    const float scale = scale_data_size == 1 ? scale_data_a[0] : scale_data_a[q];
                    const float bias = bias_data_size == 1 ? bias_data_a[0] : bias_data_a[q];

                    int i = 0;
#if __SSE2__
                    __m128 _scale = _mm_set1_ps(scale);
                    __m128 _bias = _mm_set1_ps(bias);
                    for (; i + 3 < size; i += 4)
                    {
                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                        _mm_storeu_ps(ptr, _v);

                        intptr += 4;
                        ptr += 4;
                    }
#endif  // __SSE2__
                    for (; i < size; i++) {
                        ptr[i] = intptr[i] * scale + bias;
                    }
                }
            });
        }
    } else if (dims == 4) {
        int batchsize = src.size(0);
        int channels = src.size(1);
        int h = src.size(2);
        int w = src.size(3);
        int size = w * h;
        
        for (const auto b : otter::irange(0, batchsize)) {
            auto src_a = src.accessor<int, 4>()[b];
            auto dst_a = dst.accessor<float, 4>()[b];
            
            if (bias_data_size == 0) {
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const int* intptr = src_a[q].data();
                        float* ptr = dst_a[q].data();

                        const float scale = scale_data_size == 1 ? scale_data_a[0] : scale_data_a[q];
                        
                        int i = 0;
#if __SSE2__
                        __m128 _scale = _mm_set1_ps(scale);
                        for (; i + 3 < size; i += 4) {
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_mul_ps(_v, _scale);
                            _mm_storeu_ps(ptr, _v);

                            intptr += 4;
                            ptr += 4;
                        }
#endif // __SSE2__

                        for (; i < size; i++) {
                            ptr[i] = intptr[i] * scale;
                        }
                    }
                });
            } else {
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const int* intptr = src_a[q].data();
                        float* ptr = dst_a[q].data();

                        const float scale = scale_data_size == 1 ? scale_data_a[0] : scale_data_a[q];
                        const float bias = bias_data_size == 1 ? bias_data_a[0] : bias_data_a[q];

                        int i = 0;
#if __SSE2__
                        __m128 _scale = _mm_set1_ps(scale);
                        __m128 _bias = _mm_set1_ps(bias);
                        for (; i + 3 < size; i += 4)
                        {
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                            _mm_storeu_ps(ptr, _v);

                            intptr += 4;
                            ptr += 4;
                        }
#endif // __SSE2__
                        
                        for (; i < size; i++) {
                            ptr[i] = intptr[i] * scale + bias;
                        }
                    }
                });
            }
        }
    }
    
    return dst;
}

Tensor requantize_from_int32_to_int8_x86(const Tensor& src, const Tensor& scale_in_data, const Tensor& scale_out_data, const Tensor& bias_data, int activation_type, const Tensor& activation_params, bool pack) {
    
    int dims = src.dim();
    int elempack = src.elempack();
    
    int scale_in_data_size = scale_in_data.size(0);
    const float* scale_in_data_ptr = (const float*)scale_in_data.data_ptr();
    int scale_out_data_size = scale_out_data.size(0);
    const float* scale_out_data_ptr = (const float*)scale_out_data.data_ptr();
    int bias_data_size = (bias_data.defined()) ? bias_data.size(0) : 0;
    const float* bias_data_a = (bias_data.defined()) ? bias_data.data_ptr<float>() : nullptr;
    
    Tensor dst;
    
    if (elempack == 8) {
        return requantize_from_int32_to_int8_x86(src.packing(4), scale_in_data, scale_out_data, bias_data, activation_type, activation_params, pack);
    }
    
    if (elempack == 4) {
        if (dims == 1) {
            int w = src.size(0);
            int out_elempack = pack && w * elempack % 8 == 0 ? 8 : 1;
            int outw = w * elempack / out_elempack;
            
            dst = otter::empty({outw}, get_update_scalarType(otter::ScalarType::Byte, out_elempack));
            
            const int* srcptr = (const int*)src.data_ptr();
            signed char* dstptr = (signed char*)dst.data_ptr();
            
            if (scale_in_data_size == 1 && scale_out_data_size == 1) {
                __m128 _scale_in = _mm_set1_ps(scale_in_data_ptr[0]);
                __m128 _scale_out = _mm_set1_ps(scale_out_data_ptr[0]);

                if (bias_data_size == 0) {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_mul_ps(_v, _scale_in);
                            _v = activation_sse(_v, activation_type, activation_params);
                            _v = _mm_mul_ps(_v, _scale_out);
                            int64_t v = float2int8_sse(_v, _v);
                            ptr[0] = (v >> 32) & 0xff;
                            ptr[1] = (v >> 40) & 0xff;
                            ptr[2] = (v >> 48) & 0xff;
                            ptr[3] = (v >> 56) & 0xff;
                        }
                    });
                } else if (bias_data_size == 1) {
                    __m128 _bias = _mm_set1_ps(bias_data_a[0]);

                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale_in));
                            _v = activation_sse(_v, activation_type, activation_params);
                            _v = _mm_mul_ps(_v, _scale_out);
                            int64_t v = float2int8_sse(_v, _v);
                            ptr[0] = (v >> 32) & 0xff;
                            ptr[1] = (v >> 40) & 0xff;
                            ptr[2] = (v >> 48) & 0xff;
                            ptr[3] = (v >> 56) & 0xff;
                        }
                    });
                } else {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            __m128 _bias = _mm_loadu_ps((const float*)bias_data_a + i * 4);
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale_in));
                            _v = activation_sse(_v, activation_type, activation_params);
                            _v = _mm_mul_ps(_v, _scale_out);
                            int64_t v = float2int8_sse(_v, _v);
                            ptr[0] = (v >> 32) & 0xff;
                            ptr[1] = (v >> 40) & 0xff;
                            ptr[2] = (v >> 48) & 0xff;
                            ptr[3] = (v >> 56) & 0xff;
                        }
                    });
                }
            } else if (scale_in_data_size == 1 && scale_out_data_size > 1) {
                __m128 _scale_in = _mm_set1_ps(scale_in_data_ptr[0]);

                if (bias_data_size == 0) {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            __m128 _scale_out = _mm_loadu_ps((const float*)scale_out_data_ptr + i * 4);
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_mul_ps(_v, _scale_in);
                            _v = activation_sse(_v, activation_type, activation_params);
                            _v = _mm_mul_ps(_v, _scale_out);
                            int64_t v = float2int8_sse(_v, _v);
                            ptr[0] = (v >> 32) & 0xff;
                            ptr[1] = (v >> 40) & 0xff;
                            ptr[2] = (v >> 48) & 0xff;
                            ptr[3] = (v >> 56) & 0xff;
                        }
                    });
                } else if (bias_data_size == 1) {
                    __m128 _bias = _mm_set1_ps(bias_data_a[0]);

                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            __m128 _scale_out = _mm_loadu_ps((const float*)scale_out_data_ptr + i * 4);
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale_in));
                            _v = activation_sse(_v, activation_type, activation_params);
                            _v = _mm_mul_ps(_v, _scale_out);
                            int64_t v = float2int8_sse(_v, _v);
                            ptr[0] = (v >> 32) & 0xff;
                            ptr[1] = (v >> 40) & 0xff;
                            ptr[2] = (v >> 48) & 0xff;
                            ptr[3] = (v >> 56) & 0xff;
                        }
                    });
                } else {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            __m128 _scale_out = _mm_loadu_ps((const float*)scale_out_data_ptr + i * 4);
                            __m128 _bias = _mm_loadu_ps((const float*)bias_data_a + i * 4);
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale_in));
                            _v = activation_sse(_v, activation_type, activation_params);
                            _v = _mm_mul_ps(_v, _scale_out);
                            int64_t v = float2int8_sse(_v, _v);
                            ptr[0] = (v >> 32) & 0xff;
                            ptr[1] = (v >> 40) & 0xff;
                            ptr[2] = (v >> 48) & 0xff;
                            ptr[3] = (v >> 56) & 0xff;
                        }
                    });
                }
            } else if (scale_in_data_size > 1 && scale_out_data_size == 1) {
                __m128 _scale_out = _mm_set1_ps(scale_out_data_ptr[0]);

                if (bias_data_size == 0) {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            __m128 _scale_in = _mm_loadu_ps((const float*)scale_in_data_ptr + i * 4);
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_mul_ps(_v, _scale_in);
                            _v = activation_sse(_v, activation_type, activation_params);
                            _v = _mm_mul_ps(_v, _scale_out);
                            int64_t v = float2int8_sse(_v, _v);
                            ptr[0] = (v >> 32) & 0xff;
                            ptr[1] = (v >> 40) & 0xff;
                            ptr[2] = (v >> 48) & 0xff;
                            ptr[3] = (v >> 56) & 0xff;
                        }
                    });
                } else if (bias_data_size == 1) {
                    __m128 _bias = _mm_set1_ps(bias_data_a[0]);

                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            __m128 _scale_in = _mm_loadu_ps((const float*)scale_in_data_ptr + i * 4);
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale_in));
                            _v = activation_sse(_v, activation_type, activation_params);
                            _v = _mm_mul_ps(_v, _scale_out);
                            int64_t v = float2int8_sse(_v, _v);
                            ptr[0] = (v >> 32) & 0xff;
                            ptr[1] = (v >> 40) & 0xff;
                            ptr[2] = (v >> 48) & 0xff;
                            ptr[3] = (v >> 56) & 0xff;
                        }
                    });
                } else {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            __m128 _scale_in = _mm_loadu_ps((const float*)scale_in_data_ptr + i * 4);
                            __m128 _bias = _mm_loadu_ps((const float*)bias_data_a + i * 4);
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale_in));
                            _v = activation_sse(_v, activation_type, activation_params);
                            _v = _mm_mul_ps(_v, _scale_out);
                            int64_t v = float2int8_sse(_v, _v);
                            ptr[0] = (v >> 32) & 0xff;
                            ptr[1] = (v >> 40) & 0xff;
                            ptr[2] = (v >> 48) & 0xff;
                            ptr[3] = (v >> 56) & 0xff;
                        }
                    });
                }
            } else {
                if (bias_data_size == 0) {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            __m128 _scale_in = _mm_loadu_ps((const float*)scale_in_data_ptr + i * 4);
                            __m128 _scale_out = _mm_loadu_ps((const float*)scale_out_data_ptr + i * 4);
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_mul_ps(_v, _scale_in);
                            _v = activation_sse(_v, activation_type, activation_params);
                            _v = _mm_mul_ps(_v, _scale_out);
                            int64_t v = float2int8_sse(_v, _v);
                            ptr[0] = (v >> 32) & 0xff;
                            ptr[1] = (v >> 40) & 0xff;
                            ptr[2] = (v >> 48) & 0xff;
                            ptr[3] = (v >> 56) & 0xff;
                        }
                    });
                } else if (bias_data_size == 1) {
                    __m128 _bias = _mm_set1_ps(bias_data_a[0]);

                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            __m128 _scale_in = _mm_loadu_ps((const float*)scale_in_data_ptr + i * 4);
                            __m128 _scale_out = _mm_loadu_ps((const float*)scale_out_data_ptr + i * 4);
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale_in));
                            _v = activation_sse(_v, activation_type, activation_params);
                            _v = _mm_mul_ps(_v, _scale_out);
                            int64_t v = float2int8_sse(_v, _v);
                            ptr[0] = (v >> 32) & 0xff;
                            ptr[1] = (v >> 40) & 0xff;
                            ptr[2] = (v >> 48) & 0xff;
                            ptr[3] = (v >> 56) & 0xff;
                        }
                    });
                } else {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            __m128 _scale_in = _mm_loadu_ps((const float*)scale_in_data_ptr + i * 4);
                            __m128 _scale_out = _mm_loadu_ps((const float*)scale_out_data_ptr + i * 4);
                            __m128 _bias = _mm_loadu_ps((const float*)bias_data_a + i * 4);
                            __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                            _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale_in));
                            _v = activation_sse(_v, activation_type, activation_params);
                            _v = _mm_mul_ps(_v, _scale_out);
                            int64_t v = float2int8_sse(_v, _v);
                            ptr[0] = (v >> 32) & 0xff;
                            ptr[1] = (v >> 40) & 0xff;
                            ptr[2] = (v >> 48) & 0xff;
                            ptr[3] = (v >> 56) & 0xff;
                        }
                    });
                }
            }
        } else if (dims == 2) {
            int w = src.size(1);
            int h = src.size(0);
            int out_elempack = pack && h * elempack % 8 == 0 ? 8 : 1;
            int outh = h * elempack / out_elempack;
            
            dst = otter::empty({outh, w}, get_update_scalarType(otter::ScalarType::Byte, out_elempack));
            
            auto src_a = src.accessor<int, 2>();
            auto dst_ra = dst.raw_accessor<signed char, 2>();
            
            if (out_elempack == 8) {
                if (bias_data_size == 0) {
                    otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr0 = src_a[i * 2].data();
                            const int* intptr1 = src_a[i * 2 + 1].data();
                            signed char* ptr = dst_ra[i].data();

                            __m128 _scale_in0 = scale_in_data_size == 1 ? _mm_set1_ps(scale_in_data_ptr[0]) : _mm_loadu_ps((const float*)scale_in_data_ptr + i * 8);
                            __m128 _scale_in1 = scale_in_data_size == 1 ? _mm_set1_ps(scale_in_data_ptr[0]) : _mm_loadu_ps((const float*)scale_in_data_ptr + i * 8 + 4);
                            __m128 _scale_out0 = scale_out_data_size == 1 ? _mm_set1_ps(scale_out_data_ptr[0]) : _mm_loadu_ps((const float*)scale_out_data_ptr + i * 8);
                            __m128 _scale_out1 = scale_out_data_size == 1 ? _mm_set1_ps(scale_out_data_ptr[0]) : _mm_loadu_ps((const float*)scale_out_data_ptr + i * 8 + 4);

                            for (int j = 0; j < w; j++) {
                                __m128 _v0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr0));
                                __m128 _v1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr1));
                                _v0 = _mm_mul_ps(_v0, _scale_in0);
                                _v1 = _mm_mul_ps(_v1, _scale_in1);
                                _v0 = activation_sse(_v0, activation_type, activation_params);
                                _v1 = activation_sse(_v1, activation_type, activation_params);
                                _v0 = _mm_mul_ps(_v0, _scale_out0);
                                _v1 = _mm_mul_ps(_v1, _scale_out1);
                                *(int64_t*)ptr = float2int8_sse(_v0, _v1);

                                intptr0 += 4;
                                intptr1 += 4;
                                ptr += 8;
                            }
                        }
                    });
                } else {
                    otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr0 = src_a[i * 2].data();
                            const int* intptr1 = src_a[i * 2 + 1].data();
                            signed char* ptr = dst_ra[i].data();

                            __m128 _scale_in0 = scale_in_data_size == 1 ? _mm_set1_ps(scale_in_data_ptr[0]) : _mm_loadu_ps((const float*)scale_in_data_ptr + i * 8);
                            __m128 _scale_in1 = scale_in_data_size == 1 ? _mm_set1_ps(scale_in_data_ptr[0]) : _mm_loadu_ps((const float*)scale_in_data_ptr + i * 8 + 4);
                            __m128 _scale_out0 = scale_out_data_size == 1 ? _mm_set1_ps(scale_out_data_ptr[0]) : _mm_loadu_ps((const float*)scale_out_data_ptr + i * 8);
                            __m128 _scale_out1 = scale_out_data_size == 1 ? _mm_set1_ps(scale_out_data_ptr[0]) : _mm_loadu_ps((const float*)scale_out_data_ptr + i * 8 + 4);
                            __m128 _bias0 = bias_data_size == 1 ? _mm_set1_ps(bias_data_a[0]) : _mm_loadu_ps((const float*)bias_data_a + i * 8);
                            __m128 _bias1 = bias_data_size == 1 ? _mm_set1_ps(bias_data_a[0]) : _mm_loadu_ps((const float*)bias_data_a + i * 8 + 4);

                            for (int j = 0; j < w; j++)
                            {
                                __m128 _v0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr0));
                                __m128 _v1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr1));
                                _v0 = _mm_add_ps(_bias0, _mm_mul_ps(_v0, _scale_in0));
                                _v1 = _mm_add_ps(_bias1, _mm_mul_ps(_v1, _scale_in1));
                                _v0 = activation_sse(_v0, activation_type, activation_params);
                                _v1 = activation_sse(_v1, activation_type, activation_params);
                                _v0 = _mm_mul_ps(_v0, _scale_out0);
                                _v1 = _mm_mul_ps(_v1, _scale_out1);
                                *(int64_t*)ptr = float2int8_sse(_v0, _v1);

                                intptr0 += 4;
                                intptr1 += 4;
                                ptr += 8;
                            }
                        }
                    });
                }
            }
            if (out_elempack == 1) {
                if (bias_data_size == 0) {
                    otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = src_a[i].data();
                            signed char* ptr0 = dst_ra[i * 4].data();
                            signed char* ptr1 = dst_ra[i * 4 + 1].data();
                            signed char* ptr2 = dst_ra[i * 4 + 2].data();
                            signed char* ptr3 = dst_ra[i * 4 + 3].data();

                            __m128 _scale_in = scale_in_data_size == 1 ? _mm_set1_ps(scale_in_data_ptr[0]) : _mm_loadu_ps((const float*)scale_in_data_ptr + i * 4);
                            __m128 _scale_out = scale_out_data_size == 1 ? _mm_set1_ps(scale_out_data_ptr[0]) : _mm_loadu_ps((const float*)scale_out_data_ptr + i * 4);

                            for (int j = 0; j < w; j++) {
                                __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                                _v = _mm_mul_ps(_v, _scale_in);
                                _v = activation_sse(_v, activation_type, activation_params);
                                _v = _mm_mul_ps(_v, _scale_out);
                                int64_t v = float2int8_sse(_v, _v);
                                ptr0[0] = (v >> 32) & 0xff;
                                ptr1[0] = (v >> 40) & 0xff;
                                ptr2[0] = (v >> 48) & 0xff;
                                ptr3[0] = (v >> 56) & 0xff;

                                intptr += 4;
                                ptr0 += 1;
                                ptr1 += 1;
                                ptr2 += 1;
                                ptr3 += 1;
                            }
                        }
                    });
                } else {
                    otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = src_a[i].data();
                            signed char* ptr0 = dst_ra[i * 4].data();
                            signed char* ptr1 = dst_ra[i * 4 + 1].data();
                            signed char* ptr2 = dst_ra[i * 4 + 2].data();
                            signed char* ptr3 = dst_ra[i * 4 + 3].data();

                            __m128 _scale_in = scale_in_data_size == 1 ? _mm_set1_ps(scale_in_data_ptr[0]) : _mm_loadu_ps((const float*)scale_in_data_ptr + i * 4);
                            __m128 _scale_out = scale_out_data_size == 1 ? _mm_set1_ps(scale_out_data_ptr[0]) : _mm_loadu_ps((const float*)scale_out_data_ptr + i * 4);
                            __m128 _bias = bias_data_size == 1 ? _mm_set1_ps(bias_data_a[0]) : _mm_loadu_ps((const float*)bias_data_a + i * 4);

                            for (int j = 0; j < w; j++)
                            {
                                __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                                _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale_in));
                                _v = activation_sse(_v, activation_type, activation_params);
                                _v = _mm_mul_ps(_v, _scale_out);
                                int64_t v = float2int8_sse(_v, _v);
                                ptr0[0] = (v >> 32) & 0xff;
                                ptr1[0] = (v >> 40) & 0xff;
                                ptr2[0] = (v >> 48) & 0xff;
                                ptr3[0] = (v >> 56) & 0xff;

                                intptr += 4;
                                ptr0 += 1;
                                ptr1 += 1;
                                ptr2 += 1;
                                ptr3 += 1;
                            }
                        }
                    });
                }
            }
            
        } else if (dims == 3) {
            int w = src.size(2);
            int h = src.size(1);
            int channels = src.size(0);
            int size = w * h;
            int out_elempack = pack && channels * elempack % 8 == 0 ? 8 : 1;
            int outc = channels * elempack / out_elempack;
            
            dst = otter::empty({outc, h, w}, get_update_scalarType(otter::ScalarType::Byte, out_elempack));
            
            auto src_a = src.accessor<int, 3>();
            auto dst_ra = dst.raw_accessor<signed char, 3>();
            
            if (out_elempack == 8) {
                if (bias_data_size == 0) {
                    otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                        for (const auto q : otter::irange(begin, end)) {
                            const int* intptr0 = src_a[q * 2].data();
                            const int* intptr1 = src_a[q * 2 + 1].data();
                            signed char* ptr = dst_ra[q].data();

                            __m128 _scale_in0 = scale_in_data_size == 1 ? _mm_set1_ps(scale_in_data_ptr[0]) : _mm_loadu_ps((const float*)scale_in_data_ptr + q * 8);
                            __m128 _scale_in1 = scale_in_data_size == 1 ? _mm_set1_ps(scale_in_data_ptr[0]) : _mm_loadu_ps((const float*)scale_in_data_ptr + q * 8 + 4);
                            __m128 _scale_out0 = scale_out_data_size == 1 ? _mm_set1_ps(scale_out_data_ptr[0]) : _mm_loadu_ps((const float*)scale_out_data_ptr + q * 8);
                            __m128 _scale_out1 = scale_out_data_size == 1 ? _mm_set1_ps(scale_out_data_ptr[0]) : _mm_loadu_ps((const float*)scale_out_data_ptr + q * 8 + 4);

                            for (int i = 0; i < size; i++)
                            {
                                __m128 _v0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr0));
                                __m128 _v1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr1));
                                _v0 = _mm_mul_ps(_v0, _scale_in0);
                                _v1 = _mm_mul_ps(_v1, _scale_in1);
                                _v0 = activation_sse(_v0, activation_type, activation_params);
                                _v1 = activation_sse(_v1, activation_type, activation_params);
                                _v0 = _mm_mul_ps(_v0, _scale_out0);
                                _v1 = _mm_mul_ps(_v1, _scale_out1);
                                *(int64_t*)ptr = float2int8_sse(_v0, _v1);

                                intptr0 += 4;
                                intptr1 += 4;
                                ptr += 8;
                            }
                        }
                    });
                } else {
                    otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                        for (const auto q : otter::irange(begin, end)) {
                            const int* intptr0 = src_a[q * 2].data();
                            const int* intptr1 = src_a[q * 2 + 1].data();
                            signed char* ptr = dst_ra[q].data();

                            __m128 _scale_in0 = scale_in_data_size == 1 ? _mm_set1_ps(scale_in_data_ptr[0]) : _mm_loadu_ps((const float*)scale_in_data_ptr + q * 8);
                            __m128 _scale_in1 = scale_in_data_size == 1 ? _mm_set1_ps(scale_in_data_ptr[0]) : _mm_loadu_ps((const float*)scale_in_data_ptr + q * 8 + 4);
                            __m128 _scale_out0 = scale_out_data_size == 1 ? _mm_set1_ps(scale_out_data_ptr[0]) : _mm_loadu_ps((const float*)scale_out_data_ptr + q * 8);
                            __m128 _scale_out1 = scale_out_data_size == 1 ? _mm_set1_ps(scale_out_data_ptr[0]) : _mm_loadu_ps((const float*)scale_out_data_ptr + q * 8 + 4);
                            __m128 _bias0 = bias_data_size == 1 ? _mm_set1_ps(bias_data_a[0]) : _mm_loadu_ps((const float*)bias_data_a + q * 8);
                            __m128 _bias1 = bias_data_size == 1 ? _mm_set1_ps(bias_data_a[0]) : _mm_loadu_ps((const float*)bias_data_a + q * 8 + 4);

                            for (int i = 0; i < size; i++) {
                                __m128 _v0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr0));
                                __m128 _v1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr1));
                                _v0 = _mm_add_ps(_bias0, _mm_mul_ps(_v0, _scale_in0));
                                _v1 = _mm_add_ps(_bias1, _mm_mul_ps(_v1, _scale_in1));
                                _v0 = activation_sse(_v0, activation_type, activation_params);
                                _v1 = activation_sse(_v1, activation_type, activation_params);
                                _v0 = _mm_mul_ps(_v0, _scale_out0);
                                _v1 = _mm_mul_ps(_v1, _scale_out1);
                                *(int64_t*)ptr = float2int8_sse(_v0, _v1);

                                intptr0 += 4;
                                intptr1 += 4;
                                ptr += 8;
                            }
                        }
                    });
                }
            } if (out_elempack == 1) {
                if (bias_data_size == 0) {
                    otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                        for (const auto q : otter::irange(begin, end)) {
                            const int* intptr = src_a[q].data();
                            signed char* ptr0 = dst_ra[q * 4].data();
                            signed char* ptr1 = dst_ra[q * 4 + 1].data();
                            signed char* ptr2 = dst_ra[q * 4 + 2].data();
                            signed char* ptr3 = dst_ra[q * 4 + 3].data();

                            __m128 _scale_in = scale_in_data_size == 1 ? _mm_set1_ps(scale_in_data_ptr[0]) : _mm_loadu_ps((const float*)scale_in_data_ptr + q * 4);
                            __m128 _scale_out = scale_out_data_size == 1 ? _mm_set1_ps(scale_out_data_ptr[0]) : _mm_loadu_ps((const float*)scale_out_data_ptr + q * 4);

                            for (int i = 0; i < size; i++)
                            {
                                __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                                _v = _mm_mul_ps(_v, _scale_in);
                                _v = activation_sse(_v, activation_type, activation_params);
                                _v = _mm_mul_ps(_v, _scale_out);
                                int32_t v = float2int8_sse(_v);
                                ptr0[0] = (v >> 0) & 0xff;
                                ptr1[0] = (v >> 8) & 0xff;
                                ptr2[0] = (v >> 16) & 0xff;
                                ptr3[0] = (v >> 24) & 0xff;

                                intptr += 4;
                                ptr0 += 1;
                                ptr1 += 1;
                                ptr2 += 1;
                                ptr3 += 1;
                            }
                        }
                    });
                } else {
                    otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                        for (const auto q : otter::irange(begin, end)) {
                            const int* intptr = src_a[q].data();
                            signed char* ptr0 = dst_ra[q * 4].data();
                            signed char* ptr1 = dst_ra[q * 4 + 1].data();
                            signed char* ptr2 = dst_ra[q * 4 + 2].data();
                            signed char* ptr3 = dst_ra[q * 4 + 3].data();

                            __m128 _scale_in = scale_in_data_size == 1 ? _mm_set1_ps(scale_in_data_ptr[0]) : _mm_loadu_ps((const float*)scale_in_data_ptr + q * 4);
                            __m128 _scale_out = scale_out_data_size == 1 ? _mm_set1_ps(scale_out_data_ptr[0]) : _mm_loadu_ps((const float*)scale_out_data_ptr + q * 4);
                            __m128 _bias = bias_data_size == 1 ? _mm_set1_ps(bias_data_a[0]) : _mm_loadu_ps((const float*)bias_data_a + q * 4);

                            for (int i = 0; i < size; i++)
                            {
                                __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                                _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale_in));
                                _v = activation_sse(_v, activation_type, activation_params);
                                _v = _mm_mul_ps(_v, _scale_out);
                                int32_t v = float2int8_sse(_v);
                                ptr0[0] = (v >> 0) & 0xff;
                                ptr1[0] = (v >> 8) & 0xff;
                                ptr2[0] = (v >> 16) & 0xff;
                                ptr3[0] = (v >> 24) & 0xff;

                                intptr += 4;
                                ptr0 += 1;
                                ptr1 += 1;
                                ptr2 += 1;
                                ptr3 += 1;
                            }
                        }
                    });
                }
            }
        } else if (dims == 4) {
            int batchsize = src.size(0);
            int w = src.size(3);
            int h = src.size(2);
            int channels = src.size(1);
            int size = w * h;
            int out_elempack = pack && channels * elempack % 8 == 0 ? 8 : 1;
            int outc = channels * elempack / out_elempack;
            
            dst = otter::empty({batchsize, outc, h, w}, get_update_scalarType(otter::ScalarType::Byte, out_elempack));
            
            for (const auto b : otter::irange(0, batchsize)) {
                auto src_a = src.accessor<int, 4>()[b];
                auto dst_ra = dst.raw_accessor<signed char, 4>()[b];
                
                if (out_elempack == 8) {
                    if (bias_data_size == 0) {
                        otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                            for (const auto q : otter::irange(begin, end)) {
                                const int* intptr0 = src_a[q * 2].data();
                                const int* intptr1 = src_a[q * 2 + 1].data();
                                signed char* ptr = dst_ra[q].data();

                                __m128 _scale_in0 = scale_in_data_size == 1 ? _mm_set1_ps(scale_in_data_ptr[0]) : _mm_loadu_ps((const float*)scale_in_data_ptr + q * 8);
                                __m128 _scale_in1 = scale_in_data_size == 1 ? _mm_set1_ps(scale_in_data_ptr[0]) : _mm_loadu_ps((const float*)scale_in_data_ptr + q * 8 + 4);
                                __m128 _scale_out0 = scale_out_data_size == 1 ? _mm_set1_ps(scale_out_data_ptr[0]) : _mm_loadu_ps((const float*)scale_out_data_ptr + q * 8);
                                __m128 _scale_out1 = scale_out_data_size == 1 ? _mm_set1_ps(scale_out_data_ptr[0]) : _mm_loadu_ps((const float*)scale_out_data_ptr + q * 8 + 4);

                                for (int i = 0; i < size; i++)
                                {
                                    __m128 _v0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr0));
                                    __m128 _v1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr1));
                                    _v0 = _mm_mul_ps(_v0, _scale_in0);
                                    _v1 = _mm_mul_ps(_v1, _scale_in1);
                                    _v0 = activation_sse(_v0, activation_type, activation_params);
                                    _v1 = activation_sse(_v1, activation_type, activation_params);
                                    _v0 = _mm_mul_ps(_v0, _scale_out0);
                                    _v1 = _mm_mul_ps(_v1, _scale_out1);
                                    *(int64_t*)ptr = float2int8_sse(_v0, _v1);

                                    intptr0 += 4;
                                    intptr1 += 4;
                                    ptr += 8;
                                }
                            }
                        });
                    } else {
                        otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                            for (const auto q : otter::irange(begin, end)) {
                                const int* intptr0 = src_a[q * 2].data();
                                const int* intptr1 = src_a[q * 2 + 1].data();
                                signed char* ptr = dst_ra[q].data();

                                __m128 _scale_in0 = scale_in_data_size == 1 ? _mm_set1_ps(scale_in_data_ptr[0]) : _mm_loadu_ps((const float*)scale_in_data_ptr + q * 8);
                                __m128 _scale_in1 = scale_in_data_size == 1 ? _mm_set1_ps(scale_in_data_ptr[0]) : _mm_loadu_ps((const float*)scale_in_data_ptr + q * 8 + 4);
                                __m128 _scale_out0 = scale_out_data_size == 1 ? _mm_set1_ps(scale_out_data_ptr[0]) : _mm_loadu_ps((const float*)scale_out_data_ptr + q * 8);
                                __m128 _scale_out1 = scale_out_data_size == 1 ? _mm_set1_ps(scale_out_data_ptr[0]) : _mm_loadu_ps((const float*)scale_out_data_ptr + q * 8 + 4);
                                __m128 _bias0 = bias_data_size == 1 ? _mm_set1_ps(bias_data_a[0]) : _mm_loadu_ps((const float*)bias_data_a + q * 8);
                                __m128 _bias1 = bias_data_size == 1 ? _mm_set1_ps(bias_data_a[0]) : _mm_loadu_ps((const float*)bias_data_a + q * 8 + 4);

                                for (int i = 0; i < size; i++) {
                                    __m128 _v0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr0));
                                    __m128 _v1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr1));
                                    _v0 = _mm_add_ps(_bias0, _mm_mul_ps(_v0, _scale_in0));
                                    _v1 = _mm_add_ps(_bias1, _mm_mul_ps(_v1, _scale_in1));
                                    _v0 = activation_sse(_v0, activation_type, activation_params);
                                    _v1 = activation_sse(_v1, activation_type, activation_params);
                                    _v0 = _mm_mul_ps(_v0, _scale_out0);
                                    _v1 = _mm_mul_ps(_v1, _scale_out1);
                                    *(int64_t*)ptr = float2int8_sse(_v0, _v1);

                                    intptr0 += 4;
                                    intptr1 += 4;
                                    ptr += 8;
                                }
                            }
                        });
                    }
                } if (out_elempack == 1) {
                    if (bias_data_size == 0) {
                        otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                            for (const auto q : otter::irange(begin, end)) {
                                const int* intptr = src_a[q].data();
                                signed char* ptr0 = dst_ra[q * 4].data();
                                signed char* ptr1 = dst_ra[q * 4 + 1].data();
                                signed char* ptr2 = dst_ra[q * 4 + 2].data();
                                signed char* ptr3 = dst_ra[q * 4 + 3].data();

                                __m128 _scale_in = scale_in_data_size == 1 ? _mm_set1_ps(scale_in_data_ptr[0]) : _mm_loadu_ps((const float*)scale_in_data_ptr + q * 4);
                                __m128 _scale_out = scale_out_data_size == 1 ? _mm_set1_ps(scale_out_data_ptr[0]) : _mm_loadu_ps((const float*)scale_out_data_ptr + q * 4);

                                for (int i = 0; i < size; i++)
                                {
                                    __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                                    _v = _mm_mul_ps(_v, _scale_in);
                                    _v = activation_sse(_v, activation_type, activation_params);
                                    _v = _mm_mul_ps(_v, _scale_out);
                                    int32_t v = float2int8_sse(_v);
                                    ptr0[0] = (v >> 0) & 0xff;
                                    ptr1[0] = (v >> 8) & 0xff;
                                    ptr2[0] = (v >> 16) & 0xff;
                                    ptr3[0] = (v >> 24) & 0xff;

                                    intptr += 4;
                                    ptr0 += 1;
                                    ptr1 += 1;
                                    ptr2 += 1;
                                    ptr3 += 1;
                                }
                            }
                        });
                    } else {
                        otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                            for (const auto q : otter::irange(begin, end)) {
                                const int* intptr = src_a[q].data();
                                signed char* ptr0 = dst_ra[q * 4].data();
                                signed char* ptr1 = dst_ra[q * 4 + 1].data();
                                signed char* ptr2 = dst_ra[q * 4 + 2].data();
                                signed char* ptr3 = dst_ra[q * 4 + 3].data();

                                __m128 _scale_in = scale_in_data_size == 1 ? _mm_set1_ps(scale_in_data_ptr[0]) : _mm_loadu_ps((const float*)scale_in_data_ptr + q * 4);
                                __m128 _scale_out = scale_out_data_size == 1 ? _mm_set1_ps(scale_out_data_ptr[0]) : _mm_loadu_ps((const float*)scale_out_data_ptr + q * 4);
                                __m128 _bias = bias_data_size == 1 ? _mm_set1_ps(bias_data_a[0]) : _mm_loadu_ps((const float*)bias_data_a + q * 4);

                                for (int i = 0; i < size; i++)
                                {
                                    __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                                    _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale_in));
                                    _v = activation_sse(_v, activation_type, activation_params);
                                    _v = _mm_mul_ps(_v, _scale_out);
                                    int32_t v = float2int8_sse(_v);
                                    ptr0[0] = (v >> 0) & 0xff;
                                    ptr1[0] = (v >> 8) & 0xff;
                                    ptr2[0] = (v >> 16) & 0xff;
                                    ptr3[0] = (v >> 24) & 0xff;

                                    intptr += 4;
                                    ptr0 += 1;
                                    ptr1 += 1;
                                    ptr2 += 1;
                                    ptr3 += 1;
                                }
                            }
                        });
                    }
                }
            }
        }
    }
    
    if (dims == 1) {
        int w = src.size(0);

        dst = otter::empty({w}, otter::ScalarType::Byte);

        const int* intptr = (const int*)src.data_ptr();
        signed char* ptr = (signed char*)dst.data_ptr();

        if (scale_in_data_size == 1 && scale_out_data_size == 1) {
            const float scale_in = scale_in_data_ptr[0];
            const float scale_out = scale_out_data_ptr[0];

            if (bias_data_size == 0) {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (int i = 0; i < w; i++) {
                        float v = intptr[i] * scale_in;
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                    }
                });
            } else if (bias_data_size == 1) {
                const float bias = bias_data_a[0];

                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        float v = intptr[i] * scale_in + bias;
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                    }
                });
            } else {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        float v = intptr[i] * scale_in + bias_data_a[i];
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                    }
                });
            }
        } else if (scale_in_data_size == 1 && scale_out_data_size > 1) {
            const float scale_in = scale_in_data_ptr[0];

            if (bias_data_size == 0) {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        float v = intptr[i] * scale_in;
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data_ptr[i]);
                    }
                });
            } else if (bias_data_size == 1) {
                const float bias = bias_data_a[0];

                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        float v = intptr[i] * scale_in + bias;
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data_ptr[i]);
                    }
                });
            } else {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        float v = intptr[i] * scale_in + bias_data_a[i];
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data_ptr[i]);
                    }
                });
            }
        } else if (scale_in_data_size > 1 && scale_out_data_size == 1) {
            const float scale_out = scale_out_data_ptr[0];

            if (bias_data_size == 0) {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        float v = intptr[i] * scale_in_data_ptr[i];
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                    }
                });
            } else if (bias_data_size == 1) {
                const float bias = bias_data_a[0];

                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        float v = intptr[i] * scale_in_data_ptr[i] + bias;
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                    }
                });
            } else {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        float v = intptr[i] * scale_in_data_ptr[i] + bias_data_a[i];
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                    }
                });
            }
        } else {
            if (bias_data_size == 0) {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        float v = intptr[i] * scale_in_data_ptr[i];
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data_ptr[i]);
                    }
                });
            } else if (bias_data_size == 1) {
                const float bias = bias_data_a[0];

                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        float v = intptr[i] * scale_in_data_ptr[i] + bias;
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data_ptr[i]);
                    }
                });
            } else {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        float v = intptr[i] * scale_in_data_ptr[i] + bias_data_a[i];
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data_ptr[i]);
                    }
                });
            }
        }
    } else if (dims == 2) {
        int w = src.size(1);
        int h = src.size(0);

        dst = otter::empty({h, w}, otter::ScalarType::Byte);
        
        auto src_a = src.accessor<int, 2>();
        auto dst_ra = dst.accessor<signed char, 2>();

        if (bias_data_size == 0) {
            otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    const int* intptr = src_a[i].data();
                    signed char* ptr = dst_ra[i].data();

                    const float scale_in = scale_in_data_size == 1 ? scale_in_data_ptr[0] : scale_in_data_ptr[i];
                    const float scale_out = scale_out_data_size == 1 ? scale_out_data_ptr[0] : scale_out_data_ptr[i];

                    for (int j = 0; j < w; j++) {
                        float v = intptr[j] * scale_in;
                        ptr[j] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                    }
                }
            });
        } else {
            otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    const int* intptr = src_a[i].data();
                    signed char* ptr = dst_ra[i].data();

                    const float scale_in = scale_in_data_size == 1 ? scale_in_data_ptr[0] : scale_in_data_ptr[i];
                    const float scale_out = scale_out_data_size == 1 ? scale_out_data_ptr[0] : scale_out_data_ptr[i];
                    const float bias = bias_data_size == 1 ? bias_data_a[0] : bias_data_a[i];

                    for (int j = 0; j < w; j++) {
                        float v = intptr[j] * scale_in + bias;
                        ptr[j] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                    }
                }
            });
        }
    } else if (dims == 3) {
        int w = src.size(2);
        int h = src.size(1);
        int channels = src.size(0);
        int size = w * h;
        
        dst = otter::empty({channels, h, w}, otter::ScalarType::Byte);

        auto src_a = src.accessor<int, 3>();
        auto dst_ra = dst.raw_accessor<signed char, 3>();
        
        if (bias_data_size == 0) {
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const int* intptr = src_a[q].data();
                    signed char* ptr = dst_ra[q].data();

                    const float scale_in = scale_in_data_size == 1 ? scale_in_data_ptr[0] : scale_in_data_ptr[q];
                    const float scale_out = scale_out_data_size == 1 ? scale_out_data_ptr[0] : scale_out_data_ptr[q];

                    for (int i = 0; i < size; i++) {
                        float v = intptr[i] * scale_in;
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                    }
                }
            });
        } else {
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const int* intptr = src_a[q].data();
                    signed char* ptr = dst_ra[q].data();

                    const float scale_in = scale_in_data_size == 1 ? scale_in_data_ptr[0] : scale_in_data_ptr[q];
                    const float scale_out = scale_out_data_size == 1 ? scale_out_data_ptr[0] : scale_out_data_ptr[q];
                    const float bias = bias_data_size == 1 ? bias_data_a[0] : bias_data_a[q];

                    for (int i = 0; i < size; i++) {
                        float v = intptr[i] * scale_in + bias;
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                    }
                }
            });
        }
    } else if (dims == 4) {
        int batchsize = src.size(0);
        int w = src.size(3);
        int h = src.size(2);
        int channels = src.size(1);
        int size = w * h;
        
        dst = otter::empty({batchsize, channels, h, w}, otter::ScalarType::Byte);

        for (const auto b : otter::irange(0, batchsize)) {
        
            auto src_a = src.accessor<int, 4>()[b];
            auto dst_ra = dst.raw_accessor<signed char, 4>()[b];
            
            if (bias_data_size == 0) {
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const int* intptr = src_a[q].data();
                        signed char* ptr = dst_ra[q].data();

                        const float scale_in = scale_in_data_size == 1 ? scale_in_data_ptr[0] : scale_in_data_ptr[q];
                        const float scale_out = scale_out_data_size == 1 ? scale_out_data_ptr[0] : scale_out_data_ptr[q];

                        for (int i = 0; i < size; i++) {
                            float v = intptr[i] * scale_in;
                            ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                        }
                    }
                });
            } else {
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const int* intptr = src_a[q].data();
                        signed char* ptr = dst_ra[q].data();

                        const float scale_in = scale_in_data_size == 1 ? scale_in_data_ptr[0] : scale_in_data_ptr[q];
                        const float scale_out = scale_out_data_size == 1 ? scale_out_data_ptr[0] : scale_out_data_ptr[q];
                        const float bias = bias_data_size == 1 ? bias_data_a[0] : bias_data_a[q];

                        for (int i = 0; i < size; i++) {
                            float v = intptr[i] * scale_in + bias;
                            ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                        }
                    }
                });
            }
        }
    }
    
    return dst;
}

}   // end namespace otter

#endif  // __SSE2__
