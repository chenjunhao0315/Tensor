//
//  QuantizeNeon.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/27.
//

#if __ARM_NEON__

#include "QuantizeNeon.hpp"
#include "TensorFactory.hpp"
#include "Parallel.hpp"
#include "TensorPacking.hpp"
#include "VecIntrinsic.hpp"

namespace otter {

static inline float activation_ss(float v, int activation_type, const Tensor& activation_params) {
    if (activation_type == 1) {
        v = fmax(v, 0.f);
    } else if (activation_type == 2) {
        float slope = activation_params.data_ptr<float>()[0];
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

Tensor quantize_to_int8_neon(const Tensor& src, const Tensor& scale_data, bool pack) {
    int dims = src.dim();
    int elempack = src.elempack();
    
    int scale_data_size = scale_data.size(0);
    const float* scale_data_ptr = (const float*)scale_data.data_ptr();
    
    Tensor dst;
    
    if (elempack == 4) {
        if (dims == 1) {
            int w = src.size(0);
            int out_elempack = pack && w * elempack % 8 == 0 ? 8 : 1;
            int outw = w * elempack / out_elempack;
            
            dst = otter::empty({outw}, get_update_scalarType(otter::ScalarType::Byte, out_elempack));
            
            const float* src_ptr = (const float*)src.data_ptr();
            signed char* dst_ptr = (signed char*)dst.data_ptr();
            
            if (scale_data_size == 1) {
                const float scale = scale_data_ptr[0];
                
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
                        
                        outptr[0] = float2int8(ptr0[0] * scale_data_ptr[i * 4]);
                        outptr[1] = float2int8(ptr0[1] * scale_data_ptr[i * 4 + 1]);
                        outptr[2] = float2int8(ptr0[2] * scale_data_ptr[i * 4 + 2]);
                        outptr[3] = float2int8(ptr0[3] * scale_data_ptr[i * 4 + 3]);
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
                    float32x4_t _scale = vdupq_n_f32(scale_data_ptr[0]);
                    
                    otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const float* ptr0 = src_a[i * 2].data();
                            const float* ptr1 = src_a[i * 2 + 1].data();
                            signed char* outptr = dst_ra[i].data();
                            
                            for (int j = 0; j < w; j++) {
                                float32x4_t _vlow = vld1q_f32(ptr0);
                                float32x4_t _vhigh = vld1q_f32(ptr1);
                                _vlow = vmulq_f32(_vlow, _scale);
                                _vhigh = vmulq_f32(_vhigh, _scale);
                                int8x8_t _v = float2int8(_vlow, _vhigh);
                                vst1_s8(outptr, _v);

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
                            
                            float32x4_t _scale0 = vld1q_f32((const float*)scale_data_ptr + i * 8);
                            float32x4_t _scale1 = vld1q_f32((const float*)scale_data_ptr + i * 8 + 4);
                            
                            for (int j = 0; j < w; j++) {
                                float32x4_t _vlow = vld1q_f32(ptr0);
                                float32x4_t _vhigh = vld1q_f32(ptr1);
                                _vlow = vmulq_f32(_vlow, _scale0);
                                _vhigh = vmulq_f32(_vhigh, _scale1);
                                int8x8_t _v = float2int8(_vlow, _vhigh);
                                vst1_s8(outptr, _v);

                                ptr0 += 4;
                                ptr1 += 4;
                                outptr += 8;
                            }
                        }
                    });
                }
            } else if (out_elempack == 1) {
                if (scale_data_size == 1) {
                    const float scale = scale_data_ptr[0];
                    
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
                            
                            const float s0 = scale_data_ptr[i * 4];
                            const float s1 = scale_data_ptr[i * 4 + 1];
                            const float s2 = scale_data_ptr[i * 4 + 2];
                            const float s3 = scale_data_ptr[i * 4 + 3];
                            
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
                    float32x4_t _scale = vdupq_n_f32(scale_data_ptr[0]);

                    otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                        for (const auto q : otter::irange(begin, end)) {
                            const float* ptr0 = src_a[q * 2].data();
                            const float* ptr1 = src_a[q * 2 + 1].data();
                            signed char* outptr = dst_ra[q].data();

                            int i = 0;
                            for (; i + 1 < size; i += 2) {
                                float32x4_t _v0 = vld1q_f32(ptr0);
                                float32x4_t _v1 = vld1q_f32(ptr0 + 4);
                                float32x4_t _v2 = vld1q_f32(ptr1);
                                float32x4_t _v3 = vld1q_f32(ptr1 + 4);
                                _v0 = vmulq_f32(_v0, _scale);
                                _v1 = vmulq_f32(_v1, _scale);
                                _v2 = vmulq_f32(_v2, _scale);
                                _v3 = vmulq_f32(_v3, _scale);
                                vst1_s8(outptr, float2int8(_v0, _v2));
                                vst1_s8(outptr + 8, float2int8(_v1, _v3));

                                ptr0 += 8;
                                ptr1 += 8;
                                outptr += 16;
                            }
                            for (; i < size; i++) {
                                float32x4_t _vlow = vld1q_f32(ptr0);
                                float32x4_t _vhigh = vld1q_f32(ptr1);
                                _vlow = vmulq_f32(_vlow, _scale);
                                _vhigh = vmulq_f32(_vhigh, _scale);
                                int8x8_t _v = float2int8(_vlow, _vhigh);
                                vst1_s8(outptr, _v);

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

                            float32x4_t _scale0 = vld1q_f32((const float*)scale_data_ptr + q * 8);
                            float32x4_t _scale1 = vld1q_f32((const float*)scale_data_ptr + q * 8 + 4);

                            int i = 0;
                            for (; i < size; i++) {
                                float32x4_t _vlow = vld1q_f32(ptr0);
                                float32x4_t _vhigh = vld1q_f32(ptr1);
                                _vlow = vmulq_f32(_vlow, _scale0);
                                _vhigh = vmulq_f32(_vhigh, _scale1);
                                int8x8_t _v = float2int8(_vlow, _vhigh);
                                vst1_s8(outptr, _v);

                                ptr0 += 4;
                                ptr1 += 4;
                                outptr += 8;
                            }
                        }
                    });
                }
            } else if (out_elempack == 1) {
                if (scale_data_size == 1) {
                    const float scale = scale_data_ptr[0];

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

                            const float s0 = scale_data_ptr[q * 4];
                            const float s1 = scale_data_ptr[q * 4 + 1];
                            const float s2 = scale_data_ptr[q * 4 + 2];
                            const float s3 = scale_data_ptr[q * 4 + 3];

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
                        float32x4_t _scale = vdupq_n_f32(scale_data_ptr[0]);

                        otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                            for (const auto q : otter::irange(begin, end)) {
                                const float* ptr0 = src_a[q * 2].data();
                                const float* ptr1 = src_a[q * 2 + 1].data();
                                signed char* outptr = dst_ra[q].data();

                                int i = 0;
                                for (; i + 1 < size; i += 2) {
                                    float32x4_t _v0 = vld1q_f32(ptr0);
                                    float32x4_t _v1 = vld1q_f32(ptr0 + 4);
                                    float32x4_t _v2 = vld1q_f32(ptr1);
                                    float32x4_t _v3 = vld1q_f32(ptr1 + 4);
                                    _v0 = vmulq_f32(_v0, _scale);
                                    _v1 = vmulq_f32(_v1, _scale);
                                    _v2 = vmulq_f32(_v2, _scale);
                                    _v3 = vmulq_f32(_v3, _scale);
                                    vst1_s8(outptr, float2int8(_v0, _v2));
                                    vst1_s8(outptr + 8, float2int8(_v1, _v3));

                                    ptr0 += 8;
                                    ptr1 += 8;
                                    outptr += 16;
                                }
                                for (; i < size; i++) {
                                    float32x4_t _vlow = vld1q_f32(ptr0);
                                    float32x4_t _vhigh = vld1q_f32(ptr1);
                                    _vlow = vmulq_f32(_vlow, _scale);
                                    _vhigh = vmulq_f32(_vhigh, _scale);
                                    int8x8_t _v = float2int8(_vlow, _vhigh);
                                    vst1_s8(outptr, _v);

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

                                float32x4_t _scale0 = vld1q_f32((const float*)scale_data_ptr + q * 8);
                                float32x4_t _scale1 = vld1q_f32((const float*)scale_data_ptr + q * 8 + 4);

                                int i = 0;
                                for (; i < size; i++) {
                                    float32x4_t _vlow = vld1q_f32(ptr0);
                                    float32x4_t _vhigh = vld1q_f32(ptr1);
                                    _vlow = vmulq_f32(_vlow, _scale0);
                                    _vhigh = vmulq_f32(_vhigh, _scale1);
                                    int8x8_t _v = float2int8(_vlow, _vhigh);
                                    vst1_s8(outptr, _v);

                                    ptr0 += 4;
                                    ptr1 += 4;
                                    outptr += 8;
                                }
                            }
                        });
                    }
                } else if (out_elempack == 1) {
                    if (scale_data_size == 1) {
                        const float scale = scale_data_ptr[0];

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

                                const float s0 = scale_data_ptr[q * 4];
                                const float s1 = scale_data_ptr[q * 4 + 1];
                                const float s2 = scale_data_ptr[q * 4 + 2];
                                const float s3 = scale_data_ptr[q * 4 + 3];

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
            const float scale = scale_data_ptr[0];

            otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    outptr[i] = float2int8(ptr[i] * scale);
                }
            });
        } else {
            otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                for (const auto i : otter::irange(begin, end)) {
                    outptr[i] = float2int8(ptr[i] * scale_data_ptr[i]);
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

                const float scale = scale_data_size == 1 ? scale_data_ptr[0] : scale_data_ptr[i];

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

                const float scale = scale_data_size == 1 ? scale_data_ptr[0] : scale_data_ptr[q];

                int i = 0;
    #if __ARM_NEON
                float32x4_t _scale = vdupq_n_f32(scale);
                for (; i + 15 < size; i += 16) {
                    float32x4_t _v0 = vld1q_f32(ptr);
                    float32x4_t _v1 = vld1q_f32(ptr + 4);
                    float32x4_t _v2 = vld1q_f32(ptr + 8);
                    float32x4_t _v3 = vld1q_f32(ptr + 12);
                    _v0 = vmulq_f32(_v0, _scale);
                    _v1 = vmulq_f32(_v1, _scale);
                    _v2 = vmulq_f32(_v2, _scale);
                    _v3 = vmulq_f32(_v3, _scale);
                    vst1_s8(outptr, float2int8(_v0, _v1));
                    vst1_s8(outptr + 8, float2int8(_v2, _v3));

                    ptr += 16;
                    outptr += 16;
                }
                for (; i + 7 < size; i += 8) {
                    float32x4_t _v0 = vld1q_f32(ptr);
                    float32x4_t _v1 = vld1q_f32(ptr + 4);
                    _v0 = vmulq_f32(_v0, _scale);
                    _v1 = vmulq_f32(_v1, _scale);
                    int8x8_t _v = float2int8(_v0, _v1);
                    vst1_s8(outptr, _v);

                    ptr += 8;
                    outptr += 8;
                }
    #endif // __ARM_NEON
                for (; i < size; i++) {
                    *outptr++ = float2int8(*ptr++ * scale);
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

                    const float scale = scale_data_size == 1 ? scale_data_ptr[0] : scale_data_ptr[q];

                    int i = 0;
        #if __ARM_NEON
                    float32x4_t _scale = vdupq_n_f32(scale);
                    for (; i + 15 < size; i += 16) {
                        float32x4_t _v0 = vld1q_f32(ptr);
                        float32x4_t _v1 = vld1q_f32(ptr + 4);
                        float32x4_t _v2 = vld1q_f32(ptr + 8);
                        float32x4_t _v3 = vld1q_f32(ptr + 12);
                        _v0 = vmulq_f32(_v0, _scale);
                        _v1 = vmulq_f32(_v1, _scale);
                        _v2 = vmulq_f32(_v2, _scale);
                        _v3 = vmulq_f32(_v3, _scale);
                        vst1_s8(outptr, float2int8(_v0, _v1));
                        vst1_s8(outptr + 8, float2int8(_v2, _v3));

                        ptr += 16;
                        outptr += 16;
                    }
                    for (; i + 7 < size; i += 8) {
                        float32x4_t _v0 = vld1q_f32(ptr);
                        float32x4_t _v1 = vld1q_f32(ptr + 4);
                        _v0 = vmulq_f32(_v0, _scale);
                        _v1 = vmulq_f32(_v1, _scale);
                        int8x8_t _v = float2int8(_v0, _v1);
                        vst1_s8(outptr, _v);

                        ptr += 8;
                        outptr += 8;
                    }
        #endif // __ARM_NEON
                    for (; i < size; i++) {
                        *outptr++ = float2int8(*ptr++ * scale);
                    }
                }
            });
        }
    }
    
    return dst;
}

Tensor dequantize_from_int32_neon(const Tensor& src, const Tensor& scale_data, const Tensor& bias_data, bool pack) {
    int dims = src.dim();
    int elempack = src.elempack();
    
    Tensor dst;
    
    int scale_data_size = scale_data.size(0);
    const float* scale_data_ptr = (const float*)scale_data.data_ptr();
    
    int bias_data_size = bias_data.size(0);
    const float* bias_data_ptr = (const float*)bias_data.data_ptr();
    
    if (elempack == 8) {
        if (dims == 1) {
            int w = src.size(0);
            int outw = w * 2;
            
            dst = otter::empty({outw}, otter::ScalarType::Float4);
            
            const int* srcptr = (const int*)src.data_ptr();
            float* dstptr = (float*)dst.data_ptr();

            if (scale_data_size == 1) {
                float32x4_t _scale = vdupq_n_f32(scale_data_ptr[0]);

                if (bias_data_size == 0) {
                    otter::parallel_for(0, outw, 0, [&](int64_t begin, int64_t end) {
                        for (int i = 0; i < outw; i++) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            float* ptr = (float*)dstptr + i * 4;

                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmulq_f32(_v, _scale);
                            vst1q_f32(ptr, _v);
                        }
                    });
                } else if (bias_data_size == 1) {
                    float32x4_t _bias = vdupq_n_f32(bias_data_ptr[0]);

                    otter::parallel_for(0, outw, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            float* ptr = (float*)dstptr + i * 4;

                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
    #if __aarch64__
                            _v = vfmaq_f32(_bias, _v, _scale);
    #else
                            _v = vmlaq_f32(_bias, _v, _scale);
    #endif
                            vst1q_f32(ptr, _v);
                        }
                    });
                } else {
                    otter::parallel_for(0, outw, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            float* ptr = (float*)dstptr + i * 4;

                            float32x4_t _bias = vld1q_f32((const float*)bias_data_ptr + i * 4);
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
    #if __aarch64__
                            _v = vfmaq_f32(_bias, _v, _scale);
    #else
                            _v = vmlaq_f32(_bias, _v, _scale);
    #endif
                            vst1q_f32(ptr, _v);
                        }
                    });
                }
            } else {
                if (bias_data_size == 0) {
                    otter::parallel_for(0, outw, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            float* ptr = (float*)dstptr + i * 4;

                            float32x4_t _scale = vld1q_f32((const float*)scale_data_ptr + i * 4);
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmulq_f32(_v, _scale);
                            vst1q_f32(ptr, _v);
                        }
                    });
                } else if (bias_data_size == 1) {
                    float32x4_t _bias = vdupq_n_f32(bias_data_ptr[0]);

                    otter::parallel_for(0, outw, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            float* ptr = (float*)dstptr + i * 4;

                            float32x4_t _scale = vld1q_f32((const float*)scale_data_ptr + i * 4);
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
    #if __aarch64__
                            _v = vfmaq_f32(_bias, _v, _scale);
    #else
                            _v = vmlaq_f32(_bias, _v, _scale);
    #endif
                            vst1q_f32(ptr, _v);
                        }
                    });
                } else {
                    otter::parallel_for(0, outw, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            float* ptr = (float*)dstptr + i * 4;

                            float32x4_t _scale = vld1q_f32((const float*)scale_data_ptr + i * 4);
                            float32x4_t _bias = vld1q_f32((const float*)bias_data_ptr + i * 4);
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
    #if __aarch64__
                            _v = vfmaq_f32(_bias, _v, _scale);
    #else
                            _v = vmlaq_f32(_bias, _v, _scale);
    #endif
                            vst1q_f32(ptr, _v);
                        }
                    });
                }
            }
        } else if (dims == 2) {
            int w = src.size(1);
            int h = src.size(0);
            int outh = h * 2;

            dst = otter::empty({outh, w}, otter::ScalarType::Float4);
            
            auto src_a = src.accessor<int, 2, 8>();
            auto dst_a = dst.accessor<float, 2, 4>();

            if (bias_data_size == 0) {
                otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        const int* intptr = src_a[i].data();
                        float* ptr0 = dst_a[i * 2].data();
                        float* ptr1 = dst_a[i * 2 + 1].data();

                        float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data_ptr[0]) : vld1q_f32((const float*)scale_data_ptr + i * 8);
                        float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data_ptr[0]) : vld1q_f32((const float*)scale_data_ptr + i * 8 + 4);

                        for (int j = 0; j < w; j++) {
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                            _v0 = vmulq_f32(_v0, _scale0);
                            _v1 = vmulq_f32(_v1, _scale1);
                            vst1q_f32(ptr0, _v0);
                            vst1q_f32(ptr1, _v1);

                            intptr += 8;
                            ptr0 += 4;
                            ptr1 += 4;
                        }
                    }
                });
            } else {
                otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        const int* intptr = src_a[i].data();
                        float* ptr0 = dst_a[i * 2].data();
                        float* ptr1 = dst_a[i * 2 + 1].data();

                        float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data_ptr[0]) : vld1q_f32((const float*)scale_data_ptr + i * 8);
                        float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data_ptr[0]) : vld1q_f32((const float*)scale_data_ptr + i * 8 + 4);
                        float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + i * 8);
                        float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + i * 8 + 4);

                        for (int j = 0; j < w; j++) {
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
    #if __aarch64__
                            _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                            _v1 = vfmaq_f32(_bias1, _v1, _scale1);
    #else
                            _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                            _v1 = vmlaq_f32(_bias1, _v1, _scale1);
    #endif
                            vst1q_f32(ptr0, _v0);
                            vst1q_f32(ptr1, _v1);

                            intptr += 8;
                            ptr0 += 4;
                            ptr1 += 4;
                        }
                    }
                });
            }
        } else if (dims == 3) {
            int w = src.size(2);
            int h = src.size(1);
            int channels = src.size(0);
            int size = w * h;
            int outc = channels * 2;
            
            dst = otter::empty({outc, h, w}, otter::ScalarType::Float4);

            auto src_a = src.accessor<int, 3, 8>();
            auto dst_a = dst.accessor<float, 3, 4>();
            
            if (bias_data_size == 0) {
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const int* intptr = src_a[q].data();
                        float* ptr0 = dst_a[q * 2].data();
                        float* ptr1 = dst_a[q * 2 + 1].data();

                        float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data_ptr[0]) : vld1q_f32((const float*)scale_data_ptr + q * 8);
                        float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data_ptr[0]) : vld1q_f32((const float*)scale_data_ptr + q * 8 + 4);

                        int i = 0;
                        for (; i + 1 < size; i += 2) {
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                            float32x4_t _v2 = vcvtq_f32_s32(vld1q_s32(intptr + 8));
                            float32x4_t _v3 = vcvtq_f32_s32(vld1q_s32(intptr + 12));
                            _v0 = vmulq_f32(_v0, _scale0);
                            _v1 = vmulq_f32(_v1, _scale1);
                            _v2 = vmulq_f32(_v2, _scale0);
                            _v3 = vmulq_f32(_v3, _scale1);
                            vst1q_f32(ptr0, _v0);
                            vst1q_f32(ptr0 + 4, _v2);
                            vst1q_f32(ptr1, _v1);
                            vst1q_f32(ptr1 + 4, _v3);

                            intptr += 16;
                            ptr0 += 8;
                            ptr1 += 8;
                        }
                        for (; i < size; i++) {
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                            _v0 = vmulq_f32(_v0, _scale0);
                            _v1 = vmulq_f32(_v1, _scale1);
                            vst1q_f32(ptr0, _v0);
                            vst1q_f32(ptr1, _v1);

                            intptr += 8;
                            ptr0 += 4;
                            ptr1 += 4;
                        }
                    }
                });
            } else {
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const int* intptr = src_a[q].data();
                        float* ptr0 = dst_a[q * 2].data();
                        float* ptr1 = dst_a[q * 2 + 1].data();

                        float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data_ptr[0]) : vld1q_f32((const float*)scale_data_ptr + q * 8);
                        float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data_ptr[0]) : vld1q_f32((const float*)scale_data_ptr + q * 8 + 4);
                        float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8);
                        float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8 + 4);

                        int i = 0;
                        for (; i + 1 < size; i += 2) {
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                            float32x4_t _v2 = vcvtq_f32_s32(vld1q_s32(intptr + 8));
                            float32x4_t _v3 = vcvtq_f32_s32(vld1q_s32(intptr + 12));
    #if __aarch64__
                            _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                            _v1 = vfmaq_f32(_bias1, _v1, _scale1);
                            _v2 = vfmaq_f32(_bias0, _v2, _scale0);
                            _v3 = vfmaq_f32(_bias1, _v3, _scale1);
    #else
                            _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                            _v1 = vmlaq_f32(_bias1, _v1, _scale1);
                            _v2 = vmlaq_f32(_bias0, _v2, _scale0);
                            _v3 = vmlaq_f32(_bias1, _v3, _scale1);
    #endif
                            vst1q_f32(ptr0, _v0);
                            vst1q_f32(ptr0 + 4, _v2);
                            vst1q_f32(ptr1, _v1);
                            vst1q_f32(ptr1 + 4, _v3);

                            intptr += 16;
                            ptr0 += 8;
                            ptr1 += 8;
                        }
                        for (; i < size; i++) {
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
    #if __aarch64__
                            _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                            _v1 = vfmaq_f32(_bias1, _v1, _scale1);
    #else
                            _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                            _v1 = vmlaq_f32(_bias1, _v1, _scale1);
    #endif
                            vst1q_f32(ptr0, _v0);
                            vst1q_f32(ptr1, _v1);

                            intptr += 8;
                            ptr0 += 4;
                            ptr1 += 4;
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
            int outc = channels * 2;
            
            dst = otter::empty({batchsize, outc, h, w}, otter::ScalarType::Float4);

            for (const auto b : otter::irange(0, batchsize)) {
            
                auto src_a = src.accessor<int, 4, 8>()[b];
                auto dst_a = dst.accessor<float, 4, 4>()[b];
                
                if (bias_data_size == 0) {
                    otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                        for (const auto q : otter::irange(begin, end)) {
                            const int* intptr = src_a[q].data();
                            float* ptr0 = dst_a[q * 2].data();
                            float* ptr1 = dst_a[q * 2 + 1].data();

                            float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data_ptr[0]) : vld1q_f32((const float*)scale_data_ptr + q * 8);
                            float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data_ptr[0]) : vld1q_f32((const float*)scale_data_ptr + q * 8 + 4);

                            int i = 0;
                            for (; i + 1 < size; i += 2) {
                                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                                float32x4_t _v2 = vcvtq_f32_s32(vld1q_s32(intptr + 8));
                                float32x4_t _v3 = vcvtq_f32_s32(vld1q_s32(intptr + 12));
                                _v0 = vmulq_f32(_v0, _scale0);
                                _v1 = vmulq_f32(_v1, _scale1);
                                _v2 = vmulq_f32(_v2, _scale0);
                                _v3 = vmulq_f32(_v3, _scale1);
                                vst1q_f32(ptr0, _v0);
                                vst1q_f32(ptr0 + 4, _v2);
                                vst1q_f32(ptr1, _v1);
                                vst1q_f32(ptr1 + 4, _v3);

                                intptr += 16;
                                ptr0 += 8;
                                ptr1 += 8;
                            }
                            for (; i < size; i++) {
                                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                                _v0 = vmulq_f32(_v0, _scale0);
                                _v1 = vmulq_f32(_v1, _scale1);
                                vst1q_f32(ptr0, _v0);
                                vst1q_f32(ptr1, _v1);

                                intptr += 8;
                                ptr0 += 4;
                                ptr1 += 4;
                            }
                        }
                    });
                } else {
                    otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                        for (const auto q : otter::irange(begin, end)) {
                            const int* intptr = src_a[q].data();
                            float* ptr0 = dst_a[q * 2].data();
                            float* ptr1 = dst_a[q * 2 + 1].data();

                            float32x4_t _scale0 = scale_data_size == 1 ? vdupq_n_f32(scale_data_ptr[0]) : vld1q_f32((const float*)scale_data_ptr + q * 8);
                            float32x4_t _scale1 = scale_data_size == 1 ? vdupq_n_f32(scale_data_ptr[0]) : vld1q_f32((const float*)scale_data_ptr + q * 8 + 4);
                            float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8);
                            float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8 + 4);

                            int i = 0;
                            for (; i + 1 < size; i += 2) {
                                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                                float32x4_t _v2 = vcvtq_f32_s32(vld1q_s32(intptr + 8));
                                float32x4_t _v3 = vcvtq_f32_s32(vld1q_s32(intptr + 12));
        #if __aarch64__
                                _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                                _v1 = vfmaq_f32(_bias1, _v1, _scale1);
                                _v2 = vfmaq_f32(_bias0, _v2, _scale0);
                                _v3 = vfmaq_f32(_bias1, _v3, _scale1);
        #else
                                _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                                _v1 = vmlaq_f32(_bias1, _v1, _scale1);
                                _v2 = vmlaq_f32(_bias0, _v2, _scale0);
                                _v3 = vmlaq_f32(_bias1, _v3, _scale1);
        #endif
                                vst1q_f32(ptr0, _v0);
                                vst1q_f32(ptr0 + 4, _v2);
                                vst1q_f32(ptr1, _v1);
                                vst1q_f32(ptr1 + 4, _v3);

                                intptr += 16;
                                ptr0 += 8;
                                ptr1 += 8;
                            }
                            for (; i < size; i++) {
                                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
        #if __aarch64__
                                _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                                _v1 = vfmaq_f32(_bias1, _v1, _scale1);
        #else
                                _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                                _v1 = vmlaq_f32(_bias1, _v1, _scale1);
        #endif
                                vst1q_f32(ptr0, _v0);
                                vst1q_f32(ptr1, _v1);

                                intptr += 8;
                                ptr0 += 4;
                                ptr1 += 4;
                            }
                        }
                    });
                }
            }
        }
    }
    
    if (elempack == 4) {
        if (dims == 1) {
            int w = src.size(0);
            
            dst = otter::empty({w}, otter::ScalarType::Float4);
            
            const int* srcptr = (const int*)src.data_ptr();
            float* dstptr = (float*)dst.data_ptr();
            
            if (scale_data_size == 1) {
                float32x4_t _scale = vdupq_n_f32(scale_data_ptr[0]);

                if (bias_data_size == 0) {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            float* ptr = (float*)dstptr + i * 4;

                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmulq_f32(_v, _scale);
                            vst1q_f32(ptr, _v);
                        }
                    });
                } else if (bias_data_size == 1) {
                    float32x4_t _bias = vdupq_n_f32(bias_data_ptr[0]);

                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            float* ptr = (float*)dstptr + i * 4;

                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
    #if __aarch64__
                            _v = vfmaq_f32(_bias, _v, _scale);
    #else
                            _v = vmlaq_f32(_bias, _v, _scale);
    #endif
                            vst1q_f32(ptr, _v);
                        }
                    });
                } else {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            float* ptr = (float*)dstptr + i * 4;

                            float32x4_t _bias = vld1q_f32((const float*)bias_data_ptr + i * 4);
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
    #if __aarch64__
                            _v = vfmaq_f32(_bias, _v, _scale);
    #else
                            _v = vmlaq_f32(_bias, _v, _scale);
    #endif
                            vst1q_f32(ptr, _v);
                        }
                    });
                }
            } else {
                if (bias_data_size == 0) {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            float* ptr = (float*)dstptr + i * 4;

                            float32x4_t _scale = vld1q_f32((const float*)scale_data_ptr + i * 4);
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmulq_f32(_v, _scale);
                            vst1q_f32(ptr, _v);
                        }
                    });
                } else if (bias_data_size == 1) {
                    float32x4_t _bias = vdupq_n_f32(bias_data_ptr[0]);

                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            float* ptr = (float*)dstptr + i * 4;

                            float32x4_t _scale = vld1q_f32((const float*)scale_data_ptr + i * 4);
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
    #if __aarch64__
                            _v = vfmaq_f32(_bias, _v, _scale);
    #else
                            _v = vmlaq_f32(_bias, _v, _scale);
    #endif
                            vst1q_f32(ptr, _v);
                        }
                    });
                } else {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            float* ptr = (float*)dstptr + i * 4;

                            float32x4_t _scale = vld1q_f32((const float*)scale_data_ptr + i * 4);
                            float32x4_t _bias = vld1q_f32((const float*)bias_data_ptr + i * 4);
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
    #if __aarch64__
                            _v = vfmaq_f32(_bias, _v, _scale);
    #else
                            _v = vmlaq_f32(_bias, _v, _scale);
    #endif
                            vst1q_f32(ptr, _v);
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

                        float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data_ptr[0]) : vld1q_f32((const float*)scale_data_ptr + i * 4);

                        for (int j = 0; j < w; j++) {
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmulq_f32(_v, _scale);
                            vst1q_f32(ptr, _v);

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

                        float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data_ptr[0]) : vld1q_f32((const float*)scale_data_ptr + i * 4);
                        float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + i * 4);

                        for (int j = 0; j < w; j++) {
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
    #if __aarch64__
                            _v = vfmaq_f32(_bias, _v, _scale);
    #else
                            _v = vmlaq_f32(_bias, _v, _scale);
    #endif
                            vst1q_f32(ptr, _v);

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

                        float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data_ptr[0]) : vld1q_f32((const float*)scale_data_ptr + q * 4);

                        int i = 0;
                        for (; i + 1 < size; i += 2) {
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                            _v0 = vmulq_f32(_v0, _scale);
                            _v1 = vmulq_f32(_v1, _scale);
                            vst1q_f32(ptr, _v0);
                            vst1q_f32(ptr + 4, _v1);

                            intptr += 8;
                            ptr += 8;
                        }
                        for (; i < size; i++) {
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmulq_f32(_v, _scale);
                            vst1q_f32(ptr, _v);

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

                        float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data_ptr[0]) : vld1q_f32((const float*)scale_data_ptr + q * 4);
                        float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 4);

                        int i = 0;
                        for (; i + 1 < size; i += 2) {
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
    #if __aarch64__
                            _v0 = vfmaq_f32(_bias, _v0, _scale);
                            _v1 = vfmaq_f32(_bias, _v1, _scale);
    #else
                            _v0 = vmlaq_f32(_bias, _v0, _scale);
                            _v1 = vmlaq_f32(_bias, _v1, _scale);
    #endif
                            vst1q_f32(ptr, _v0);
                            vst1q_f32(ptr + 4, _v1);

                            intptr += 8;
                            ptr += 8;
                        }
                        for (; i < size; i++) {
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
    #if __aarch64__
                            _v = vfmaq_f32(_bias, _v, _scale);
    #else
                            _v = vmlaq_f32(_bias, _v, _scale);
    #endif
                            vst1q_f32(ptr, _v);

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

                            float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data_ptr[0]) : vld1q_f32((const float*)scale_data_ptr + q * 4);

                            int i = 0;
                            for (; i + 1 < size; i += 2) {
                                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                                _v0 = vmulq_f32(_v0, _scale);
                                _v1 = vmulq_f32(_v1, _scale);
                                vst1q_f32(ptr, _v0);
                                vst1q_f32(ptr + 4, _v1);

                                intptr += 8;
                                ptr += 8;
                            }
                            for (; i < size; i++) {
                                float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                                _v = vmulq_f32(_v, _scale);
                                vst1q_f32(ptr, _v);

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

                            float32x4_t _scale = scale_data_size == 1 ? vdupq_n_f32(scale_data_ptr[0]) : vld1q_f32((const float*)scale_data_ptr + q * 4);
                            float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 4);

                            int i = 0;
                            for (; i + 1 < size; i += 2) {
                                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
        #if __aarch64__
                                _v0 = vfmaq_f32(_bias, _v0, _scale);
                                _v1 = vfmaq_f32(_bias, _v1, _scale);
        #else
                                _v0 = vmlaq_f32(_bias, _v0, _scale);
                                _v1 = vmlaq_f32(_bias, _v1, _scale);
        #endif
                                vst1q_f32(ptr, _v0);
                                vst1q_f32(ptr + 4, _v1);

                                intptr += 8;
                                ptr += 8;
                            }
                            for (; i < size; i++) {
                                float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
        #if __aarch64__
                                _v = vfmaq_f32(_bias, _v, _scale);
        #else
                                _v = vmlaq_f32(_bias, _v, _scale);
        #endif
                                vst1q_f32(ptr, _v);

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
            const float scale = scale_data_ptr[0];

            if (bias_data_size == 0) {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        ptr[i] = intptr[i] * scale;
                    }
                });
            } else if (bias_data_size == 1) {
                const float bias = bias_data_ptr[0];

                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        ptr[i] = intptr[i] * scale + bias;
                    }
                });
            } else {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        ptr[i] = intptr[i] * scale + bias_data_ptr[i];
                    }
                });
            }
        } else {
            if (bias_data_size == 0) {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        ptr[i] = intptr[i] * scale_data_ptr[i];
                    }
                });
            } else if (bias_data_size == 1) {
                const float bias = bias_data_ptr[0];

                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        ptr[i] = intptr[i] * scale_data_ptr[i] + bias;
                    }
                });
            } else {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        ptr[i] = intptr[i] * scale_data_ptr[i] + bias_data_ptr[i];
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

                    const float scale = scale_data_size == 1 ? scale_data_ptr[0] : scale_data_ptr[i];

                    int j = 0;
    #if __ARM_NEON
                    float32x4_t _scale = vdupq_n_f32(scale);
                    for (; j + 3 < w; j += 4)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        vst1q_f32(ptr, _v);

                        intptr += 4;
                        ptr += 4;
                    }
    #endif // __ARM_NEON
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

                    const float scale = scale_data_size == 1 ? scale_data_ptr[0] : scale_data_ptr[i];
                    const float bias = bias_data_size == 1 ? bias_data_ptr[0] : bias_data_ptr[i];

                    int j = 0;
    #if __ARM_NEON
                    float32x4_t _scale = vdupq_n_f32(scale);
                    float32x4_t _bias = vdupq_n_f32(bias);
                    for (; j + 3 < w; j += 4)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
    #if __aarch64__
                        _v = vfmaq_f32(_bias, _v, _scale);
    #else
                        _v = vmlaq_f32(_bias, _v, _scale);
    #endif
                        vst1q_f32(ptr, _v);

                        intptr += 4;
                        ptr += 4;
                    }
    #endif // __ARM_NEON
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

                    const float scale = scale_data_size == 1 ? scale_data_ptr[0] : scale_data_ptr[q];

                    int i = 0;
    #if __ARM_NEON
                    float32x4_t _scale = vdupq_n_f32(scale);
                    for (; i + 7 < size; i += 8)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                        _v0 = vmulq_f32(_v0, _scale);
                        _v1 = vmulq_f32(_v1, _scale);
                        vst1q_f32(ptr, _v0);
                        vst1q_f32(ptr + 4, _v1);

                        intptr += 8;
                        ptr += 8;
                    }
                    for (; i + 3 < size; i += 4)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        vst1q_f32(ptr, _v);

                        intptr += 4;
                        ptr += 4;
                    }
    #endif // __ARM_NEON
                    for (; i < size; i++)
                    {
                        *ptr++ = *intptr++ * scale;
                    }
                }
            });
        } else {
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const int* intptr = src_a[q].data();
                    float* ptr = dst_a[q].data();

                    const float scale = scale_data_size == 1 ? scale_data_ptr[0] : scale_data_ptr[q];
                    const float bias = bias_data_size == 1 ? bias_data_ptr[0] : bias_data_ptr[q];

                    int i = 0;
    #if __ARM_NEON
                    float32x4_t _scale = vdupq_n_f32(scale);
                    float32x4_t _bias = vdupq_n_f32(bias);
                    for (; i + 7 < size; i += 8)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
    #if __aarch64__
                        _v0 = vfmaq_f32(_bias, _v0, _scale);
                        _v1 = vfmaq_f32(_bias, _v1, _scale);
    #else
                        _v0 = vmlaq_f32(_bias, _v0, _scale);
                        _v1 = vmlaq_f32(_bias, _v1, _scale);
    #endif
                        vst1q_f32(ptr, _v0);
                        vst1q_f32(ptr + 4, _v1);

                        intptr += 8;
                        ptr += 8;
                    }
                    for (; i + 3 < size; i += 4)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
    #if __aarch64__
                        _v = vfmaq_f32(_bias, _v, _scale);
    #else
                        _v = vmlaq_f32(_bias, _v, _scale);
    #endif
                        vst1q_f32(ptr, _v);

                        intptr += 4;
                        ptr += 4;
                    }
    #endif // __ARM_NEON
                    for (; i < size; i++)
                    {
                        *ptr++ = *intptr++ * scale + bias;
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

                        const float scale = scale_data_size == 1 ? scale_data_ptr[0] : scale_data_ptr[q];

                        int i = 0;
        #if __ARM_NEON
                        float32x4_t _scale = vdupq_n_f32(scale);
                        for (; i + 7 < size; i += 8)
                        {
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                            _v0 = vmulq_f32(_v0, _scale);
                            _v1 = vmulq_f32(_v1, _scale);
                            vst1q_f32(ptr, _v0);
                            vst1q_f32(ptr + 4, _v1);

                            intptr += 8;
                            ptr += 8;
                        }
                        for (; i + 3 < size; i += 4)
                        {
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmulq_f32(_v, _scale);
                            vst1q_f32(ptr, _v);

                            intptr += 4;
                            ptr += 4;
                        }
        #endif // __ARM_NEON
                        for (; i < size; i++)
                        {
                            *ptr++ = *intptr++ * scale;
                        }
                    }
                });
            } else {
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const int* intptr = src_a[q].data();
                        float* ptr = dst_a[q].data();

                        const float scale = scale_data_size == 1 ? scale_data_ptr[0] : scale_data_ptr[q];
                        const float bias = bias_data_size == 1 ? bias_data_ptr[0] : bias_data_ptr[q];

                        int i = 0;
        #if __ARM_NEON
                        float32x4_t _scale = vdupq_n_f32(scale);
                        float32x4_t _bias = vdupq_n_f32(bias);
                        for (; i + 7 < size; i += 8)
                        {
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
        #if __aarch64__
                            _v0 = vfmaq_f32(_bias, _v0, _scale);
                            _v1 = vfmaq_f32(_bias, _v1, _scale);
        #else
                            _v0 = vmlaq_f32(_bias, _v0, _scale);
                            _v1 = vmlaq_f32(_bias, _v1, _scale);
        #endif
                            vst1q_f32(ptr, _v0);
                            vst1q_f32(ptr + 4, _v1);

                            intptr += 8;
                            ptr += 8;
                        }
                        for (; i + 3 < size; i += 4)
                        {
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
        #if __aarch64__
                            _v = vfmaq_f32(_bias, _v, _scale);
        #else
                            _v = vmlaq_f32(_bias, _v, _scale);
        #endif
                            vst1q_f32(ptr, _v);

                            intptr += 4;
                            ptr += 4;
                        }
        #endif // __ARM_NEON
                        for (; i < size; i++)
                        {
                            *ptr++ = *intptr++ * scale + bias;
                        }
                    }
                });
            }
        }
    }
    
    return dst;
}

static void requantize_relu_pack8_neon(const Tensor& bottom_blob, Tensor& top_blob, const Tensor& scale_in_data, const Tensor& scale_out_data, const Tensor& bias_data) {
    int w = bottom_blob.size(2);
    int h = bottom_blob.size(1);
    int channels = bottom_blob.size(0);
    int size = w * h;

    int scale_in_data_size = scale_in_data.size(0);
    int scale_out_data_size = scale_out_data.size(0);
    int bias_data_size = bias_data.size(0);

    // int8(relu(v * scale_in) * scale_out)
    // int8_relu(v * (scale_in * scale_out))

    // int8(relu(v * scale_in + bias) * scale_out)
    // int8_relu(v * (scale_in * scale_out) + (bias * scale_out))
    
    auto bottom_blob_a = bottom_blob.accessor<int, 3, 8>();
    auto top_blob_a = top_blob.accessor<signed char, 3, 8>();
    const float* scale_in_data_ptr = (const float*)scale_in_data.data_ptr();
    const float* scale_out_data_ptr = (const float*)scale_out_data.data_ptr();
    const float* bias_data_ptr = (const float*)bias_data.data_ptr();

    if (bias_data_size == 0) {
        otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
            for (const auto q : otter::irange(begin, end))
            {
                const int* intptr = bottom_blob_a[q].data();
                signed char* ptr = top_blob_a[q].data();

                float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8);
                float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8 + 4);
                float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8);
                float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8 + 4);

                float32x4_t _scale0 = vmulq_f32(_scale_in0, _scale_out0);
                float32x4_t _scale1 = vmulq_f32(_scale_in1, _scale_out1);

                int i = 0;
    #if __aarch64__
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                    float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                    float32x4_t _v2 = vcvtq_f32_s32(vld1q_s32(intptr + 8));
                    float32x4_t _v3 = vcvtq_f32_s32(vld1q_s32(intptr + 12));
                    float32x4_t _v4 = vcvtq_f32_s32(vld1q_s32(intptr + 16));
                    float32x4_t _v5 = vcvtq_f32_s32(vld1q_s32(intptr + 20));
                    float32x4_t _v6 = vcvtq_f32_s32(vld1q_s32(intptr + 24));
                    float32x4_t _v7 = vcvtq_f32_s32(vld1q_s32(intptr + 28));
                    _v0 = vmulq_f32(_v0, _scale0);
                    _v1 = vmulq_f32(_v1, _scale1);
                    _v2 = vmulq_f32(_v2, _scale0);
                    _v3 = vmulq_f32(_v3, _scale1);
                    _v4 = vmulq_f32(_v4, _scale0);
                    _v5 = vmulq_f32(_v5, _scale1);
                    _v6 = vmulq_f32(_v6, _scale0);
                    _v7 = vmulq_f32(_v7, _scale1);
                    vst1_s8(ptr, float2int8relu(_v0, _v1));
                    vst1_s8(ptr + 8, float2int8relu(_v2, _v3));
                    vst1_s8(ptr + 16, float2int8relu(_v4, _v5));
                    vst1_s8(ptr + 24, float2int8relu(_v6, _v7));

                    intptr += 32;
                    ptr += 32;
                }
    #endif // __aarch64__
                for (; i + 1 < size; i += 2)
                {
                    float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                    float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                    float32x4_t _v2 = vcvtq_f32_s32(vld1q_s32(intptr + 8));
                    float32x4_t _v3 = vcvtq_f32_s32(vld1q_s32(intptr + 12));
                    _v0 = vmulq_f32(_v0, _scale0);
                    _v1 = vmulq_f32(_v1, _scale1);
                    _v2 = vmulq_f32(_v2, _scale0);
                    _v3 = vmulq_f32(_v3, _scale1);
                    vst1_s8(ptr, float2int8relu(_v0, _v1));
                    vst1_s8(ptr + 8, float2int8relu(_v2, _v3));

                    intptr += 16;
                    ptr += 16;
                }
                for (; i < size; i++)
                {
                    float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                    float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                    _v0 = vmulq_f32(_v0, _scale0);
                    _v1 = vmulq_f32(_v1, _scale1);
                    vst1_s8(ptr, float2int8relu(_v0, _v1));

                    intptr += 8;
                    ptr += 8;
                }
            }
        });
    } else {
        otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
            for (const auto q : otter::irange(begin, end))
            {
                const int* intptr = bottom_blob_a[q].data();
                signed char* ptr = top_blob_a[q].data();

                float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8);
                float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8 + 4);
                float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8);
                float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8 + 4);
                float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8);
                float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8 + 4);

                float32x4_t _scale0 = vmulq_f32(_scale_in0, _scale_out0);
                float32x4_t _scale1 = vmulq_f32(_scale_in1, _scale_out1);
                _bias0 = vmulq_f32(_bias0, _scale_out0);
                _bias1 = vmulq_f32(_bias1, _scale_out1);

                int i = 0;
    #if __aarch64__
                for (; i + 3 < size; i += 4)
                {
                    float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                    float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                    float32x4_t _v2 = vcvtq_f32_s32(vld1q_s32(intptr + 8));
                    float32x4_t _v3 = vcvtq_f32_s32(vld1q_s32(intptr + 12));
                    float32x4_t _v4 = vcvtq_f32_s32(vld1q_s32(intptr + 16));
                    float32x4_t _v5 = vcvtq_f32_s32(vld1q_s32(intptr + 20));
                    float32x4_t _v6 = vcvtq_f32_s32(vld1q_s32(intptr + 24));
                    float32x4_t _v7 = vcvtq_f32_s32(vld1q_s32(intptr + 28));

                    _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                    _v1 = vfmaq_f32(_bias1, _v1, _scale1);
                    _v2 = vfmaq_f32(_bias0, _v2, _scale0);
                    _v3 = vfmaq_f32(_bias1, _v3, _scale1);
                    _v4 = vfmaq_f32(_bias0, _v4, _scale0);
                    _v5 = vfmaq_f32(_bias1, _v5, _scale1);
                    _v6 = vfmaq_f32(_bias0, _v6, _scale0);
                    _v7 = vfmaq_f32(_bias1, _v7, _scale1);

                    vst1_s8(ptr, float2int8relu(_v0, _v1));
                    vst1_s8(ptr + 8, float2int8relu(_v2, _v3));
                    vst1_s8(ptr + 16, float2int8relu(_v4, _v5));
                    vst1_s8(ptr + 24, float2int8relu(_v6, _v7));

                    intptr += 32;
                    ptr += 32;
                }
    #endif // __aarch64__
                for (; i + 1 < size; i += 2)
                {
                    float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                    float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                    float32x4_t _v2 = vcvtq_f32_s32(vld1q_s32(intptr + 8));
                    float32x4_t _v3 = vcvtq_f32_s32(vld1q_s32(intptr + 12));

    #if __aarch64__
                    _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                    _v1 = vfmaq_f32(_bias1, _v1, _scale1);
                    _v2 = vfmaq_f32(_bias0, _v2, _scale0);
                    _v3 = vfmaq_f32(_bias1, _v3, _scale1);
    #else  // __aarch64__
                    _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                    _v1 = vmlaq_f32(_bias1, _v1, _scale1);
                    _v2 = vmlaq_f32(_bias0, _v2, _scale0);
                    _v3 = vmlaq_f32(_bias1, _v3, _scale1);
    #endif // __aarch64__

                    vst1_s8(ptr, float2int8relu(_v0, _v1));
                    vst1_s8(ptr + 8, float2int8relu(_v2, _v3));

                    intptr += 16;
                    ptr += 16;
                }
                for (; i < size; i++)
                {
                    float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                    float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
    #if __aarch64__
                    _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                    _v1 = vfmaq_f32(_bias1, _v1, _scale1);
    #else  // __aarch64__
                    _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                    _v1 = vmlaq_f32(_bias1, _v1, _scale1);
    #endif // __aarch64__
                    vst1_s8(ptr, float2int8relu(_v0, _v1));

                    intptr += 8;
                    ptr += 8;
                }
            }
        });
    }
}

static void requantize_leakyrelu_pack8_neon(const Tensor& bottom_blob, Tensor& top_blob, const Tensor& scale_in_data, const Tensor& scale_out_data, const Tensor& bias_data, float slope)
{
    int w = bottom_blob.size(2);
    int h = bottom_blob.size(1);
    int channels = bottom_blob.size(0);
    int size = w * h;

    int scale_in_data_size = scale_in_data.size(0);
    int scale_out_data_size = scale_out_data.size(0);
    int bias_data_size = bias_data.size(0);

    // int8(leakyrelu(v * scale_in, slope) * scale_out)
    // int8_leakyrelu(v * (scale_in * scale_out), slope)

    // int8(leakyrelu(v * scale_in + bias, slope) * scale_out)
    // int8_leakyrelu(v * (scale_in * scale_out) + (bias * scale_out), slope)
    
    auto bottom_blob_a = bottom_blob.accessor<int, 3, 8>();
    auto top_blob_a = top_blob.accessor<signed char, 3, 8>();
    const float* scale_in_data_ptr = (const float*)scale_in_data.data_ptr();
    const float* scale_out_data_ptr = (const float*)scale_out_data.data_ptr();
    const float* bias_data_ptr = (const float*)bias_data.data_ptr();

    if (bias_data_size == 0)
    {
        otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
        for (const auto q : otter::irange(begin, end))
        {
            const int* intptr = bottom_blob_a[q].data();
            signed char* ptr = top_blob_a[q].data();

            float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8);
            float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8 + 4);
            float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8);
            float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8 + 4);

            float32x4_t _scale0 = vmulq_f32(_scale_in0, _scale_out0);
            float32x4_t _scale1 = vmulq_f32(_scale_in1, _scale_out1);
            float32x4_t _slope = vdupq_n_f32(slope);

            int i = 0;
#if __aarch64__
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                float32x4_t _v2 = vcvtq_f32_s32(vld1q_s32(intptr + 8));
                float32x4_t _v3 = vcvtq_f32_s32(vld1q_s32(intptr + 12));
                float32x4_t _v4 = vcvtq_f32_s32(vld1q_s32(intptr + 16));
                float32x4_t _v5 = vcvtq_f32_s32(vld1q_s32(intptr + 20));
                float32x4_t _v6 = vcvtq_f32_s32(vld1q_s32(intptr + 24));
                float32x4_t _v7 = vcvtq_f32_s32(vld1q_s32(intptr + 28));
                _v0 = vmulq_f32(_v0, _scale0);
                _v1 = vmulq_f32(_v1, _scale1);
                _v2 = vmulq_f32(_v2, _scale0);
                _v3 = vmulq_f32(_v3, _scale1);
                _v4 = vmulq_f32(_v4, _scale0);
                _v5 = vmulq_f32(_v5, _scale1);
                _v6 = vmulq_f32(_v6, _scale0);
                _v7 = vmulq_f32(_v7, _scale1);
                vst1_s8(ptr, float2int8leakyrelu(_v0, _v1, _slope));
                vst1_s8(ptr + 8, float2int8leakyrelu(_v2, _v3, _slope));
                vst1_s8(ptr + 16, float2int8leakyrelu(_v4, _v5, _slope));
                vst1_s8(ptr + 24, float2int8leakyrelu(_v6, _v7, _slope));

                intptr += 32;
                ptr += 32;
            }
#endif // __aarch64__
            for (; i + 1 < size; i += 2)
            {
                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                float32x4_t _v2 = vcvtq_f32_s32(vld1q_s32(intptr + 8));
                float32x4_t _v3 = vcvtq_f32_s32(vld1q_s32(intptr + 12));
                _v0 = vmulq_f32(_v0, _scale0);
                _v1 = vmulq_f32(_v1, _scale1);
                _v2 = vmulq_f32(_v2, _scale0);
                _v3 = vmulq_f32(_v3, _scale1);
                vst1_s8(ptr, float2int8leakyrelu(_v0, _v1, _slope));
                vst1_s8(ptr + 8, float2int8leakyrelu(_v2, _v3, _slope));

                intptr += 16;
                ptr += 16;
            }
            for (; i < size; i++)
            {
                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                _v0 = vmulq_f32(_v0, _scale0);
                _v1 = vmulq_f32(_v1, _scale1);
                vst1_s8(ptr, float2int8leakyrelu(_v0, _v1, _slope));

                intptr += 8;
                ptr += 8;
            }
        }
        });
    } else {
        otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
        for (const auto q : otter::irange(begin, end))
        {
            const int* intptr = bottom_blob_a[q].data();
            signed char* ptr = top_blob_a[q].data();

            float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8);
            float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8 + 4);
            float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8);
            float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8 + 4);
            float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8);
            float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8 + 4);

            float32x4_t _scale0 = vmulq_f32(_scale_in0, _scale_out0);
            float32x4_t _scale1 = vmulq_f32(_scale_in1, _scale_out1);
            _bias0 = vmulq_f32(_bias0, _scale_out0);
            _bias1 = vmulq_f32(_bias1, _scale_out1);
            float32x4_t _slope = vdupq_n_f32(slope);

            int i = 0;
#if __aarch64__
            for (; i + 3 < size; i += 4)
            {
                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                float32x4_t _v2 = vcvtq_f32_s32(vld1q_s32(intptr + 8));
                float32x4_t _v3 = vcvtq_f32_s32(vld1q_s32(intptr + 12));
                float32x4_t _v4 = vcvtq_f32_s32(vld1q_s32(intptr + 16));
                float32x4_t _v5 = vcvtq_f32_s32(vld1q_s32(intptr + 20));
                float32x4_t _v6 = vcvtq_f32_s32(vld1q_s32(intptr + 24));
                float32x4_t _v7 = vcvtq_f32_s32(vld1q_s32(intptr + 28));

                _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                _v1 = vfmaq_f32(_bias1, _v1, _scale1);
                _v2 = vfmaq_f32(_bias0, _v2, _scale0);
                _v3 = vfmaq_f32(_bias1, _v3, _scale1);
                _v4 = vfmaq_f32(_bias0, _v4, _scale0);
                _v5 = vfmaq_f32(_bias1, _v5, _scale1);
                _v6 = vfmaq_f32(_bias0, _v6, _scale0);
                _v7 = vfmaq_f32(_bias1, _v7, _scale1);

                vst1_s8(ptr, float2int8leakyrelu(_v0, _v1, _slope));
                vst1_s8(ptr + 8, float2int8leakyrelu(_v2, _v3, _slope));
                vst1_s8(ptr + 16, float2int8leakyrelu(_v4, _v5, _slope));
                vst1_s8(ptr + 24, float2int8leakyrelu(_v6, _v7, _slope));

                intptr += 32;
                ptr += 32;
            }
#endif // __aarch64__
            for (; i + 1 < size; i += 2)
            {
                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
                float32x4_t _v2 = vcvtq_f32_s32(vld1q_s32(intptr + 8));
                float32x4_t _v3 = vcvtq_f32_s32(vld1q_s32(intptr + 12));

#if __aarch64__
                _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                _v1 = vfmaq_f32(_bias1, _v1, _scale1);
                _v2 = vfmaq_f32(_bias0, _v2, _scale0);
                _v3 = vfmaq_f32(_bias1, _v3, _scale1);
#else  // __aarch64__
                _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                _v1 = vmlaq_f32(_bias1, _v1, _scale1);
                _v2 = vmlaq_f32(_bias0, _v2, _scale0);
                _v3 = vmlaq_f32(_bias1, _v3, _scale1);
#endif // __aarch64__

                vst1_s8(ptr, float2int8leakyrelu(_v0, _v1, _slope));
                vst1_s8(ptr + 8, float2int8leakyrelu(_v2, _v3, _slope));

                intptr += 16;
                ptr += 16;
            }
            for (; i < size; i++)
            {
                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr + 4));
#if __aarch64__
                _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                _v1 = vfmaq_f32(_bias1, _v1, _scale1);
#else  // __aarch64__
                _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                _v1 = vmlaq_f32(_bias1, _v1, _scale1);
#endif // __aarch64__
                vst1_s8(ptr, float2int8leakyrelu(_v0, _v1, _slope));

                intptr += 8;
                ptr += 8;
            }
        }
        });
    }
}

static void requantize_relu_pack4_neon(const Tensor& bottom_blob, Tensor& top_blob, const Tensor& scale_in_data, const Tensor& scale_out_data, const Tensor& bias_data)
{
    int w = bottom_blob.size(2);
    int h = bottom_blob.size(1);
    int channels = bottom_blob.size(0);
    int size = w * h;
    int outc = top_blob.size(0);
    int out_elempack = top_blob.elempack();

    int scale_in_data_size = scale_in_data.size(0);
    int scale_out_data_size = scale_out_data.size(0);
    int bias_data_size = bias_data.size(0);

    // int8(relu(v * scale_in) * scale_out)
    // int8_relu(v * (scale_in * scale_out))

    // int8(relu(v * scale_in + bias) * scale_out)
    // int8_relu(v * (scale_in * scale_out) + (bias * scale_out))
    
    auto bottom_blob_a = bottom_blob.accessor<int, 3, 8>();
    auto top_blob_ra = top_blob.raw_accessor<signed char, 3>();
    const float* scale_in_data_ptr = (const float*)scale_in_data.data_ptr();
    const float* scale_out_data_ptr = (const float*)scale_out_data.data_ptr();
    const float* bias_data_ptr = (const float*)bias_data.data_ptr();

    if (out_elempack == 8) {
        if (bias_data_size == 0) {
            otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const int* intptr0 = bottom_blob_a[q * 2].data();
                    const int* intptr1 = bottom_blob_a[q * 2 + 1].data();
                    signed char* ptr = top_blob_ra[q].data();

                    float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8);
                    float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8 + 4);
                    float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8);
                    float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8 + 4);

                    float32x4_t _scale0 = vmulq_f32(_scale_in0, _scale_out0);
                    float32x4_t _scale1 = vmulq_f32(_scale_in1, _scale_out1);

                    int i = 0;
    #if __aarch64__
                    for (; i + 3 < size; i += 4)
                    {
                        float32x4_t _v00 = vcvtq_f32_s32(vld1q_s32(intptr0));
                        float32x4_t _v01 = vcvtq_f32_s32(vld1q_s32(intptr0 + 4));
                        float32x4_t _v02 = vcvtq_f32_s32(vld1q_s32(intptr0 + 8));
                        float32x4_t _v03 = vcvtq_f32_s32(vld1q_s32(intptr0 + 12));
                        float32x4_t _v10 = vcvtq_f32_s32(vld1q_s32(intptr1));
                        float32x4_t _v11 = vcvtq_f32_s32(vld1q_s32(intptr1 + 4));
                        float32x4_t _v12 = vcvtq_f32_s32(vld1q_s32(intptr1 + 8));
                        float32x4_t _v13 = vcvtq_f32_s32(vld1q_s32(intptr1 + 12));
                        _v00 = vmulq_f32(_v00, _scale0);
                        _v01 = vmulq_f32(_v01, _scale0);
                        _v02 = vmulq_f32(_v02, _scale0);
                        _v03 = vmulq_f32(_v03, _scale0);
                        _v10 = vmulq_f32(_v10, _scale1);
                        _v11 = vmulq_f32(_v11, _scale1);
                        _v12 = vmulq_f32(_v12, _scale1);
                        _v13 = vmulq_f32(_v13, _scale1);
                        vst1_s8(ptr, float2int8relu(_v00, _v10));
                        vst1_s8(ptr + 8, float2int8relu(_v01, _v11));
                        vst1_s8(ptr + 16, float2int8relu(_v02, _v12));
                        vst1_s8(ptr + 24, float2int8relu(_v03, _v13));

                        intptr0 += 16;
                        intptr1 += 16;
                        ptr += 32;
                    }
    #endif // __aarch64__
                    for (; i < size; i++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr0));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr1));
                        _v0 = vmulq_f32(_v0, _scale0);
                        _v1 = vmulq_f32(_v1, _scale1);
                        vst1_s8(ptr, float2int8relu(_v0, _v1));

                        intptr0 += 4;
                        intptr1 += 4;
                        ptr += 8;
                    }
                }
            });
        }
        else
        {
            otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const int* intptr0 = bottom_blob_a[q * 2].data();
                    const int* intptr1 = bottom_blob_a[q * 2 + 1].data();
                    signed char* ptr = top_blob_ra[q].data();

                    float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8);
                    float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8 + 4);
                    float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8);
                    float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8 + 4);
                    float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8);
                    float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8 + 4);

                    float32x4_t _scale0 = vmulq_f32(_scale_in0, _scale_out0);
                    float32x4_t _scale1 = vmulq_f32(_scale_in1, _scale_out1);
                    _bias0 = vmulq_f32(_bias0, _scale_out0);
                    _bias1 = vmulq_f32(_bias1, _scale_out1);

                    int i = 0;
    #if __aarch64__
                    for (; i + 3 < size; i += 4)
                    {
                        float32x4_t _v00 = vcvtq_f32_s32(vld1q_s32(intptr0));
                        float32x4_t _v01 = vcvtq_f32_s32(vld1q_s32(intptr0 + 4));
                        float32x4_t _v02 = vcvtq_f32_s32(vld1q_s32(intptr0 + 8));
                        float32x4_t _v03 = vcvtq_f32_s32(vld1q_s32(intptr0 + 12));
                        float32x4_t _v10 = vcvtq_f32_s32(vld1q_s32(intptr1));
                        float32x4_t _v11 = vcvtq_f32_s32(vld1q_s32(intptr1 + 4));
                        float32x4_t _v12 = vcvtq_f32_s32(vld1q_s32(intptr1 + 8));
                        float32x4_t _v13 = vcvtq_f32_s32(vld1q_s32(intptr1 + 12));
                        _v00 = vfmaq_f32(_bias0, _v00, _scale0);
                        _v01 = vfmaq_f32(_bias0, _v01, _scale0);
                        _v02 = vfmaq_f32(_bias0, _v02, _scale0);
                        _v03 = vfmaq_f32(_bias0, _v03, _scale0);
                        _v10 = vfmaq_f32(_bias1, _v10, _scale1);
                        _v11 = vfmaq_f32(_bias1, _v11, _scale1);
                        _v12 = vfmaq_f32(_bias1, _v12, _scale1);
                        _v13 = vfmaq_f32(_bias1, _v13, _scale1);
                        vst1_s8(ptr, float2int8relu(_v00, _v10));
                        vst1_s8(ptr + 8, float2int8relu(_v01, _v11));
                        vst1_s8(ptr + 16, float2int8relu(_v02, _v12));
                        vst1_s8(ptr + 24, float2int8relu(_v03, _v13));

                        intptr0 += 16;
                        intptr1 += 16;
                        ptr += 32;
                    }
    #endif // __aarch64__
                    for (; i + 1 < size; i += 2)
                    {
                        float32x4_t _v00 = vcvtq_f32_s32(vld1q_s32(intptr0));
                        float32x4_t _v01 = vcvtq_f32_s32(vld1q_s32(intptr0 + 4));
                        float32x4_t _v10 = vcvtq_f32_s32(vld1q_s32(intptr1));
                        float32x4_t _v11 = vcvtq_f32_s32(vld1q_s32(intptr1 + 4));
    #if __aarch64__
                        _v00 = vfmaq_f32(_bias0, _v00, _scale0);
                        _v01 = vfmaq_f32(_bias0, _v01, _scale0);
                        _v10 = vfmaq_f32(_bias1, _v10, _scale1);
                        _v11 = vfmaq_f32(_bias1, _v11, _scale1);
    #else  // __aarch64__
                        _v00 = vmlaq_f32(_bias0, _v00, _scale0);
                        _v01 = vmlaq_f32(_bias0, _v01, _scale0);
                        _v10 = vmlaq_f32(_bias1, _v10, _scale1);
                        _v11 = vmlaq_f32(_bias1, _v11, _scale1);
    #endif // __aarch64__
                        vst1_s8(ptr, float2int8relu(_v00, _v10));
                        vst1_s8(ptr + 8, float2int8relu(_v01, _v11));

                        intptr0 += 8;
                        intptr1 += 8;
                        ptr += 16;
                    }
                    for (; i < size; i++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr0));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr1));
    #if __aarch64__
                        _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                        _v1 = vfmaq_f32(_bias1, _v1, _scale1);
    #else  // __aarch64__
                        _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                        _v1 = vmlaq_f32(_bias1, _v1, _scale1);
    #endif // __aarch64__
                        vst1_s8(ptr, float2int8relu(_v0, _v1));

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
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const int* intptr = bottom_blob_a[q].data();
                    signed char* ptr0 = top_blob_ra[q * 4].data();
                    signed char* ptr1 = top_blob_ra[q * 4 + 1].data();
                    signed char* ptr2 = top_blob_ra[q * 4 + 2].data();
                    signed char* ptr3 = top_blob_ra[q * 4 + 3].data();

                    float32x4_t _scale_in = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 4);
                    float32x4_t _scale_out = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 4);

                    float32x4_t _scale = vmulq_f32(_scale_in, _scale_out);

                    int i = 0;
                    for (; i < size; i++)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        int8x8_t v = float2int8relu(_v, _v);
                        ptr0[0] = vget_lane_s8(v, 0);
                        ptr1[0] = vget_lane_s8(v, 1);
                        ptr2[0] = vget_lane_s8(v, 2);
                        ptr3[0] = vget_lane_s8(v, 3);

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
                for (const auto q : otter::irange(begin, end))
                {
                    const int* intptr = bottom_blob_a[q].data();
                    signed char* ptr0 = top_blob_ra[q * 4].data();
                    signed char* ptr1 = top_blob_ra[q * 4 + 1].data();
                    signed char* ptr2 = top_blob_ra[q * 4 + 2].data();
                    signed char* ptr3 = top_blob_ra[q * 4 + 3].data();

                    float32x4_t _scale_in = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 4);
                    float32x4_t _scale_out = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 4);
                    float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 4);

                    float32x4_t _scale = vmulq_f32(_scale_in, _scale_out);
                    _bias = vmulq_f32(_bias, _scale_out);

                    int i = 0;
                    for (; i < size; i++)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
    #if __aarch64__
                        _v = vfmaq_f32(_bias, _v, _scale);
    #else
                        _v = vmlaq_f32(_bias, _v, _scale);
    #endif
                        int8x8_t v = float2int8relu(_v, _v);
                        ptr0[0] = vget_lane_s8(v, 0);
                        ptr1[0] = vget_lane_s8(v, 1);
                        ptr2[0] = vget_lane_s8(v, 2);
                        ptr3[0] = vget_lane_s8(v, 3);

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

static void requantize_leakyrelu_pack4_neon(const Tensor& bottom_blob, Tensor& top_blob, const Tensor& scale_in_data, const Tensor& scale_out_data, const Tensor& bias_data, float slope)
{
    int w = bottom_blob.size(2);
    int h = bottom_blob.size(1);
    int channels = bottom_blob.size(0);
    int size = w * h;
    int outc = top_blob.size(0);
    int out_elempack = top_blob.elempack();

    int scale_in_data_size = scale_in_data.size(0);
    int scale_out_data_size = scale_out_data.size(0);
    int bias_data_size = bias_data.size(0);

    // int8(leakyrelu(v * scale_in, slope) * scale_out)
    // int8_leakyrelu(v * (scale_in * scale_out), slope)

    // int8(leakyrelu(v * scale_in + bias, slope) * scale_out)
    // int8_leakyrelu(v * (scale_in * scale_out) + (bias * scale_out), slope)
    
    auto bottom_blob_a = bottom_blob.accessor<int, 3, 4>();
    auto top_blob_ra = top_blob.raw_accessor<signed char, 3>();
    const float* scale_in_data_ptr = (const float*)scale_in_data.data_ptr();
    const float* scale_out_data_ptr = (const float*)scale_out_data.data_ptr();
    const float* bias_data_ptr = (const float*)bias_data.data_ptr();

    if (out_elempack == 8) {
        if (bias_data_size == 0) {
            otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end))
                {
                    const int* intptr0 = bottom_blob_a[q * 2].data();
                    const int* intptr1 = bottom_blob_a[q * 2 + 1].data();
                    signed char* ptr = top_blob_ra[q].data();

                    float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8);
                    float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8 + 4);
                    float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8);
                    float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8 + 4);

                    float32x4_t _scale0 = vmulq_f32(_scale_in0, _scale_out0);
                    float32x4_t _scale1 = vmulq_f32(_scale_in1, _scale_out1);
                    float32x4_t _slope = vdupq_n_f32(slope);

                    int i = 0;
    #if __aarch64__
                    for (; i + 3 < size; i += 4)
                    {
                        float32x4_t _v00 = vcvtq_f32_s32(vld1q_s32(intptr0));
                        float32x4_t _v01 = vcvtq_f32_s32(vld1q_s32(intptr0 + 4));
                        float32x4_t _v02 = vcvtq_f32_s32(vld1q_s32(intptr0 + 8));
                        float32x4_t _v03 = vcvtq_f32_s32(vld1q_s32(intptr0 + 12));
                        float32x4_t _v10 = vcvtq_f32_s32(vld1q_s32(intptr1));
                        float32x4_t _v11 = vcvtq_f32_s32(vld1q_s32(intptr1 + 4));
                        float32x4_t _v12 = vcvtq_f32_s32(vld1q_s32(intptr1 + 8));
                        float32x4_t _v13 = vcvtq_f32_s32(vld1q_s32(intptr1 + 12));
                        _v00 = vmulq_f32(_v00, _scale0);
                        _v01 = vmulq_f32(_v01, _scale0);
                        _v02 = vmulq_f32(_v02, _scale0);
                        _v03 = vmulq_f32(_v03, _scale0);
                        _v10 = vmulq_f32(_v10, _scale1);
                        _v11 = vmulq_f32(_v11, _scale1);
                        _v12 = vmulq_f32(_v12, _scale1);
                        _v13 = vmulq_f32(_v13, _scale1);
                        vst1_s8(ptr, float2int8leakyrelu(_v00, _v10, _slope));
                        vst1_s8(ptr + 8, float2int8leakyrelu(_v01, _v11, _slope));
                        vst1_s8(ptr + 16, float2int8leakyrelu(_v02, _v12, _slope));
                        vst1_s8(ptr + 24, float2int8leakyrelu(_v03, _v13, _slope));

                        intptr0 += 16;
                        intptr1 += 16;
                        ptr += 32;
                    }
    #endif // __aarch64__
                    for (; i < size; i++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr0));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr1));
                        _v0 = vmulq_f32(_v0, _scale0);
                        _v1 = vmulq_f32(_v1, _scale1);
                        vst1_s8(ptr, float2int8leakyrelu(_v0, _v1, _slope));

                        intptr0 += 4;
                        intptr1 += 4;
                        ptr += 8;
                    }
                }
            });
        }
        else
        {
            otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end))
                {
                    const int* intptr0 = bottom_blob_a[q * 2].data();
                    const int* intptr1 = bottom_blob_a[q * 2 + 1].data();
                    signed char* ptr = top_blob_ra[q].data();

                    float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8);
                    float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8 + 4);
                    float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8);
                    float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8 + 4);
                    float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8);
                    float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8 + 4);

                    float32x4_t _scale0 = vmulq_f32(_scale_in0, _scale_out0);
                    float32x4_t _scale1 = vmulq_f32(_scale_in1, _scale_out1);
                    _bias0 = vmulq_f32(_bias0, _scale_out0);
                    _bias1 = vmulq_f32(_bias1, _scale_out1);
                    float32x4_t _slope = vdupq_n_f32(slope);

                    int i = 0;
    #if __aarch64__
                    for (; i + 3 < size; i += 4)
                    {
                        float32x4_t _v00 = vcvtq_f32_s32(vld1q_s32(intptr0));
                        float32x4_t _v01 = vcvtq_f32_s32(vld1q_s32(intptr0 + 4));
                        float32x4_t _v02 = vcvtq_f32_s32(vld1q_s32(intptr0 + 8));
                        float32x4_t _v03 = vcvtq_f32_s32(vld1q_s32(intptr0 + 12));
                        float32x4_t _v10 = vcvtq_f32_s32(vld1q_s32(intptr1));
                        float32x4_t _v11 = vcvtq_f32_s32(vld1q_s32(intptr1 + 4));
                        float32x4_t _v12 = vcvtq_f32_s32(vld1q_s32(intptr1 + 8));
                        float32x4_t _v13 = vcvtq_f32_s32(vld1q_s32(intptr1 + 12));
                        _v00 = vfmaq_f32(_bias0, _v00, _scale0);
                        _v01 = vfmaq_f32(_bias0, _v01, _scale0);
                        _v02 = vfmaq_f32(_bias0, _v02, _scale0);
                        _v03 = vfmaq_f32(_bias0, _v03, _scale0);
                        _v10 = vfmaq_f32(_bias1, _v10, _scale1);
                        _v11 = vfmaq_f32(_bias1, _v11, _scale1);
                        _v12 = vfmaq_f32(_bias1, _v12, _scale1);
                        _v13 = vfmaq_f32(_bias1, _v13, _scale1);
                        vst1_s8(ptr, float2int8leakyrelu(_v00, _v10, _slope));
                        vst1_s8(ptr + 8, float2int8leakyrelu(_v01, _v11, _slope));
                        vst1_s8(ptr + 16, float2int8leakyrelu(_v02, _v12, _slope));
                        vst1_s8(ptr + 24, float2int8leakyrelu(_v03, _v13, _slope));

                        intptr0 += 16;
                        intptr1 += 16;
                        ptr += 32;
                    }
    #endif // __aarch64__
                    for (; i + 1 < size; i += 2)
                    {
                        float32x4_t _v00 = vcvtq_f32_s32(vld1q_s32(intptr0));
                        float32x4_t _v01 = vcvtq_f32_s32(vld1q_s32(intptr0 + 4));
                        float32x4_t _v10 = vcvtq_f32_s32(vld1q_s32(intptr1));
                        float32x4_t _v11 = vcvtq_f32_s32(vld1q_s32(intptr1 + 4));
    #if __aarch64__
                        _v00 = vfmaq_f32(_bias0, _v00, _scale0);
                        _v01 = vfmaq_f32(_bias0, _v01, _scale0);
                        _v10 = vfmaq_f32(_bias1, _v10, _scale1);
                        _v11 = vfmaq_f32(_bias1, _v11, _scale1);
    #else  // __aarch64__
                        _v00 = vmlaq_f32(_bias0, _v00, _scale0);
                        _v01 = vmlaq_f32(_bias0, _v01, _scale0);
                        _v10 = vmlaq_f32(_bias1, _v10, _scale1);
                        _v11 = vmlaq_f32(_bias1, _v11, _scale1);
    #endif // __aarch64__
                        vst1_s8(ptr, float2int8leakyrelu(_v00, _v10, _slope));
                        vst1_s8(ptr + 8, float2int8leakyrelu(_v01, _v11, _slope));

                        intptr0 += 8;
                        intptr1 += 8;
                        ptr += 16;
                    }
                    for (; i < size; i++)
                    {
                        float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr0));
                        float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr1));
    #if __aarch64__
                        _v0 = vfmaq_f32(_bias0, _v0, _scale0);
                        _v1 = vfmaq_f32(_bias1, _v1, _scale1);
    #else  // __aarch64__
                        _v0 = vmlaq_f32(_bias0, _v0, _scale0);
                        _v1 = vmlaq_f32(_bias1, _v1, _scale1);
    #endif // __aarch64__
                        vst1_s8(ptr, float2int8leakyrelu(_v0, _v1, _slope));

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
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const int* intptr = bottom_blob_a[q].data();
                    signed char* ptr0 = top_blob_ra[q * 4].data();
                    signed char* ptr1 = top_blob_ra[q * 4 + 1].data();
                    signed char* ptr2 = top_blob_ra[q * 4 + 2].data();
                    signed char* ptr3 = top_blob_ra[q * 4 + 3].data();

                    float32x4_t _scale_in = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 4);
                    float32x4_t _scale_out = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 4);

                    float32x4_t _scale = vmulq_f32(_scale_in, _scale_out);
                    float32x4_t _slope = vdupq_n_f32(slope);

                    int i = 0;
                    for (; i < size; i++)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                        _v = vmulq_f32(_v, _scale);
                        int8x8_t v = float2int8leakyrelu(_v, _v, _slope);
                        ptr0[0] = vget_lane_s8(v, 0);
                        ptr1[0] = vget_lane_s8(v, 1);
                        ptr2[0] = vget_lane_s8(v, 2);
                        ptr3[0] = vget_lane_s8(v, 3);

                        intptr += 4;
                        ptr0 += 1;
                        ptr1 += 1;
                        ptr2 += 1;
                        ptr3 += 1;
                    }
                }
            });
        }
        else
        {
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end))
                {
                    const int* intptr = bottom_blob_a[q].data();
                    signed char* ptr0 = top_blob_ra[q * 4].data();
                    signed char* ptr1 = top_blob_ra[q * 4 + 1].data();
                    signed char* ptr2 = top_blob_ra[q * 4 + 2].data();
                    signed char* ptr3 = top_blob_ra[q * 4 + 3].data();

                    float32x4_t _scale_in = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 4);
                    float32x4_t _scale_out = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 4);
                    float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 4);

                    float32x4_t _scale = vmulq_f32(_scale_in, _scale_out);
                    _bias = vmulq_f32(_bias, _scale_out);
                    float32x4_t _slope = vdupq_n_f32(slope);

                    int i = 0;
                    for (; i < size; i++)
                    {
                        float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
    #if __aarch64__
                        _v = vfmaq_f32(_bias, _v, _scale);
    #else
                        _v = vmlaq_f32(_bias, _v, _scale);
    #endif
                        int8x8_t v = float2int8leakyrelu(_v, _v, _slope);
                        ptr0[0] = vget_lane_s8(v, 0);
                        ptr1[0] = vget_lane_s8(v, 1);
                        ptr2[0] = vget_lane_s8(v, 2);
                        ptr3[0] = vget_lane_s8(v, 3);

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

Tensor requantize_from_int32_to_int8_neon(const Tensor& src, const Tensor& scale_in_data, const Tensor& scale_out_data, const Tensor& bias_data, int activation_type, const Tensor& activation_params, bool pack) {
    int dims = src.dim();
    int elempack = src.elempack();
    
    int scale_in_data_size = scale_in_data.size(0);
    const float* scale_in_data_ptr = (const float*)scale_in_data.data_ptr();
    int scale_out_data_size = scale_out_data.size(0);
    const float* scale_out_data_ptr = (const float*)scale_out_data.data_ptr();
    int bias_data_size = (bias_data.defined()) ? bias_data.size(0) : 0;
    const float* bias_data_ptr = (bias_data.defined()) ? bias_data.data_ptr<float>() : nullptr;
    
    Tensor dst;
    
    if (elempack == 8) {
        if (dims == 1) {
            int w = src.size(0);
            
            dst = otter::empty({w}, otter::ScalarType::Byte8);
            
            const int* srcptr = (const int*)src.data_ptr();
            float* dstptr = (float*)dst.data_ptr();

            if (scale_in_data_size == 1 && scale_out_data_size == 1) {
                float32x4_t _scale_in = vdupq_n_f32(scale_in_data_ptr[0]);
                float32x4_t _scale_out = vdupq_n_f32(scale_out_data_ptr[0]);

                if (bias_data_size == 0) {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 8;
                            signed char* ptr = (signed char*)dstptr + i * 8;

                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                            _v0 = vmulq_f32(_v0, _scale_in);
                            _v1 = vmulq_f32(_v1, _scale_in);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out);
                            _v1 = vmulq_f32(_v1, _scale_out);
                            vst1_s8(ptr, float2int8(_v0, _v1));
                        }
                    });
                } else if (bias_data_size == 1) {
                    float32x4_t _bias = vdupq_n_f32(bias_data_ptr[0]);

                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 8;
                            signed char* ptr = (signed char*)dstptr + i * 8;

                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                            _v0 = vmlaq_f32(_bias, _v0, _scale_in);
                            _v1 = vmlaq_f32(_bias, _v1, _scale_in);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out);
                            _v1 = vmulq_f32(_v1, _scale_out);
                            vst1_s8(ptr, float2int8(_v0, _v1));
                        }
                    });
                } else {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (int i = 0; i < w; i++) {
                            const int* intptr = (const int*)srcptr + i * 8;
                            signed char* ptr = (signed char*)dstptr + i * 8;

                            float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + i * 8);
                            float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + i * 8 + 4);
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                            _v0 = vmlaq_f32(_bias0, _v0, _scale_in);
                            _v1 = vmlaq_f32(_bias1, _v1, _scale_in);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out);
                            _v1 = vmulq_f32(_v1, _scale_out);
                            vst1_s8(ptr, float2int8(_v0, _v1));
                        }
                    });
                }
            } else if (scale_in_data_size == 1 && scale_out_data_size > 1) {
                float32x4_t _scale_in = vdupq_n_f32(scale_in_data_ptr[0]);

                if (bias_data_size == 0) {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 8;
                            signed char* ptr = (signed char*)dstptr + i * 8;

                            float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8);
                            float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8 + 4);
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                            _v0 = vmulq_f32(_v0, _scale_in);
                            _v1 = vmulq_f32(_v1, _scale_in);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out0);
                            _v1 = vmulq_f32(_v1, _scale_out1);
                            vst1_s8(ptr, float2int8(_v0, _v1));
                        }
                    });
                } else if (bias_data_size == 1) {
                    float32x4_t _bias = vdupq_n_f32(bias_data_ptr[0]);

                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 8;
                            signed char* ptr = (signed char*)dstptr + i * 8;

                            float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8);
                            float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8 + 4);
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                            _v0 = vmlaq_f32(_bias, _v0, _scale_in);
                            _v1 = vmlaq_f32(_bias, _v1, _scale_in);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out0);
                            _v1 = vmulq_f32(_v1, _scale_out1);
                            vst1_s8(ptr, float2int8(_v0, _v1));
                        }
                    });
                } else {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 8;
                            signed char* ptr = (signed char*)dstptr + i * 8;

                            float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8);
                            float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8 + 4);
                            float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + i * 8);
                            float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + i * 8 + 4);
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                            _v0 = vmlaq_f32(_bias0, _v0, _scale_in);
                            _v1 = vmlaq_f32(_bias1, _v1, _scale_in);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out0);
                            _v1 = vmulq_f32(_v1, _scale_out1);
                            vst1_s8(ptr, float2int8(_v0, _v1));
                        }
                    });
                }
            } else if (scale_in_data_size > 1 && scale_out_data_size == 1) {
                float32x4_t _scale_out = vdupq_n_f32(scale_out_data_ptr[0]);

                if (bias_data_size == 0) {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 8;
                            signed char* ptr = (signed char*)dstptr + i * 8;

                            float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8);
                            float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8 + 4);
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                            _v0 = vmulq_f32(_v0, _scale_in0);
                            _v1 = vmulq_f32(_v1, _scale_in1);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out);
                            _v1 = vmulq_f32(_v1, _scale_out);
                            vst1_s8(ptr, float2int8(_v0, _v1));
                        }
                    });
                } else if (bias_data_size == 1) {
                    float32x4_t _bias = vdupq_n_f32(bias_data_ptr[0]);

                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 8;
                            signed char* ptr = (signed char*)dstptr + i * 8;

                            float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8);
                            float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8 + 4);
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                            _v0 = vmlaq_f32(_bias, _v0, _scale_in0);
                            _v1 = vmlaq_f32(_bias, _v1, _scale_in1);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out);
                            _v1 = vmulq_f32(_v1, _scale_out);
                            vst1_s8(ptr, float2int8(_v0, _v1));
                        }
                    });
                } else {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 8;
                            signed char* ptr = (signed char*)dstptr + i * 8;

                            float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8);
                            float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8 + 4);
                            float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + i * 8);
                            float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + i * 8 + 4);
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                            _v0 = vmlaq_f32(_bias0, _v0, _scale_in0);
                            _v1 = vmlaq_f32(_bias1, _v1, _scale_in1);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out);
                            _v1 = vmulq_f32(_v1, _scale_out);
                            vst1_s8(ptr, float2int8(_v0, _v1));
                        }
                    });
                }
            } else {
                if (bias_data_size == 0) {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 8;
                            signed char* ptr = (signed char*)dstptr + i * 8;

                            float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8);
                            float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8 + 4);
                            float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8);
                            float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8 + 4);
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                            _v0 = vmulq_f32(_v0, _scale_in0);
                            _v1 = vmulq_f32(_v1, _scale_in1);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out0);
                            _v1 = vmulq_f32(_v1, _scale_out1);
                            vst1_s8(ptr, float2int8(_v0, _v1));
                        }
                    });
                } else if (bias_data_size == 1) {
                    float32x4_t _bias = vdupq_n_f32(bias_data_ptr[0]);

                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 8;
                            signed char* ptr = (signed char*)dstptr + i * 8;

                            float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8);
                            float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8 + 4);
                            float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8);
                            float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8 + 4);
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                            _v0 = vmlaq_f32(_bias, _v0, _scale_in0);
                            _v1 = vmlaq_f32(_bias, _v1, _scale_in1);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out0);
                            _v1 = vmulq_f32(_v1, _scale_out1);
                            vst1_s8(ptr, float2int8(_v0, _v1));
                        }
                    });
                } else {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 8;
                            signed char* ptr = (signed char*)dstptr + i * 8;

                            float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8);
                            float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8 + 4);
                            float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8);
                            float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8 + 4);
                            float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + i * 8);
                            float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + i * 8 + 4);
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                            _v0 = vmlaq_f32(_bias0, _v0, _scale_in0);
                            _v1 = vmlaq_f32(_bias1, _v1, _scale_in1);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out0);
                            _v1 = vmulq_f32(_v1, _scale_out1);
                            vst1_s8(ptr, float2int8(_v0, _v1));
                        }
                    });
                }
            }
        } else if (dims == 2) {
            int w = src.size(1);
            int h = src.size(0);
            
            dst = otter::empty({h, w}, otter::ScalarType::Byte8);

            auto src_a = src.accessor<int, 2, 8>();
            auto dst_a = dst.accessor<signed char, 2, 8>();

            if (bias_data_size == 0) {
                otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        const int* intptr = src_a[i].data();
                        signed char* ptr = dst_a[i].data();

                        float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8);
                        float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8 + 4);
                        float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8);
                        float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8 + 4);

                        for (int j = 0; j < w; j++)
                        {
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                            _v0 = vmulq_f32(_v0, _scale_in0);
                            _v1 = vmulq_f32(_v1, _scale_in1);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out0);
                            _v1 = vmulq_f32(_v1, _scale_out1);
                            vst1_s8(ptr, float2int8(_v0, _v1));

                            intptr += 8;
                            ptr += 8;
                        }
                    }
                });
            } else {
                otter::parallel_for(0, h, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end))
                    {
                        const int* intptr = src_a[i].data();
                        signed char* ptr = dst_a[i].data();

                        float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8);
                        float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8 + 4);
                        float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8);
                        float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8 + 4);
                        float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + i * 8);
                        float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + i * 8 + 4);

                        for (int j = 0; j < w; j++)
                        {
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                            _v0 = vmlaq_f32(_bias0, _v0, _scale_in0);
                            _v1 = vmlaq_f32(_bias1, _v1, _scale_in1);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out0);
                            _v1 = vmulq_f32(_v1, _scale_out1);
                            vst1_s8(ptr, float2int8(_v0, _v1));

                            intptr += 8;
                            ptr += 8;
                        }
                    }
                });
            }
        } else if (dims == 3) {
            int w = src.size(2);
            int h = src.size(1);
            int channels = src.size(0);
            int size = w * h;
            
            dst = otter::empty({channels, h, w}, otter::ScalarType::Byte8);
            
            if (activation_type == 1) {
                requantize_relu_pack8_neon(src, dst, scale_in_data, scale_out_data, bias_data);
                
                return dst;
            }

            if (activation_type == 2 && activation_params.data_ptr<float>()[0] > 0.f) {
                requantize_leakyrelu_pack8_neon(src, dst, scale_in_data, scale_out_data, bias_data, activation_params.data_ptr<float>()[0]);
                
                return dst;
            }
            
            auto src_a = src.accessor<int, 3, 8>();
            auto dst_a = dst.accessor<signed char, 3, 8>();

            if (bias_data_size == 0) {
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const int* intptr = src_a[q].data();
                        signed char* ptr = dst_a[q].data();

                        float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8);
                        float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8 + 4);
                        float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8);
                        float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8 + 4);

                        for (int i = 0; i < size; i++)
                        {
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                            _v0 = vmulq_f32(_v0, _scale_in0);
                            _v1 = vmulq_f32(_v1, _scale_in1);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out0);
                            _v1 = vmulq_f32(_v1, _scale_out1);
                            vst1_s8(ptr, float2int8(_v0, _v1));

                            intptr += 8;
                            ptr += 8;
                        }
                    }
                });
            } else {
                otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const int* intptr = src_a[q].data();
                        signed char* ptr = dst_a[q].data();

                        float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8);
                        float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8 + 4);
                        float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8);
                        float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8 + 4);
                        float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8);
                        float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8 + 4);

                        for (int i = 0; i < size; i++)
                        {
                            float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                            float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                            _v0 = vmlaq_f32(_bias0, _v0, _scale_in0);
                            _v1 = vmlaq_f32(_bias1, _v1, _scale_in1);
                            _v0 = activation_ps(_v0, activation_type, activation_params);
                            _v1 = activation_ps(_v1, activation_type, activation_params);
                            _v0 = vmulq_f32(_v0, _scale_out0);
                            _v1 = vmulq_f32(_v1, _scale_out1);
                            vst1_s8(ptr, float2int8(_v0, _v1));

                            intptr += 8;
                            ptr += 8;
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
            
            dst = otter::empty({batchsize, channels, h, w}, otter::ScalarType::Byte8);
            
            for (const auto b : otter::irange(0, batchsize)) {
                if (activation_type == 1) {
                    const Tensor& src_b = src[b];
                    Tensor dst_b = dst[b];
                    requantize_relu_pack8_neon(src_b, dst_b, scale_in_data, scale_out_data, bias_data);
                    
                    return dst;
                }

                if (activation_type == 2 && activation_params.data_ptr<float>()[0] > 0.f) {
                    const Tensor& src_b = src[b];
                    Tensor dst_b = dst[b];
                    requantize_leakyrelu_pack8_neon(src_b, dst_b, scale_in_data, scale_out_data, bias_data, activation_params.data_ptr<float>()[0]);
                    
                    return dst;
                }
                
                auto src_a = src.accessor<int, 4, 8>()[b];
                auto dst_a = dst.accessor<signed char, 4, 8>()[b];

                if (bias_data_size == 0) {
                    otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                        for (const auto q : otter::irange(begin, end)) {
                            const int* intptr = src_a[q].data();
                            signed char* ptr = dst_a[q].data();

                            float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8);
                            float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8 + 4);
                            float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8);
                            float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8 + 4);

                            for (int i = 0; i < size; i++)
                            {
                                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                                _v0 = vmulq_f32(_v0, _scale_in0);
                                _v1 = vmulq_f32(_v1, _scale_in1);
                                _v0 = activation_ps(_v0, activation_type, activation_params);
                                _v1 = activation_ps(_v1, activation_type, activation_params);
                                _v0 = vmulq_f32(_v0, _scale_out0);
                                _v1 = vmulq_f32(_v1, _scale_out1);
                                vst1_s8(ptr, float2int8(_v0, _v1));

                                intptr += 8;
                                ptr += 8;
                            }
                        }
                    });
                } else {
                    otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                        for (const auto q : otter::irange(begin, end)) {
                            const int* intptr = src_a[q].data();
                            signed char* ptr = dst_a[q].data();

                            float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8);
                            float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8 + 4);
                            float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8);
                            float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8 + 4);
                            float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8);
                            float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8 + 4);

                            for (int i = 0; i < size; i++)
                            {
                                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr));
                                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32((intptr + 4)));
                                _v0 = vmlaq_f32(_bias0, _v0, _scale_in0);
                                _v1 = vmlaq_f32(_bias1, _v1, _scale_in1);
                                _v0 = activation_ps(_v0, activation_type, activation_params);
                                _v1 = activation_ps(_v1, activation_type, activation_params);
                                _v0 = vmulq_f32(_v0, _scale_out0);
                                _v1 = vmulq_f32(_v1, _scale_out1);
                                vst1_s8(ptr, float2int8(_v0, _v1));

                                intptr += 8;
                                ptr += 8;
                            }
                        }
                    });
                }
            }
        }
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
                float32x4_t _scale_in = vdupq_n_f32(scale_in_data_ptr[0]);
                float32x4_t _scale_out = vdupq_n_f32(scale_out_data_ptr[0]);

                if (bias_data_size == 0) {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmulq_f32(_v, _scale_in);
                            _v = activation_ps(_v, activation_type, activation_params);
                            _v = vmulq_f32(_v, _scale_out);
                            int8x8_t v = float2int8(_v, _v);
                            ptr[0] = vget_lane_s8(v, 0);
                            ptr[1] = vget_lane_s8(v, 1);
                            ptr[2] = vget_lane_s8(v, 2);
                            ptr[3] = vget_lane_s8(v, 3);
                        }
                    });
                } else if (bias_data_size == 1) {
                    float32x4_t _bias = vdupq_n_f32(bias_data_ptr[0]);

                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmlaq_f32(_bias, _v, _scale_in);
                            _v = activation_ps(_v, activation_type, activation_params);
                            _v = vmulq_f32(_v, _scale_out);
                            int8x8_t v = float2int8(_v, _v);
                            ptr[0] = vget_lane_s8(v, 0);
                            ptr[1] = vget_lane_s8(v, 1);
                            ptr[2] = vget_lane_s8(v, 2);
                            ptr[3] = vget_lane_s8(v, 3);
                        }
                    });
                } else {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            float32x4_t _bias = vld1q_f32((const float*)bias_data_ptr + i * 4);
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmlaq_f32(_bias, _v, _scale_in);
                            _v = activation_ps(_v, activation_type, activation_params);
                            _v = vmulq_f32(_v, _scale_out);
                            int8x8_t v = float2int8(_v, _v);
                            ptr[0] = vget_lane_s8(v, 0);
                            ptr[1] = vget_lane_s8(v, 1);
                            ptr[2] = vget_lane_s8(v, 2);
                            ptr[3] = vget_lane_s8(v, 3);
                        }
                    });
                }
            } else if (scale_in_data_size == 1 && scale_out_data_size > 1) {
                float32x4_t _scale_in = vdupq_n_f32(scale_in_data_ptr[0]);

                if (bias_data_size == 0) {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            float32x4_t _scale_out = vld1q_f32((const float*)scale_out_data_ptr + i * 4);
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmulq_f32(_v, _scale_in);
                            _v = activation_ps(_v, activation_type, activation_params);
                            _v = vmulq_f32(_v, _scale_out);
                            int8x8_t v = float2int8(_v, _v);
                            ptr[0] = vget_lane_s8(v, 0);
                            ptr[1] = vget_lane_s8(v, 1);
                            ptr[2] = vget_lane_s8(v, 2);
                            ptr[3] = vget_lane_s8(v, 3);
                        }
                    });
                } else if (bias_data_size == 1) {
                    float32x4_t _bias = vdupq_n_f32(bias_data_ptr[0]);

                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            float32x4_t _scale_out = vld1q_f32((const float*)scale_out_data_ptr + i * 4);
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmlaq_f32(_bias, _v, _scale_in);
                            _v = activation_ps(_v, activation_type, activation_params);
                            _v = vmulq_f32(_v, _scale_out);
                            int8x8_t v = float2int8(_v, _v);
                            ptr[0] = vget_lane_s8(v, 0);
                            ptr[1] = vget_lane_s8(v, 1);
                            ptr[2] = vget_lane_s8(v, 2);
                            ptr[3] = vget_lane_s8(v, 3);
                        }
                    });
                } else {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            float32x4_t _scale_out = vld1q_f32((const float*)scale_out_data_ptr + i * 4);
                            float32x4_t _bias = vld1q_f32((const float*)bias_data_ptr + i * 4);
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmlaq_f32(_bias, _v, _scale_in);
                            _v = activation_ps(_v, activation_type, activation_params);
                            _v = vmulq_f32(_v, _scale_out);
                            int8x8_t v = float2int8(_v, _v);
                            ptr[0] = vget_lane_s8(v, 0);
                            ptr[1] = vget_lane_s8(v, 1);
                            ptr[2] = vget_lane_s8(v, 2);
                            ptr[3] = vget_lane_s8(v, 3);
                        }
                    });
                }
            } else if (scale_in_data_size > 1 && scale_out_data_size == 1) {
                float32x4_t _scale_out = vdupq_n_f32(scale_out_data_ptr[0]);

                if (bias_data_size == 0) {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            float32x4_t _scale_in = vld1q_f32((const float*)scale_in_data_ptr + i * 4);
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmulq_f32(_v, _scale_in);
                            _v = activation_ps(_v, activation_type, activation_params);
                            _v = vmulq_f32(_v, _scale_out);
                            int8x8_t v = float2int8(_v, _v);
                            ptr[0] = vget_lane_s8(v, 0);
                            ptr[1] = vget_lane_s8(v, 1);
                            ptr[2] = vget_lane_s8(v, 2);
                            ptr[3] = vget_lane_s8(v, 3);
                        }
                    });
                } else if (bias_data_size == 1) {
                    float32x4_t _bias = vdupq_n_f32(bias_data_ptr[0]);

                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            float32x4_t _scale_in = vld1q_f32((const float*)scale_in_data_ptr + i * 4);
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmlaq_f32(_bias, _v, _scale_in);
                            _v = activation_ps(_v, activation_type, activation_params);
                            _v = vmulq_f32(_v, _scale_out);
                            int8x8_t v = float2int8(_v, _v);
                            ptr[0] = vget_lane_s8(v, 0);
                            ptr[1] = vget_lane_s8(v, 1);
                            ptr[2] = vget_lane_s8(v, 2);
                            ptr[3] = vget_lane_s8(v, 3);
                        }
                    });
                } else {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            float32x4_t _scale_in = vld1q_f32((const float*)scale_in_data_ptr + i * 4);
                            float32x4_t _bias = vld1q_f32((const float*)bias_data_ptr + i * 4);
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmlaq_f32(_bias, _v, _scale_in);
                            _v = activation_ps(_v, activation_type, activation_params);
                            _v = vmulq_f32(_v, _scale_out);
                            int8x8_t v = float2int8(_v, _v);
                            ptr[0] = vget_lane_s8(v, 0);
                            ptr[1] = vget_lane_s8(v, 1);
                            ptr[2] = vget_lane_s8(v, 2);
                            ptr[3] = vget_lane_s8(v, 3);
                        }
                    });
                }
            } else {
                if (bias_data_size == 0) {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            float32x4_t _scale_in = vld1q_f32((const float*)scale_in_data_ptr + i * 4);
                            float32x4_t _scale_out = vld1q_f32((const float*)scale_out_data_ptr + i * 4);
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmulq_f32(_v, _scale_in);
                            _v = activation_ps(_v, activation_type, activation_params);
                            _v = vmulq_f32(_v, _scale_out);
                            int8x8_t v = float2int8(_v, _v);
                            ptr[0] = vget_lane_s8(v, 0);
                            ptr[1] = vget_lane_s8(v, 1);
                            ptr[2] = vget_lane_s8(v, 2);
                            ptr[3] = vget_lane_s8(v, 3);
                        }
                    });
                } else if (bias_data_size == 1) {
                    float32x4_t _bias = vdupq_n_f32(bias_data_ptr[0]);

                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            float32x4_t _scale_in = vld1q_f32((const float*)scale_in_data_ptr + i * 4);
                            float32x4_t _scale_out = vld1q_f32((const float*)scale_out_data_ptr + i * 4);
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmlaq_f32(_bias, _v, _scale_in);
                            _v = activation_ps(_v, activation_type, activation_params);
                            _v = vmulq_f32(_v, _scale_out);
                            int8x8_t v = float2int8(_v, _v);
                            ptr[0] = vget_lane_s8(v, 0);
                            ptr[1] = vget_lane_s8(v, 1);
                            ptr[2] = vget_lane_s8(v, 2);
                            ptr[3] = vget_lane_s8(v, 3);
                        }
                    });
                } else {
                    otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr = (const int*)srcptr + i * 4;
                            signed char* ptr = (signed char*)dstptr + i * 4;

                            float32x4_t _scale_in = vld1q_f32((const float*)scale_in_data_ptr + i * 4);
                            float32x4_t _scale_out = vld1q_f32((const float*)scale_out_data_ptr + i * 4);
                            float32x4_t _bias = vld1q_f32((const float*)bias_data_ptr + i * 4);
                            float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                            _v = vmlaq_f32(_bias, _v, _scale_in);
                            _v = activation_ps(_v, activation_type, activation_params);
                            _v = vmulq_f32(_v, _scale_out);
                            int8x8_t v = float2int8(_v, _v);
                            ptr[0] = vget_lane_s8(v, 0);
                            ptr[1] = vget_lane_s8(v, 1);
                            ptr[2] = vget_lane_s8(v, 2);
                            ptr[3] = vget_lane_s8(v, 3);
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
            
            auto src_a = src.accessor<int, 2, 4>();
            auto dst_ra = dst.raw_accessor<signed char, 2>();
            
            if (out_elempack == 8) {
                if (bias_data_size == 0) {
                    otter::parallel_for(0, outh, 0, [&](int64_t begin, int64_t end) {
                        for (const auto i : otter::irange(begin, end)) {
                            const int* intptr0 = src_a[i * 2].data();
                            const int* intptr1 = src_a[i * 2 + 1].data();
                            signed char* ptr = dst_ra[i].data();

                            float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8);
                            float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8 + 4);
                            float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8);
                            float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8 + 4);

                            for (int j = 0; j < w; j++)
                            {
                                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr0));
                                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr1));
                                _v0 = vmulq_f32(_v0, _scale_in0);
                                _v1 = vmulq_f32(_v1, _scale_in1);
                                _v0 = activation_ps(_v0, activation_type, activation_params);
                                _v1 = activation_ps(_v1, activation_type, activation_params);
                                _v0 = vmulq_f32(_v0, _scale_out0);
                                _v1 = vmulq_f32(_v1, _scale_out1);
                                vst1_s8(ptr, float2int8(_v0, _v1));

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

                            float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8);
                            float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 8 + 4);
                            float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8);
                            float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 8 + 4);
                            float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + i * 8);
                            float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + i * 8 + 4);

                            for (int j = 0; j < w; j++)
                            {
                                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr0));
                                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr1));
                                _v0 = vmlaq_f32(_bias0, _v0, _scale_in0);
                                _v1 = vmlaq_f32(_bias1, _v1, _scale_in1);
                                _v0 = activation_ps(_v0, activation_type, activation_params);
                                _v1 = activation_ps(_v1, activation_type, activation_params);
                                _v0 = vmulq_f32(_v0, _scale_out0);
                                _v1 = vmulq_f32(_v1, _scale_out1);
                                vst1_s8(ptr, float2int8(_v0, _v1));

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

                            float32x4_t _scale_in = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 4);
                            float32x4_t _scale_out = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 4);

                            for (int j = 0; j < w; j++)
                            {
                                float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                                _v = vmulq_f32(_v, _scale_in);
                                _v = activation_ps(_v, activation_type, activation_params);
                                _v = vmulq_f32(_v, _scale_out);
                                int8x8_t v = float2int8(_v, _v);
                                ptr0[0] = vget_lane_s8(v, 0);
                                ptr1[0] = vget_lane_s8(v, 1);
                                ptr2[0] = vget_lane_s8(v, 2);
                                ptr3[0] = vget_lane_s8(v, 3);

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

                            float32x4_t _scale_in = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + i * 4);
                            float32x4_t _scale_out = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + i * 4);
                            float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + i * 4);

                            for (int j = 0; j < w; j++)
                            {
                                float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                                _v = vmlaq_f32(_bias, _v, _scale_in);
                                _v = activation_ps(_v, activation_type, activation_params);
                                _v = vmulq_f32(_v, _scale_out);
                                int8x8_t v = float2int8(_v, _v);
                                ptr0[0] = vget_lane_s8(v, 0);
                                ptr1[0] = vget_lane_s8(v, 1);
                                ptr2[0] = vget_lane_s8(v, 2);
                                ptr3[0] = vget_lane_s8(v, 3);

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
            
            if (activation_type == 1) {
                requantize_relu_pack4_neon(src, dst, scale_in_data, scale_out_data, bias_data);
                
                return dst;
            }

            if (activation_type == 2 && activation_params.data_ptr<float>()[0] > 0.f) {
                requantize_leakyrelu_pack4_neon(src, dst, scale_in_data, scale_out_data, bias_data, activation_params.data_ptr<float>()[0]);
                
                return dst;
            }
            
            auto src_a = src.accessor<int, 3, 4>();
            auto dst_ra = dst.raw_accessor<signed char, 3>();
            
            if (out_elempack == 8) {
                if (bias_data_size == 0) {
                    otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                        for (const auto q : otter::irange(begin, end)) {
                            const int* intptr0 = src_a[q * 2].data();
                            const int* intptr1 = src_a[q * 2 + 1].data();
                            signed char* ptr = dst_ra[q].data();

                            float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8);
                            float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8 + 4);
                            float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8);
                            float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8 + 4);

                            for (int i = 0; i < size; i++)
                            {
                                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr0));
                                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr1));
                                _v0 = vmulq_f32(_v0, _scale_in0);
                                _v1 = vmulq_f32(_v1, _scale_in1);
                                _v0 = activation_ps(_v0, activation_type, activation_params);
                                _v1 = activation_ps(_v1, activation_type, activation_params);
                                _v0 = vmulq_f32(_v0, _scale_out0);
                                _v1 = vmulq_f32(_v1, _scale_out1);
                                vst1_s8(ptr, float2int8(_v0, _v1));

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

                            float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8);
                            float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8 + 4);
                            float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8);
                            float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8 + 4);
                            float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8);
                            float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8 + 4);

                            for (int i = 0; i < size; i++)
                            {
                                float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr0));
                                float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr1));
                                _v0 = vmlaq_f32(_bias0, _v0, _scale_in0);
                                _v1 = vmlaq_f32(_bias1, _v1, _scale_in1);
                                _v0 = activation_ps(_v0, activation_type, activation_params);
                                _v1 = activation_ps(_v1, activation_type, activation_params);
                                _v0 = vmulq_f32(_v0, _scale_out0);
                                _v1 = vmulq_f32(_v1, _scale_out1);
                                vst1_s8(ptr, float2int8(_v0, _v1));

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

                            float32x4_t _scale_in = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 4);
                            float32x4_t _scale_out = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 4);

                            for (int i = 0; i < size; i++)
                            {
                                float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                                _v = vmulq_f32(_v, _scale_in);
                                _v = activation_ps(_v, activation_type, activation_params);
                                _v = vmulq_f32(_v, _scale_out);
                                int8x8_t v = float2int8(_v, _v);
                                ptr0[0] = vget_lane_s8(v, 0);
                                ptr1[0] = vget_lane_s8(v, 1);
                                ptr2[0] = vget_lane_s8(v, 2);
                                ptr3[0] = vget_lane_s8(v, 3);

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

                            float32x4_t _scale_in = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 4);
                            float32x4_t _scale_out = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 4);
                            float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 4);

                            for (int i = 0; i < size; i++)
                            {
                                float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                                _v = vmlaq_f32(_bias, _v, _scale_in);
                                _v = activation_ps(_v, activation_type, activation_params);
                                _v = vmulq_f32(_v, _scale_out);
                                int8x8_t v = float2int8(_v, _v);
                                ptr0[0] = vget_lane_s8(v, 0);
                                ptr1[0] = vget_lane_s8(v, 1);
                                ptr2[0] = vget_lane_s8(v, 2);
                                ptr3[0] = vget_lane_s8(v, 3);

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
                if (activation_type == 1) {
                    const Tensor& src_b = src[b];
                    Tensor dst_b = dst[b];
                    
                    requantize_relu_pack4_neon(src_b, dst_b, scale_in_data, scale_out_data, bias_data);
                    
                    return dst;
                }

                if (activation_type == 2 && activation_params.data_ptr<float>()[0] > 0.f) {
                    const Tensor& src_b = src[b];
                    Tensor dst_b = dst[b];
                    
                    requantize_leakyrelu_pack4_neon(src_b, dst_b, scale_in_data, scale_out_data, bias_data, activation_params.data_ptr<float>()[0]);
                    
                    return dst;
                }
                
                auto src_a = src.accessor<int, 4, 4>()[b];
                auto dst_ra = dst.raw_accessor<signed char, 4>()[b];
                
                if (out_elempack == 8) {
                    if (bias_data_size == 0) {
                        otter::parallel_for(0, outc, 0, [&](int64_t begin, int64_t end) {
                            for (const auto q : otter::irange(begin, end)) {
                                const int* intptr0 = src_a[q * 2].data();
                                const int* intptr1 = src_a[q * 2 + 1].data();
                                signed char* ptr = dst_ra[q].data();

                                float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8);
                                float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8 + 4);
                                float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8);
                                float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8 + 4);

                                for (int i = 0; i < size; i++)
                                {
                                    float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr0));
                                    float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr1));
                                    _v0 = vmulq_f32(_v0, _scale_in0);
                                    _v1 = vmulq_f32(_v1, _scale_in1);
                                    _v0 = activation_ps(_v0, activation_type, activation_params);
                                    _v1 = activation_ps(_v1, activation_type, activation_params);
                                    _v0 = vmulq_f32(_v0, _scale_out0);
                                    _v1 = vmulq_f32(_v1, _scale_out1);
                                    vst1_s8(ptr, float2int8(_v0, _v1));

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

                                float32x4_t _scale_in0 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8);
                                float32x4_t _scale_in1 = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 8 + 4);
                                float32x4_t _scale_out0 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8);
                                float32x4_t _scale_out1 = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 8 + 4);
                                float32x4_t _bias0 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8);
                                float32x4_t _bias1 = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 8 + 4);

                                for (int i = 0; i < size; i++)
                                {
                                    float32x4_t _v0 = vcvtq_f32_s32(vld1q_s32(intptr0));
                                    float32x4_t _v1 = vcvtq_f32_s32(vld1q_s32(intptr1));
                                    _v0 = vmlaq_f32(_bias0, _v0, _scale_in0);
                                    _v1 = vmlaq_f32(_bias1, _v1, _scale_in1);
                                    _v0 = activation_ps(_v0, activation_type, activation_params);
                                    _v1 = activation_ps(_v1, activation_type, activation_params);
                                    _v0 = vmulq_f32(_v0, _scale_out0);
                                    _v1 = vmulq_f32(_v1, _scale_out1);
                                    vst1_s8(ptr, float2int8(_v0, _v1));

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

                                float32x4_t _scale_in = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 4);
                                float32x4_t _scale_out = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 4);

                                for (int i = 0; i < size; i++)
                                {
                                    float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                                    _v = vmulq_f32(_v, _scale_in);
                                    _v = activation_ps(_v, activation_type, activation_params);
                                    _v = vmulq_f32(_v, _scale_out);
                                    int8x8_t v = float2int8(_v, _v);
                                    ptr0[0] = vget_lane_s8(v, 0);
                                    ptr1[0] = vget_lane_s8(v, 1);
                                    ptr2[0] = vget_lane_s8(v, 2);
                                    ptr3[0] = vget_lane_s8(v, 3);

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

                                float32x4_t _scale_in = scale_in_data_size == 1 ? vdupq_n_f32(scale_in_data_ptr[0]) : vld1q_f32((const float*)scale_in_data_ptr + q * 4);
                                float32x4_t _scale_out = scale_out_data_size == 1 ? vdupq_n_f32(scale_out_data_ptr[0]) : vld1q_f32((const float*)scale_out_data_ptr + q * 4);
                                float32x4_t _bias = bias_data_size == 1 ? vdupq_n_f32(bias_data_ptr[0]) : vld1q_f32((const float*)bias_data_ptr + q * 4);

                                for (int i = 0; i < size; i++)
                                {
                                    float32x4_t _v = vcvtq_f32_s32(vld1q_s32(intptr));
                                    _v = vmlaq_f32(_bias, _v, _scale_in);
                                    _v = activation_ps(_v, activation_type, activation_params);
                                    _v = vmulq_f32(_v, _scale_out);
                                    int8x8_t v = float2int8(_v, _v);
                                    ptr0[0] = vget_lane_s8(v, 0);
                                    ptr1[0] = vget_lane_s8(v, 1);
                                    ptr2[0] = vget_lane_s8(v, 2);
                                    ptr3[0] = vget_lane_s8(v, 3);

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
        
        return dst;
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
                const float bias = bias_data_ptr[0];

                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        float v = intptr[i] * scale_in + bias;
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                    }
                });
            } else {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        float v = intptr[i] * scale_in + bias_data_ptr[i];
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
                const float bias = bias_data_ptr[0];

                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        float v = intptr[i] * scale_in + bias;
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data_ptr[i]);
                    }
                });
            } else {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        float v = intptr[i] * scale_in + bias_data_ptr[i];
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
                const float bias = bias_data_ptr[0];

                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        float v = intptr[i] * scale_in_data_ptr[i] + bias;
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                    }
                });
            } else {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        float v = intptr[i] * scale_in_data_ptr[i] + bias_data_ptr[i];
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
                const float bias = bias_data_ptr[0];

                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        float v = intptr[i] * scale_in_data_ptr[i] + bias;
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out_data_ptr[i]);
                    }
                });
            } else {
                otter::parallel_for(0, w, 0, [&](int64_t begin, int64_t end) {
                    for (const auto i : otter::irange(begin, end)) {
                        float v = intptr[i] * scale_in_data_ptr[i] + bias_data_ptr[i];
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
                    const float bias = bias_data_size == 1 ? bias_data_ptr[0] : bias_data_ptr[i];

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
                    const float bias = bias_data_size == 1 ? bias_data_ptr[0] : bias_data_ptr[q];

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
                        const float bias = bias_data_size == 1 ? bias_data_ptr[0] : bias_data_ptr[q];

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

}

#endif
