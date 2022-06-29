//
//  Quantize.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/2.
//

#include "Quantize.hpp"
#include "TensorFactory.hpp"
#include "Parallel.hpp"
#include "VecIntrinsic.hpp"

#if __SSE2__
#include "QuantizeX86.hpp"
#endif

#if __ARM_NEON__
#include "QuantizeNeon.hpp"
#endif

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

Tensor quantize_to_int8(const Tensor& src, const Tensor& scale_data, bool pack) {
    
#if __SSE2__
    return quantize_to_int8_x86(src, scale_data, pack);
#elif __ARM_NEON__
    return quantize_to_int8_neon(src, scale_data, pack);
#else
    auto dst = otter::empty_like(src, otter::ScalarType::Byte);
    int scale_data_size = scale_data.size(0);
    auto scale_data_a = scale_data.accessor<float, 1>();
    
    if (src.dim() == 1) {
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
    } else if (src.dim() == 2) {
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
    } else if (src.dim() == 3) {
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
    } else if (src.dim() == 4) {
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
#endif
}

Tensor dequantize_from_int32(const Tensor& src, const Tensor& scale_data, const Tensor& bias_data, bool pack) {
    
#if __SSE2__
    return dequantize_from_int32_x86(src, scale_data, bias_data, pack);
#elif __ARM_NEON
    return dequantize_from_int32_neon(src, scale_data, bias_data, pack);
#else
    auto dst = otter::empty_like(src, otter::ScalarType::Float);
    
    int scale_data_size = scale_data.size(0);
    auto scale_data_a = scale_data.accessor<float, 1>();
    int bias_data_size = bias_data.size(0);
    auto bias_data_a = bias_data.accessor<float, 1>();
    
    if (src.dim() == 1) {
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
    } else if (src.dim() == 2) {
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
                    for (; j < w; j++)
                    {
                        *ptr++ = *intptr++ * scale + bias;
                    }
                }
            });
        }
    } else if (src.dim() == 3) {
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
                    for (; i < size; i++) {
                        ptr[i] = intptr[i] * scale + bias;
                    }
                }
            });
        }
    } else if (src.dim() == 4) {
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
                        for (; i < size; i++) {
                            ptr[i] = intptr[i] * scale + bias;
                        }
                    }
                });
            }
        }
    }
    
    return dst;
#endif
}

Tensor requantize_from_int32_to_int8(const Tensor& src, const Tensor& scale_in_data, const Tensor& scale_out_data, const Tensor& bias_data, int activation_type, const Tensor& activation_params, bool pack) {
    
#if __SSE2__
    return requantize_from_int32_to_int8_x86(src, scale_in_data, scale_out_data, bias_data, activation_type, activation_params, pack);
#elif __ARM_NEON__
    return requantize_from_int32_to_int8_neon(src, scale_in_data, scale_out_data, bias_data, activation_type, activation_params, pack);
#else
    int dims = src.dim();
    
    int scale_in_data_size = scale_in_data.size(0);
    const float* scale_in_data_ptr = (const float*)scale_in_data.data_ptr();
    int scale_out_data_size = scale_out_data.size(0);
    const float* scale_out_data_ptr = (const float*)scale_out_data.data_ptr();
    int bias_data_size = (bias_data.defined()) ? bias_data.size(0) : 0;
    const float* bias_data_a = (bias_data.defined()) ? bias_data.data_ptr<float>() : nullptr;
    
    Tensor dst;
    
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
                    for (const auto i : otter::irange(begin, end)) {
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
#endif
}

}   // end namespace otter
