//
//  Quantize.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/2.
//

#include "Quantize.hpp"
#include "TensorFactory.hpp"
#include "Parallel.hpp"

#include "QuantizeX86.hpp"

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

Tensor quantize_to_int8(const Tensor& src, const Tensor& scale_data) {
    auto dst = otter::empty_like(src, otter::ScalarType::Byte);
    int scale_data_size = scale_data.size(0);
    auto scale_data_a = scale_data.accessor<float, 1>();
    
    if (src.dim() == 2) {
        OTTER_CHECK(false, "quantize 1D unimplement");
    } else if (src.dim() == 3) {
        OTTER_CHECK(false, "quantize 2D unimplement");
    } else if (src.dim() == 4) {
        int channels = src.size(1);
        int h = src.size(2);
        int w = src.size(3);
        int size = w * h;
        
        auto src_a = src.accessor<float, 4>()[0];
        auto dst_a = dst.accessor<unsigned char, 4>()[0];
        
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
    
    return dst;
}

Tensor dequantize_from_int32(const Tensor& src, const Tensor& scale_data, const Tensor& bias_data) {
    auto dst = otter::empty_like(src, otter::ScalarType::Float);
    
    int scale_data_size = scale_data.size(0);
    auto scale_data_a = scale_data.accessor<float, 1>();
    int bias_data_size = bias_data.size(0);
    auto bias_data_a = bias_data.accessor<float, 1>();
    
    if (src.dim() == 2) {
        OTTER_CHECK(false, "dequantize 1D unimplement");
    } else if (src.dim() == 3) {
        OTTER_CHECK(false, "dequantize 2D unimplement");
    } else if (src.dim() == 4) {
        int channels = src.size(1);
        int h = src.size(2);
        int w = src.size(3);
        int size = w * h;
        
        auto src_a = src.accessor<int, 4>()[0];
        auto dst_a = dst.accessor<float, 4>()[0];
        
        if (bias_data_size == 0) {
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const int* intptr = src_a[q].data();
                    float* ptr = dst_a[q].data();

                    const float scale = scale_data_size == 1 ? scale_data_a[0] : scale_data_a[q];

                    for (int i = 0; i < size; i++) {
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

                    for (int i = 0; i < size; i++) {
                        ptr[i] = intptr[i] * scale + bias;
                    }
                }
            });
        }
    }
    
    return dst;
}

Tensor requantize_from_int32_to_int8(const Tensor& src, const Tensor& scale_in_data, const Tensor& scale_out_data, const Tensor& bias_data, int activation_type, const Tensor& activation_params) {
    auto dst = otter::empty_like(src, otter::ScalarType::Byte);
    
    int scale_in_data_size = scale_in_data.size(0);
    auto scale_in_data_a = scale_in_data.accessor<float, 1>();
    int scale_out_data_size = scale_out_data.size(0);
    auto scale_out_data_a = scale_out_data.accessor<float, 1>();
    int bias_data_size = (bias_data.defined()) ? bias_data.size(0) : 0;
    const float* bias_data_a = (bias_data.defined()) ? bias_data.data_ptr<float>() : nullptr;
    
    if (src.dim() == 2) {
        OTTER_CHECK(false, "dequantize 1D unimplement");
    } else if (src.dim() == 3) {
        OTTER_CHECK(false, "dequantize 2D unimplement");
    } else if (src.dim() == 4) {
        int channels = src.size(1);
        int h = src.size(2);
        int w = src.size(3);
        int size = w * h;
        
        auto src_a = src.accessor<int, 4>()[0];
        auto dst_a = dst.accessor<unsigned char, 4>()[0];
        
        if (bias_data_size == 0) {
            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(begin, end)) {
                    const int* intptr = src_a[q].data();
                    signed char* ptr = (signed char*)dst_a[q].data();

                    const float scale_in = scale_in_data_size == 1 ? scale_in_data_a[0] : scale_in_data_a[q];
                    const float scale_out = scale_out_data_size == 1 ? scale_out_data_a[0] : scale_out_data_a[q];

                    for (int i = 0; i < size; i++) {
                        float v = intptr[i] * scale_in;
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                    }
                }
            });
        } else {
//            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
                for (const auto q : otter::irange(0, channels)) {
                    const int* intptr = src_a[q].data();
                    signed char* ptr = (signed char*)dst_a[q].data();

                    const float scale_in = scale_in_data_size == 1 ? scale_in_data_a[0] : scale_in_data_a[q];
                    const float scale_out = scale_out_data_size == 1 ? scale_out_data_a[0] : scale_out_data_a[q];
                    const float bias = bias_data_size == 1 ? bias_data_a[0] : bias_data_a[q];

                    for (int i = 0; i < size; i++) {
                        float v = intptr[i] * scale_in + bias;
                        ptr[i] = float2int8(activation_ss(v, activation_type, activation_params) * scale_out);
                    }
                }
//            });
        }
    }
    
    return dst;
}

}   // end namespace otter
