//
//  Quantize.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/2.
//

#include "Quantize.hpp"
#include "TensorFactory.hpp"
#include "Parallel.hpp"

namespace otter {

static inline signed char float2int8(float v) {
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
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
        
//        if (scale_data_size == 1) {
//            const float scale = scale_data_a[0];
//
//            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
//                for (const auto q : otter::irange(begin, end)) {
//                    const float* ptr0 = src_a[q].data();
//                    signed char* outptr0 = (signed char*)dst_a[q * 4 + 0].data();
//                    signed char* outptr1 = (signed char*)dst_a[q * 4 + 1].data();
//                    signed char* outptr2 = (signed char*)dst_a[q * 4 + 2].data();
//                    signed char* outptr3 = (signed char*)dst_a[q * 4 + 3].data();
//
//                    for (int i = 0; i < size; i++)
//                    {
//                        outptr0[0] = float2int8(ptr0[0] * scale);
//                        outptr1[0] = float2int8(ptr0[1] * scale);
//                        outptr2[0] = float2int8(ptr0[2] * scale);
//                        outptr3[0] = float2int8(ptr0[3] * scale);
//
//                        ptr0 += 4;
//                        outptr0 += 1;
//                        outptr1 += 1;
//                        outptr2 += 1;
//                        outptr3 += 1;
//                    }
//                }
//            });
//        } else {
//            otter::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
//                for (const auto q : otter::irange(begin, end)) {
//                    const float* ptr0 = src_a[q].data();
//                    signed char* outptr0 = (signed char*)dst_a[q * 4 + 0].data();
//                    signed char* outptr1 = (signed char*)dst_a[q * 4 + 1].data();
//                    signed char* outptr2 = (signed char*)dst_a[q * 4 + 2].data();
//                    signed char* outptr3 = (signed char*)dst_a[q * 4 + 3].data();
//
//                    const float s0 = scale_data_a[q * 4];
//                    const float s1 = scale_data_a[q * 4 + 1];
//                    const float s2 = scale_data_a[q * 4 + 2];
//                    const float s3 = scale_data_a[q * 4 + 3];
//
//                    for (int i = 0; i < size; i++) {
//                        outptr0[0] = float2int8(ptr0[0] * s0);
//                        outptr1[0] = float2int8(ptr0[1] * s1);
//                        outptr2[0] = float2int8(ptr0[2] * s2);
//                        outptr3[0] = float2int8(ptr0[3] * s3);
//
//                        ptr0 += 4;
//                        outptr0 += 1;
//                        outptr1 += 1;
//                        outptr2 += 1;
//                        outptr3 += 1;
//                    }
//                }
//            });
//        }
    }
    
    return dst;
}

}   // end namespace otter
