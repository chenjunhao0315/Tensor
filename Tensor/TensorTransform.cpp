//
//  TensorTransform.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/4.
//

#include "TensorTransform.hpp"
#include "TensorFactory.hpp"
#include "Dispatch.hpp"
#include "Parallel.hpp"
#include "TensorPacking.hpp"
#include "VecIntrinsic.hpp"
#include "TensorShape.hpp"
#include "Formatting.hpp"

namespace otter {

static void cut_border(const Tensor& src_, Tensor &dst_, int64_t top, int64_t left) {
    auto src = (src_.dim() == 1) ? src_.unsqueeze(0) : src_;
    auto dst = (dst_.dim() == 1) ? dst_.unsqueeze(0) : dst_;
    
    int64_t output_height = dst.size(0);
    int64_t output_width  = dst.size(1);
    int64_t input_width   = src.size(1);
    
    OTTER_DISPATCH_ALL_TYPES(src.scalar_type(), "cut_border", [&] {
        auto src_a = src.accessor<scalar_t, 2>();
        
        const scalar_t* src_ptr = src_a[top].data() + left;
        scalar_t* dst_ptr = dst.data_ptr<scalar_t>();
        
        for (const auto y : otter::irange(0, output_height)) {
            (void)y;
            if (output_width < 12) {
                for (const auto x : otter::irange(0, output_width)) {
                    dst_ptr[x] = src_ptr[x];
                }
            } else {
                memcpy(dst_ptr, src_ptr, sizeof(scalar_t) * output_width);
            }
            dst_ptr += output_width;
            src_ptr += input_width;
        }
    });
}

std::vector<int64_t> resolve_roi(std::vector<int64_t> shape, IntArrayRef border) {
    if (shape.size() == 1) {
        int64_t input_width = shape[0];
        
        OTTER_CHECK(border.size() >= 2, "ROI need at least 2 parameters but get", border);
        
        int64_t left  = border[0];
        int64_t right = border[1];
        
        int64_t output_width = input_width - left - right;
        
        return { output_width };
    } else if (shape.size() == 2) {
        int64_t input_height = shape[0];
        int64_t input_width  = shape[1];
        
        OTTER_CHECK(border.size() >= 4, "ROI need at least 4 parameters but get", border);
        
        int64_t left   = border[0];
        int64_t right  = border[1];
        int64_t top    = border[2];
        int64_t bottom = border[3];
        
        int64_t output_height = input_height - top - bottom;
        int64_t output_width  = input_width - left - right;
        
        return { output_height, output_width };
    } else if (shape.size() == 3) {
        int64_t input_channels = shape[0];
        int64_t input_height   = shape[1];
        int64_t input_width    = shape[2];
        
        OTTER_CHECK(border.size() >= 4, "ROI need at least 4 parameters but get", border);
        
        int64_t left   = border[0];
        int64_t right  = border[1];
        int64_t top    = border[2];
        int64_t bottom = border[3];
        
        int64_t channel_front = 0;
        int64_t channel_rear  = 0;
        
        if (border.size() > 4) {
            OTTER_CHECK(border.size() >= 6, "ROI need at least 6 parameters but get", border);
            channel_front = border[4];
            channel_rear  = border[5];
        }
        
        int64_t output_channels = input_channels - channel_front - channel_rear;
        int64_t output_height   = input_height - top - bottom;
        int64_t output_width    = input_width - left - right;
        
        return { output_channels, output_height, output_width };
    } else if (shape.size() == 4) {
        int64_t input_batch    = shape[0];
        int64_t input_channels = shape[1];
        int64_t input_height   = shape[2];
        int64_t input_width    = shape[3];
        
        OTTER_CHECK(border.size() >= 4, "ROI need at least 4 parameters but get", border);
        
        int64_t left   = border[0];
        int64_t right  = border[1];
        int64_t top    = border[2];
        int64_t bottom = border[3];
        
        int64_t channel_front = 0;
        int64_t channel_rear  = 0;
        int64_t batch_front   = 0;
        int64_t batch_rear    = 0;
        
        if (border.size() > 4) {
            OTTER_CHECK(border.size() >= 6, "ROI need at least 6 parameters but get", border);
            channel_front = border[4];
            channel_rear  = border[5];
        }
        if (border.size() > 6) {
            OTTER_CHECK(border.size() >= 8, "ROI need at least 6 parameters but get", border);
            batch_front = border[6];
            batch_rear  = border[7];
        }

        int64_t output_batch    = input_batch - batch_front - batch_rear;
        int64_t output_channels = input_channels - channel_front - channel_rear;
        int64_t output_height   = input_height - top - bottom;
        int64_t output_width    = input_width - left - right;
        
        return { output_batch, output_channels, output_height, output_width };
    }
    
    OTTER_CHECK(false, "Unsupport roi resolve");
    return {};
}

#if __SSE2__
Tensor& crop_x86_(const Tensor& input, IntArrayRef border, Tensor& output);
#elif __ARM_NEON__
Tensor& crop_neon_(const Tensor& input, IntArrayRef border, Tensor& output);
#endif

Tensor& crop_(const Tensor& input, IntArrayRef border, Tensor& output) {
    OTTER_CHECK(input.dim() <= 4, "Expect input dim <= 4 but get", input.dim());
    
    int elempack = input.elempack();
    ScalarType dtype = input.scalar_type();
    
    if (elempack != 1) {
        if (dtype == ScalarType::Float || dtype == ScalarType::Float4 || dtype == ScalarType::Float8) {
#if __SSE2__
            return crop_x86_(input, border, output);
#elif __ARM_NEON__
            return crop_neon_(input, border, output);
#endif
        } else {
            return crop_(input.packing(1), border, output);
        }
    }
    
    auto output_shape = resolve_roi(input.shape(), border);
    output = otter::empty(output_shape, input.options());
    
    if (input.dim() == 1) {
        if (output_shape[0] == input.size(0)) {
            output = input;
            
            return output;
        }
        cut_border(input.contiguous(), output, 0, border[0]);
    } else if (input.dim() == 2) {
        if (output_shape[0] == input.size(0) && output_shape[1] == input.size(1)) {
            output = input;
            
            return output;
        }
        cut_border(input.contiguous(), output, border[2], border[0]);
    } else if (input.dim() == 3) {
        if (output_shape[0] == input.size(0) && output_shape[1] == input.size(1) && output_shape[2] == input.size(2)) {
            output = input;
            
            return output;
        }
        
        int64_t channel_front = (output_shape[0] != input.size(0)) ? border[4] : 0;
        
        otter::parallel_for(channel_front, channel_front + output_shape[0], 0, [&](int64_t begin, int64_t end) {
            for (const auto c : otter::irange(begin, end)) {
                const auto input_c = input[c];
                auto output_c = output[c - channel_front];
                
                cut_border(input_c, output_c, border[2], border[0]);
            }
        });
    } else if (input.dim() == 4) {
        if (output_shape[0] == input.size(0) && output_shape[1] == input.size(1) && output_shape[2] == input.size(2) && output_shape[3] == input.size(3)) {
            output = input;
            
            return output;
        }
        
        int64_t batch_front   = (output_shape[0] != input.size(0)) ? border[6] : 0;
        int64_t channel_front = (output_shape[1] != input.size(1)) ? border[4] : 0;
        
        otter::parallel_for(batch_front, batch_front + output_shape[0], 0, [&](int64_t begin, int64_t end) {
            for (const auto b : otter::irange(begin, end)) {
                for (const auto c : otter::irange(channel_front, channel_front + output_shape[1])) {
                    const auto input_c = input[b][c];
                    auto output_c = output[b - batch_front][c - channel_front];
                    
                    cut_border(input_c, output_c, border[2], border[0]);
                }
            }
        });
    }
    
    return output;
}

Tensor crop(const Tensor& input, IntArrayRef border) {
    Tensor output;
    
    return crop_(input, border, output);
}

#if __SSE2__
static void crop_pack4_sse(const Tensor& src_, Tensor& dst_, int top, int left) {
    auto src = (src_.dim() == 1) ? src_.unsqueeze(0) : src_;
    auto dst = (dst_.dim() == 1) ? dst_.unsqueeze(0) : dst_;
    
    int w = dst.size(1);
    int h = dst.size(0);
    int right = src.size(1) - dst.size(1) - left;

    const float* ptr = (const float*)src[top].data_ptr() + left * 4;
    float* outptr = (float*)dst.data_ptr();

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            __m128 _p = _mm_loadu_ps(ptr);
            _mm_storeu_ps(outptr, _p);
            ptr += 4;
            outptr += 4;
        }

        ptr += (left + right) * 4;
    }
}

Tensor& crop_x86_(const Tensor& input, IntArrayRef border, Tensor& output) {
    int elempack = input.elempack();
    int dims = input.dim();
    ScalarType dtype = input.scalar_type();
    
    auto output_shape = resolve_roi(input.shape(), border);
    
    if (elempack == 4) {
        if (dims == 1) {
            int w = input.size(0);
            int outw = output_shape[0];
            int woffset = border[0];
            
            int out_elempack = outw % 4 == 0 ? 4 : 1;
            
            if (outw / out_elempack == w && out_elempack == 4) {
                output = input;
                
                return output;
            }
            
            if (woffset % 4 == 0 && out_elempack == 4) {
                output = otter::empty({outw / out_elempack}, otter::get_update_scalarType(dtype, out_elempack));

                crop_pack4_sse(input, output, 0, woffset / elempack);

                return output;
            }
        } else if (dims == 2) {
            int h = input.size(0);
            int w = input.size(1);
            int outh = output_shape[0];
            int outw = output_shape[1];
            int hoffset = border[2];
            int woffset = border[0];
            
            int out_elempack = outw % 4 == 0 ? 4 : 1;
            
            if (outw == w && outh / out_elempack == h && out_elempack == 4) {
                output = input;
                
                return output;
            }
            
            if (hoffset % 4 == 0 && out_elempack == 4) {
                output = otter::empty({outh / out_elempack, outw}, otter::get_update_scalarType(dtype, out_elempack));

                crop_pack4_sse(input, output, hoffset / elempack, woffset);

                return output;
            }
        } else if (dims == 3) {
            int channels = input.size(0);
            int h = input.size(1);
            int w = input.size(2);
            int outc = output_shape[0];
            int outh = output_shape[1];
            int outw = output_shape[2];
            
            int woffset = border[0];
            int hoffset = border[2];
            int coffset = (border.size() > 4) ? border[4] : 0;
            
            int out_elempack = outc % 4 == 0 ? 4 : 1;

            if (outw == w && outh == h && outc / out_elempack == channels && out_elempack == 4) {
                output = input;
                
                return output;
            }

            if (coffset % 4 == 0 && out_elempack == 4) {
                const Tensor bottom_blob_sliced = otter::native::slice(input, 0, coffset, coffset + outc, 1);
                
                if (outw == w && outh == h) {
                    output = bottom_blob_sliced.clone();
                }
                
                output = otter::empty({outc / out_elempack, outh, outw}, otter::get_update_scalarType(dtype, out_elempack));

                otter::parallel_for(0, outc / out_elempack, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const auto m = bottom_blob_sliced[q];
                        auto borderm = output[q];

                        crop_pack4_sse(m, borderm, hoffset, woffset);
                    }
                });

                return output;
            }
        } else if (dims == 4) {
            int b = input.size(0);
            int channels = input.size(1);
            int h = input.size(2);
            int w = input.size(3);
            int outb = output_shape[0];
            int outc = output_shape[1];
            int outh = output_shape[2];
            int outw = output_shape[3];
            
            int woffset = border[0];
            int hoffset = border[2];
            int coffset = (border.size() > 4) ? border[4] : 0;
            int boffset = (border.size() > 6) ? border[6] : 0;
            
            int out_elempack = outc % 4 == 0 ? 4 : 1;

            if (outw == w && outh == h && outb == b && outc / out_elempack == channels && out_elempack == 4) {
                output = input;
                
                return output;
            }
            
            output = otter::empty({outb, outc / out_elempack, outh, outw}, otter::get_update_scalarType(dtype, out_elempack));

            if (coffset % 4 == 0 && out_elempack == 4) {
                for (const auto q : otter::irange(boffset, boffset + outb)) {
                    const auto bottom_blob_sliced = otter::native::slice(input[q], 0, coffset, coffset + outc, 1);
                    auto output_sliced = output[q];
                    otter::parallel_for(0, outc / out_elempack, 0, [&](int64_t begin, int64_t end) {
                        for (const auto c : otter::irange(begin, end)) {
                            const auto m = bottom_blob_sliced[c];
                            auto borderm = output_sliced[c];

                            crop_pack4_sse(m, borderm, hoffset, woffset);
                        }
                    });
                }

                return output;
            }
        }
    }
    
    return crop_(input.packing(1), border, output);
}
#elif __ARM_NEON__
static void crop_pack4_neon(const Tensor& src_, Tensor& dst_, int top, int left) {
    auto src = (src_.dim() == 1) ? src_.unsqueeze(0) : src_;
    auto dst = (dst_.dim() == 1) ? dst_.unsqueeze(0) : dst_;
    
    int w = dst.size(1);
    int h = dst.size(0);
    int right = src.size(1) - dst.size(1) - left;

    const float* ptr = (const float*)src[top].data_ptr() + left * 4;
    float* outptr = (float*)dst.data_ptr();

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float32x4_t _p = vld1q_f32(ptr);
            vst1q_f32(outptr, _p);
            ptr += 4;
            outptr += 4;
        }

        ptr += (left + right) * 4;
    }
}

Tensor& crop_neon_(const Tensor& input, IntArrayRef border, Tensor& output) {
    int elempack = input.elempack();
    int dims = input.dim();
    ScalarType dtype = input.scalar_type();
    
    auto output_shape = resolve_roi(input.shape(), border);
    
    if (elempack == 4) {
        if (dims == 1) {
            int w = input.size(0);
            int outw = output_shape[0];
            int woffset = border[0];
            
            int out_elempack = outw % 4 == 0 ? 4 : 1;
            
            if (outw / out_elempack == w && out_elempack == 4) {
                output = input;
                
                return output;
            }
            
            if (woffset % 4 == 0 && out_elempack == 4) {
                output = otter::empty({outw / out_elempack}, otter::get_update_scalarType(dtype, out_elempack));

                crop_pack4_neon(input, output, 0, woffset / elempack);

                return output;
            }
        } else if (dims == 2) {
            int h = input.size(0);
            int w = input.size(1);
            int outh = output_shape[0];
            int outw = output_shape[1];
            int hoffset = border[2];
            int woffset = border[0];
            
            int out_elempack = outw % 4 == 0 ? 4 : 1;
            
            if (outw == w && outh / out_elempack == h && out_elempack == 4) {
                output = input;
                
                return output;
            }
            
            if (hoffset % 4 == 0 && out_elempack == 4) {
                output = otter::empty({outh / out_elempack, outw}, otter::get_update_scalarType(dtype, out_elempack));

                crop_pack4_neon(input, output, hoffset / elempack, woffset);

                return output;
            }
        } else if (dims == 3) {
            int channels = input.size(0);
            int h = input.size(1);
            int w = input.size(2);
            int outc = output_shape[0];
            int outh = output_shape[1];
            int outw = output_shape[2];
            
            int woffset = border[0];
            int hoffset = border[2];
            int coffset = (border.size() > 4) ? border[4] : 0;
            
            int out_elempack = outc % 4 == 0 ? 4 : 1;

            if (outw == w && outh == h && outc / out_elempack == channels && out_elempack == 4) {
                output = input;
                
                return output;
            }

            if (coffset % 4 == 0 && out_elempack == 4) {
                const Tensor bottom_blob_sliced = otter::native::slice(input, 0, coffset, coffset + outc, 1);
                
                if (outw == w && outh == h) {
                    output = bottom_blob_sliced.clone();
                }
                
                output = otter::empty({outc / out_elempack, outh, outw}, otter::get_update_scalarType(dtype, out_elempack));

                otter::parallel_for(0, outc / out_elempack, 0, [&](int64_t begin, int64_t end) {
                    for (const auto q : otter::irange(begin, end)) {
                        const auto m = bottom_blob_sliced[q];
                        auto borderm = output[q];

                        crop_pack4_neon(m, borderm, hoffset, woffset);
                    }
                });

                return output;
            }
        } else if (dims == 4) {
            int b = input.size(0);
            int channels = input.size(1);
            int h = input.size(2);
            int w = input.size(3);
            int outb = output_shape[0];
            int outc = output_shape[1];
            int outh = output_shape[2];
            int outw = output_shape[3];
            
            int woffset = border[0];
            int hoffset = border[2];
            int coffset = (border.size() > 4) ? border[4] : 0;
            int boffset = (border.size() > 6) ? border[6] : 0;
            
            int out_elempack = outc % 4 == 0 ? 4 : 1;

            if (outw == w && outh == h && outb == b && outc / out_elempack == channels && out_elempack == 4) {
                output = input;
                
                return output;
            }
            
            output = otter::empty({outb, outc / out_elempack, outh, outw}, otter::get_update_scalarType(dtype, out_elempack));

            if (coffset % 4 == 0 && out_elempack == 4) {
                for (const auto q : otter::irange(boffset, boffset + outb)) {
                    const auto bottom_blob_sliced = otter::native::slice(input[q], 0, coffset, coffset + outc, 1);
                    auto output_sliced = output[q];
                    otter::parallel_for(0, outc / out_elempack, 0, [&](int64_t begin, int64_t end) {
                        for (const auto c : otter::irange(begin, end)) {
                            const auto m = bottom_blob_sliced[c];
                            auto borderm = output_sliced[c];

                            crop_pack4_neon(m, borderm, hoffset, woffset);
                        }
                    });
                }

                return output;
            }
        }
    }
    
    return crop_(input.packing(1), border, output);
}
#endif

}   // end namespace otter
