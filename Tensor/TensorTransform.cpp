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

Tensor& crop_(const Tensor& input, IntArrayRef border, Tensor& output) {
    OTTER_CHECK(input.dim() <= 4, "Expect input dim <= 4 but get", input.dim());
    
    int elempack = input.elempack();
    
    if (elempack != 1) {
#if __SSE2__
        return crop_(input.packing(1), border, output);
#else
        return crop_(input.packing(1), border, output);
#endif
    }
    
    auto output_shape = resolve_roi(input.shape(), border);
    output = otter::empty(output_shape, input.options());
    
    if (input.dim() == 1) {
        if (output_shape[0] == input.size(0)) {
            output.copy_(input);
            
            return output;
        }
        cut_border(input.contiguous(), output, 0, border[0]);
    } else if (input.dim() == 2) {
        if (output_shape[0] == input.size(0) && output_shape[1] == input.size(1)) {
            output.copy_(input);
            
            return output;
        }
        cut_border(input.contiguous(), output, border[2], border[0]);
    } else if (input.dim() == 3) {
        if (output_shape[0] == input.size(0) && output_shape[1] == input.size(1) && output_shape[2] == input.size(2)) {
            output.copy_(input);
            
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
            output.copy_(input);
            
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

}   // end namespace otter
