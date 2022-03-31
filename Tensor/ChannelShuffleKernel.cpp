//
//  ChannelShuffleKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/31.
//

#include "ChannelShuffle.hpp"
#include "ChannelShuffleKernel.hpp"
#include "Tensor.hpp"
#include "Vec.hpp"
#include "Parallel.hpp"
#include "Utils.hpp"
#include "Dispatch.hpp"

namespace otter {

template <typename scalar_t>
void cpu_channel_shuffle(Tensor& output, const Tensor& input, int64_t groups) {
    auto input_data = input.data_ptr<scalar_t>();
    auto output_data = output.data_ptr<scalar_t>();
    
    int64_t nbatch = input.size(0);
    int64_t channels = input.size(1);
    int64_t channels_per_group = channels / groups;
    int64_t image_size = input.numel() / nbatch / channels;
    
    // treat input tensor as shape of [n, g, oc, ...]
    // output tensor as shape of [n, oc, g, ...]
    //
    // 3d, 4d, 5d: parallel on dimension of n, c
    using Vec = vec::Vectorized<scalar_t>;
    int64_t inner_size = image_size - (image_size % Vec::size());
    otter::parallel_for (0, nbatch * /* oc*g */channels, 0, [&](int64_t begin, int64_t end) {
        int64_t n = 0;
        int64_t oc = 0;
        int64_t g = 0;
        data_index_init(begin, n, nbatch, oc, channels_per_group, g, groups);
        
        for (const auto i : otter::irange(begin, end)) {
            scalar_t* output_ptr = output_data + i * image_size;
            scalar_t* input_ptr = input_data + n * channels * image_size + g * channels_per_group * image_size + oc * image_size;
            
            int64_t d = 0;
            for (; d < inner_size; d += Vec::size()) {
                Vec data_vec = Vec::loadu(input_ptr + d);
                data_vec.store(output_ptr + d);
            }
            for (; d < image_size; d++) {
                output_ptr[d] = input_ptr[d];
            }
            
            // move on to next output index
            data_index_step(n, nbatch, oc, channels_per_group, g, groups);
        }
    });
}

template <typename scalar_t>
void cpu_channel_shuffle_cl(Tensor& output, const Tensor& input, int64_t groups) {
    auto input_data = input.data_ptr<scalar_t>();
    auto output_data = output.data_ptr<scalar_t>();
    
    int64_t nbatch = input.size(0);
    int64_t channels = input.size(1);
    int64_t channels_per_group = channels / groups;
    int64_t image_size = input.numel() / nbatch / channels;
    
    // 4d: parallel on dimension of n, h, w
    // 5d: parallel on dimension of n, d, h, w
    otter::parallel_for(0, nbatch * image_size, 0, [&](int64_t begin, int64_t end) {
        for (const auto i : otter::irange(begin, end)) {
            scalar_t* output_ptr = output_data + i * channels;
            scalar_t* input_ptr = input_data + i * channels;
            
            // transpose each channel lane:
            // from [groups, channels_per_group] to [channels_per_group, groups]
            otter::utils::transpose(groups, channels_per_group, input_ptr, channels_per_group, output_ptr, groups);
        }
    });
}

void channel_shuffle_kernel_impl(Tensor& output, const Tensor& input, int64_t groups) {
    switch (input.suggest_memory_format()) {
        case MemoryFormat::Contiguous: {
            OTTER_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, input.scalar_type(), "channel_shuffle", [&] {
                cpu_channel_shuffle<scalar_t>(output, input, groups);
            });
            break;
        }
        case MemoryFormat::ChannelsLast:
        case MemoryFormat::ChannelsLast3d: {
            OTTER_DISPATCH_ALL_TYPES_AND(ScalarType::Bool, input.scalar_type(), "channel_shuffle_cl", [&] {
                cpu_channel_shuffle_cl<scalar_t>(output, input, groups);
            });
            break;
        }
        default:
            OTTER_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, ChannelsLast3d, Contiguous");
    }
}

REGISTER_DISPATCH(channel_shuffle_stub, &channel_shuffle_kernel_impl);

}   // end namespace otter
