//
//  MaxPoolKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/2.
//

#include "Pool.hpp"
#include "MaxPoolKernel.hpp"
#include "Tensor.hpp"
#include "Dispatch.hpp"
#include "Parallel.hpp"
#include "Utils.hpp"
#include "Vec.hpp"

namespace otter {

template <typename scalar_t, typename accscalar_t>
void cpu_max_pool_impl(
    const Tensor& output_,
    const Tensor& indices_,
    const Tensor& input_,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH) {
    
    auto input = input_.contiguous();
    auto output = output_.contiguous();
    auto indices = indices_.contiguous();
    
    auto input_data = input.data_ptr<scalar_t>();
    auto output_data = output.data_ptr<scalar_t>();
    auto indices_data = indices.data_ptr<int64_t>();
    
    int64_t numel = output.numel();
    int64_t ndim = input.dim();
    // treat batch size and channels as one dimension
    int64_t channels = ndim == 3 ? input.size(0) : input.size(0) * input.size(1);
    int64_t input_height = input.size(-2);
    int64_t input_width = input.size(-1);
    int64_t output_height = output.size(-2);
    int64_t output_width = output.size(-1);
    
    otter::parallel_for(0, numel, 0, [&](int64_t begin, int64_t end) {
        int64_t c = 0;
        int64_t oh = 0;
        int64_t ow = 0;
        data_index_init(begin, c, channels, oh, output_height, ow, output_width);
        
        for (const auto i : otter::irange(begin, end)) {
            int64_t ih0 = oh * dH - padH;
            int64_t iw0 = ow * dW - padW;
            int64_t ih1 = std::min(ih0 + (kH - 1) * dilationH + 1, input_height);
            int64_t iw1 = std::min(iw0 + (kW - 1) * dilationW + 1, input_width);
            while(ih0 < 0) { ih0 += dilationH; }
            while(iw0 < 0) { iw0 += dilationW; }

            // local pointers
            scalar_t* input_ptr = input_data + c * input_height * input_width;

            // compute local max
            int64_t maxindex = ih0 * input_width + iw0;
            accscalar_t maxval = -std::numeric_limits<accscalar_t>::infinity();
            for (int64_t ih = ih0; ih < ih1; ih += dilationH) {
                for (int64_t iw = iw0; iw < iw1; iw += dilationW) {
                    int64_t index = ih * input_width + iw;
                    accscalar_t val = accscalar_t(input_ptr[index]);
                    if ((val > maxval) || std::isnan(val)) {
                        maxval = val;
                        maxindex = index;
                    }
                }
            }

            // set output to local max and store location of max
            output_data[i] = scalar_t(maxval);
            indices_data[i] = maxindex;

            // move on to next output index
            data_index_step(c, channels, oh, output_height, ow, output_width);
        }
    });

    if (!output_.is_contiguous()) {
        output_.copy_(output);
    }
    if (!indices_.is_contiguous()) {
        indices_.copy_(indices);
    }
} 

template <typename scalar_t>
void cpu_max_pool_channels_last_impl(
    const Tensor& output_,
    const Tensor& indices_,
    const Tensor& input_,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH) {
    
    OTTER_CHECK(input_.dim() == 4, "max pooling with channels last format supports tensors with 4 dims");
    auto memory_format = MemoryFormat::ChannelsLast;
    auto input = input_.contiguous(memory_format);
    auto output = output_.contiguous(memory_format);
    auto indices = indices_.contiguous(memory_format);

    auto input_data = input.data_ptr<scalar_t>();
    auto output_data = output.data_ptr<scalar_t>();
    auto indices_data = indices.data_ptr<int64_t>();

    int64_t nbatch = input.size(0);
    int64_t channels = input.size(1);
    int64_t input_height = input.size(2);
    int64_t input_width = input.size(3);
    int64_t output_height = output.size(2);
    int64_t output_width = output.size(3);

    using Vec = vec::Vectorized<scalar_t>;
    using integer_t = vec::int_same_size_t<scalar_t>;
    using iVec = vec::Vectorized<integer_t>;
    OTTER_INTERNAL_ASSERT(input_height * input_width <= std::numeric_limits<integer_t>::max());

    // parallel on dim N, H, W
    otter::parallel_for(0, nbatch * output_height * output_width, 0, [&](int64_t begin, int64_t end) {
        int64_t n = 0;
        int64_t oh = 0;
        int64_t ow = 0;
        data_index_init(begin, n, nbatch, oh, output_height, ow, output_width);

        int64_t size = channels;
        int64_t len = size - (size % Vec::size());
        
        std::unique_ptr<integer_t []> index_buffer(new integer_t[len]);

        for (const auto i : otter::irange(begin, end)) {
            int64_t ih0 = oh * dH - padH;
            int64_t iw0 = ow * dW - padW;
            int64_t ih1 = std::min(ih0 + (kH - 1) * dilationH + 1, input_height);
            int64_t iw1 = std::min(iw0 + (kW - 1) * dilationW + 1, input_width);
            while(ih0 < 0) { ih0 += dilationH; }
            while(iw0 < 0) { iw0 += dilationW; }

            scalar_t* out = output_data + i * channels;
            int64_t* ind = indices_data + i * channels;

            // Pass I: init out lane
            iVec index0_vec = iVec(static_cast<int>(ih0 * input_width + iw0));
            Vec out_vec = Vec(-std::numeric_limits<scalar_t>::infinity());
            int64_t d1 = 0;
            for (; d1 < len; d1 += Vec::size()) {
                index0_vec.store(index_buffer.get() + d1);
                out_vec.store(out + d1);
            }
            for (; d1 < size; d1++) {
                ind[d1] = ih0 * input_width + iw0;
                out[d1] = -std::numeric_limits<scalar_t>::infinity();
            }
            // Pass II: compute local max
            for (int64_t ih = ih0; ih < ih1; ih += dilationH) {
                for (int64_t iw = iw0; iw < iw1; iw += dilationW) {
                    scalar_t* in = input_data + n * input_height * input_width * channels + ih * input_width * channels + iw * channels;

                    int64_t d2 = 0;
                    for (; d2 < len; d2 += Vec::size()) {
                        iVec index_vec = iVec(static_cast<int>(ih * input_width + iw));
                        Vec val_vec = Vec::loadu(in + d2);
                        iVec maxindex_vec = iVec::loadu(index_buffer.get() + d2);
                        Vec maxval_vec = Vec::loadu(out + d2);

                        // true = all ones, false = all zeros
                        Vec mask = (val_vec > maxval_vec) | val_vec.isnan();
                        iVec imask = vec::cast<integer_t>(mask);
                        Vec out_vec = Vec::blendv(maxval_vec, val_vec, mask);
                        iVec ind_vec = iVec::blendv(maxindex_vec, index_vec, imask);

                        out_vec.store(out + d2);
                        ind_vec.store(index_buffer.get() + d2);
                    }
                    for (; d2 < size; d2++) {
                        int64_t index = ih * input_width + iw;
                        scalar_t val = in[d2];
                        int64_t maxindex = ind[d2];
                        scalar_t maxval = out[d2];

                        bool mask = (val > maxval) || std::isnan(val);
                        out[d2] = mask ? val : maxval;
                        ind[d2] = mask ? index : maxindex;
                    }
                }
            }
            // convert indice data type
            vec::convert<integer_t, int64_t>(index_buffer.get(), ind, len);

            // move on to next output index
            data_index_step(n, nbatch, oh, output_height, ow, output_width);
        }
    });

    if (!output_.is_contiguous(memory_format)) {
        output_.copy_(output);
    }
    if (!indices_.is_contiguous(memory_format)) {
        indices_.copy_(indices);
    }
}

void max_pool2d_kernel(
    const Tensor& output,
    const Tensor& indices,
    const Tensor& input,
    int kW, int kH,
    int dW, int dH,
    int padW, int padH,
    int dilationW, int dilationH) {
    
    switch (input.suggest_memory_format()) {
        case MemoryFormat::Contiguous:
            OTTER_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cpu", [&] {
                cpu_max_pool_impl<scalar_t, scalar_t>(output, indices, input, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
            });
            break;
        case MemoryFormat::ChannelsLast:
            OTTER_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_cpu", [&] {
                cpu_max_pool_channels_last_impl<scalar_t>(output, indices, input, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
            });
            break;
        default:
            break;
    }
}

REGISTER_DISPATCH(max_pool2d_stub, &max_pool2d_kernel);

}   // end namespace otter
