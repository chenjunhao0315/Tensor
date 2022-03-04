//
//  im2col.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/24.
//

#include "im2col.hpp"
#include "Tensor.hpp"
#include "TensorFactory.hpp"
#include "Dispatch.hpp"
#include "Parallel.hpp"
#include "Unfold2D.hpp"

namespace otter {

void im2col_out_cpu_template(Tensor& output, const Tensor& input_, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
    OTTER_CHECK(kernel_size.size() == 2, "Expect kernel_size is 2, but get", kernel_size.size());
    OTTER_CHECK(stride.size() == 2, "Expect stride is 2, but get", stride.size());
    OTTER_CHECK(padding.size() == 2, "Expect padding is 2, but get", padding.size());
    OTTER_CHECK(dilation.size() == 2, "Expect dilation is 2, but get", dilation.size());
    
    int64_t kernel_height = kernel_size[0];
    int64_t kernel_width = kernel_size[1];
    int64_t stride_height = stride[0];
    int64_t stride_width = stride[1];
    int64_t pad_height = padding[0];
    int64_t pad_width = padding[1];
    int64_t dilation_height = dilation[0];
    int64_t dilation_width = dilation[1];
    
    im2col_shape_check(
                       input_,
                       Tensor(),
                       kernel_height, kernel_width,
                       dilation_height, dilation_width,
                       pad_height, pad_width,
                       stride_height, stride_width);
    
    Tensor input = input_.contiguous();

    bool batched_input = true;

    if (input.dim() == 3) {
        batched_input = false;
        input = input.view({1, input.size(0), input.size(1), input.size(2)});
    }
    
    int64_t batch_size = input.size(0);
    int64_t n_input_plane = input.size(1);
    int64_t input_height = input.size(2);
    int64_t input_width = input.size(3);

    int64_t output_height = (input_height + 2 * pad_height - (dilation_height * (kernel_height - 1) + 1)) / stride_height + 1;
    int64_t output_width = (input_width + 2 * pad_width - (dilation_width * (kernel_width - 1) + 1)) / stride_width + 1;
    int64_t n_output_plane = n_input_plane * kernel_width * kernel_height;
    int64_t output_length = output_height * output_width;

    output.resize_({batch_size, n_output_plane, output_length});
    output.zero_();
    
    OTTER_DISPATCH_ALL_TYPES(input.scalar_type(), "im2col_cpu_out", [&] {
        Tensor input_n;
        Tensor output_n;
        
        for (const auto elt : otter::irange(batch_size)) {
            input_n = input.select(0, elt);
            output_n = output.select(0, elt);

            im2col<scalar_t>(
                             input_n.data_ptr<scalar_t>(),
                             n_input_plane,
                             input_height,
                             input_width,
                             output_height,
                             output_width,
                             kernel_height,
                             kernel_width,
                             pad_height,
                             pad_width,
                             stride_height,
                             stride_width,
                             dilation_height,
                             dilation_width,
                             output_n.data_ptr<scalar_t>());
        }

        if (!batched_input) {
            output.resize_({n_output_plane, output_length});
        }
    });
}

Tensor& im2col_out_cpu_unfold2d_template(const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor& output) {
    int64_t batch_size = input.size(0);
    int64_t n_input_planes = input.size(1);
    int64_t input_height = input.size(2);
    int64_t input_width = input.size(3);
    int64_t kernel_height = kernel_size[0];
    int64_t kernel_width = kernel_size[1];
    int64_t stride_height = stride[0];
    int64_t stride_width = stride[1];
    int64_t pad_height = padding[0];
    int64_t pad_width = padding[1];
    int64_t dilation_height = dilation[0];
    int64_t dilation_width = dilation[1];
    int64_t output_height = (input_height + 2 * pad_height - (dilation_height * (kernel_height - 1) + 1)) / stride_height + 1;
    int64_t output_width = (input_width + 2 * pad_width - (dilation_width * (kernel_width - 1) + 1)) / stride_width + 1;
    
    output = otter::empty({batch_size, n_input_planes * kernel_height * kernel_width, output_height * output_width}, input.options());
    OTTER_DISPATCH_ALL_TYPES(input.scalar_type(), "im2col_cpu", [&] {
        auto input_a   = input.accessor<scalar_t, 4>();
        auto output_a = output.accessor<scalar_t, 3>();
        
        otter::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
            for (const auto t : otter::irange(start, end)) {
                auto input_t   = input_a[t];
                auto columns_t = output_a[t];
                
                unfold2d_copy_stub(
                    Device::CPU,
                    otter::CppTypeToScalarType<scalar_t>::value,
                    columns_t.data(),
                    input_t.data(),
                    kernel_height, kernel_width,
                    stride_height, stride_width,
                    pad_height, pad_width,
                    n_input_planes, input_height, input_width,
                    output_height, output_width);
            }
        });
    });
    return output;
}

Tensor& im2col_out_cpu(const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, Tensor& output) {
    
//    im2col_out_cpu_template(output, input, kernel_size, stride, padding, dilation);
    
    im2col_out_cpu_unfold2d_template(input, kernel_size, stride, padding, dilation, output);
    
    return output;
}

Tensor im2col_cpu(const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
//    Tensor output = otter::empty_like(input);
//    im2col_out_cpu_template(output, input, kernel_size, stride, padding, dilation);
    
    Tensor output;
    im2col_out_cpu_unfold2d_template(input, kernel_size, stride, padding, dilation, output);
    
    return output;
}

}   // end namespace otter
