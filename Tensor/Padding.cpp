//
//  Padding.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/2.
//

#include "Padding.hpp"
#include "Tensor.hpp"
#include "TensorFactory.hpp"

namespace otter {

Tensor constant_pad(const Tensor& self, IntArrayRef pad, const Scalar& value) {
    OTTER_CHECK(pad.size() % 2 == 0, "Length of pad must be even but instead it equals ", pad.size());
    
    auto input_sizes = self.sizes();
    auto l_inp = self.dim();

    auto l_pad = pad.size() / 2;
    auto l_diff = l_inp - l_pad;
    OTTER_CHECK(l_inp >= (int64_t)l_pad, "Length of pad should be no more than twice the number of "
                "dimensions of the input. Pad length is ", pad.size(), "while the input has ",
                l_inp, "dimensions.");

    std::vector<int64_t> new_shape;
    
    bool all_pads_non_positive = true;

    auto c_input = self;
    for (const auto i : otter::irange(l_diff, l_inp)) {
        auto pad_idx = 2 * (l_inp - i - 1);
        if (pad[pad_idx] < 0) {
            c_input = c_input.narrow(i, -pad[pad_idx], c_input.size(i) + pad[pad_idx]);
        } else if (pad[pad_idx] != 0) {
            all_pads_non_positive = false;
        }
        if (pad[pad_idx + 1] < 0) {
            c_input = c_input.narrow(i, 0, c_input.size(i) + pad[pad_idx + 1]);
        } else if (pad[pad_idx + 1] != 0) {
            all_pads_non_positive = false;
        }
    }

    // if none of the pads are positive we can optimize and just return the result
    // of calling .narrow() on the input
    if (all_pads_non_positive) {
        return c_input.clone();
    }
    
    for (size_t i = 0; i < (size_t)l_diff; i ++) {
        new_shape.emplace_back(input_sizes[i]);
    }

    for (const auto i : otter::irange((size_t)l_pad)) {
        auto pad_idx = pad.size() - ((i + 1) * 2);
        auto new_dim = input_sizes[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1];
        OTTER_CHECK(new_dim > 0, "The input size ", input_sizes[l_diff + i], ", plus negative padding ",
                    pad[pad_idx], " and ", pad[pad_idx + 1], " resulted in a negative output size, "
                    "which is invalid. Check dimension ", l_diff + i, " of your input.");
        new_shape.emplace_back(new_dim);
    }

    Tensor output;
    const auto memory_format = self.suggest_memory_format();
    output = otter::empty(new_shape, self.options().memory_format(memory_format));
    output.fill_(value);

    auto c_output = output;
    for (const auto i : otter::irange(l_diff, l_inp)) {
        auto pad_idx = 2 * (l_inp - i - 1);
        if (pad[pad_idx] > 0) {
            c_output = c_output.narrow(i, pad[pad_idx], c_output.size(i) - pad[pad_idx]);
        }
        if (pad[pad_idx + 1] > 0) {
            c_output = c_output.narrow(i, 0, c_output.size(i) - pad[pad_idx + 1]);
        }
    }
    c_output.copy_(c_input);
    return output;
}

}   // end namespace otter
