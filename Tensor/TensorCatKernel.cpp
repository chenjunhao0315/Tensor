//
//  TensorCatKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/19.
//

#include "TensorCat.hpp"
#include "TensorCatKernel.hpp"
#include "Dispatch.hpp"
#include "Tensor.hpp"
#include "Vec.hpp"

namespace otter {

struct InputMeta {
    void* data_ptr;
    int64_t inner_size;

    InputMeta(const Tensor& t, int64_t dim, int64_t inner) : data_ptr(t.data_ptr()) , inner_size(t.sizes()[dim] * inner) {}
};

template <typename scalar_t>
void cat_serial_kernel_impl(Tensor& result, TensorList tensors, int64_t dim) {
    assert(dim >= 0 && dim < result.dim()); // "dim out of range in cat_serial_kernel_impl");
    int64_t outer = result.numel() / (result.sizes()[dim] * result.strides()[dim]);
    scalar_t* result_data = result.data_ptr<scalar_t>();
    int64_t ninputs = tensors.size();
    std::vector<InputMeta> inputs;
    inputs.reserve(ninputs);
    for (auto const &tensor : tensors) {
        inputs.emplace_back(tensor, dim, result.strides()[dim]);
    }
    
    using Vec = vec::Vectorized<scalar_t>;
    scalar_t* result_ptr = result_data;
    for (const auto i : otter::irange(outer)) {
        for (const auto j : otter::irange(ninputs)) {
            int64_t local_inner = inputs[j].inner_size;
            scalar_t* input_ptr = (scalar_t*)(inputs[j].data_ptr) + i * local_inner;
            int64_t d = 0;
            for (; d < local_inner - (local_inner % Vec::size()); d += Vec::size()) {
                Vec in_vec = Vec::loadu(input_ptr + d);
                in_vec.store(result_ptr + d);
            }
        #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
        # pragma unroll
        #endif
            for (; d < local_inner; d++) {
                result_ptr[d] = input_ptr[d];
            }
            result_ptr += local_inner;
        }
    }
}

void cat_serial_kernel(Tensor& result, TensorList tensors, int64_t dim) {
    OTTER_DISPATCH_FLOATING_TYPES(result.scalar_type(), "cat_serial_kernel", [&]() {
        cat_serial_kernel_impl<scalar_t>(result, tensors, dim);
    });
}

REGISTER_DISPATCH(cat_serial_stub, &cat_serial_kernel);

}   // end namespace otter
