//
//  UnaryOpsKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/6.
//

#include "Dispatch.hpp"
#include "Loop.hpp"
#include "UnaryOps.hpp"
#include "Dispatch.hpp"
#include "UnaryOpsKernel.hpp"

namespace otter {

void sin_kernel(TensorIterator& iter) {
    OTTER_DISPATCH_ALL_TYPES(iter.dtype(), "sin_cpu", [&]() {
        cpu_kernel(iter, [=](scalar_t a) -> scalar_t {
            return std::sin(a);
        });
    });
}

REGISTER_DISPATCH(sin_stub, &sin_kernel);


}
