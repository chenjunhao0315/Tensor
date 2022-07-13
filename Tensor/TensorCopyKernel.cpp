//
//  TensorCopyKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/10.
//

#include "TensorCopy.hpp"
#include "TensorCopyKernel.hpp"
#include "Dispatch.hpp"
#include "Loop.hpp"
#include "Parallel.hpp"
#include "TypeCast.hpp"

namespace otter {

void direct_copy_kernel(TensorIterator& iter) {
    OTTER_DISPATCH_ALL_TYPES_AND(otter::ScalarType::Bool, iter.dtype(), "copy_kernel", [&]() {
        cpu_kernel(iter, [=](scalar_t a) -> scalar_t {
            return a;
        });
    });
}

void copy_same_dtype(TensorIterator& iter) {
    direct_copy_kernel(iter);
}

void copy_kernel(TensorIterator& iter, bool /*non_blocking*/) {
    ScalarType dtype = iter.dtype(0);
    
    if (dtype == iter.dtype(1)) {
        copy_same_dtype(iter);
    } else {
        OTTER_DISPATCH_ALL_TYPES_AND(otter::ScalarType::Bool, dtype, "copy_", [&]() {
            using dest_t = scalar_t;
            OTTER_DISPATCH_ALL_TYPES_AND(otter::ScalarType::Bool, iter.dtype(1), "copy_", [&]() {
                cpu_kernel(iter, [=](scalar_t src) -> dest_t {
                    return otter::static_cast_with_inter_type<dest_t, scalar_t>::apply(src);;
                });
            });
        });
    }
}


REGISTER_DISPATCH(copy_stub, &copy_kernel);

}
