//
//  BinaryOpsKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/30.
//

#include "Dispatch.hpp"
#include "Loop.hpp"
#include "BinaryOps.hpp"
#include "Dispatch.hpp"
#include "BinaryOpsKernel.hpp"

namespace otter {

void add_kernel(TensorIterator& iter, const Scalar& alpha_scalar) {
    if (iter.dtype() == ScalarType::Bool) {
        using scalar_t = bool;
        auto alpha = alpha_scalar.to<scalar_t>();
        cpu_kernel(iter, [=](scalar_t a, scalar_t b) -> scalar_t { return a + alpha * b; });
    } else {
        OTTER_DISPATCH_ALL_TYPES(iter.dtype(), "add_cpu", [&]() {
            auto alpha = alpha_scalar.to<scalar_t>();
            cpu_kernel(iter, [=](scalar_t a, scalar_t b) -> scalar_t {
                return a + alpha * b; });
        });
    }
}

void sub_kernel(TensorIterator& iter, const Scalar& alpha_scalar) {
    add_kernel(iter, -alpha_scalar);
}

void mul_kernel(TensorIterator& iter) {
    if (iter.dtype() == ScalarType::Bool) {
        cpu_kernel(iter, [=](bool a, bool b) -> bool {
            return a && b;
        });
    } else {
        OTTER_DISPATCH_ALL_TYPES(iter.dtype(), "mul_cpu", [&]() {
            cpu_kernel(iter, [=](scalar_t a, scalar_t b) -> scalar_t {
                return a * b;
            });
        });
    }
}

void div_true_kernel(TensorIterator& iter) {
    OTTER_DISPATCH_ALL_TYPES(iter.dtype(), "div_true_cpu", [&]() {
        cpu_kernel(iter, [=](scalar_t a, scalar_t b) {
            return a / b;
        });
    });
}

REGISTER_DISPATCH(add_stub, &add_kernel);
REGISTER_DISPATCH(sub_stub, &sub_kernel);
REGISTER_DISPATCH(mul_stub, &mul_kernel);
REGISTER_DISPATCH(div_true_stub, &div_true_kernel);


}
