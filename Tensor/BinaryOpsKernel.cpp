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
#include "ScalarType.hpp"

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

void remainder_kernel(TensorIterator& iter) {
    if (isIntegralType(iter.common_dtype(), false)) {
        OTTER_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "remainder_cpu", [&]() {
            cpu_kernel(iter, [=](scalar_t a, scalar_t b) -> scalar_t {
                scalar_t r = a % b;
                if ((r != 0) && ((r < 0) != (b < 0))) {
                    r += b;
                }
                return r;
            });
        });
    } else {
        OTTER_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "remainder_cpu", [&]() {
            cpu_kernel(iter, [=](scalar_t a, scalar_t b) -> scalar_t {
                scalar_t mod = std::fmod(a, b);
                if ((mod != 0) && ((b < 0) != (mod < 0))) mod += b;
                return mod;
            });
        });
    }
}

void bitwise_and_kernel(TensorIterator& iter) {
    if (iter.dtype() == ScalarType::Bool) {
        cpu_kernel(iter, [=](bool a, bool b) {
            return a && b;
        });
    } else {
        OTTER_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_and_cpu", [&]() {
            cpu_kernel(iter, [=](scalar_t a, scalar_t b) -> scalar_t {
                return a & b;
            });
        });
    }
}

void bitwise_or_kernel(TensorIterator& iter) {
    if (iter.dtype() == ScalarType::Bool) {
        cpu_kernel(iter, [=](bool a, bool b) {
            return a || b;
        });
    } else {
        OTTER_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_and_cpu", [&]() {
            cpu_kernel(iter, [=](scalar_t a, scalar_t b) -> scalar_t {
                return a | b;
            });
        });
    }
}

void bitwise_xor_kernel(TensorIterator& iter) {
    if (iter.dtype() == ScalarType::Bool) {
        cpu_kernel(iter, [=](bool a, bool b) {
            return a != b;
        });
    } else {
        OTTER_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_and_cpu", [&]() {
            cpu_kernel(iter, [=](scalar_t a, scalar_t b) -> scalar_t {
                return a ^ b;
            });
        });
    }
}

REGISTER_DISPATCH(add_stub, &add_kernel);
REGISTER_DISPATCH(sub_stub, &sub_kernel);
REGISTER_DISPATCH(mul_stub, &mul_kernel);
REGISTER_DISPATCH(div_true_stub, &div_true_kernel);
REGISTER_DISPATCH(remainder_stub, &remainder_kernel);
REGISTER_DISPATCH(bitwise_and_stub, &bitwise_and_kernel);
REGISTER_DISPATCH(bitwise_or_stub, &bitwise_or_kernel);
REGISTER_DISPATCH(bitwise_xor_stub, &bitwise_xor_kernel);

}
