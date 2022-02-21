//
//  UnaryOpsKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/6.
//

#include "Dispatch.hpp"
#include "Loop.hpp"
#include "UnaryOps.hpp"
#include "UnaryOpsKernel.hpp"
#include "Math.hpp"
#include "VecBase.hpp"

namespace otter {

void bitwise_not_kernel(TensorIterator& iter) {
    if (iter.dtype() == ScalarType::Bool) {
        cpu_kernel(iter, [=](bool a) -> bool {
            return !a;
        });
    } else {
        OTTER_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_not_cpu", [&]() {
            cpu_kernel(iter, [=](scalar_t a) -> scalar_t {
                return ~a;
            });
//            cpu_kernel_vec(iter, [=](scalar_t a) -> scalar_t { return ~a; }, [=](vec::Vectorized<scalar_t> a) { return ~a; });
        });
    }
}

void neg_kernel(TensorIterator& iter) {
    OTTER_DISPATCH_ALL_TYPES(iter.dtype(), "neg_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](scalar_t a) -> scalar_t { return -a; },
            [=](vec::Vectorized<scalar_t> a) { return a.neg(); }
        );
    });
}

void abs_kernel(TensorIterator& iter) {
    OTTER_DISPATCH_ALL_TYPES(iter.dtype(), "abs_cpu", [&]() {
        cpu_kernel_vec(
            iter,
            [=](scalar_t a) -> scalar_t { return abs_impl(a); },
            [=](vec::Vectorized<scalar_t> a) { return a.abs(); }
        );
    });
}

void sin_kernel(TensorIterator& iter) {
    OTTER_DISPATCH_ALL_TYPES(iter.dtype(), "sin_cpu", [&]() {
        cpu_kernel(iter, [=](scalar_t a) -> scalar_t {
            return std::sin(a);
        });
//        cpu_kernel_vec(
//            iter,
//            [=](scalar_t a) -> scalar_t { return std::sin(a); },
//            [=](vec::Vectorized<scalar_t> a) { return a.sin(); }
//        );
    });
}

void cos_kernel(TensorIterator& iter) {
    OTTER_DISPATCH_ALL_TYPES(iter.dtype(), "cos_cpu", [&]() {
        cpu_kernel(iter, [=](scalar_t a) -> scalar_t {
            return std::cos(a);
        });
//        cpu_kernel_vec(
//            iter,
//            [=](scalar_t a) -> scalar_t { return std::cos(a); },
//            [=](vec::Vectorized<scalar_t> a) { return a.cos(); }
//        );
    });
}

void tan_kernel(TensorIterator& iter) {
    OTTER_DISPATCH_ALL_TYPES(iter.dtype(), "tan_cpu", [&]() {
        cpu_kernel(iter, [=](scalar_t a) -> scalar_t {
            return std::tan(a);
        });
//        cpu_kernel_vec(
//            iter,
//            [=](scalar_t a) -> scalar_t { return std::tan(a); },
//            [=](vec::Vectorized<scalar_t> a) { return a.tan(); }
//        );
    });
}

void exp_kernel(TensorIterator& iter) {
    OTTER_DISPATCH_ALL_TYPES(iter.dtype(), "tan_cpu", [&]() {
        cpu_kernel(iter, [=](scalar_t a) -> scalar_t {
            return std::exp(a);
        });
//        cpu_kernel_vec(
//            iter,
//            [=](scalar_t a) -> scalar_t { return std::exp(a); },
//            [=](vec::Vectorized<scalar_t> a) { return a.exp(); }
//        );
    });
}

void sqrt_kernel(TensorIterator& iter) {
    OTTER_DISPATCH_ALL_TYPES(iter.dtype(), "sqrt_cpu", [&]() {
        cpu_kernel(iter, [=](scalar_t a) -> scalar_t {
            return std::sqrt(a);
        });
    });
}

REGISTER_DISPATCH(bitwise_not_stub, &bitwise_not_kernel);
REGISTER_DISPATCH(neg_stub, &neg_kernel);
REGISTER_DISPATCH(abs_stub, &abs_kernel);
REGISTER_DISPATCH(sin_stub, &sin_kernel);
REGISTER_DISPATCH(cos_stub, &cos_kernel);
REGISTER_DISPATCH(tan_stub, &tan_kernel);
REGISTER_DISPATCH(exp_stub, &exp_kernel);
REGISTER_DISPATCH(sqrt_stub, &sqrt_kernel);


}
