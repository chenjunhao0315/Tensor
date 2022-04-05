//
//  TensorCompareKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/6.
//

#include "TensorCompare.hpp"
#include "TensorCompareKernel.hpp"
#include "Dispatch.hpp"
#include "TensorIterator.hpp"
#include "Vec.hpp"
#include "Loop.hpp"

using namespace otter::vec;

namespace otter {

static void clamp_kernel_impl(TensorIterator& iter) {
    OTTER_DISPATCH_ALL_TYPES(iter.common_dtype(), "clamp_cpu", [&]() {
        cpu_kernel_vec(iter, [](scalar_t a, scalar_t min, scalar_t max) -> scalar_t {
            return std::min(std::max(a, min), max);
        },[](Vectorized<scalar_t> a, Vectorized<scalar_t> min, Vectorized<scalar_t> max) {
            return vec::clamp(a, min, max);
        });
    });
}

static void clamp_scalar_kernel_impl(TensorIterator& iter, const Scalar& min_, const Scalar& max_) {
    OTTER_DISPATCH_ALL_TYPES(iter.common_dtype(), "clamp_scalar_cpu", [&]() {
        const auto min = min_.to<scalar_t>();
        const auto max = max_.to<scalar_t>();
        const Vectorized<scalar_t> min_vec(min);
        const Vectorized<scalar_t> max_vec(max);
        cpu_kernel_vec(iter, [=](scalar_t a) -> scalar_t {
            return std::min(std::max(a, min), max);
        }, [=](Vectorized<scalar_t> a) {
            return vec::clamp(a, min_vec, max_vec);
        });
    });
}

static void clamp_max_kernel_impl(TensorIterator& iter) {
    OTTER_DISPATCH_ALL_TYPES(iter.common_dtype(), "clamp_max_cpu", [&]() {
        cpu_kernel_vec(iter, [](scalar_t a, scalar_t max) -> scalar_t {
            return std::min(a, max);
        }, [](Vectorized<scalar_t> a, Vectorized<scalar_t> max) {
            return vec::clamp_max(a, max);
        });
    });
}

static void clamp_max_scalar_kernel_impl(TensorIterator& iter, Scalar max_) {
    OTTER_DISPATCH_ALL_TYPES(iter.common_dtype(), "clamp_max_scalar_cpu", [&]() {
        const auto max = max_.to<scalar_t>();
        const Vectorized<scalar_t> max_vec(max);
        cpu_kernel_vec(iter, [=](scalar_t a) -> scalar_t {
            return std::min(a, max);
        }, [=](Vectorized<scalar_t> a) {
            return vec::clamp_max(a, max_vec);
        });
    });
}

static void clamp_min_kernel_impl(TensorIterator& iter) {
    OTTER_DISPATCH_ALL_TYPES(iter.common_dtype(), "clamp_min_cpu", [&]() {
        cpu_kernel_vec(iter, [](scalar_t a, scalar_t min) -> scalar_t {
            return std::max(a, min);
        }, [](Vectorized<scalar_t> a, Vectorized<scalar_t> min) {
            return vec::clamp_min(a, min);
        });
    });
}

static void clamp_min_scalar_kernel_impl(TensorIterator& iter, Scalar min_) {
    OTTER_DISPATCH_ALL_TYPES(iter.common_dtype(), "clamp_min_cpu", [&]() {
        const auto min = min_.to<scalar_t>();
        const Vectorized<scalar_t> min_vec(min);
        cpu_kernel_vec(iter, [=](scalar_t a) -> scalar_t {
            return std::max(a, min);
        }, [=](Vectorized<scalar_t> a) {
            return vec::clamp_min(a, min_vec);
        });
    });
}

REGISTER_DISPATCH(clamp_stub, &clamp_kernel_impl);
REGISTER_DISPATCH(clamp_min_stub, &clamp_min_kernel_impl);
REGISTER_DISPATCH(clamp_max_stub, &clamp_max_kernel_impl);
REGISTER_DISPATCH(clamp_scalar_stub, &clamp_scalar_kernel_impl);
REGISTER_DISPATCH(clamp_min_scalar_stub, &clamp_min_scalar_kernel_impl);
REGISTER_DISPATCH(clamp_max_scalar_stub, &clamp_max_scalar_kernel_impl);

}   // end namespace otter
