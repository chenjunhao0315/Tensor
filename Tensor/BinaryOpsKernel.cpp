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
#include "ScalarOps.hpp"
#include "TensorFactory.hpp"

namespace otter {

void add_kernel(TensorIterator& iter, const Scalar& alpha_scalar) {
    if (iter.dtype() == ScalarType::Bool) {
        using scalar_t = bool;
        auto alpha = alpha_scalar.to<scalar_t>();
        cpu_kernel(iter, [=](scalar_t a, scalar_t b) -> scalar_t { return a + alpha * b; });
    } else {
        OTTER_DISPATCH_ALL_TYPES(iter.dtype(), "add_cpu", [&]() {
            auto alpha = alpha_scalar.to<scalar_t>();
            auto alpha_vec = Vectorized<scalar_t>(alpha);
            cpu_kernel_vec(
                iter,
                [=](scalar_t a, scalar_t b) __ubsan_ignore_undefined__ -> scalar_t { return a + alpha * b; },
                [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) __ubsan_ignore_undefined__ { return vec::fmadd(b, alpha_vec, a);
            });
        });
    }
}

void sub_kernel(TensorIterator& iter, const Scalar& alpha_scalar) {
    add_kernel(iter, -alpha_scalar);
}

void add_clamp_kernel(TensorIterator& iter, const Scalar& alpha_scalar, const Scalar& min_val, const Scalar& max_val) {
    OTTER_DISPATCH_ALL_TYPES(iter.dtype(), "add_clamp_cpu", [&]() {
        auto alpha = alpha_scalar.to<scalar_t>();
        auto alpha_vec = Vectorized<scalar_t>(alpha);
        auto min_scalar = min_val.to<scalar_t>();
        auto min_vec = Vectorized<scalar_t>(min_scalar);
        auto max_scalar = max_val.to<scalar_t>();
        auto max_vec = Vectorized<scalar_t>(max_scalar);
        cpu_kernel_vec(iter,
            [=](scalar_t a, scalar_t b) __ubsan_ignore_undefined__ -> scalar_t {
                return std::min(max_scalar, std::max(min_scalar, static_cast<scalar_t>(a + alpha * b)));
            },
            [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) __ubsan_ignore_undefined__ {
                auto add_clamp_res = vec::fmadd(b, alpha_vec, a);
                add_clamp_res = vec::clamp_min(add_clamp_res, min_vec);
                add_clamp_res = vec::clamp_max(add_clamp_res, max_vec);
            
                return add_clamp_res;
        });
    });
}

void mul_kernel(TensorIterator& iter) {
    if (iter.dtype() == ScalarType::Bool) {
        cpu_kernel(iter, [=](bool a, bool b) -> bool {
            return a && b;
        });
    } else {
        OTTER_DISPATCH_ALL_TYPES(iter.dtype(), "mul_cpu", [&]() {
            cpu_kernel_vec(iter,
                           [=](scalar_t a, scalar_t b) -> scalar_t { return a * b; },
                           [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a * b; }
                           );
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

void lt_kernel(TensorIterator& iter) {
    // See Note [special-case bool outputs]
    if (iter.dtype() == ScalarType::Bool) {
        OTTER_DISPATCH_ALL_TYPES_AND(otter::ScalarType::Bool, iter.common_dtype(), "lt_cpu", [&]() {
            cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool {
                return a < b;
            });
        });
    } else {
        OTTER_DISPATCH_ALL_TYPES(iter.common_dtype(), "lt_cpu", [&]() {
            cpu_kernel_vec(iter,
                [](scalar_t a, scalar_t b) -> scalar_t {
                    return a < b;
                },
                [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
                    return a.lt(b);
                });
        });
    }
}
void le_kernel(TensorIterator& iter) {
    // See Note [special-case bool outputs]
    if (iter.dtype() == ScalarType::Bool) {
        OTTER_DISPATCH_ALL_TYPES_AND(otter::ScalarType::Bool, iter.common_dtype(), "le_cpu", [&]() {
            cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool {
                return a <= b;
            });
        });
    } else {
        OTTER_DISPATCH_ALL_TYPES(iter.common_dtype(), "le_cpu", [&]() {
            cpu_kernel_vec(iter,
                [](scalar_t a, scalar_t b) -> scalar_t {
                    return a <= b;
                },
                [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
                    return a.le(b);
                });
        });
    }
}
void gt_kernel(TensorIterator& iter) {
    // See Note [special-case bool outputs]
    if (iter.dtype() == ScalarType::Bool) {
        OTTER_DISPATCH_ALL_TYPES_AND(otter::ScalarType::Bool, iter.common_dtype(), "gt_cpu", [&]() {
            cpu_kernel(iter,
                       [](scalar_t a, scalar_t b) -> bool {
                return a > b;
            });
        });
    } else {
        OTTER_DISPATCH_ALL_TYPES(iter.common_dtype(), "gt_cpu", [&]() {
            cpu_kernel_vec(
                           iter,
                           [](scalar_t a, scalar_t b) -> scalar_t {
                               return a > b;
                           },
                           [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
                               return a.gt(b);
                           });
        });
    }
}
void ge_kernel(TensorIterator& iter) {
    // See Note [special-case bool outputs]
    if (iter.dtype() == ScalarType::Bool) {
        OTTER_DISPATCH_ALL_TYPES_AND(otter::ScalarType::Bool, iter.common_dtype(), "ge_cpu", [&]() {
            cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool {
                return a >= b;
            });
        });
    } else {
        OTTER_DISPATCH_ALL_TYPES(iter.common_dtype(), "ge_cpu", [&]() {
            cpu_kernel_vec(iter,
                [](scalar_t a, scalar_t b) -> scalar_t {
                    return a >= b;
                },
                [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
                    return a.ge(b);
                });
        });
    }
}
void eq_kernel(TensorIterator& iter) {
    // See Note [special-case bool outputs]
    if (iter.dtype() == ScalarType::Bool) {
        OTTER_DISPATCH_ALL_TYPES_AND(otter::ScalarType::Bool, iter.common_dtype(), "eq_cpu", [&]() {
            cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool {
                return a == b;
            });
        });
    } else {
        OTTER_DISPATCH_ALL_TYPES(iter.common_dtype(), "eq_cpu", [&]() {
            cpu_kernel_vec(iter,
                [](scalar_t a, scalar_t b) -> scalar_t {
                    return a == b;
                },
                [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
                    return a.eq(b);
                });
        });
    }
}
void ne_kernel(TensorIterator& iter) {
    // See Note [special-case bool outputs]
    if (iter.dtype() == ScalarType::Bool) {
        OTTER_DISPATCH_ALL_TYPES_AND(otter::ScalarType::Bool, iter.common_dtype(), "ne_cpu", [&]() {
            cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool {
                return a != b;
            });
        });
    } else {
        OTTER_DISPATCH_ALL_TYPES(iter.common_dtype(), "ne_cpu", [&]() {
            cpu_kernel_vec(iter,
                [](scalar_t a, scalar_t b) -> scalar_t {
                    return a != b;
                },
                [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) -> Vectorized<scalar_t> {
                    return a.ne(b);
                });
        });
    }
}

REGISTER_DISPATCH(add_stub, &add_kernel);
REGISTER_DISPATCH(sub_stub, &sub_kernel);
REGISTER_DISPATCH(add_clamp_stub, &add_clamp_kernel);
REGISTER_DISPATCH(mul_stub, &mul_kernel);
REGISTER_DISPATCH(div_true_stub, &div_true_kernel);
REGISTER_DISPATCH(remainder_stub, &remainder_kernel);
REGISTER_DISPATCH(bitwise_and_stub, &bitwise_and_kernel);
REGISTER_DISPATCH(bitwise_or_stub, &bitwise_or_kernel);
REGISTER_DISPATCH(bitwise_xor_stub, &bitwise_xor_kernel);
REGISTER_DISPATCH(lt_stub, &lt_kernel);
REGISTER_DISPATCH(le_stub, &le_kernel);
REGISTER_DISPATCH(gt_stub, &gt_kernel);
REGISTER_DISPATCH(ge_stub, &ge_kernel);
REGISTER_DISPATCH(eq_stub, &eq_kernel);
REGISTER_DISPATCH(ne_stub, &ne_kernel);

Tensor& add_relu_impl(Tensor& result, const Tensor& self, const Tensor& other, const Scalar& alpha) {
    auto iter = TensorIterator::binary_op(result, self, other);
    Scalar min_val;
    Scalar max_val;
    if (self.scalar_type() == otter::ScalarType::Int) {
        min_val = 0;
        max_val = std::numeric_limits<int32_t>::max();
    } else if (self.scalar_type() == otter::ScalarType::Long) {
        min_val = 0;
        max_val = std::numeric_limits<int64_t>::max();
    } else if (self.scalar_type() == otter::ScalarType::Short) {
        min_val = 0;
        max_val = std::numeric_limits<int16_t>::max();
    } else if (self.scalar_type() == otter::ScalarType::Char) {
        min_val = 0;
        max_val = std::numeric_limits<int8_t>::max();
    } else if (self.scalar_type() == otter::ScalarType::Float) {
        min_val = 0.0;
        max_val = std::numeric_limits<float>::max();
    } else if (self.scalar_type() == otter::ScalarType::Double) {
        min_val = 0.0;
        max_val = std::numeric_limits<double>::max();
    } else {
        OTTER_INTERNAL_ASSERT(false, "Unsupported datatype for add_relu:", self.dtype().name());
    }

    result = iter.output();
    add_clamp_stub(Device::CPU, iter, alpha, min_val, max_val);
    return result;
}

Tensor& add_relu_out(const Tensor& self, const Tensor& other, const Scalar& alpha, Tensor& result) {
    return add_relu_impl(result, self, other, alpha);
}

Tensor add_relu(const Tensor& self, const Tensor& other, const Scalar& alpha) {
    Tensor result = otter::empty_like(self);
    return add_relu_impl(result, self, other, alpha);
}

Tensor add_relu(const Tensor& self, const Scalar& other, const Scalar& alpha) {
    return add_relu(self, native::wrapped_scalar_tensor(other), alpha);
}

Tensor& add_relu_(Tensor& self, const Tensor& other, const Scalar& alpha) {
    return add_relu_impl(self, self, other, alpha);
}

Tensor& add_relu_(Tensor& self, const Scalar& other, const Scalar& alpha) {
    return add_relu_(self, native::wrapped_scalar_tensor(other), alpha);
}

}   // end namespace otter
