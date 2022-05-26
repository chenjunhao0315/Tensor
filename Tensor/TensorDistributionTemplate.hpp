//
//  TensorDistributionTemplate.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/18.
//

#ifndef TensorDistributionTemplate_hpp
#define TensorDistributionTemplate_hpp

#include "Tensor.hpp"
#include "Dispatch.hpp"
#include "Generator.hpp"
#include "TensorIterator.hpp"
#include "Loop.hpp"
#include "DistributionsHelper.hpp"
#include "TensorFactory.hpp"
#include "ExpandUtils.hpp"
#include "TensorResize.hpp"

#include "VecIntrinsic.hpp"
#ifdef CPU_CAPABILITY_AVX2
#include "Avx_Math.hpp"
#endif

namespace otter {
namespace native {
namespace templates {

//#define CHECK_NORMAL_TENSOR_STD(std) \
//do { \
//    OTTER_CHECK( \
//    std.numel() == 0 || std.min().ge(0).item<bool>(), \
//    "normal expects all elements of std >= 0.0"); \
//} while (0)

#define CHECK_NORMAL_TENSOR_STD(std) \
do { \
OTTER_CHECK( \
std.numel() == 0, \
"normal expects all elements of std >= 0.0"); \
} while (0)

#define CHECK_NORMAL_STD(std) \
OTTER_CHECK(std >= 0.0, "normal expects std >= 0.0, but found std ", std);

template<template<typename> class normal_kernel, typename RNG>
Tensor& normal_impl_(Tensor& self, double mean, double std, Generator gen) {
    CHECK_NORMAL_STD(std);
    
    normal_kernel<RNG>()(self, mean, std, gen);
    
    return self;
}

template<template<typename> class normal_kernel, typename RNG>
Tensor& normal_out_impl(Tensor& output, const Tensor& mean, double std, Generator gen) {
    CHECK_NORMAL_STD(std);
    auto std_tensor = otter::empty_like(output, MemoryFormat::Contiguous);
    auto shape = otter::infer_size_dimvector(mean.sizes(), std_tensor.sizes());
    otter::native::resize_output(output, shape);
    normal_impl_<normal_kernel, RNG>(output, 0, std, gen);
    output.add_(mean);
    return output;
}

template<template<typename> class normal_kernel, typename RNG>
Tensor& normal_out_impl(Tensor& output, double mean, const Tensor& std, Generator gen) {
    CHECK_NORMAL_TENSOR_STD(std);
    auto mean_tensor = otter::full({}, mean, output.options());
    auto shape = otter::infer_size_dimvector(mean_tensor.sizes(), std.sizes());
    otter::native::resize_output(output, shape);
    normal_impl_<normal_kernel, RNG>(output, 0, 1, gen);
    // CUDA NB: addcmul_out copies the tensor to be added into the output.
    // Please look at aten/src/THC/generic/THCTensorMathPointwise.cu
    // The previous function here was addcmul_out(output, mean_tensor, output, std, 1);
    // The third argument is not a constant reference and hence the samples in output are overwritten.
    // Consequently, the computation performed is mean_tensor + mean_tensor * std instead of mean_tensor + output * std
    output.mul_(std).add_(mean_tensor);
    return output;
}

template<template<typename> class normal_kernel, typename RNG>
Tensor& normal_out_impl(Tensor& output, const Tensor& mean, const Tensor& std, Generator gen) {
    CHECK_NORMAL_TENSOR_STD(std);
    auto shape = otter::infer_size_dimvector(mean.sizes(), std.sizes());
    otter::native::resize_output(output, shape);
    normal_impl_<normal_kernel, RNG>(output, 0, 1, gen);
    // CUDA NB: addcmul_out copies the tensor to be added into the output.
    // Please look at aten/src/THC/generic/THCTensorMathPointwise.cu
    // The previous function here was addcmul_out(output, mean, output, std, 1);
    // The third argument is not a constant reference and hence the samples in output are overwritten.
    // Consequently, the computation performed is mean + mean * std instead of mean + output * std
    output.mul_(std).add_(mean);
    return output;
}

template<template<typename> class normal_kernel, typename RNG>
Tensor normal_impl(const Tensor& mean, double std, Generator gen) {
    CHECK_NORMAL_STD(std);
    Tensor ret = otter::empty_like(mean, MemoryFormat::Contiguous);
    normal_out_impl<normal_kernel, RNG>(ret, mean, std, gen);
    return ret;
}

template<template<typename> class normal_kernel, typename RNG>
Tensor normal_impl(double mean, const Tensor& std, Generator gen) {
    CHECK_NORMAL_TENSOR_STD(std);
    Tensor ret = otter::empty_like(std, MemoryFormat::Contiguous);
    normal_out_impl<normal_kernel, RNG>(ret, mean, std, gen);
    return ret;
}

template<template<typename> class normal_kernel, typename RNG>
Tensor normal_impl(const Tensor& mean, const Tensor& std, Generator gen) {
    CHECK_NORMAL_TENSOR_STD(std);
    auto shape = otter::infer_size_dimvector(mean.sizes(), std.sizes());
    Tensor ret = otter::empty(shape, mean.options(), MemoryFormat::Contiguous);
    normal_out_impl<normal_kernel, RNG>(ret, mean, std, gen);
    return ret;
}

#define CHECK_OUT_OF_BOUNDS(var, name, min, max) \
OTTER_CHECK(var >= min && var <= max, name , " is out of bounds"); \

template<template<typename> class uniform_kernel, typename RNG>
Tensor& uniform_impl_(Tensor& self, double from, double to, Generator generator) {
    OTTER_DISPATCH_FLOATING_TYPES(self.scalar_type(), "check_uniform_bounds", [&] {
//        const auto dtype = self.dtype();
        const auto min = static_cast<double>(std::numeric_limits<scalar_t>::lowest());
        const auto max = static_cast<double>(std::numeric_limits<scalar_t>::max());
        CHECK_OUT_OF_BOUNDS(from, "from", min, max);
        CHECK_OUT_OF_BOUNDS(to, "to", min, max);
        OTTER_CHECK(from <= to, "uniform_ expects to return a [from, to) range, but found from=", from, " > to=", to);
        OTTER_CHECK((to - from) <= std::numeric_limits<scalar_t>::max(),
                    "uniform_ expects to-from <= std::numeric_limits<", toString(self.scalar_type()),
                    ">::max(), but found to=", to, " and from=", from,
                    " which result in to-from to exceed the limit");
        from = std::min(std::max(from, min), max);
        to = std::max(std::min(to, max), min);
    });
    auto iter = TensorIterator::borrowing_nullary_op(self);
    uniform_kernel<RNG>()(iter, from, to, generator);
    return self;
}

template<typename scalar_t>
int64_t update_from(int64_t from) {
    static_assert(
                  std::is_floating_point<scalar_t>::value, "scalar_t must be floating-point type");
    const auto from_plus_1 = static_cast<int64_t>(static_cast<scalar_t>(from + 1));
    if (from_plus_1 < from) {
        int64_t from_ = std::abs(from + 1);
        int n = 0;
        while (from_ >>= 1) ++n;
        // NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult)
        from = from_plus_1 + (1LL << (n - std::numeric_limits<scalar_t>::digits + 1));
    }
    return from;
}
template<typename scalar_t>
int64_t update_to(int64_t to) {
    static_assert(
                  std::is_floating_point<scalar_t>::value, "scalar_t must be floating-point type");
    const auto to_minus_1 = static_cast<int64_t>(static_cast<scalar_t>(to - 1));
    if (to_minus_1 >= to) {
        int64_t to_ = std::abs(to - 1);
        int n = 0;
        while (to_ >>= 1) ++n;
        // NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult)
        to = to_minus_1 - (1LL << (n - std::numeric_limits<scalar_t>::digits + 1));
    }
    return to;
}

template<template<typename> class random_kernel, typename RNG>
Tensor& random_impl(Tensor& self, Generator generator) {
    auto iter = otter::TensorIterator::borrowing_nullary_op(self);
    random_kernel<RNG>()(iter, generator);
    return self;
}

#define CHECK_OUT_OF_BOUNDS(var, name, min, max) \
OTTER_CHECK(var >= min && var <= max, name , " is out of bounds"); \

#define WARN_OUT_OF_BOUNDS(var, name, digits) \
if (var < -(1LL << digits) || var > (1LL << digits)) { \
OTTER_CHECK(false, "Out out bound!") \
}

static inline void check_from_to_in_range(int64_t from, int64_t to_inc, otter::TypeMeta dtype) {
    const auto scalar_type = typeMetaToScalarType(dtype);
    if (isFloatingType(scalar_type)) {
        OTTER_DISPATCH_FLOATING_TYPES(scalar_type, "check_random_fp_bounds", [&] {
            const auto min = static_cast<double>(std::numeric_limits<scalar_t>::lowest());
            const auto max = static_cast<double>(std::numeric_limits<scalar_t>::max());
            CHECK_OUT_OF_BOUNDS(from, "from", min, max);
            CHECK_OUT_OF_BOUNDS(to_inc, "to - 1", min, max);
            constexpr auto digits = std::numeric_limits<scalar_t>::digits;
            WARN_OUT_OF_BOUNDS(from, "from", digits);
            WARN_OUT_OF_BOUNDS(to_inc, "to - 1", digits);
        });
    } else if (isIntegralType(scalar_type, /*includeBool=*/true)) {
        OTTER_DISPATCH_INTEGRAL_TYPES(scalar_type, "check_random_integral_bounds", [&]() {
            const auto min = static_cast<int64_t>(std::numeric_limits<scalar_t>::lowest());
            const auto max = static_cast<int64_t>(std::numeric_limits<scalar_t>::max());
            CHECK_OUT_OF_BOUNDS(from, "from", min, max);
            CHECK_OUT_OF_BOUNDS(to_inc, "to - 1", min, max);
        });
    } else {
        OTTER_CHECK(false, "check_random_bounds handles only integral, floating-point and boolean types");
    }
}
template<template<typename> class random_from_to_kernel, typename RNG>
Tensor& random_from_to_impl(Tensor& self, int64_t from, int64_t to, Generator generator) {
    uint64_t range = 0;
    auto iter = otter::TensorIterator::borrowing_nullary_op(self);
    
    // [from, to)
    OTTER_CHECK(from < to, "random_ expects 'from' to be less than 'to', but got from=", from, " >= to=", to);
    if (isFloatingType(iter.dtype())) {
        OTTER_DISPATCH_FLOATING_TYPES(self.scalar_type(), "random_update_from_to", [&] {
            from = update_from<scalar_t>(from);
            to = update_to<scalar_t>(to);
            OTTER_CHECK(from < to, "random_ expects 'from' casted to dtype to be less than 'to' casted to dtype, but got from=", from, " >= to=", to);
        });
    }
    check_from_to_in_range(from, to - 1, self.dtype());
    range = static_cast<uint64_t>(to) - static_cast<uint64_t>(from);
    random_from_to_kernel<RNG>()(iter, range, from, generator);
    
    return self;
}

namespace cpu {

template<typename RNG>
void random_from_to_kernel(TensorIterator& iter, uint64_t range, int64_t base, RNG generator) {
    OTTER_DISPATCH_ALL_TYPES(iter.dtype(), "random_from_to_kernel_cpu", [&] {
        std::lock_guard<std::mutex> lock(generator->mutex_);
        cpu_serial_kernel(iter, [range, base, generator]() -> scalar_t {
            uniform_int_from_to_distribution<scalar_t> random(range, base);
            return random(generator);
        });
    });
}
// This is the special kernel to handle single specific case:
// from(inclusive) = std::numeric_limits<int64_t>::lowest()
// to(exclusive) = None (= std::numeric_limits<int64_t>::max() + 1)
template<typename RNG>
void random_full_64_bits_range_kernel(TensorIterator& iter, RNG generator) {
    OTTER_DISPATCH_ALL_TYPES(iter.dtype(), "random_full_64_bits_range_kernel_cpu", [&] {
        std::lock_guard<std::mutex> lock(generator->mutex_);
        if (std::is_same<scalar_t, int64_t>::value ||
            std::is_same<scalar_t, double>::value ||
            std::is_same<scalar_t, float>::value) {
            cpu_serial_kernel(iter, [generator]() -> scalar_t {
                uniform_int_full_range_distribution<scalar_t> random;
                return random(generator);
            });
        } else {
            OTTER_CHECK(false, "random_full_64_bits_range_kernel_cpu handles only int64, double, float and bfloat16");
        }
    });
}
template<typename RNG>
struct RandomFromToKernel {
    void operator()(TensorIterator& iter, uint64_t range, int64_t base, Generator gen) {
        random_from_to_kernel(iter, range, base, check_generator<RNG>(gen));
    }
    void operator()(TensorIterator& iter, Generator gen) {
        random_full_64_bits_range_kernel(iter, check_generator<RNG>(gen));
    }
};
template<typename RNG>
void random_kernel(TensorIterator& iter, RNG generator) {
    std::lock_guard<std::mutex> lock(generator->mutex_);
    OTTER_DISPATCH_ALL_TYPES(iter.dtype(), "random_kernel_cpu", [&] {
        cpu_serial_kernel(iter, [generator]() -> scalar_t {
            uniform_int_distribution<scalar_t> random;
            return random(generator);
        });
    });
}
template<typename RNG>
struct RandomKernel {
    void operator()(TensorIterator& iter, Generator gen) {
        random_kernel(iter, check_generator<RNG>(gen));
    }
};


#ifdef CPU_CAPABILITY_AVX2
static inline void normal_fill_16_AVX2(float *data,
                                const __m256* two_pi,
                                const __m256* one,
                                const __m256* minus_two,
                                const __m256* mean,
                                const __m256* std_v) {
    const __m256 u1 = _mm256_sub_ps(*one, _mm256_loadu_ps(data));
    const __m256 u2 = _mm256_loadu_ps(data + 8);
    // sincos256_ps and log256_ps are from avx_mathfun.h
    const __m256 radius = _mm256_sqrt_ps(_mm256_mul_ps(*minus_two, log256_ps(u1)));
    const __m256 theta = _mm256_mul_ps(*two_pi, u2);
    __m256 sintheta, costheta;
    sincos256_ps(theta, &sintheta, &costheta);
    const __m256 n1 = _mm256_mul_ps(radius, costheta);
    const __m256 n2 = _mm256_mul_ps(radius, sintheta);
    _mm256_storeu_ps(data, _mm256_fmadd_ps(n1, *std_v, *mean));
    _mm256_storeu_ps(data + 8, _mm256_fmadd_ps(n2, *std_v, *mean));
}
template<typename RNG>
void normal_fill_AVX2(const TensorBase &self, const float mean, const float std, RNG generator) {
    float *data = self.data_ptr<float>();
    auto size = self.numel();
    std::lock_guard<std::mutex> lock(generator->mutex_);
    for (const auto i : otter::irange(size)) {
        otter::uniform_real_distribution<float> uniform(0, 1);
        data[i] = uniform(generator);
    }
    const __m256 two_pi = _mm256_set1_ps(2.0f * static_cast<double>(M_PI));
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 minus_two = _mm256_set1_ps(-2.0f);
    const __m256 mean_v = _mm256_set1_ps(mean);
    const __m256 std_v = _mm256_set1_ps(std);
    for (int64_t i = 0; i < size - 15; i += 16) {
        normal_fill_16_AVX2(data + i, &two_pi, &one, &minus_two, &mean_v, &std_v);
    }
    if (size % 16 != 0) {
        // Recompute the last 16 values.
        data = data + size - 16;
        for (const auto i : otter::irange(16)) {
            otter::uniform_real_distribution<float> uniform(0, 1);
            data[i] = uniform(generator);
        }
        normal_fill_16_AVX2(data, &two_pi, &one, &minus_two, &mean_v, &std_v);
    }
}
#endif
template <typename scalar_t>
static void normal_fill_16(scalar_t *data, const scalar_t mean, const scalar_t std) {
    for (const auto j : otter::irange(8)) {
        const scalar_t u1 = 1 - data[j]; // [0, 1) -> (0, 1] for log.
        const scalar_t u2 = data[j + 8];
        const scalar_t radius = std::sqrt(-2 * std::log(u1));
        const scalar_t theta = 2.0f * static_cast<double>(M_PI) * u2;
        data[j] = radius * std::cos(theta) * std + mean;
        data[j + 8] = radius * std::sin(theta) * std + mean;
    }
}
template <typename scalar_t, typename RNG>
void normal_fill(const TensorBase &self, const scalar_t mean, const scalar_t std, RNG generator) {
    scalar_t *data = self.data_ptr<scalar_t>();
    auto size = self.numel();
    std::lock_guard<std::mutex> lock(generator->mutex_);
    for (const auto i : otter::irange(size)) {
        otter::uniform_real_distribution<scalar_t> uniform(0, 1);
        data[i] = uniform(generator);
    }
    for (int64_t i = 0; i < size - 15; i += 16) {
        normal_fill_16<scalar_t>(data + i, mean, std);
    }
    if (size % 16 != 0) {
        // Recompute the last 16 values.
        data = data + size - 16;
        for (const auto i : otter::irange(16)) {
            otter::uniform_real_distribution<scalar_t> uniform(0, 1);
            data[i] = uniform(generator);
        }
        normal_fill_16<scalar_t>(data, mean, std);
    }
}
template<typename RNG>
void normal_kernel(const TensorBase &self, double mean, double std, RNG generator) {
    auto size = self.numel();
    if (self.scalar_type() == ScalarType::Float && size >= 16 && self.is_contiguous()) {
#ifdef CPU_CAPABILITY_AVX2
        normal_fill_AVX2(self, static_cast<float>(mean), static_cast<float>(std), generator);
#else
        normal_fill(self, static_cast<float>(mean), static_cast<float>(std), generator);
#endif
    } else {
        OTTER_DISPATCH_FLOATING_TYPES(self.scalar_type(), "normal_kernel_cpu", [&] {
            if (size >= 16 && self.is_contiguous()) {
                normal_fill<scalar_t>(self, static_cast<scalar_t>(mean), static_cast<scalar_t>(std), generator);
            } else {
                auto iter = TensorIterator::borrowing_nullary_op(self);
                std::lock_guard<std::mutex> lock(generator->mutex_);
                cpu_serial_kernel(iter, [mean, std, generator]() -> scalar_t {
                    otter::normal_distribution<double> normal(mean, std);
                    return static_cast<scalar_t>(normal(generator));
                });
            }
        });
    }
}
template<typename RNG>
struct NormalKernel {
    void operator()(Tensor& self, double mean, double std, Generator gen) {
        normal_kernel(self, mean, std, check_generator<RNG>(gen));
    }
};

template<typename RNG>
void uniform_kernel(TensorIterator& iter, double from_, double to_, RNG generator) {
    OTTER_DISPATCH_FLOATING_TYPES(iter.dtype(), "uniform_kernel_cpu", [&]() {
        std::lock_guard<std::mutex> lock(generator->mutex_);
        auto from = static_cast<scalar_t>(from_);
        auto to = static_cast<scalar_t>(to_);
        otter::uniform_real_distribution<scalar_t> uniform(from, to);
        cpu_serial_kernel(iter, [&uniform, generator]() -> scalar_t {
            return static_cast<scalar_t>(uniform(generator));
        });
    });
}
template<typename RNG>
struct UniformKernel {
    void operator()(TensorIterator& iter, double from, double to, Generator gen) {
        uniform_kernel(iter, from, to, check_generator<RNG>(gen));
    }
};

}   // end namespace cpu

}   // end namespace templates
}   // end namespace native
}   // end namespace otter

#endif /* TensorDistributionTemplate_hpp */
