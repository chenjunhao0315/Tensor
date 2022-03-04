//
//  Loop.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef Loop_hpp
#define Loop_hpp

#include <stdint.h>

#include "Function_Trait.hpp"
#include "Utils.hpp"
#include "C++17.hpp"
#include "TensorIterator.hpp"
#include "Vec.hpp"
#include "IsContiguous.hpp"
#include "TensorIteratorDynamicCasting.hpp"


namespace otter {

using namespace vec;

template <typename traits, std::size_t... INDEX>
typename traits::ArgsTuple dereference_impl(char* __restrict__ data[], const int64_t* strides, int64_t i, std::index_sequence<INDEX...>) {
    return std::make_tuple(
                           *(typename traits::template arg<INDEX>::type*)
                           (data[INDEX] + i * strides[INDEX])...);
}

template <typename traits>
typename traits::ArgsTuple dereference(char* __restrict__ data[], const int64_t* strides, int64_t i) {
    using Indices = std::make_index_sequence<traits::arity>;
    return dereference_impl<traits>(data, strides, i, Indices{});
}

template <typename traits, std::size_t... INDEX>
typename traits::ArgsTuple
dereference_vec_impl(char* __restrict__ data[],
                     const typename traits::result_type& opt_scalar,
                     size_t S,
                     int64_t i,
                     std::index_sequence<INDEX...>) {
    using Vec = typename traits::result_type;
    using scalar_t = typename Vec::value_type;
    return std::make_tuple(
                           S == INDEX + 1 ?
                           opt_scalar :
                           Vec::loadu(data[INDEX] + i * sizeof(scalar_t))...);
}

template <typename traits>
typename traits::ArgsTuple
dereference_vec(char* __restrict__ data[], const typename traits::result_type& opt_scalar, size_t S, int64_t i) {
    using Indices = std::make_index_sequence<traits::arity>;
    return dereference_vec_impl<traits>(data, opt_scalar, S, i, Indices{});
}

template <typename func_t,
typename std::enable_if<!std::is_void<typename function_traits<func_t>::result_type>::value>::type* = nullptr>
static inline void execute_op(char* __restrict__ data[], const int64_t* strides, int64_t i, int64_t n, func_t&& op) {
    using traits = function_traits<func_t>;
    using result_type = typename traits::result_type;
    for (; i < n; i++) {
        result_type* out_ptr = (result_type*)(data[0] + i * strides[0]);
        *out_ptr = otter::apply(std::forward<func_t>(op), dereference<traits>(&data[1], &strides[1], i));
    }
}

template <typename func_t,
typename std::enable_if<std::is_void<typename function_traits<func_t>::result_type>::value>::type* = nullptr>
static inline void execute_op(char* __restrict__ data[], const int64_t* strides, int64_t i, int64_t n, func_t&& op) {
    using traits = function_traits<func_t>;
    for (; i < n; i++) {
        otter::apply(std::forward<func_t>(op), dereference<traits>(&data[0], &strides[0], i));
    }
}

template <typename func_t>
static inline void
basic_loop(char* __restrict__ data[], const int64_t* strides_, int64_t i, int64_t n, func_t&& op) {
    using traits = function_traits<func_t>;
    constexpr int ntensors = traits::arity + 1;
    
    int64_t strides[ntensors];
    for (const auto arg : otter::irange(ntensors)) {
        strides[arg] = strides_[arg];
    }
    
    execute_op(data, strides, i, n, std::forward<func_t>(op));
}

template <typename func_t, typename vec_func_t>
static inline void
vectorized_loop(char** __restrict__ data_, int64_t n, int64_t S, func_t&& op, vec_func_t&& vop) {
    using traits = function_traits<vec_func_t>;
    using scalar_t = typename function_traits<func_t>::result_type;
    using Vec = Vectorized<scalar_t>;
    constexpr int ntensors = traits::arity + 1;
    
    char* __restrict__ data[ntensors];
    for (const auto arg : otter::irange(ntensors)) {
        data[arg] = data_[arg];
    }
    
    Vec opt_scalar = Vec(S > 0 ? *(scalar_t*)data[S] : scalar_t(0));
    int64_t i = 0;
    for (; i <= n - 2 * Vec::size(); i += 2 * Vec::size()) {
        auto args1 = dereference_vec<traits>(&data[1], opt_scalar, S, i);
        auto args2 = dereference_vec<traits>(&data[1], opt_scalar, S, i + Vec::size());
        auto out1 = otter::apply(std::forward<vec_func_t>(vop), std::move(args1));
        auto out2 = otter::apply(std::forward<vec_func_t>(vop), std::move(args2));
        out1.store(data[0] + i * sizeof(scalar_t));
        out2.store(data[0] + (i + Vec::size()) * sizeof(scalar_t));
    }
    if (i < n) {
        int64_t strides[ntensors];
        for (const auto arg : otter::irange(ntensors)) {
            strides[arg] = (S > 0 && arg == S) ? 0 : sizeof(scalar_t);
        }
        basic_loop(data, strides, i, n, std::forward<func_t>(op));
    }
}

template <typename traits, typename cb_t>
static inline void unroll_contiguous_scalar_checks(
    const int64_t* strides,
    std::index_sequence<>,
    cb_t&& cb) {
    
    cb(0);
}

template <typename traits, typename cb_t, size_t INDEX0, size_t ...INDEX>
static inline void unroll_contiguous_scalar_checks(
    const int64_t* strides,
    std::index_sequence<INDEX0, INDEX...>,
    cb_t&& cb) {
    
    if (is_contiguous_scalar<traits, INDEX0 + 1>(strides)) {
        cb(INDEX0 + 1);
    } else {
        unroll_contiguous_scalar_checks<traits>(strides, std::index_sequence<INDEX...>{}, std::forward<cb_t>(cb));
    }
}

template <typename op_t, typename vop_t>
struct VectorizedLoop2d {
    op_t op;
    vop_t vop;
    
    using traits = function_traits<op_t>;
    static constexpr int ntensors = traits::arity + 1;
    using data_t = std::array<char*, ntensors>;
    
    VectorizedLoop2d(const op_t &op, const vop_t &vop):
    op(op), vop(vop) {}
    
    static void advance(data_t &data, const int64_t *outer_strides) {
        for (const auto arg : otter::irange(data.size())) {
            data[arg] += outer_strides[arg];
        }
    }
    
    void operator()(char** base, const int64_t *strides, int64_t size0, int64_t size1) {
        data_t data;
        std::copy_n(base, ntensors, data.data());
        const int64_t *outer_strides = &strides[ntensors];
        
        if (is_contiguous<traits>(strides)) {
            for (const auto i : otter::irange(size1)) {
                (void)i;
                vectorized_loop(data.data(), size0, 0, op, vop);
                advance(data, outer_strides);
            }
        } else {
            using Indices = std::make_index_sequence<traits::arity>;
            unroll_contiguous_scalar_checks<traits>(strides, Indices{}, [&](size_t idx) {
                if (idx) {
                    for (const auto i : otter::irange(size1)) {
                        (void)i;
                        vectorized_loop(data.data(), size0, idx, op, vop);
                        advance(data, outer_strides);
                    }
                } else {
                    for (const auto i : otter::irange(size1)) {
                        (void)i;
                        basic_loop(data.data(), strides, 0, size0, op);
                        advance(data, outer_strides);
                    }
                }
            });
        }
    }
};

template <typename op_t, typename vop_t>
VectorizedLoop2d<op_t, vop_t> make_vectorized_loop2d(const op_t &op, const vop_t &vop) {
    return VectorizedLoop2d<op_t, vop_t>(op, vop);
}

template <typename func_t>
void cpu_kernel(TensorIterator& iter, func_t&& op, int64_t grain_size = otter::GRAIN_SIZE) {
    using traits = function_traits<func_t>;
    
    OTTER_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
    OTTER_INTERNAL_ASSERT(iter.noutputs() == 1);
    OTTER_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));
    
    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
        basic_loop(data, strides, 0, n, std::forward<func_t>(op));
    }, grain_size);
    iter.cast_outputs();
}

template <bool check_dynamic_cast=true, typename func_t, typename vec_func_t>
void cpu_kernel_vec(TensorIterator& iter, func_t&& op, vec_func_t&& vop, int64_t grain_size = otter::GRAIN_SIZE) {
    using traits = function_traits<func_t>;
    
    OTTER_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
    OTTER_INTERNAL_ASSERT(iter.noutputs() == 1);
    
    otter::if_constexpr<check_dynamic_cast>([&] {
        OTTER_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));
    });
    
    iter.for_each(make_vectorized_loop2d(op, vop), grain_size);
    iter.cast_outputs();
}

template <typename func_t>
void cpu_serial_kernel(TensorIterator& iter, func_t&& op, const Range& range) {
    using traits = function_traits<func_t>;
    constexpr bool result_void = std::is_void<typename traits::result_type>::value;
    OTTER_INTERNAL_ASSERT(iter.ninputs() == traits::arity && ((result_void && iter.noutputs() == 0) || (!result_void && iter.noutputs() == 1)));
    
    iter.serial_for_each([&](char** data, const int64_t* strides, int64_t n) {
        basic_loop(data, strides, 0, n, std::forward<func_t>(op));
    }, range);
    iter.cast_outputs();
}

template <typename func_t>
void cpu_serial_kernel(TensorIterator& iter, func_t&& op) {
    cpu_serial_kernel(iter, op, {0, iter.numel()});
}

template <typename func_t, typename vec_func_t>
void cpu_serial_kernel_vec(TensorIterator& iter, func_t&& op, vec_func_t&& vop, const Range& range) {
    using traits = function_traits<func_t>;
    // this could be extended to work with void return types
    OTTER_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
    OTTER_INTERNAL_ASSERT(iter.noutputs() == 1);
    // dynamic casting not currently supported on CPU
    OTTER_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

    iter.serial_for_each(make_vectorized_loop2d(op, vop), range);
    iter.cast_outputs();
}

template <typename func_t, typename vec_func_t>
void cpu_serial_kernel_vec(TensorIterator& iter, func_t&& op, vec_func_t&& vop) {
    cpu_serial_kernel_vec(iter, op, vop, {0, iter.numel()});
}



}   // end namespace otter

#endif /* Loop_hpp */
