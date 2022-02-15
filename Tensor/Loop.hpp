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


namespace otter {

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

template <typename func_t>
void cpu_kernel(TensorIterator& iter, func_t&& op, int64_t grain_size = otter::GRAIN_SIZE) {
    using traits = function_traits<func_t>;
    
    assert(iter.ninputs() == traits::arity);
    assert(iter.noutputs() == 1);
    
    iter.for_each([&](char** data, const int64_t* strides, int64_t n) {
        basic_loop(data, strides, 0, n, std::forward<func_t>(op));
    }, grain_size);
    iter.cast_outputs();
}

template <typename func_t>
void cpu_serial_kernel(TensorIterator& iter, func_t&& op, const Range& range) {
    using traits = function_traits<func_t>;
    constexpr bool result_void = std::is_void<typename traits::result_type>::value;
    assert(iter.ninputs() == traits::arity && ((result_void && iter.noutputs() == 0) || (!result_void && iter.noutputs() == 1)));

    iter.serial_for_each([&](char** data, const int64_t* strides, int64_t n) {
        basic_loop(data, strides, 0, n, std::forward<func_t>(op));
    }, range);
    iter.cast_outputs();
}

template <typename func_t>
void cpu_serial_kernel(TensorIterator& iter, func_t&& op) {
    cpu_serial_kernel(iter, op, {0, iter.numel()});
}

}   // end namespace otter

#endif /* Loop_hpp */
