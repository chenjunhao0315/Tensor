//
//  TensorIteratorDynamicCasting.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/16.
//

#ifndef TensorIteratorDynamicCasting_hpp
#define TensorIteratorDynamicCasting_hpp

#include "TensorIterator.hpp"
#include "C++17.hpp"
#include "Function_Trait.hpp"

namespace otter {

template<typename func_t, int nargs=function_traits<func_t>::arity>
struct needs_dynamic_casting {
  static bool check(TensorIterator& iter) {
    using traits = function_traits<func_t>;
    using cpp_type = typename traits::template arg<nargs - 1>::type;
    using cpp_map = otter::CppTypeToScalarType<cpp_type>;

    if (iter.input_dtype(nargs-1) != cpp_map::value) {
      return true;
    }
    return needs_dynamic_casting<func_t, nargs - 1>::check(iter);
  }
};

template<typename func_t>
struct needs_dynamic_casting<func_t, 0> {
  static bool check(TensorIterator& iter) {
    using traits = function_traits<func_t>;
    using cpp_type = typename traits::result_type;

    // we could assert output numbers are correct here, but checks
    // (including arity) are currently pushed outside of this struct.
    return otter::if_constexpr<std::is_void<cpp_type>::value>([]() {
      return false;
    }, /* else */ [&](auto _) {
      // decltype(_) is used to delay computation
      using delayed_type = typename decltype(_)::template type_identity<cpp_type>;
      return iter.dtype(0) != otter::CppTypeToScalarType<delayed_type>::value;
    });
  }
};

}

#endif /* TensorIteratorDynamicCasting_hpp */
