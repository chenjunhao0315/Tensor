//
//  FunctionRef.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef FunctionRef_hpp
#define FunctionRef_hpp

#include <stdio.h>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace otter {

template <typename Fn>
class FunctionRef;

template <typename Ret, typename ...Params>
class FunctionRef<Ret(Params...)> {
    Ret (*callback)(intptr_t callable, Params... params) = nullptr;
    intptr_t callable;
    
    template <typename Callable>
    static Ret callback_fn(intptr_t callable, Params... params) {
        return (*reinterpret_cast<Callable*>(callable))(std::forward<Params>(params)...);
    }
public:
    FunctionRef() = default;
    FunctionRef(std::nullptr_t) {}
    
    // Check
    template <typename Callable>
    FunctionRef(Callable&& callable, typename std::enable_if<!std::is_same<typename std::remove_reference<Callable>::type, FunctionRef>::value>::type* = nullptr, typename std::enable_if<std::is_convertible<typename std::result_of<Callable && (Params && ...)>::type, Ret>::value>::type* = nullptr) : callback(callback_fn<typename std::remove_reference<Callable>::type>), callable(reinterpret_cast<intptr_t>(&callable)) {}
    
    Ret operator()(Params... params) const {
        return callback(callable, std::forward<Params>(params)...);
    }
    
    operator bool() const {
        return callback;
    }
};


}

#endif /* FunctionRef_hpp */
