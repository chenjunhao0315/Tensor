//
//  Scalar.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/30.
//

#ifndef Scalar_hpp
#define Scalar_hpp

#include <stdexcept>

#include "DType.hpp"
#include "ScalarType.hpp"

namespace otter {

class Scalar {
public:
    Scalar() : Scalar(int64_t(0)) {}
   
// Constructor
#define DEFINE_IMPLICIT_CTOR(type, name) \
Scalar(type vv) : Scalar(vv, true) {}
    OTTER_ALL_SCALAR_TYPES_WO_BOOL(DEFINE_IMPLICIT_CTOR)
#undef DEFINE_IMPLICIT_CTOR
    
    template <typename T, typename std::enable_if<std::is_same<T, bool>::value, bool>::type* = nullptr>
    Scalar(T vv) : tag(Tag::HAS_BOOL) {
        v.i = convert<int64_t, bool>(vv);
    }
//
    
// Accessor
#define DEFINE_ACCESSOR(type, name) \
type to##name() const {                                                 \
    if (Tag::HAS_DOUBLE == tag) {                                       \
        return checked_convert<type, double>(v.d, #type);               \
    }                                                                   \
    if (Tag::HAS_BOOL == tag) {                                         \
        return checked_convert<type, bool>(v.i, #type);                 \
    } else {                                                            \
        return checked_convert<type, int64_t>(v.i, #type);              \
    }                                                                   \
}
    OTTER_ALL_SCALAR_TYPES(DEFINE_ACCESSOR)
#undef DEFINE_ACCESSOR
//
    
    template <typename T>
    T to() const;
    
    bool isFloatingPoint() const {
        return Tag::HAS_DOUBLE == tag;
    }

    bool isIntegral(bool includeBool) const {
        return Tag::HAS_INTEGRAL == tag || (includeBool && isBoolean());
    }

    bool isBoolean() const {
        return Tag::HAS_BOOL == tag;
    }
    
    Scalar operator-() const;
    Scalar log() const;
    
    bool equal(bool num) const {
        if (isBoolean()) {
            return static_cast<bool>(v.i) == num;
        } else {
            return false;
        }
    }

    ScalarType type() const {
        if (isFloatingPoint()) {
            return ScalarType::Double;
        } else if (isIntegral(false)) {
            return ScalarType::Long;
        } else if (isBoolean()) {
            return ScalarType::Bool;
        } else {
            throw std::runtime_error("Unknown scalar type.");
        }
      }
    
    
private:
    template <typename T, typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value, bool>::type* = nullptr>
    Scalar(T vv, bool) : tag(Tag::HAS_INTEGRAL) {
        v.i = convert<decltype(v.i), T>(vv);
    }
    
    template <typename T, typename std::enable_if<!std::is_integral<T>::value, bool>::type* = nullptr>
    Scalar(T vv, bool) : tag(Tag::HAS_DOUBLE) {
        v.d = convert<decltype(v.d), T>(vv);
    }
    
    
    enum class Tag { HAS_INTEGRAL, HAS_DOUBLE, HAS_BOOL };
    Tag tag;
    union v_t {
        double d;
        int64_t i;
        v_t() {} // default constructor
    } v;
};

#define DEFINE_TO(T, name)          \
template <>                         \
inline T Scalar::to<T>() const {    \
    return to##name();              \
}
    OTTER_ALL_SCALAR_TYPES(DEFINE_TO)
#undef DEFINE_TO

}   // namespace otter

#endif /* Scalar_hpp */
