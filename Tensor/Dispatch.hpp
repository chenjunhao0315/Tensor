//
//  Dispatch.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/30.
//

#ifndef Dispatch_hpp
#define Dispatch_hpp

#include "ScalarType.hpp"

namespace otter {

namespace detail {

inline ScalarType scalar_type(ScalarType s) {
    return s;
}

}

#define OTTER_CASE_TYPE_HINT(ENUM_TYPE, TYPE, HINT, ...)    \
    case ENUM_TYPE: {                                       \
        using HINT = TYPE;                                  \
        return __VA_ARGS__();                               \
    }


#define OTTER_PRIVATE_CASE_TYPE_USING_HINT(enum_type, HINT, ...)    \
    case enum_type: {                                               \
        using HINT = otter::ScalarTypeToCPPTypeT<enum_type>;        \
        return __VA_ARGS__();                                       \
    }

#define OTTER_CASE_TYPE(ENUM_TYPE, TYPE, ...)   \
    OTTER_CASE_TYPE_HINT(ENUM_TYPE, TYPE, scalar_t, __VA_ARGS__)

#define OTTER_DISPATCH_ALL_TYPES(TYPE, NAME, ...)               \
    [&] {                                                       \
    const auto& the_type = TYPE;                                \
    ScalarType _st = otter::detail::scalar_type(the_type);           \
    switch(_st) {                                               \
        OTTER_CASE_TYPE(ScalarType::Byte, uint8_t, __VA_ARGS__)      \
        OTTER_CASE_TYPE(ScalarType::Char, int8_t, __VA_ARGS__)       \
        OTTER_CASE_TYPE(ScalarType::Short, int16_t, __VA_ARGS__)     \
        OTTER_CASE_TYPE(ScalarType::Int, int, __VA_ARGS__)           \
        OTTER_CASE_TYPE(ScalarType::Long, int64_t, __VA_ARGS__)      \
        OTTER_CASE_TYPE(ScalarType::Float, float, __VA_ARGS__)       \
        OTTER_CASE_TYPE(ScalarType::Double, double, __VA_ARGS__)     \
        default:                                                \
            OTTER_ERROR(                                                    \
            '"',                                                            \
            NAME,                                                           \
            "\" not implemented for '",                                     \
            toString(_st),                                                  \
            "'");                                                           \
    }                                                           \
    }()

#define OTTER_DISPATCH_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)               \
    [&] {                                                       \
    const auto& the_type = TYPE;                                \
    ScalarType _st = otter::detail::scalar_type(the_type);           \
    switch(_st) {                                               \
        OTTER_CASE_TYPE(ScalarType::Byte, uint8_t, __VA_ARGS__)      \
        OTTER_CASE_TYPE(ScalarType::Char, int8_t, __VA_ARGS__)       \
        OTTER_CASE_TYPE(ScalarType::Short, int16_t, __VA_ARGS__)     \
        OTTER_CASE_TYPE(ScalarType::Int, int, __VA_ARGS__)           \
        OTTER_CASE_TYPE(ScalarType::Long, int64_t, __VA_ARGS__)      \
        OTTER_CASE_TYPE(ScalarType::Float, float, __VA_ARGS__)       \
        OTTER_CASE_TYPE(ScalarType::Double, double, __VA_ARGS__)     \
        OTTER_CASE_TYPE(SCALARTYPE, decltype(ScalarTypeToCPPType<SCALARTYPE>::t), __VA_ARGS__)  \
        default:                                                \
            OTTER_ERROR(                                                    \
            '"',                                                            \
            NAME,                                                           \
            "\" not implemented for '",                                     \
            toString(_st),                                                  \
            "'");                                                           \
    }                                                           \
    }()

#define OTTER_DISPATCH_ALL_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)               \
    [&] {                                                       \
    const auto& the_type = TYPE;                                \
    ScalarType _st = otter::detail::scalar_type(the_type);           \
    switch(_st) {                                               \
        OTTER_CASE_TYPE(ScalarType::Byte, uint8_t, __VA_ARGS__)      \
        OTTER_CASE_TYPE(ScalarType::Char, int8_t, __VA_ARGS__)       \
        OTTER_CASE_TYPE(ScalarType::Short, int16_t, __VA_ARGS__)     \
        OTTER_CASE_TYPE(ScalarType::Int, int, __VA_ARGS__)           \
        OTTER_CASE_TYPE(ScalarType::Long, int64_t, __VA_ARGS__)      \
        OTTER_CASE_TYPE(ScalarType::Float, float, __VA_ARGS__)       \
        OTTER_CASE_TYPE(ScalarType::Double, double, __VA_ARGS__)     \
        OTTER_CASE_TYPE(SCALARTYPE1, decltype(ScalarTypeToCPPType<SCALARTYPE1>::t), __VA_ARGS__)  \
        OTTER_CASE_TYPE(SCALARTYPE2, decltype(ScalarTypeToCPPType<SCALARTYPE2>::t), __VA_ARGS__)  \
        default:                                                \
            OTTER_ERROR(                                                    \
            '"',                                                            \
            NAME,                                                           \
            "\" not implemented for '",                                     \
            toString(_st),                                                  \
            "'");                                                           \
    }                                                           \
    }()

#define OTTER_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...)               \
    [&] {                                                       \
    const auto& the_type = TYPE;                                \
    ScalarType _st = otter::detail::scalar_type(the_type);           \
    switch(_st) {                                               \
        OTTER_CASE_TYPE(ScalarType::Byte, uint8_t, __VA_ARGS__)      \
        OTTER_CASE_TYPE(ScalarType::Char, int8_t, __VA_ARGS__)       \
        OTTER_CASE_TYPE(ScalarType::Short, int16_t, __VA_ARGS__)     \
        OTTER_CASE_TYPE(ScalarType::Int, int, __VA_ARGS__)           \
        OTTER_CASE_TYPE(ScalarType::Long, int64_t, __VA_ARGS__)      \
        default:                                                \
            OTTER_ERROR(                                                    \
            '"',                                                            \
            NAME,                                                           \
            "\" not implemented for '",                                     \
            toString(_st),                                                  \
            "'");                                                           \
    }                                                           \
    }()

#define OTTER_DISPATCH_INTEGRAL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)               \
    [&] {                                                       \
    const auto& the_type = TYPE;                                \
    ScalarType _st = otter::detail::scalar_type(the_type);           \
    switch(_st) {                                               \
        OTTER_CASE_TYPE(ScalarType::Byte, uint8_t, __VA_ARGS__)      \
        OTTER_CASE_TYPE(ScalarType::Char, int8_t, __VA_ARGS__)       \
        OTTER_CASE_TYPE(ScalarType::Short, int16_t, __VA_ARGS__)     \
        OTTER_CASE_TYPE(ScalarType::Int, int, __VA_ARGS__)           \
        OTTER_CASE_TYPE(ScalarType::Long, int64_t, __VA_ARGS__)      \
        OTTER_CASE_TYPE(SCALARTYPE, decltype(ScalarTypeToCPPType<SCALARTYPE>::t), __VA_ARGS__)  \
        default:                                                \
            OTTER_ERROR(                                                    \
            '"',                                                            \
            NAME,                                                           \
            "\" not implemented for '",                                     \
            toString(_st),                                                  \
            "'");                                                           \
    }                                                           \
    }()

#define OTTER_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)               \
    [&] {                                                       \
    const auto& the_type = TYPE;                                \
    ScalarType _st = otter::detail::scalar_type(the_type);           \
    switch(_st) {                                               \
        OTTER_CASE_TYPE(ScalarType::Float, float, __VA_ARGS__)       \
        OTTER_CASE_TYPE(ScalarType::Double, double, __VA_ARGS__)     \
        default:                                                \
            OTTER_ERROR(                                                    \
            '"',                                                            \
            NAME,                                                           \
            "\" not implemented for '",                                     \
            toString(_st),                                                  \
            "'");                                                           \
    }                                                           \
    }()

#define OTTER_DISPATCH_FLOATING_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)               \
    [&] {                                                       \
    const auto& the_type = TYPE;                                \
    ScalarType _st = otter::detail::scalar_type(the_type);           \
    switch(_st) {                                               \
        OTTER_CASE_TYPE(ScalarType::Float, float, __VA_ARGS__)       \
        OTTER_CASE_TYPE(ScalarType::Double, double, __VA_ARGS__)     \
        OTTER_CASE_TYPE(SCALARTYPE, decltype(ScalarTypeToCPPType<SCALARTYPE>::t), __VA_ARGS__)  \
        default:                                                \
            OTTER_ERROR(                                                    \
            '"',                                                            \
            NAME,                                                           \
            "\" not implemented for '",                                     \
            toString(_st),                                                  \
            "'");                                                           \
    }                                                           \
    }()

#define OTTER_DISPATCH_ALL_TYPES_HINT(TYPE, HINT, NAME, ...)    \
    [&] {                                                       \
    const auto& the_type = TYPE;                                \
    ScalarType _st = otter::detail::scalar_type(the_type);           \
    switch(_st) {                                               \
        OTTER_CASE_TYPE_HINT(ScalarType::Byte, uint8_t, HINT, __VA_ARGS__)      \
        OTTER_CASE_TYPE_HINT(ScalarType::Char, int8_t, HINT, __VA_ARGS__)       \
        OTTER_CASE_TYPE_HINT(ScalarType::Short, int16_t, HINT, __VA_ARGS__)     \
        OTTER_CASE_TYPE_HINT(ScalarType::Int, int, HINT, __VA_ARGS__)           \
        OTTER_CASE_TYPE_HINT(ScalarType::Long, int64_t, HINT, __VA_ARGS__)      \
        OTTER_CASE_TYPE_HINT(ScalarType::Float, float, HINT, __VA_ARGS__)       \
        OTTER_CASE_TYPE_HINT(ScalarType::Double, double, HINT, __VA_ARGS__)     \
        default:                                                \
            OTTER_ERROR(                                                    \
            '"',                                                            \
            NAME,                                                           \
            "\" not implemented for '",                                     \
            toString(_st),                                                  \
            "'");                                                           \
    }                                                           \
    }()

#define OTTER_DISPATCH_ALL_TYPES_STD_MATH(TYPE, NAME, ...)               \
    [&] {                                                       \
    const auto& the_type = TYPE;                                \
    ScalarType _st = otter::detail::scalar_type(the_type);           \
    switch(_st) {                                               \
        OTTER_CASE_TYPE(ScalarType::Float, float, __VA_ARGS__)       \
        OTTER_CASE_TYPE(ScalarType::Double, double, __VA_ARGS__)     \
        default:                                                \
            OTTER_ERROR(                                                    \
            '"',                                                            \
            NAME,                                                           \
            "\" not implemented for '",                                     \
            toString(_st),                                                  \
            "'");                                                           \
    }                                                           \
    }()

#define OTTER_DISPATCH_SWITCH(TYPE, NAME, ...)                              \
  [&] {                                                                     \
    const auto& the_type = TYPE;                                            \
    /* don't use TYPE again in case it is an expensive or side-effect op */ \
    otter::ScalarType _st = otter::detail::scalar_type(the_type);           \
    switch (_st) {                                                          \
      __VA_ARGS__                                                           \
      default:                                                              \
        OTTER_ERROR(                                                    \
        '"',                                                            \
        NAME,                                                           \
        "\" not implemented for '",                                     \
        toString(_st),                                                  \
        "'");                                                           \
    }                                                                       \
  }()

#define OTTER_DISPATCH_INDEX_TYPES(TYPE, NAME, ...)     \
  OTTER_DISPATCH_SWITCH(                                \
      TYPE,                                             \
      NAME,                                             \
      OTTER_PRIVATE_CASE_TYPE_USING_HINT(               \
          otter::ScalarType::Int, index_t, __VA_ARGS__) \
              OTTER_PRIVATE_CASE_TYPE_USING_HINT(       \
              otter::ScalarType::Long, index_t, __VA_ARGS__))

}   // end namespace otter


#endif /* Dispatch_hpp */
