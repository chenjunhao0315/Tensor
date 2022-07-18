//
//  AccmulateType.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/18.
//

#include "AccmulateType.hpp"

namespace otter {

ScalarType toAccumulateType(ScalarType type, bool is_cuda) {
  switch (type) {
#define DEFINE_CASE(scalar_t, TypeNum)                                  \
    case ScalarType::TypeNum:                                           \
      return is_cuda ?                                                  \
          CppTypeToScalarType<acc_type<scalar_t, true>>::value :    \
          CppTypeToScalarType<acc_type<scalar_t, false>>::value;
    OTTER_ALL_SCALAR_TYPES(DEFINE_CASE)
#undef DEFINE_CASE
    default: OTTER_INTERNAL_ASSERT(false, "Unrecognized ScalarType: ", type);
  }
}

}   // end namespace otter
