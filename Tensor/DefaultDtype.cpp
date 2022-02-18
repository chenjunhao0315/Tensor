//
//  DefaultDtype.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/18.
//

#include "DefaultDtype.hpp"

namespace otter {

static auto default_dtype = TypeMeta::Make<float>();
static auto default_dtype_as_scalartype = default_dtype.toScalarType();

void set_default_dtype(TypeMeta dtype) {
    default_dtype = dtype;
    default_dtype_as_scalartype = default_dtype.toScalarType();
}

const TypeMeta get_default_dtype() {
    return default_dtype;
}

ScalarType get_default_dtype_as_scalartype() {
    return default_dtype_as_scalartype;
}

}   // end namespace otter
