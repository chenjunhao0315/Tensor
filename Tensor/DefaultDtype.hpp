//
//  DefaultDtype.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/18.
//

#ifndef DefaultDtype_hpp
#define DefaultDtype_hpp

#include "DType.hpp"
#include "ScalarType.hpp"

namespace otter {

void set_default_dtype(TypeMeta dtype);
const TypeMeta get_default_dtype();
ScalarType get_default_dtype_as_scalartype();



}   // end namespace otter

#endif /* DefaultDtype_hpp */
