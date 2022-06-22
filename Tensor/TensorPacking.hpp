//
//  TensorPacking.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/16.
//

#ifndef TensorPacking_hpp
#define TensorPacking_hpp

#include "Tensor.hpp"

namespace otter {

int64_t get_elempack_from_type(const ScalarType& src);

ScalarType get_update_scalarType(const ScalarType& src, int out_elempack);

void convertPacking(const Tensor& src, Tensor& dst, int out_elempack);

}   // end namespace otter

#endif /* TensorPacking_hpp */
