//
//  TypeProperties.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/24.
//

#ifndef TypeProperties_hpp
#define TypeProperties_hpp

#include "ArrayRef.hpp"
#include "ScalarType.hpp"

namespace otter {
namespace native {

struct ResultTypeState {
    ScalarType dimResult = ScalarType::Undefined;
    ScalarType wrappedResult = ScalarType::Undefined;
    ScalarType zeroResult = ScalarType::Undefined;
};

ResultTypeState update_result_type_state(const Tensor& tensor, const ResultTypeState& in_state);
ScalarType result_type(const ResultTypeState& state);
ScalarType result_type(TensorList tensors);

}
}

#endif /* TypeProperties_hpp */
