//
//  TensorConversion.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/6.
//

#include "TensorFactory.hpp"
#include "TensorConversion.hpp"
#include "Parallel.hpp"
#include "Loop.hpp"
#include "Dispatch.hpp"

namespace otter {
namespace native {
Tensor to(const Tensor& self, ScalarType dtype) {
    auto result = empty_like(self, dtype);
    
    otter::parallel_for(0, self.numel(), 0, [&](int64_t begin, int64_t end) {
        OTTER_DISPATCH_ALL_TYPES_HINT(self.scalar_type(), scalar_t_1, "to_dtype_1", [&]() {
            scalar_t_1 *type1_data = (scalar_t_1*)self.data_ptr();
            OTTER_DISPATCH_ALL_TYPES_HINT(dtype, scalar_t_2, "to_dtype_2", [&]() {
                scalar_t_2 *type2_data = (scalar_t_2*)result.data_ptr();
                for (auto i : otter::irange(begin, end)) {
                    type2_data[i] = type1_data[i];
                }
            });
        });
    });
    
    return result;
}
}





}
