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
#include "TypeCast.hpp"

namespace otter {
namespace native {
Tensor to(const Tensor& self, ScalarType dtype) {
    auto result = empty_like(self, dtype);
    
    otter::parallel_for(0, self.numel(), 0, [&](int64_t begin, int64_t end) {
        OTTER_DISPATCH_ALL_TYPES(self.scalar_type(), "to_src_t", [&]() {
            using src_t = scalar_t;
            src_t *src_data = (src_t*)self.data_ptr();
            OTTER_DISPATCH_ALL_TYPES(dtype, "to_dest_t", [&]() {
                scalar_t *dst_data = (scalar_t*)result.data_ptr();
                for (auto i : otter::irange(begin, end)) {
                    dst_data[i] = static_cast_with_inter_type<scalar_t, src_t>::apply(src_data[i]);
                }
            });
        });
    });
    
    return result;
}
}





}
