//
//  RangeFactoryKernel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/9.
//

#include "RangeFactory.hpp"
#include "RangeFactoryKernel.hpp"
#include "Dispatch.hpp"
#include "Loop.hpp"
#include "Parallel.hpp"

namespace otter {

void linspace_kernel(TensorIterator& iter, const Scalar& scalar_start, const Scalar& scalar_end, int64_t steps) {
    OTTER_DISPATCH_ALL_TYPES(iter.dtype(), "linspace_cpu", [&]() {
        using step_t = std::conditional_t<std::is_integral<scalar_t>::value, double, scalar_t>;
        const scalar_t start = scalar_start.to<scalar_t>();
        const scalar_t end = scalar_end.to<scalar_t>();
        
        const step_t step = (static_cast<step_t>(end) - static_cast<step_t>(start)) / (steps - 1);
        int64_t halfway = steps / 2;
        otter::parallel_for(0, steps, otter::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
            int64_t idx(p_begin);
            TensorIterator it(iter);
            cpu_serial_kernel(it, [start, end, step, halfway, steps, &idx]() -> scalar_t {
                if (idx < halfway) {
                    return start + step * (idx++);
                } else {
                    return end - step * (steps - (idx++) - 1);
                }
            }, {p_begin, p_end});
        });
    });
}

REGISTER_DISPATCH(linspace_stub, &linspace_kernel);

}
