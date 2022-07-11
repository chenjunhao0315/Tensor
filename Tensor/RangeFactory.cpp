//
//  RangeFactory.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/9.
//

#include "Tensor.hpp"
#include "TensorIterator.hpp"
#include "RangeFactory.hpp"
#include "Dispatch.hpp"
#include "Parallel.hpp"

namespace otter {

Tensor& linspace_out(const Scalar& start, const Scalar& end, int64_t steps, Tensor& result) {
    OTTER_CHECK(steps > 0, "Steps must > 0 but get ", steps);
    
    if (result.numel() != steps) {
        result.resize_({steps});
    }
    
    if (steps == 0) {
        // skip
    } else if (steps == 1) {
        result.fill_(start);
    } else {
        Tensor r = result;
        auto iter = TensorIterator::borrowing_nullary_op(r);
        linspace_stub(Device::CPU, iter, start, end, steps);
      }
    
    return result;
}

Tensor& range_out(const Scalar& start, const Scalar& end, const Scalar& step, Tensor& result) {
    OTTER_DISPATCH_ALL_TYPES(result.scalar_type(), "range_cpu", [&] {
        // TODO: accumulate type
        auto xstart = start.to<int>();
        auto xend = end.to<int>();
        auto xstep = step.to<int>();
        
        OTTER_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
        OTTER_CHECK(std::isfinite(static_cast<double>(xstart)) &&
                    std::isfinite(static_cast<double>(xend)),
                    "unsupported range: ", xstart, " -> ", xend);
        OTTER_CHECK(((xstep > 0) && (xend >= xstart)) || ((xstep < 0) && (xend <= xstart)),
                    "upper bound and larger bound inconsistent with step sign");
        int64_t size = static_cast<int64_t>(((xend - xstart) / xstep) + 1);
        if (result.numel() != size) {
            result.resize_({size});
        }
        Tensor r = result.is_contiguous() ? result : result.contiguous();
        scalar_t *data_ptr = r.data_ptr<scalar_t>();

        otter::parallel_for(0, size, otter::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
            int64_t is = p_begin;
            for (int64_t i = p_begin; i < p_end; ++i, ++is) {
                data_ptr[i] = static_cast<scalar_t>(xstart + is * xstep);
            }
        });
        if (!result.is_contiguous()) {
            result.copy_(r);
        }
    });
    return result;
}

Tensor& arange_out(const Scalar& start, const Scalar& end, const Scalar& step, Tensor& result) {
    OTTER_DISPATCH_ALL_TYPES(result.scalar_type(), "arange_cpu", [&]() {
        auto xstart = start.to<scalar_t>();
        auto xend = end.to<scalar_t>();
        auto xstep = step.to<scalar_t>();

        OTTER_CHECK(xstep > 0 || xstep < 0, "step must be nonzero");
        OTTER_CHECK(std::isfinite(static_cast<double>(xstart)) &&
                    std::isfinite(static_cast<double>(xend)),
                    "unsupported range: ", xstart, " -> ", xend);
        OTTER_CHECK(((xstep > 0) && (xend >= xstart)) || ((xstep < 0) && (xend <= xstart)),
                    "upper bound and larger bound inconsistent with step sign");

        // we use double precision for (start - end) / step
        // to compute size_d for consistency across devices.
        // The problem with using accscalar_t is that accscalar_t might be float32 on gpu for a float32 scalar_t,
        // but double on cpu for the same,
        // and the effective output size starts differing on CPU vs GPU because of precision issues, which
        // we dont want.
        // the corner-case we do want to take into account is int64_t, which has higher precision than double
        double size_d;
        // NOLINTNEXTLINE(bugprone-branch-clone)
        if (std::is_same<scalar_t, int64_t>::value) {
            int64_t sgn = (xstep > 0) - (xstep < 0);
            size_d = std::ceil((xend - xstart + xstep - sgn) / xstep);
        } else {
            size_d = std::ceil(static_cast<double>(end.to<double>() - start.to<double>())
                               / step.to<double>());
        }

        OTTER_CHECK(size_d >= 0 && size_d <= static_cast<double>(std::numeric_limits<int64_t>::max()),
                    "invalid size, possible overflow?");

        int64_t size = static_cast<int64_t>(size_d);
        int64_t numel = result.numel();

        if (numel != size) {
            if(numel > 1){
                printf("The number of elements in the out tensor of shape!\n");
            }
            result.resize_({size});
        }

        Tensor r = result.is_contiguous() ? result : result.contiguous();
        auto iter = TensorIterator::borrowing_nullary_op(r);
        arange_stub(Device::CPU, iter, start, size, step);
        if (!result.is_contiguous()) {
            result.copy_(r);
        }
    });

    return result;
}

DEFINE_DISPATCH(arange_stub);
DEFINE_DISPATCH(linspace_stub);

}
