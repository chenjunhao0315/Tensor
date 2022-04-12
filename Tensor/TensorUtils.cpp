//
//  TensorUtils.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/11.
//

#include "Tensor.hpp"
#include "Accumulator.hpp"
#include "TensorUtils.hpp"

namespace otter {

void check_dim_size(const Tensor& tensor, int64_t dim, int64_t dim_size, int64_t size) {
    (void)tensor;
    (void)dim;
    (void)dim_size;
    (void)size;
    assert(tensor.dim() == dim && tensor.size(dim_size) == size);
}

template <typename ResultVec, typename NewShapeVec>
ResultVec computeStride_impl(IntArrayRef oldshape, IntArrayRef oldstride, const NewShapeVec& newshape, ResultVec toResult(const IntArrayRef&)) {
    if (oldshape.empty()) {
        return ResultVec(newshape.size(), 1);
    }
    
    const int64_t numel = otter::multiply_integers(oldshape);
    if (numel == 0 && oldshape.equals(newshape)) {
        return toResult(oldstride);
    }
    
    ResultVec newstride(newshape.size());
    if (numel == 0) {
        for (int64_t view_d = newshape.size() - 1; view_d >= 0; view_d--) {
            if (view_d == (int64_t)(newshape.size() - 1)) {
                newstride[view_d] = 1;
            } else {
                newstride[view_d] = std::max<int64_t>(newshape[view_d+1], 1) * newstride[view_d+1];
            }
        }
        return newstride;
    }
    
    int64_t view_d = (int64_t)newshape.size() - 1;
    // stride for each subspace in the chunk
    int64_t chunk_base_stride = oldstride.back();
    // numel in current chunk
    int64_t tensor_numel = 1;
    int64_t view_numel = 1;
    for (int64_t tensor_d = oldshape.size() - 1; tensor_d >= 0; tensor_d--) {
        tensor_numel *= oldshape[tensor_d];
        // if end of tensor size chunk, check view
        if ((tensor_d == 0) || (oldshape[tensor_d - 1] != 1 && oldstride[tensor_d - 1] != tensor_numel * chunk_base_stride)) {
            while (view_d >= 0 && (view_numel < tensor_numel || newshape[view_d] == 1)) {
                newstride[view_d] = view_numel * chunk_base_stride;
                view_numel *= newshape[view_d];
                view_d--;
            }
            if (view_numel != tensor_numel) {
                return DimVector();
            }
            if (tensor_d > 0) {
                chunk_base_stride = oldstride[tensor_d - 1];
                tensor_numel = 1;
                view_numel = 1;
            }
        }
    }
    if (view_d != -1) {
        return DimVector();
    }
    return newstride;
}

DimVector computeStride(IntArrayRef oldshape, IntArrayRef oldstride, const DimVector& newshape) {
    auto toResult = [](const IntArrayRef& a) { return DimVector(a); };
    return computeStride_impl<DimVector, DimVector>(oldshape, oldstride, newshape, toResult);
}


}
