//
//  ExpandUtils.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/2/12.
//

#include "ExpandUtils.hpp"

namespace otter {

// For TensorIterator
template <typename Container>
Container infer_size_impl(IntArrayRef a, IntArrayRef b) {
    size_t dimsA = a.size();
    size_t dimsB = b.size();
    size_t ndim = dimsA > dimsB ? dimsA : dimsB;
    Container expandedSizes(ndim);
    
    for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
        ptrdiff_t offset = ndim - 1 - i;
        ptrdiff_t dimA = dimsA - 1 - offset;
        ptrdiff_t dimB = dimsB - 1 - offset;
        int64_t sizeA = (dimA >= 0) ? a[dimA] : 1;
        int64_t sizeB = (dimB >= 0) ? b[dimB] : 1;
        
        assert(sizeA == sizeB || sizeA == 1 || sizeB == 1);
        
        expandedSizes[i] = sizeA == 1 ? sizeB : sizeA;
    }
    
    return expandedSizes;
}

DimVector infer_size_dimvector(IntArrayRef a, IntArrayRef b) {
    return infer_size_impl<DimVector>(a, b);
}
//

// For Tensor view
template <typename ResultVec>
inline void infer_size_impl(IntArrayRef shape, int64_t numel, ResultVec &res) {
    int64_t newsize = 1;
    int64_t *infer_dim = NULL;
    for (int64_t dim = 0, ndim = shape.size(); dim != ndim; dim++) {
        if (shape[dim] == -1) {
            if (infer_dim) {
                // Only one dimension can be inferred
                assert(false);
            }
            infer_dim = new int64_t;
            *infer_dim = dim;
        } else if (shape[dim] >= 0) {
            newsize *= shape[dim];
        } else {
            // Invalid shape dimension : < 0 and != -1
            assert(false);
        }
    }
    
    if (numel == newsize || (infer_dim && newsize > 0 && numel % newsize == 0)) {
        if (infer_dim) {
            // Cannot reshape tensor of 0 elements
            assert(newsize != 0);
            res[*infer_dim] = numel / newsize;
        }
        delete infer_dim;
        return;
    }
    
    delete infer_dim;
    // Invalid shape for input of size
    assert(false);
}

DimVector infer_size_dimvector(IntArrayRef shape, int64_t numel) {
    auto res = DimVector(shape);
    infer_size_impl(shape, numel, res);
    return res;
}
//

template<typename Container>
inline InferExpandGeometryResult<Container> inferExpandGeometryImpl(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides,
    IntArrayRef sizes) {
    int64_t ndim = sizes.size();
    int64_t tensor_dim = tensor_sizes.size();

    if (tensor_dim == 0) {
        return InferExpandGeometryResult<Container>(sizes, ndim);
    }

    InferExpandGeometryResult<Container> result(ndim);
    auto& expandedSizes = result.sizes;
    auto& expandedStrides = result.strides;

    // create a new geometry for the tensors
    for (int64_t i = ndim - 1; i >= 0; --i) {
        int64_t offset = ndim - 1 - i;
        int64_t dim = tensor_dim - 1 - offset;
        int64_t size = (dim >= 0) ? tensor_sizes[dim] : 1;
        int64_t stride = (dim >= 0) ? tensor_strides[dim] : expandedSizes[i + 1] * expandedStrides[i + 1];
        int64_t targetSize = sizes[i];
        if (targetSize == -1) {
            OTTER_CHECK(dim >= 0, "The expanded size of the tensor (", targetSize, ") isn't allowed in a leading, non-existing dimension ", i);
            targetSize = size;
        }
        if (size != targetSize) {
            OTTER_CHECK(size == 1, "The expanded size of the tensor (", targetSize, ") must match the existing size (", size, ") at non-singleton dimension ", i, ".  Target sizes: ", sizes, ".  Tensor sizes: ", tensor_sizes);
            size = targetSize;
            stride = 0;
        }
        expandedSizes[i] = size;
        expandedStrides[i] = stride;
    }
    return result;
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> inferExpandGeometry(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides,
    IntArrayRef sizes) {
    auto result = inferExpandGeometryImpl<std::vector<int64_t>>(tensor_sizes, tensor_strides, sizes);
    return std::make_tuple(std::move(result.sizes), std::move(result.strides));
}

InferExpandGeometryResult<DimVector> inferExpandGeometry_dimvector(
    IntArrayRef tensor_sizes,
    IntArrayRef tensor_strides,
    IntArrayRef sizes) {
    return inferExpandGeometryImpl<DimVector>(tensor_sizes, tensor_strides, sizes);
}

}
