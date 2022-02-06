//
//  Utils.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "Utils.hpp"

namespace otter {

std::ostream& operator<<(std::ostream& out, const Range& range) {
    out << "Range[" << range.begin << ", " << range.end << "]";
    return out;
}

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

}   // end namespace otter
