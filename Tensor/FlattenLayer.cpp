//
//  FlattenLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/7/7.
//

#include "FlattenLayer.hpp"
#include "TensorMaker.hpp"

namespace otter {

FlattenLayer::FlattenLayer() {
    one_blob_only = true;
    support_inplace = true;
    
}

int FlattenLayer::parse_param(LayerOption& option, ParamDict& pd) {
    int start_dim = opt_find_int(option, "start_dim", 0);
    int end_dim = opt_find_int(option, "end_dim", -1);
    
    pd.set((int)FlattenParam::StartDim, start_dim);
    pd.set((int)FlattenParam::EndDim, end_dim);
    
    return 0;
}

int FlattenLayer::compute_output_shape(ParamDict& pd) {
    auto shape_a = bottom_shapes[0].accessor<int, 2>()[0];
    
    int dims = bottom_shapes[0][0].numel();
    int start_dim = pd.get((int)FlattenParam::StartDim, 0);
    int end_dim = pd.get((int)FlattenParam::EndDim, -1);
    
    start_dim = maybe_wrap_dim(start_dim, dims);
    end_dim = maybe_wrap_dim(end_dim, dims);
    
    if (dims == 0) {
        pd.set(OUTPUT_SHAPE_HINT, otter::tensor({1}, ScalarType::Int).view({1, -1}));
        
        return 0;
    }
    if (start_dim == end_dim) {
        pd.set(OUTPUT_SHAPE_HINT, bottom_shapes[0][0].clone().view({1, -1}));;
        
        return 0;
    }
    
    int slice_numel = 1;
    for (int i = start_dim; i <= end_dim; ++i) {
        slice_numel *= shape_a[i];
    }
    std::vector<int64_t> shape;
    shape.reserve(dims - end_dim + start_dim);
    for (const auto i : otter::irange(start_dim)) {
        shape.push_back(shape_a[i]);
    }
    shape.push_back(slice_numel);
    for (const auto i : otter::irange(end_dim + 1, dims)) {
        shape.push_back(shape_a[i]);
    }
    
    pd.set(OUTPUT_SHAPE_HINT, otter::tensor({shape}, otter::ScalarType::Int).view({1, -1}));
    
    return 0;
}

int FlattenLayer::load_param(const ParamDict& pd) {
    start_dim = pd.get((int)FlattenParam::StartDim, 0);
    end_dim = pd.get((int)FlattenParam::EndDim, -1);
    
    return 0;
}

int FlattenLayer::forward_inplace(Tensor& bottom_blob, const NetOption& opt) const {
    bottom_blob = bottom_blob.flatten(start_dim, end_dim);
    
    return 0;
}

}   // end namespace otter
