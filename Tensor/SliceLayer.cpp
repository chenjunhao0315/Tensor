//
//  SliceLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/5/31.
//

#include "SliceLayer.hpp"
#include "TensorFactory.hpp"
#include "TensorShape.hpp"

namespace otter {

SliceLayer::SliceLayer() {
    one_blob_only = false;
    support_inplace = false;
}

int SliceLayer::parse_param(LayerOption& option, ParamDict& pd) {
    pd.clear();
    int axis = opt_find_int(option, "axis", 1);
    
    Tensor slice;
    
    if (!opt_check_string(option, "slice")) {
        slice = otter::zeros({1}, otter::ScalarType::Int);
    } else {
        int num_slice = (int)std::count(option["slice"].begin(), option["slice"].end(), ',') + 1;
        int num_output = (int)std::count(option["output"].begin(), option["output"].end(), ',') + 1;
        OTTER_CHECK(num_slice == num_output, "[SliceLayer] Number of slice need to equal to number of output, but get ", num_slice, " != ", num_output);
        slice = otter::zeros({num_slice}, otter::ScalarType::Int);
        auto slice_a = slice.accessor<int, 1>();
        std::stringstream ss;
        ss << option["slice"];
        int n;
        char c;
        for (const auto i : otter::irange(num_slice)) {
            ss >> n >> c;
            slice_a[i] = n;
        }
    }
    
    pd.set((int)SliceParam::Axis, axis);
    pd.set((int)SliceParam::Slice, slice);
    
    return 0;
}

int SliceLayer::compute_output_shape(ParamDict& pd) {
    int axis = pd.get((int)SliceParam::Axis, 1);
    auto input_shape = bottom_shapes[0][0];
    auto input_shape_a = input_shape.accessor<int, 1>();
    auto slice = pd.get((int)SliceParam::Slice, otter::zeros({1}, otter::ScalarType::Int));
    auto slice_a = slice.accessor<int, 1>();
    
    int total_slice = input_shape_a[axis];
    std::vector<Tensor> output_shape_v;
    
    for (auto i : otter::irange(slice.size(0))) {
        int s = slice_a[i];
        auto out_shape = input_shape.clone();
        out_shape[axis] = slice_a[i] = (s > 0) ? s : total_slice;
        output_shape_v.push_back(out_shape.view({1, -1}));
        total_slice -= s;
    }
    
    auto output_shape = otter::native::cat(output_shape_v, 0);
    
    pd.set(OUTPUT_SHAPE_HINT, output_shape);
    
    return 0;
}

int SliceLayer::load_param(const ParamDict &pd) {
    
    axis = pd.get((int)SliceParam::Axis, 1);
    slice = pd.get((int)SliceParam::Slice, otter::zeros({1}, otter::ScalarType::Int));
    
    return 0;
}

int SliceLayer::forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& /*opt*/) const {
    auto slice_a = slice.accessor<int, 1>();
    
    int start = 0;
    int end = 0;
    for (const auto i : otter::irange(slice.size(0))) {
        end = start + slice_a[i];
        top_blobs[i] = otter::native::slice(bottom_blobs[0], axis, start, end);
        start = end;
    }
    
    return 0;
}


}   // end namespace otter
