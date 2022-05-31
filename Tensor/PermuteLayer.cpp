//
//  PermuteLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/1.
//

#include "PermuteLayer.hpp"
#include "TensorFactory.hpp"
#include "TensorMaker.hpp"

namespace otter {

PermuteLayer::PermuteLayer() {
    one_blob_only = true;
    support_inplace = true;
}

int PermuteLayer::parse_param(LayerOption& option, ParamDict& pd) {
    pd.clear();
    
    Tensor permute;
    
    if (!opt_check_string(option, "permute")) {
        permute = otter::range(0, bottom_shapes[0][0].size(0), 1, otter::ScalarType::Int);
    } else {
        int permute_length = (int)std::count(option["permute"].begin(), option["permute"].end(), ',') + 1;
        permute = otter::empty({permute_length}, otter::ScalarType::Int);
        auto permute_a = permute.accessor<int, 1>();
        std::stringstream ss;
        ss << option["permute"];
        int n; char c;
        for (const auto i : otter::irange(permute_length)) {
            ss >> n >> c;
            permute_a[i] = n;
        }
    }
    
    pd.set((int)PermuteParam::Permute, permute);
    
    return 0;
}

int PermuteLayer::compute_output_shape(ParamDict& pd) {
    int input_dim = bottom_shapes[0][0].size(0);
    auto input_shape_a = bottom_shapes[0].accessor<int, 2>()[0];
    auto permute = pd.get((int)PermuteParam::Permute, otter::full({1}, -1, otter::ScalarType::Int));
    auto permute_a = permute.accessor<int, 1>();
    
    std::vector<int> output_shape(input_dim);
    for (const auto i : otter::irange(input_dim)) {
        output_shape[i] = input_shape_a[permute_a[i]];
    }
    auto output_shape_t = otter::tensor(output_shape, otter::ScalarType::Int).view({1, -1});
    
    pd.set((int)PermuteParam::Permute, permute);
    pd.set(OUTPUT_SHAPE_HINT, output_shape_t);
    
    return 0;
}

int PermuteLayer::load_param(const ParamDict &pd) {
    auto output_permute = pd.get((int)PermuteParam::Permute, Tensor());
    auto output_permute_a = output_permute.accessor<int, 1>();
    
    permute.resize(output_permute.size(0));
    
    for (const auto i : otter::irange(output_permute.size(0))) {
        permute[i] = output_permute_a[i];
    }
    
    return 0;
}

int PermuteLayer::forward_inplace(Tensor& bottom_blob, const NetOption& opt) const {
    bottom_blob = bottom_blob.permute(permute).contiguous();
    
    return 0;
}

}   // end namespace otter
