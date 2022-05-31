//
//  ReshapeLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/6/1.
//

#include "ReshapeLayer.hpp"
#include "TensorFactory.hpp"

namespace otter {

ReshapeLayer::ReshapeLayer() {
    one_blob_only = true;
    support_inplace = true;
}

int ReshapeLayer::parse_param(LayerOption& option, ParamDict& pd) {
    pd.clear();
    
    Tensor shape;
    
    if (!opt_check_string(option, "reshape")) {
        shape = otter::full({1}, -1, otter::ScalarType::Int);
    } else {
        int shape_length = (int)std::count(option["reshape"].begin(), option["reshape"].end(), ',') + 1;
        shape = otter::full({shape_length}, -1, otter::ScalarType::Int);
        auto shape_a = shape.accessor<int, 1>();
        std::stringstream ss;
        ss << option["reshape"];
        int n; char c;
        for (const auto i : otter::irange(shape_length)) {
            ss >> n >> c;
            shape_a[i] = n;
        }
    }
    
    pd.set((int)ReshapeParam::Shape, shape);
    
    return 0;
}

int ReshapeLayer::compute_output_shape(ParamDict& pd) {
    auto input_shape_a = bottom_shapes[0].accessor<int, 2>()[0];
    auto shape = pd.get((int)ReshapeParam::Shape, otter::full({1}, -1, otter::ScalarType::Int));
    auto shape_a = shape.accessor<int, 1>();
    
    int input_numel = 1;
    for (const auto i : otter::irange(bottom_shapes[0][0].size(0))) {
        input_numel *= input_shape_a[i];
    }

    int numel = 1;
    int auto_place_index = -1;

    for (const auto i : otter::irange(shape.size(0))) {
        int c = shape_a[i];
        if (c == -1 && auto_place_index == -1) {
            auto_place_index = i;
        } else if (c == -1) {
            OTTER_CHECK(false, "[ReshapeLayer] Ilegal shape!\n");
        } else {
            numel *= c;
        }
    }
    if (auto_place_index != -1) {
        shape_a[auto_place_index] = input_numel / numel;
    }
    
    pd.set((int)ReshapeParam::Shape, shape);
    pd.set(OUTPUT_SHAPE_HINT, shape.view({1, -1}));
    
    return 0;
}

int ReshapeLayer::load_param(const ParamDict &pd) {
    auto output_shape = pd.get((int)ReshapeParam::Shape, Tensor());
    auto output_shape_a = output_shape.accessor<int, 1>();
    
    shape.resize(output_shape.size(0));
    
    for (const auto i : otter::irange(output_shape.size(0))) {
        shape[i] = output_shape_a[i];
    }
    
    return 0;
}

int ReshapeLayer::forward_inplace(Tensor& bottom_blob, const NetOption& opt) const {
    bottom_blob = bottom_blob.view(shape);
    
    return 0;
}

}   // end namespace otter
