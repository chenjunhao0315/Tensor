//
//  Yolov3DetectionOutputLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/6.
//

#include "Yolov3DetectionOutputLayer.hpp"
#include "LayerRegistry.hpp"

namespace otter {

Yolov3DetectionOutputLayer::Yolov3DetectionOutputLayer() {
    one_blob_only = false;
    support_inplace = false;
}

int Yolov3DetectionOutputLayer::parse_param(LayerOption& option, ParamDict& pd) {
    return 0;
}

int Yolov3DetectionOutputLayer::compute_output_shape(ParamDict &pd) {
    return 0;
}

int Yolov3DetectionOutputLayer::load_param(const ParamDict &pd) {
    return 0;
}

int Yolov3DetectionOutputLayer::forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& opt) const {
    return 0;
}

REGISTER_LAYER_CLASS(Yolov3DetectionOutput);

}   // end namespace otter
