//
//  Yolov3DetectionOutputLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/6.
//

#include "Yolov3DetectionOutputLayer.hpp"
#include "LayerRegistry.hpp"
#include "Parallel.hpp"

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
    
    std::vector<BBox> all_bbox;
    
    auto mask_a = mask.accessor<float, 1>();
    auto biases_a = biases.accessor<float, 1>();
    auto anchors_scale_a = anchors_scale.accessor<float, 1>();
    
    for (const auto i : otter::irange(bottom_blobs.size())) {
        std::vector<BBox> bbox_list;
        bbox_list.resize(num_box);
        
        const Tensor& bottom = bottom_blobs[i];
        auto bottom_a = bottom.accessor<float, 4>()[0]; // assume batch_size = 1
        
        int channels = bottom.size(1);
        int height   = bottom.size(2);
        int width    = bottom.size(3);
        const int channels_per_box = channels / num_box;
        
        if (channels_per_box != 4 + 1 + num_class) {
            fprintf(stderr, "[Yolov3DetectionOutput] Channel unmatched!\n");
            return -1;
        }
        
        size_t mask_offset = i * num_box;
        int net_h = (int)(anchors_scale_a[i] * width);
        int net_w = (int)(anchors_scale_a[i] * height);
        
        otter::parallel_for(0, num_box, 0, [&](int64_t start, int64_t end) {
            for (const auto pp : otter::irange(start, end)) {
                int p = pp * channels_per_box;
                int biases_index = static_cast<int>(mask_a[pp + mask_offset]);
                
                const float bias_w = biases_a[biases_index * 2];
                const float bias_h = biases_a[biases_index * 2 + 1];
                
                const float* xptr = bottom_a[p].data();
                const float* yptr = bottom_a[p + 1].data();
                const float* wptr = bottom_a[p + 2].data();
                const float* hptr = bottom_a[p + 3].data();
                
                const float* box_score_ptr = bottom_a[p + 4].data();
                
                Tensor scores = bottom[0].slice(0, p + 5, p + 5 + num_class);
            }
        });
    }
    
    return 0;
}

REGISTER_LAYER_CLASS(Yolov3DetectionOutput);

}   // end namespace otter
