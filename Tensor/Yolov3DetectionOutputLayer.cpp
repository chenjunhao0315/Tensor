//
//  Yolov3DetectionOutputLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/3/6.
//

#include "Yolov3DetectionOutputLayer.hpp"
#include "LayerRegistry.hpp"
#include "Parallel.hpp"
#include "TensorFactory.hpp"

#include <float.h>

namespace otter {

Yolov3DetectionOutputLayer::Yolov3DetectionOutputLayer() {
    one_blob_only = false;
    support_inplace = false;
}

int Yolov3DetectionOutputLayer::parse_param(LayerOption& option, ParamDict& pd) {
    int num_class = opt_find_int(option, "num_class", 80);
    int num_box = opt_find_int(option, "num_box", 3);
    float confidence_threshold = opt_find_float(option, "confidence_threshold", 0.25f);
    float nms_threshold = opt_find_float(option, "nms_threshold", 0.45f);
    float scale_x_y = opt_find_float(option, "scale_x_y", 1);
    
    int input_height = opt_find_int(option, "input_height", 416);
    int input_width = opt_find_int(option, "input_width", 416);
    
    Tensor mask;
    
    if (!opt_check_string(option, "mask")) {
        mask = otter::range(0, 2, 1, otter::ScalarType::Int);
    } else {
        int num_anchors = (int)std::count(option["mask"].begin(), option["mask"].end(), ',') + 1;
        mask = otter::empty({num_anchors}, otter::ScalarType::Int);
        auto mask_a = mask.accessor<int, 1>();
        std::stringstream ss;
        ss << option["mask"];
        int n;
        char c;
        for (const auto i : otter::irange(num_anchors)) {
            ss >> n >> c;
            mask_a[i] = n;
        }
    }
    
    int total_anchors = (int)std::count(option["anchor"].begin(), option["anchor"].end(), ',');
    Tensor biases = otter::empty({total_anchors * 2}, otter::ScalarType::Int);
    {
        auto biases_a = biases.accessor<int, 1>();
        
        std::stringstream ss;
        ss << option["anchor"];
        int w, h;
        char c;
        for (int i = 0; i < total_anchors; ++i) {
            ss >> w >> c >> h;
            biases_a[i * 2 + 0] = w;
            biases_a[i * 2 + 1] = h;
        }
    }
    
    int num_anchors_scale = (int)std::count(option["anchors_scale"].begin(), option["anchors_scale"].end(), ',') + 1;
    Tensor anchors_scale = otter::empty({num_anchors_scale}, otter::ScalarType::Float);
    {
        auto anchors_scale_a = anchors_scale.accessor<float, 1>();
        
        std::stringstream ss;
        ss << option["anchors_scale"];
        float scale;
        char c;
        for (int i = 0; i < num_anchors_scale; ++i) {
            ss >> scale >> c;
            anchors_scale_a[i] = scale;
        }
    }
    
    
    pd.set((int)Yolov3DetectionParam::Num_class, num_class);
    pd.set((int)Yolov3DetectionParam::Num_box, num_box);
    pd.set((int)Yolov3DetectionParam::Confidence_threshold, confidence_threshold);
    pd.set((int)Yolov3DetectionParam::Nms_threshold, nms_threshold);
    pd.set((int)Yolov3DetectionParam::Scale_x_y, scale_x_y);
    pd.set((int)Yolov3DetectionParam::Input_height, input_height);
    pd.set((int)Yolov3DetectionParam::Input_width, input_width);
    pd.set((int)Yolov3DetectionParam::Biases, biases);
    pd.set((int)Yolov3DetectionParam::Mask, mask);
    pd.set((int)Yolov3DetectionParam::Anchors_scale, anchors_scale);
    
    return 0;
}

int Yolov3DetectionOutputLayer::compute_output_shape(ParamDict &pd) {
    return 0;
}

int Yolov3DetectionOutputLayer::load_param(const ParamDict &pd) {
    num_class = pd.get((int)Yolov3DetectionParam::Num_class, 80);
    num_box = pd.get((int)Yolov3DetectionParam::Num_box, 90);
    confidence_threshold = pd.get((int)Yolov3DetectionParam::Confidence_threshold, 0.25f);
    nms_threshold = pd.get((int)Yolov3DetectionParam::Nms_threshold, 0.45f);
    
    biases = pd.get((int)Yolov3DetectionParam::Biases, Tensor());
    mask = pd.get((int)Yolov3DetectionParam::Mask, Tensor());
    anchors_scale = pd.get((int)Yolov3DetectionParam::Anchors_scale, Tensor());
    
    return 0;
}

static inline float intersection_area(const Yolov3DetectionOutputLayer::BBox& a, const Yolov3DetectionOutputLayer::BBox& b) {
    if (a.xmin > b.xmax || a.xmax < b.xmin || a.ymin > b.ymax || a.ymax < b.ymin) {
        // no intersection
        return 0.f;
    }
    
    float inter_width = std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin);
    float inter_height = std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin);
    
    return inter_width * inter_height;
}

void Yolov3DetectionOutputLayer::qsort_descent_inplace(std::vector<BBox>& datas, int left, int right) const {
    int i = left;
    int j = right;
    float p = datas[(left + right) / 2].score;
    
    while (i <= j) {
        while (datas[i].score > p)
            i++;
        
        while (datas[j].score < p)
            j--;
        
        if (i <= j) {
            std::swap(datas[i], datas[j]);
            
            i++;
            j--;
        }
    }
    
    if (left < j)
        qsort_descent_inplace(datas, left, j);
    
    if (i < right)
        qsort_descent_inplace(datas, i, right);
}

void Yolov3DetectionOutputLayer::qsort_descent_inplace(std::vector<BBox>& datas) const {
    if (datas.empty())
        return;
    
    qsort_descent_inplace(datas, 0, static_cast<int>(datas.size() - 1));
}

void Yolov3DetectionOutputLayer::nms_sorted_bboxes(std::vector<BBox>& bboxes, std::vector<size_t>& picked, float nms_threshold) const {
    picked.clear();
    
    const size_t n = bboxes.size();
    
    for (size_t i = 0; i < n; i++) {
        const BBox& a = bboxes[i];
        
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const BBox& b = bboxes[picked[j]];
            
            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = a.area + b.area - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area > nms_threshold * union_area) {
                keep = 0;
                break;
            }
        }
        
        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

int Yolov3DetectionOutputLayer::forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& opt) const {
    
    std::vector<BBox> all_bbox;
    
    auto mask_a = mask.accessor<int, 1>();
    auto biases_a = biases.accessor<int, 1>();
    auto anchors_scale_a = anchors_scale.accessor<float, 1>();
    
    for (const auto i : otter::irange(bottom_blobs.size())) {
        std::vector<std::vector<BBox> > bbox_map;
        bbox_map.resize(num_box);
        
        const Tensor& bottom = bottom_blobs[i];
        auto bottom_a = bottom.accessor<float, 4>()[0]; // assume batch_size = 1
        
        int channels = (int)bottom.size(1);
        int height   = (int)bottom.size(2);
        int width    = (int)bottom.size(3);
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
                int p = (int)pp * channels_per_box;
                int biases_index = static_cast<int>(mask_a[pp + mask_offset]);
                
                const float bias_w = biases_a[biases_index * 2];
                const float bias_h = biases_a[biases_index * 2 + 1];
                
                const float* xptr = bottom_a[p].data();
                const float* yptr = bottom_a[p + 1].data();
                const float* wptr = bottom_a[p + 2].data();
                const float* hptr = bottom_a[p + 3].data();
                
                const float* box_score_ptr = bottom_a[p + 4].data();
                
                Tensor scores = bottom[0].slice(0, p + 5, p + 5 + num_class);
                auto scores_a = scores.accessor<float, 3>();
                
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        // find class index with max class score
                        int class_index = 0;
                        float class_score = -FLT_MAX;
                        for (int q = 0; q < num_class; q++) {
                            float score = scores_a[q][i][j];
                            if (score > class_score) {
                                class_index = q;
                                class_score = score;
                            }
                        }
                        
                        //sigmoid(box_score) * sigmoid(class_score)
                        float confidence = 1.f / ((1.f + exp(-box_score_ptr[0]) * (1.f + exp(-class_score))));
                        if (confidence >= confidence_threshold) {
                            // region box
                            float bbox_cx = (j + sigmoid(xptr[0])) / width;
                            float bbox_cy = (i + sigmoid(yptr[0])) / height;
                            float bbox_w = static_cast<float>(exp(wptr[0]) * bias_w / net_w);
                            float bbox_h = static_cast<float>(exp(hptr[0]) * bias_h / net_h);
                            
                            float bbox_xmin = bbox_cx - bbox_w * 0.5f;
                            float bbox_ymin = bbox_cy - bbox_h * 0.5f;
                            float bbox_xmax = bbox_cx + bbox_w * 0.5f;
                            float bbox_ymax = bbox_cy + bbox_h * 0.5f;
                            
                            float area = bbox_w * bbox_h;
                            
                            BBox c = {class_index, confidence, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, area};
                            bbox_map[pp].push_back(c);
                        }
                        
                        xptr++;
                        yptr++;
                        wptr++;
                        hptr++;
                        
                        box_score_ptr++;
                    }
                }
            }
        });
        
        for (int i = 0; i < num_box; i++) {
            const std::vector<BBox>& bbox_list = bbox_map[i];
            
            all_bbox.insert(all_bbox.end(), bbox_list.begin(), bbox_list.end());
        }
    }
    
    // global sort inplace
    qsort_descent_inplace(all_bbox);
    
    // apply nms
    std::vector<size_t> picked;
    nms_sorted_bboxes(all_bbox, picked, nms_threshold);
    
    // select
    std::vector<BBox> bbox_selected;
    
    for (size_t i = 0; i < picked.size(); i++)
    {
        size_t z = picked[i];
        bbox_selected.push_back(all_bbox[z]);
    }
    
    // fill result
    int num_detected = static_cast<int>(bbox_selected.size());
    if (num_detected == 0)
        return 0;
    
    Tensor& top_blob = top_blobs[0];
    top_blob = otter::empty({num_detected, 6}, otter::ScalarType::Float);
    if (top_blob.defined())
        return -100;
    
    auto top_blob_a = top_blob.accessor<float, 2>();
    
    for (int i = 0; i < num_detected; i++) {
        const BBox& r = bbox_selected[i];
        float score = r.score;
        float* outptr = top_blob_a[i].data();
        
        outptr[0] = static_cast<float>(r.label + 1); // +1 for prepend background class
        outptr[1] = score;
        outptr[2] = r.xmin;
        outptr[3] = r.ymin;
        outptr[4] = r.xmax;
        outptr[5] = r.ymax;
    }
    
    return 0;
}

REGISTER_LAYER_CLASS(Yolov3DetectionOutput);

}   // end namespace otter
