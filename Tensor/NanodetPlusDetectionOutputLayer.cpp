//
//  NanodetPlusDetectionOutputLayer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/5.
//

#include "NanodetPlusDetectionOutputLayer.hpp"
#include "TensorFactory.hpp"
#include "TensorMaker.hpp"
#include "TensorInterpolation.hpp"
#include "Padding.hpp"
#include <float.h>

namespace otter {

NanodetPlusDetectionOutputLayer::NanodetPlusDetectionOutputLayer() {
    one_blob_only = false;
    support_inplace = false;
}

int NanodetPlusDetectionOutputLayer::parse_param(LayerOption& option, ParamDict& pd) {
    float prob_threshold = opt_find_float(option, "prob_threshold", 0.4f);
    float nms_threshold = opt_find_float(option, "nms_threshold", 0.5f);
    
    Tensor stride;
    
    if (!opt_check_string(option, "stride")) {
        stride = otter::tensor({8, 16, 32, 64}, otter::ScalarType::Int);
    } else {
        int num_stride = (int)std::count(option["stride"].begin(), option["stride"].end(), ',') + 1;
        stride = otter::empty({num_stride}, otter::ScalarType::Int);
        auto stride_a = stride.accessor<int, 1>();
        std::stringstream ss;
        ss << option["stride"];
        int n;
        char c;
        for (const auto i : otter::irange(num_stride)) {
            ss >> n >> c;
            stride_a[i] = n;
        }
    }
    
    pd.set((int)NanodetPlusParam::Prob_threshold, prob_threshold);
    pd.set((int)NanodetPlusParam::Nms_threshold, nms_threshold);
    pd.set((int)NanodetPlusParam::Stride, stride);
    
    return 0;
}

int NanodetPlusDetectionOutputLayer::compute_output_shape(ParamDict& /*pd*/) {
    return 0;
}

int NanodetPlusDetectionOutputLayer::load_param(const ParamDict &pd) {
    prob_threshold = pd.get((int)NanodetPlusParam::Prob_threshold, 0.45f);
    nms_threshold = pd.get((int)NanodetPlusParam::Nms_threshold, 0.5f);
    stride = pd.get((int)NanodetPlusParam::Stride, otter::tensor({8, 16, 32, 64}, otter::ScalarType::Int));
    
    return 0;
}

int NanodetPlusDetectionOutputLayer::forward(const std::vector<Tensor>& bottom_blobs, std::vector<Tensor>& top_blobs, const NetOption& /*opt*/) const {
    
    std::vector<Object> proposals;
    
    auto stride_a = stride.accessor<int, 1>();
    for (const auto i : otter::irange(bottom_blobs.size())) {
        std::vector<Object> objects;
        
        generate_proposals(bottom_blobs[i], stride_a[i], prob_threshold, objects);
        proposals.insert(proposals.end(), objects.begin(), objects.end());
    }
    
    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = (int)picked.size();
    
    otter::Tensor& top_blob = top_blobs[0];
    top_blob = otter::empty({count, 6}, otter::ScalarType::Float);
    if (!top_blob.defined())
        return -100;
    
    auto top_blob_a = top_blob.accessor<float, 2>();
    
    for (const auto i : otter::irange(count)) {
        const Object& r = proposals[picked[i]];
        float score = r.prob;
        float* outptr = top_blob_a[i].data();
        
        outptr[0] = static_cast<float>(r.label + 1); // +1 for prepend background class
        outptr[1] = score;
        outptr[2] = r.x;
        outptr[3] = r.y;
        outptr[4] = r.width;
        outptr[5] = r.height;
    }
    
    return 0;
}

using Object = NanodetPlusDetectionOutputLayer::Object;

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

void NanodetPlusDetectionOutputLayer::generate_proposals(const otter::Tensor& pred, int stride, float prob_threshold, std::vector<Object>& objects) const {
//    const int num_grid = pred.size(2);
    
    int num_grid_x = (int)pred.size(3);
    int num_grid_y = (int)pred.size(2);
    
    const int num_class = 80; // number of classes. 80 for COCO
    const int reg_max_1 = ((int)pred.size(1) - num_class) / 4;
    
    auto pred_a = pred.accessor<float, 4>()[0];
    
    for (int i = 0; i < num_grid_y; i++) {
        for (int j = 0; j < num_grid_x; j++) {
            // find label with max score
            int label = -1;
            float score = -FLT_MAX;
            for (int k = 0; k < num_class; k++) {
                float s = pred_a[k][i][j];
                if (s > score) {
                    label = k;
                    score = s;
                }
            }
            
            score = sigmoid(score);
            
            if (score >= prob_threshold) {
                otter::Tensor bbox_pred = otter::empty({4, reg_max_1}, otter::ScalarType::Float);
                auto bbox_pred_a = bbox_pred.accessor<float, 2>();
                float* ptr = bbox_pred.data_ptr<float>();
                for (int k = 0; k < reg_max_1 * 4; k++) {
                    ptr[k] = pred_a[num_class + k][i][j];
                }
                // TODO: Softmax
                {
                    int w = reg_max_1;
                    int h = 4;
                    
                    for (int i = 0; i < h; i++) {
                        float* ptr = bbox_pred_a[i].data();
                        float m = -FLT_MAX;
                        for (int j = 0; j < w; j++) {
                            m = std::max(m, ptr[j]);
                        }
                        
                        float s = 0.f;
                        for (int j = 0; j < w; j++) {
                            ptr[j] = static_cast<float>(exp(ptr[j] - m));
                            s += ptr[j];
                        }
                        
                        for (int j = 0; j < w; j++) {
                            ptr[j] /= s;
                        }
                    }
                }
                
                float pred_ltrb[4];
                for (int k = 0; k < 4; k++) {
                    float dis = 0.f;
                    const float* dis_after_sm = bbox_pred_a[k].data();
                    for (int l = 0; l < reg_max_1; l++) {
                        dis += l * dis_after_sm[l];
                    }
                    
                    pred_ltrb[k] = dis * stride;
                }
                
                float pb_cx = j * stride;
                float pb_cy = i * stride;
                
                float x0 = pb_cx - pred_ltrb[0];
                float y0 = pb_cy - pred_ltrb[1];
                float x1 = pb_cx + pred_ltrb[2];
                float y1 = pb_cy + pred_ltrb[3];
                
                Object obj;
                obj.x = x0;
                obj.y = y0;
                obj.width = x1 - x0;
                obj.height = y1 - y0;
                obj.label = label;
                obj.prob = score;
                
                objects.push_back(obj);
            }
        }
    }
}

static inline float intersection_area(const Object& a, const Object& b) {
    float axmin = a.x, axmax = a.x + a.width;
    float aymin = a.y, aymax = a.y + a.height;
    float bxmin = b.x, bxmax = b.x + b.width;
    float bymin = b.y, bymax = b.y + b.height;
    
    if (axmin > bxmax || axmax < bxmin || aymin > bymax || aymax < bymin) {
        // no intersection
        return 0.f;
    }
    
    float inter_width = std::min(axmax, bxmax) - std::max(axmin, bxmin);
    float inter_height = std::min(aymax, bymax) - std::max(aymin, bymin);
    
    return inter_width * inter_height;
}

void NanodetPlusDetectionOutputLayer::nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold) const {
    picked.clear();
    
    const int n = (int)faceobjects.size();
    
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].width * faceobjects[i].height;
    }
    
    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];
        
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];
            
            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }
        
        if (keep)
            picked.push_back(i);
    }
}

void NanodetPlusDetectionOutputLayer::qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right) const {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;
    
    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;
        
        while (faceobjects[j].prob < p)
            j--;
        
        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);
            
            i++;
            j--;
        }
    }
    
#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void NanodetPlusDetectionOutputLayer::qsort_descent_inplace(std::vector<Object>& faceobjects) const {
    if (faceobjects.empty())
        return;
    
    qsort_descent_inplace(faceobjects, 0, (int)faceobjects.size() - 1);
}

Tensor nanodet_pre_process(const Tensor& img, int target_size, float& scale, int& wpad, int& hpad) {
    int width  = (int)img.size(3);
    int height = (int)img.size(2);
    
    int w = width;
    int h = height;
    scale = 1.f;
    if (w > h) {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }
    
    auto resize = otter::Interpolate(img, {h, w}, {0, 0}, otter::InterpolateMode::BILINEAR, false);
    
    // padding to multiply of 32
    wpad = (w + 31) / 32 * 32 - w;
    hpad = (h + 31) / 32 * 32 - h;
    
    auto resize_pad = otter::constant_pad(resize, {wpad / 2, wpad - wpad / 2, hpad / 2, hpad - hpad / 2}, 0);
    
    const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[3] = {0.017429f, 0.017507f, 0.017125f};

    resize_pad[0][0] -= mean_vals[0];
    resize_pad[0][1] -= mean_vals[1];
    resize_pad[0][2] -= mean_vals[2];

    resize_pad[0][0] *= norm_vals[0];
    resize_pad[0][1] *= norm_vals[1];
    resize_pad[0][2] *= norm_vals[2];
    
    return resize_pad;
}

Tensor nanodet_post_process(const Tensor& pred, int image_width, int image_height, float scale, int wpad, int hpad) {
    int count = (int)pred.size(0);
    std::vector<Object> objects(count);
    
    otter::Tensor pred_fix = otter::empty_like(pred);
    
    auto pred_a = pred.accessor<float, 2>();
    auto pred_fix_a = pred_fix.accessor<float, 2>();
    
    for (int i = 0; i < count; ++i) {
        Object& object = objects[i];
        auto obj = pred_a[i];
        auto obj_fix = pred_fix_a[i];
        
        object.label  = obj[0];
        object.prob   = obj[1];
        object.x      = obj[2];
        object.y      = obj[3];
        object.width  = obj[4];
        object.height = obj[5];
        
        // adjust offset to original unpadded
        float x0 = (obj[2] - (wpad / 2)) / scale;
        float y0 = (obj[3] - (hpad / 2)) / scale;
        float x1 = (obj[2] + obj[4] - (wpad / 2)) / scale;
        float y1 = (obj[3] + obj[5] - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(image_width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(image_height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(image_width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(image_height - 1)), 0.f);
        
        obj_fix[0] = obj[0];
        obj_fix[1] = obj[1];
        obj_fix[2] = x0;
        obj_fix[3] = y0;
        obj_fix[4] = x1 - x0;
        obj_fix[5] = y1 - y0;
    }
    
    return pred_fix;
}

}   // end namespace otter
