//
//  main.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "OTensor.hpp"
#include "Clock.hpp"
#include "Vision.hpp"
#include "Drawing.hpp"
#include <float.h>

using namespace std;

struct Object
{
    otter::cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    otter::cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;
    
    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;
        
        while (faceobjects[j].prob < p)
            j--;
        
        if (i <= j)
        {
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

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;
    
    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    
    const int n = faceobjects.size();
    
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
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

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

static void generate_proposals(const otter::Tensor& pred, int stride, const otter::Tensor& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = pred.size(2);
    
    int num_grid_x = pred.size(3);
    int num_grid_y = pred.size(2);
    
    const int num_class = 80; // number of classes. 80 for COCO
    const int reg_max_1 = (pred.size(1) - num_class) / 4;
    
    auto pred_a = pred.accessor<float, 4>()[0];
    
    for (int i = 0; i < num_grid_y; i++)
    {
        for (int j = 0; j < num_grid_x; j++)
        {
            // find label with max score
            int label = -1;
            float score = -FLT_MAX;
            for (int k = 0; k < num_class; k++)
            {
                float s = pred_a[k][i][j];
                if (s > score)
                {
                    label = k;
                    score = s;
                }
            }
            
            score = sigmoid(score);
            
            if (score >= prob_threshold)
            {
                otter::Tensor bbox_pred = otter::empty({4, reg_max_1}, otter::ScalarType::Float);
                auto bbox_pred_a = bbox_pred.accessor<float, 2>();
                float* ptr = bbox_pred.data_ptr<float>();
                for (int k = 0; k < reg_max_1 * 4; k++)
                {
                    ptr[k] = pred_a[num_class + k][i][j];
                }
                {
                    int w = reg_max_1;
                    int h = 4;
                    
                    for (int i = 0; i < h; i++)
                    {
                        float* ptr = bbox_pred_a[i].data();
                        float m = -FLT_MAX;
                        for (int j = 0; j < w; j++)
                        {
                            m = std::max(m, ptr[j]);
                        }
                        
                        float s = 0.f;
                        for (int j = 0; j < w; j++)
                        {
                            ptr[j] = static_cast<float>(exp(ptr[j] - m));
                            s += ptr[j];
                        }
                        
                        for (int j = 0; j < w; j++)
                        {
                            ptr[j] /= s;
                        }
                    }
                }
                
                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    const float* dis_after_sm = bbox_pred_a[k].data();
                    for (int l = 0; l < reg_max_1; l++)
                    {
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
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = label;
                obj.prob = score;
                
                objects.push_back(obj);
            }
        }
    }
}

int main(int argc, const char * argv[]) {
    
    otter::Net net;
    net.load_otter("nanodet-plus.otter", otter::CompileMode::Inference);
    net.summary();

    net.load_weight("nanodet-plus-m-1.5x_416-opt.bin");

    std::vector<Object> objects;

    otter::Clock l;
    auto img = otter::cv::load_image_rgb("5D4A0550cj.jpg");
    l.stop_and_show();
    
    int width = img.size(3);
    int height = img.size(2);

    //     const int target_size = 320;
    const int target_size = 416;
    const float prob_threshold = 0.4f;
    const float nms_threshold = 0.5f;

    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    auto in = otter::Interpolate(img, {h, w}, {0, 0}, otter::InterpolateMode::BILINEAR, false);

    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;

    auto in_pad = otter::constant_pad(in, {wpad / 2, wpad - wpad / 2, hpad / 2, hpad - hpad / 2}, 0);

    in_pad.print();

    const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
    const float norm_vals[3] = {0.017429f, 0.017507f, 0.017125f};

    in_pad[0][0] -= mean_vals[0];
    in_pad[0][1] -= mean_vals[1];
    in_pad[0][2] -= mean_vals[2];

    in_pad[0][0] *= norm_vals[0];
    in_pad[0][1] *= norm_vals[1];
    in_pad[0][2] *= norm_vals[2];

    auto ex = net.create_extractor();

    otter::Clock c;
    ex.input("data_1", in_pad);

    std::vector<Object> proposals;

    {
        otter::Tensor pred;
        ex.extract("conv_95", pred, 0);

        std::vector<Object> objects8;
        generate_proposals(pred, 8, in_pad, prob_threshold, objects8);

        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }
    {
        otter::Tensor pred;
        ex.extract("conv_100", pred, 0);

        std::vector<Object> objects16;
        generate_proposals(pred, 16, in_pad, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }
    {
        otter::Tensor pred;
        ex.extract("conv_105", pred, 0);

        std::vector<Object> objects32;
        generate_proposals(pred, 32, in_pad, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }
    {
        otter::Tensor pred;
        ex.extract("conv_110", pred, 0);

        std::vector<Object> objects64;
        generate_proposals(pred, 64, in_pad, prob_threshold, objects64);

        proposals.insert(proposals.end(), objects64.begin(), objects64.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    c.stop_and_show();

    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    otter::Clock i;
    auto image = img.to(otter::ScalarType::Byte).permute({0, 2, 3, 1}).squeeze(0).contiguous();
    i.stop_and_show();
    
    for (size_t i = 0; i < objects.size(); i++) {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        otter::cv::rectangle(image, obj.rect, otter::cv::Color(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        otter::cv::Size label_size = otter::cv::getTextSize(text, otter::cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.size(1))
            x = image.size(1) - label_size.width;

        otter::cv::rectangle(image, otter::cv::Rect(otter::cv::Point(x, y), otter::cv::Size(label_size.width, label_size.height + baseLine)),
                      otter::cv::Color(255, 255, 255), -1);

        otter::cv::putText(image, text, otter::cv::Point(x, y + label_size.height),
                    otter::cv::FONT_HERSHEY_SIMPLEX, 0.5, otter::cv::Color(0, 0, 0));
    }

    otter::cv::save_image(image, "test");
    
    return 0;
}
