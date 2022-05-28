//
//  DrawDetection.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/5.
//

#include "DrawDetection.hpp"
#include "Drawing.hpp"
#include <cmath>

namespace otter {

float get_color(int c, int x, int max);

static const char* coco_class_names[] = {
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

Object::Object(const otter::Tensor& obj) {
    OTTER_CHECK(obj.scalar_type() == otter::ScalarType::Float, "Expect float type");
    OTTER_CHECK(obj.dim() == 1 && obj.numel() == 6, "Unsupport format, expect [6] but get", obj.sizes());
    
    auto data = obj.accessor<float, 1>();
    label       = data[0];
    prob        = data[1];
    rect.x      = data[2];
    rect.y      = data[3];
    rect.width  = data[4];
    rect.height = data[5];
}

void draw_coco_detection(otter::Tensor& image, const otter::Tensor& pred, int width, int height) {
#if OTTER_OPENCV_DRAW
    size_t count = pred.size(0);
    
    std::vector<Object> objects(count);
    
    auto pred_a = pred.accessor<float, 2>();
    for (size_t i = 0; i < count; ++i) {
        Object& object = objects[i];
        auto obj = pred_a[i];
        
        object.label       = obj[0];
        object.prob        = obj[1];
        object.rect.x      = obj[2];
        object.rect.y      = obj[3];
        object.rect.width  = obj[4];
        object.rect.height = obj[5];
    }
    
    for (const auto i : otter::irange(objects.size())) {
        const Object& obj = objects[i];

        fprintf(stderr, "Label: %s (%.5f) x: %.2f y: %.2f width: %.2f height: %.2f\n", coco_class_names[obj.label], obj.prob, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
        
        int x1 = obj.rect.x;
        int y1 = obj.rect.y;
        int x2 = obj.rect.x + obj.rect.width;
        int y2 = obj.rect.y + obj.rect.height;

        int h = height * 0.02;
        if (h < 22)
            h = 22;
        
        const int classes = 80;
        int offset = obj.label * 123457 % classes;
        float red = get_color(2, offset, classes);
        float green = get_color(1, offset, classes);
        float blue = get_color(0, offset, classes);
        
        int text_size = std::max(height / 500, 1);
        int line_width = std::floor(std::min(width, height) / 1000.0) + 1;
        
        otter::cv::Color color(red * 255, green * 255, blue * 255);

        auto rect_size = getTextSize(coco_class_names[obj.label], otter::cv::FONT_HERSHEY_SIMPLEX, getFontScaleFromHeight(otter::cv::FONT_HERSHEY_SIMPLEX, h, text_size), text_size, nullptr);
        
        otter::cv::Rect rect((x1 - line_width / 2 < 0) ? 0 : x1 - line_width / 2, ((y1 - h * 1.8 + line_width / 2) < 0) ? 0 : y1 - h * 1.8 + line_width / 2, rect_size.width * 1.2, rect_size.height * 1.8);
        otter::cv::rectangle(image, rect, color, -1);
        otter::cv::rectangle(image, otter::cv::Point(x1, y1), otter::cv::Point(x2, y2), color, line_width);

        otter::cv::putText(image, coco_class_names[obj.label], otter::cv::Point((x1 - line_width / 2 < 0) ? 0 : x1 + rect_size.width * 0.1 - line_width / 2, ((y1 - h * 0.4) < 0) ? 0 : y1 - h * 0.4), otter::cv::FONT_HERSHEY_SIMPLEX, h, otter::cv::Color(0, 0, 0), text_size, otter::cv::LINE_AA, false);
    }
#endif // OTTER_OPENCV_DRAW
}

float get_color(int c, int x, int max) {
    float ratio = ((float)x / max) * 5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
    return r;
}

}   // end namespace otter
