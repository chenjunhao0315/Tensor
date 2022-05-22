//
//  DrawDetection.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/5.
//

#ifndef DrawDetection_hpp
#define DrawDetection_hpp

#include "Tensor.hpp"
#include "GraphicAPI.hpp"

namespace otter {

struct Object {
    otter::cv::Rect_<float> rect;
    int label;
    float prob;
};

static float colors[6][3] = { {1,0,1}, {0,0,1}, {0,1,1}, {0,1,0}, {1,1,0}, {1,0,0} };

float get_color(int c, int x, int max);

void draw_coco_detection(otter::Tensor& image, const otter::Tensor& pred, int width, int height);

}   // end namespace otter 

#endif /* DrawDetection_hpp */
