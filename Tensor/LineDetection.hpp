//
//  LineDetection.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/12.
//

#ifndef LineDetection_hpp
#define LineDetection_hpp

#include "Tensor.hpp"
#include "GraphicAPI.hpp"

namespace otter {
namespace cv {

struct most_param {
    most_param(float t = 0, int m = 0) : theta(t), most(m) {}
    float theta;
    int most;
};

void HoughLinesStandard(const otter::Tensor& img, std::vector<Vec2f>& lines, float rho, float theta, int threshold, int linesMax, float min_theta, float max_theta);
most_param mostline(std::vector<Vec2f>& lines);
void draw_alllines(otter::Tensor& cdst, std::vector<Vec2f>& lines);
void draw_mostline(otter::Tensor& cdst, std::vector<Vec2f>& lines, float theta);
void demo(otter::Tensor& dst, otter::Tensor& cdst, std::vector<Vec2f>& lines, float mul);

}   // end namespace cv
}   // end namespace otter

#endif /* LineDetection_hpp */
