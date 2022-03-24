#ifndef LineDection_HPP
#define LineDection_HPP

#include "OTensor.hpp"

namespace otter {
namespace cv {

struct Vec2f {
    Vec2f(float v0, float v1) {
        val[0] = v0;
        val[1] = v1;
    }

    float operator[](int index) {
        return val[index];
    }

    float val[2];
};

struct most_param {
    most_param(float t = 0, int m = 0) : theta(t), most(m) {}
    float theta;
    int most;
};

void HoughLinesStandard(const otter::Tensor& img, std::vector<Vec2f>& lines, float rho, float theta, int threshold, int linesMax, float min_theta, float max_theta);
most_param mostline(std::vector<Vec2f>& lines);
void draw_alllines(otter::Tensor& cdst, std::vector<Vec2f>& lines);
void draw_mostline(otter::Tensor& cdst, std::vector<Vec2f>& lines, float theta);
}
}

#endif