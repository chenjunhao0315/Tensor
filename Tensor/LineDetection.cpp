//
//  LineDetection.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/12.
//

#include "LineDetection.hpp"
#include "Drawing.hpp"
#include "TensorFactory.hpp"
#include "TensorOperator.hpp"
#include "Parallel.hpp"
#include <cmath>

namespace otter {
namespace cv {

using namespace std;

struct LinePolar {
    float rho;
    float angle;
};

struct hough_cmp_gt {
    hough_cmp_gt(const int* _aux) : aux(_aux) {}
    inline bool operator()(int l1, int l2) const {
        return aux[l1] > aux[l2] || (aux[l1] == aux[l2] && l1 < l2);
    }
    const int* aux;
};

void findLocalMaximums(int numrho, int numangle, int threshold, const int *accum, std::vector<int>& sort_buf) {
    for (int r = 0; r < numrho; r++) {
        for (int n = 0; n < numangle; n++) {
            int base = (n + 1) * (numrho + 2) + r + 1;
            if (accum[base] > threshold && accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] && accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2]) {
                sort_buf.push_back(base);
            }
        }
    }
}

void HoughLinesStandard(const otter::Tensor& img, std::vector<Vec2f>& lines, float rho, float theta, int threshold, int linesMax, float min_theta, float max_theta) {
    float irho = 1 / rho;
    
    const unsigned char* image = img.data_ptr<unsigned char>();
    int step = (int)(img.size(1) * img.size(2) * img.itemsize());
    int width = img.size(1);
    int height = img.size(0);
    
    // OTTER_CHECK(max_theta >= min_theta, "max_theta must be larger than min_theta");
    
    int numangle = std::round((max_theta - min_theta) / theta);
    int numrho = std::round(((width + height) * 2 + 1) / rho);
    
    auto _accum = otter::zeros({(numangle + 2), (numrho + 2), 1}, otter::ScalarType::Int);
    std::vector<int> _sort_buf;
    int *accum = _accum.data_ptr<int>();
    
    auto ang = otter::linspace(static_cast<float>(min_theta), static_cast<float>(max_theta), numangle, otter::ScalarType::Float);
    auto _sinTab = otter::sin(ang) * irho;
    auto _cosTab = otter::cos(ang) * irho;
    ang.reset();
    float *tabSin = _sinTab.data_ptr<float>(), *tabCos = _cosTab.data_ptr<float>();
    
    // Fix i for parallel since accum
    for (int i = 0; i < height; i++) {
        const unsigned char *image_row = image + i * step;
        otter::parallel_for(0, width, 0, [&](int64_t begin, int64_t end) {
            for (const auto j : otter::irange(begin, end)) {
                if (image_row[j] != 0) {
                    for (int n = 0; n < numangle; n++) {
                        int r = std::round(j * tabCos[n] + i * tabSin[n]);
                        r += (numrho - 1) / 2;
                        accum[(n + 1) * (numrho + 2) + r + 1]++;
                    }
                }
            }
        });
    }
    
    findLocalMaximums(numrho, numangle, threshold, accum, _sort_buf);
    
    std::sort(_sort_buf.begin(), _sort_buf.end(), hough_cmp_gt(accum));
    
    linesMax = std::min(linesMax, (int)_sort_buf.size());
    
    float scale = 1.0 / (numrho + 2);
    for (int i = 0; i < linesMax; i++) {
        LinePolar line;
        int idx = _sort_buf[i];
        int n = std::floor(idx * scale) - 1;
        int r = idx - (n + 1) * (numrho + 2) - 1;
        line.rho = (r - (numrho - 1) * 0.5f) * rho;
        line.angle = static_cast<float>(min_theta) + n * theta;
        lines.push_back(Vec2f(line.rho, line.angle));
    }
    
}

void draw_alllines(otter::Tensor& cdst, std::vector<Vec2f>& lines) {
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        
        otter::cv::Point pt1, pt2;
        float a = std::cos(theta), b = std::sin(theta);
        float x0 = a * rho, y0 = b * rho;
        pt1.x = std::round(x0 + 5000 * (-b));
        pt1.y = std::round(y0 + 5000 * a);
        pt2.x = std::round(x0 - 5000 * (-b));
        pt2.y = std::round(y0 - 5000 * a);
        otter::cv::line(cdst, pt1, pt2, otter::cv::getDefaultColor(otter::cv::GOLD), 1, otter::cv::LINE_AA);
    }
}

most_param mostline(std::vector<Vec2f>& lines) {
    int num_diff = 0;
    std::vector<std::vector<float>> angle(lines.size());
    for (int i = 0; i < lines.size(); ++i) {
        angle[i].resize(2);
    }
    
    for (size_t i = 0; i < lines.size(); i++) {
//        float rho = lines[i][0];
        float theta = lines[i][1];
        int same = -1;
        size_t j;
        
        for (j = 0; j < num_diff; j++) {
            if (theta == angle[j][0]) {
                same = j;
            }
        }
        if (same == -1) {
            angle[num_diff][0] = theta;
            num_diff++;
        } else {
            angle[same][1]++;
        }
    }
    
    int most = 0;
    for (int i = 0; i < num_diff; i++) {
        angle[i][1]++;
    }
    for (int i = 0; i < num_diff; i++) {
        if (angle[i][1] > angle[most][1]) {
            most = i;
        }
    }
    
    if (angle.size() == 0) {
        return {0, 0};
    }
    
    return {angle[most][0], (int)angle[most][1]};
}

void draw_mostline(otter::Tensor& cdst, std::vector<Vec2f>& lines, float theta) {
    for (size_t i = 0; i < lines.size(); i++) {
        if (lines[i][1] == theta) {
            float rho = lines[i][0];
            
            otter::cv::Point pt1, pt2;
            float a = std::cos(theta), b = std::sin(theta);
            float x0 = a * rho, y0 = b * rho;
            pt1.x = std::round(x0 + 5000 * (-b));
            pt1.y = std::round(y0 + 5000 * a);
            pt2.x = std::round(x0 - 5000 * (-b));
            pt2.y = std::round(y0 - 5000 * a);
            otter::cv::line(cdst, pt1, pt2, otter::cv::getDefaultColor(otter::cv::BLUE), 1, otter::cv::LINE_AA);
        }
    }
}

void demo(otter::Tensor& dst, otter::Tensor& cdst, std::vector<Vec2f>& lines, float mul) {
    otter::cv::most_param param;
    
    otter::cv::HoughLinesStandard(dst, lines, 2, M_PI / 90, dst.size(1) * mul, INT_MAX, 0, M_PI);
    param = mostline(lines);
    while (param.most <= 0) {
        lines.clear();
        mul = mul - 0.05;
        otter::cv::HoughLinesStandard(dst, lines, 2, M_PI / 90, dst.size(1) * mul, INT_MAX, 0, M_PI);
        param = mostline(lines);
    }
    while (param.most >= 20) {
        lines.clear();
        mul = mul + 0.05;
        otter::cv::HoughLinesStandard(dst, lines, 2, M_PI / 90, dst.size(1) * mul, INT_MAX, 0, M_PI);
        param = mostline(lines);
    }
    
    draw_mostline(cdst, lines, param.theta);
}

}   // end namespace cv
}   // end namespace otter
