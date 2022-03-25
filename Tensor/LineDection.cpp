#include "LineDection.hpp"
#include "Drawing.hpp"
#include <Cmath>

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
    int i, j;
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

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            if (image[i * step + j] != 0) {
                for (int n = 0; n < numangle; n++) {
                    int r = std::round(j * tabCos[n] + i * tabSin[n]);
                    r += (numrho - 1) / 2;
                    accum[(n + 1) * (numrho + 2) + r + 1]++;
                }
            }
        }
    }


    findLocalMaximums(numrho, numangle, threshold, accum, _sort_buf);

    std::sort(_sort_buf.begin(), _sort_buf.end(), hough_cmp_gt(accum));

    linesMax = std::min(linesMax, (int)_sort_buf.size());

    float scale = 1.0 / (numrho + 2);
    for (i = 0; i < linesMax; i++) {
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
    float angle[lines.size()][2] = {0};

    for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
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

    float theta = angle[most][0];
    cout << theta << ": " << angle[most][1] << endl;

    return {theta, angle[most][1]};

    // while (angle[most][1] <= 0) {
    //     threshold = threshold - 0.05;
    //     cout << "<0" <<endl;
    //     HoughLinesStandard(cdst, lines, 1, M_PI / 180, cdst.size(1) * threshold, INT_MAX, 0, M_PI);
    //     draw_mostline(cdst, lines, threshold);
    // } 
    // while (angle[most][1] >= 20) {
    //     threshold = threshold + 0.05;
    //     cout << ">20" <<endl;
    //     HoughLinesStandard(cdst, lines, 1, M_PI / 180, cdst.size(1) * threshold, INT_MAX, 0, M_PI);
    //     draw_mostline(cdst, lines, threshold);
    // }

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

}
}