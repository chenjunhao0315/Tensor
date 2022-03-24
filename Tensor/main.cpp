//
//  main.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "OTensor.hpp"
#include "Clock.hpp"
#include "Net.hpp"
#include "Vision.hpp"
#include "Drawing.hpp"
#include "Exception.hpp"
#include "LineDection.hpp"

using namespace std;

int main(int argc, const char * argv[]) {

    otter::Tensor cdst;
    otter::Tensor dst;

    auto t1 = otter::cv::load_image_pixel(argv[1]);
    t1 /= 255;

    dst = t1;
    cdst = otter::native::cat({t1 * 255, t1 * 255, t1 * 255}, 2);

    vector<otter::cv::Vec2f>lines;

    float mul = 0.5;
    otter::cv::most_param param;
    cout << "mul: " << mul << endl;

	otter::cv::HoughLinesStandard(dst, lines, 1, M_PI / 180, dst.size(1) * mul, INT_MAX, 0, M_PI);
    cout << lines.size() << endl;
    param = mostline(lines);
    cout << "most: " << param.most << endl;
    while (param.most >= 20) {
        cout << mul << endl;
        mul = mul + 0.05;
        cout << "++" << endl;
        cout << mul << endl;
        otter::cv::HoughLinesStandard(dst, lines, 1, M_PI / 180, dst.size(1) * mul, INT_MAX, 0, M_PI);
        param = mostline(lines);
        cout << "most: " << param.most << endl;
    }


    while (param.most <= 0) {
        cout << mul << endl;
        mul = mul - 0.05;
        cout << "--" << endl;
        cout << mul << endl;
        otter::cv::HoughLinesStandard(dst, lines, 1, M_PI / 180, dst.size(1) * mul, INT_MAX, 0, M_PI);
        param = mostline(lines);
        cout << "most: " << param.most << endl;
    }

//    draw_alllines(cdst, lines);
    draw_mostline(cdst, lines, param.theta);

    otter::cv::save_image(cdst, "shark");

    return 0;
}