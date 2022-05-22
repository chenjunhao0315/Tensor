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
#include "KalmanTracker.hpp"
#include <set>
#include "NanodetPlusDetectionOutputLayer.hpp"
#include "Hungarian.hpp"
#include "DrawDetection.hpp"
#include "Stabilizer.hpp"

using namespace std;

otter::Net net;

vector<otter::Object> read_image_and_predict(const char* file) {
    auto img = otter::cv::load_image_rgb(file);

    int width = img.size(3);
    int height = img.size(2);
    const int target_size = 416;

    float scale;
    int wpad, hpad;
    auto resize_pad = otter::nanodet_pre_process(img, target_size, scale, wpad, hpad);
    printf("Resize input (%d, %d) to (%d, %d)\n", width, height, (int)resize_pad.size(3), (int)resize_pad.size(2));

    auto ex = net.create_extractor();
    ex.input("data", resize_pad);
    otter::Tensor pred;
    ex.extract("nanodet", pred, 0);

    auto pred_fix = otter::nanodet_post_process(pred, width, height, scale, wpad, hpad);

    vector<otter::Object> objects;

    for (int i = 0; i < pred_fix.size(0); ++i) {
        auto obj_data = pred_fix[i].data_ptr<float>();

        otter::Object obj;
        obj.label = obj_data[0];
        obj.prob = obj_data[1];
        obj.rect.x = obj_data[2];
        obj.rect.y = obj_data[3];
        obj.rect.width = obj_data[4];
        obj.rect.height = obj_data[5];
        objects.push_back(obj);
    }

    return objects;
}

int main(int argc, const char * argv[]) {

    net.load_otter("nanodet-plus-m-1.5x_416.otter", otter::CompileMode::Inference);
    net.load_weight("nanodet-plus-m-1.5x_416-opt.bin", otter::Net::WeightType::Ncnn);

    otter::cv::KalmanTracker::kf_count = 0; // tracking id relies on this, so we have to reset it in each seq.

    vector<otter::core::TrackingBox> frameTrackingResult;

    vector<vector<otter::Object>> dets;

    auto img = otter::cv::load_image_pixel("img1.jpg");

    for (int i = 1; i <= 10; ++i) {
        auto detection = read_image_and_predict(("img" + to_string(i) + ".jpg").c_str());

        for (int j = 0; j < detection.size(); ++j) {
            otter::cv::rectangle(img, detection[j].rect, otter::cv::getDefaultColor(otter::cv::WHITE), 10);
            otter::cv::putText(img, to_string(i), otter::cv::Point(detection[j].rect.x, detection[j].rect.y), otter::cv::FONT_HERSHEY_SCRIPT_COMPLEX, 50, otter::cv::getDefaultColor(otter::cv::WHITE), 8);
        }

        dets.push_back(detection);
    }
    
    otter::core::Stabilizer stabilizer;
    
    for (int iter = 0; iter < dets.size(); ++iter) {
        
        frameTrackingResult = stabilizer.track(dets[iter]);
    
        for (auto tb : frameTrackingResult)
            cout << tb.frame << "," << tb.id << "," << tb.obj.label << "," << tb.obj.prob << "," << tb.obj.rect.x << "," << tb.obj.rect.y << "," << tb.obj.rect.width << "," << tb.obj.rect.height << ",1,-1,-1,-1" << endl;

        for (auto tb : frameTrackingResult) {
            int offset = tb.id * 123457 % 80;
            float red = otter::get_color(2, offset, 80);
            float green = otter::get_color(1, offset, 80);
            float blue = otter::get_color(0, offset, 80);
            otter::cv::putText(img, to_string(tb.id), otter::cv::Point(tb.obj.rect.x, tb.obj.rect.y), otter::cv::FONT_HERSHEY_SIMPLEX, 50, otter::cv::getDefaultColor(otter::cv::GOLD), 3);
            otter::cv::rectangle(img, tb.obj.rect, otter::cv::Color(red * 255, green * 255, blue * 255), 2);
        }
    }

    otter::cv::save_image(img, "final");

    return 0;
}
