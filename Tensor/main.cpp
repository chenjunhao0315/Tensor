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
#include "NanodetPlusDetectionOutputLayer.hpp"
#include "DrawDetection.hpp"

using namespace std;

int main(int argc, const char * argv[]) {
    
    otter::Net net;
    net.load_otter("nanodet-plus.otter", otter::CompileMode::Inference);
    net.summary();

    net.load_weight("nanodet-plus-m-1.5x_416-opt.bin", otter::Net::WeightType::Ncnn);

    otter::Clock l;
    auto img = otter::cv::load_image_rgb("5D4A0550cj.jpg");
    l.stop_and_show();
    
    int width = img.size(3);
    int height = img.size(2);
    const int target_size = 320;
    
    float scale;
    int wpad, hpad;
    auto resize_pad = otter::nanodet_pre_process(img, target_size, scale, wpad, hpad);
    printf("Resize input (%d, %d) to (%d, %d)\n", width, height, (int)resize_pad.size(3), (int)resize_pad.size(2));

    auto ex = net.create_extractor();

    otter::Clock c;
    ex.input("data", resize_pad);
    
    otter::Tensor pred;
    ex.extract("nanodet", pred, 0);
    
    auto pred_fix = otter::nanodet_post_process(pred, width, height, scale, wpad, hpad);

    c.stop_and_show();

    otter::Clock i;
    auto image = img.to(otter::ScalarType::Byte).permute({0, 2, 3, 1}).squeeze(0).contiguous();
    i.stop_and_show();
    
    otter::draw_coco_detection(image, pred_fix, width, height);
    otter::cv::save_image(image, "test");
    
    return 0;
}
