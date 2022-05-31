#include "OTensor.hpp"
#include "Clock.hpp"
#include "Vision.hpp"
#include "Drawing.hpp"
#include "NanodetPlusDetectionOutputLayer.hpp"
#include "DrawDetection.hpp"

using namespace std;

int main(int argc, const char * argv[]) {
    
    if (argc < 2) {
        printf("Usage: %s image_path\n", argv[0]);
        return -1;
    }
    
    otter::Net net;
    net.load_otter("nanodet-plus-m-1.5x_416-opt.otter", otter::CompileMode::Inference);
    net.summary();

    net.load_weight("nanodet-plus-m-1.5x_416-opt.bin", otter::Net::WeightType::Ncnn);

    otter::Clock l;
    auto img = otter::cv::load_image_rgb(argv[1]);
    l.stop_and_show("ms (read image)");

    int width = img.size(3);
    int height = img.size(2);
    const int target_size = (argc > 2) ? std::atoi(argv[2]) : 416;

    float scale;
    int wpad, hpad;
    auto resize_pad = otter::nanodet_pre_process(img, target_size, scale, wpad, hpad);
    printf("Resize input (%d, %d) to (%d, %d)\n", width, height, (int)resize_pad.size(3), (int)resize_pad.size(2));

    auto ex = net.create_extractor();

    otter::Clock c;
    ex.input("data_1", resize_pad);

    otter::Tensor pred;
    ex.extract("nanodet", pred, 0);

    auto pred_fix = otter::nanodet_post_process(pred, width, height, scale, wpad, hpad);

    c.stop_and_show("ms (nanodet)");

    otter::Clock i;
    auto image = img.to(otter::ScalarType::Byte).permute({0, 2, 3, 1}).squeeze(0).contiguous();
    i.stop_and_show("ms (nchw -> nhwc)");

    otter::draw_coco_detection(image, pred_fix, width, height);
    otter::cv::save_image(image, "nanodet-plus");
    
    return 0;
}

