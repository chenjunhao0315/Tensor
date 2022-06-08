#include "OTensor.hpp"
#include "Clock.hpp"
#include "Vision.hpp"
#include "Drawing.hpp"
#include "NanodetPlusDetectionOutputLayer.hpp"
#include "DrawDetection.hpp"

using namespace std;

std::string model[] = {"nanodet-plus-m-1.5x_416_fused.otter", "nanodet-plus-m-1.5x_416_int8_fused.otter", "nanodet-plus-m-1.5x_416_int8_mixed.otter"};
std::string weight[] = {"nanodet-plus-m-1.5x_416_fused.bin", "nanodet-plus-m-1.5x_416_int8_fused.bin", "nanodet-plus-m-1.5x_416_int8_mixed.bin"};

int main(int argc, const char * argv[]) {
    
    if (argc < 2) {
        printf("Usage: %s image_path\n", argv[0]);
        return -1;
    }
    
   std::string model_name = model[0];
   std::string weight_name = weight[0];
    if (argc > 2) {
        model_name = model[std::atoi(argv[2])];
        weight_name = weight[std::atoi(argv[2])];
    }
    
    otter::Net net;
    net.load_otter(model_name.c_str(), otter::CompileMode::Inference);
    net.summary();

    int ret = net.load_weight(weight_name.c_str(), otter::Net::WeightType::Ncnn);
    if (ret) {
        exit(-1);
    }

    otter::Clock l;
    auto img = otter::cv::load_image_rgb(argv[1]);
    l.stop_and_show("ms (read image)");

    int width = img.size(3);
    int height = img.size(2);
    const int target_size = (argc > 3) ? std::atoi(argv[3]) : 416;

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

