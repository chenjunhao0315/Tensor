#include "OTensor.hpp"
#include "Clock.hpp"
#include "Vision.hpp"
#include "Drawing.hpp"
#include "NanodetPlusDetectionOutputLayer.hpp"
#include "TensorTransform.hpp"
#include "PoseEstimation.hpp"
#include "DrawDetection.hpp"
#include "PackedData.hpp"

using namespace std;

int main(int argc, const char * argv[]) {
    otter::Net net;
    net.load_otter("nanodet-plus-m-1.5x_416_int8_mixed.otter", otter::CompileMode::Inference);
    int ret = net.load_weight("nanodet-plus-m-1.5x_416_int8_mixed.bin", otter::Net::WeightType::Ncnn);
    if (ret) {
        exit(-1);
    }
    auto net_profiler = net.create_extractor();
    net_profiler.benchmark_info("data_1", "nanodet", {1, 3, 416, 416});

    otter::Net pose;
    pose.load_otter("simplepose_fused.otter", otter::CompileMode::Inference);
    ret = pose.load_weight("simplepose-opt.bin", otter::Net::WeightType::Ncnn);
    if (ret) {
        exit(-1);
    }

    otter::Clock l;
    auto img = otter::cv::load_image_rgb("大合照.jpg");
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
    auto image_final = img.to(otter::ScalarType::Byte).permute({0, 2, 3, 1}).squeeze(0).contiguous();
    i.stop_and_show("ms (nchw -> nhwc)");

    otter::Clock p;
    int person_count = 0;
    auto pred_fix_a = pred_fix.accessor<float, 2>();
    for (int i = 0; i < pred_fix.size(0); ++i) {
        if (pred_fix_a[i][0] == 1) {
            person_count++;

            auto pose_input = pose_pre_process(pred_fix[i], img);

            otter::Clock p;
            auto ex = pose.create_extractor();
            ex.input("data_1", pose_input.image);

            otter::Tensor out;
            ex.extract("conv_56", out, 0);
            p.stop_and_show("ms (pose net)");

            auto keypoints = otter::pose_post_process(out, pose_input);

            otter::draw_pose_detection(image_final, keypoints);
        }
    }
    p.stop_and_show("ms (pose net total)");

    otter::cv::save_image(image_final, "simplepose");
    
    return 0;
}

