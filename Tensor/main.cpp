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
#include "TensorTransform.hpp"
#include "DrawDetection.hpp"
#include "PoseEstimation.hpp"
#include "AutoBuffer.hpp"
#include "TensorLinearAlgebra.hpp"
#include "PackedData.hpp"
#include "EmptyTensor.hpp"

using namespace std;

int main(int argc, const char * argv[]) {
    
//    auto A = otter::tensor({2926.36, 2607.79, 1.,
//        587.093, 2616.89, 1.,
//        537.031, 250.311, 1.,
//        1160.53, 1265.21, 1.}, otter::ScalarType::Float).view({4, 3});
//
//    auto B = otter::tensor({320.389, 208.197, 1.,
//        247.77, 209.726, 1.,
//        242.809, 283.182, 1.,
//        263.152, 253.715, 1.}, otter::ScalarType::Float).view({4, 3});
//
//    otter::Tensor X;
//    std::cout << "A=" << std::endl << A << std::endl;
//    std::cout << "B=" << std::endl << B << std::endl;
//
//    otter::solve(A, B, X, otter::DECOMP_SVD);
//
//    std::cout << "X=" << std::endl << X << std::endl;
    
    otter::Net net;
    net.load_otter("nanodet-plus-m-1.5x_416.otter", otter::CompileMode::Inference);

    net.load_weight("nanodet-plus-m-1.5x_416-opt.bin", otter::Net::WeightType::Ncnn);

    otter::Net pose;
    pose.load_otter("simplepose.otter", otter::CompileMode::Inference);
    pose.load_weight("simplepose-opt.bin", otter::Net::WeightType::Ncnn);

    otter::Clock l;
    auto img = otter::cv::load_image_rgb("5D4A0550cj.jpg");
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
    ex.input("data", resize_pad);

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

            auto preprocess = otter::pose_pre_process(pred_fix[i], img);
            auto in = preprocess.image;

            otter::Clock p;
            auto ex = pose.create_extractor();
            ex.input("data_1", in);

            otter::Tensor out;
            ex.extract("conv_56", out, 0);
            p.stop_and_show("ms (pose net)");

            auto keypoints = otter::pose_post_process(out, preprocess);

            otter::draw_pose_detection(image_final, keypoints);

        }
    }
    p.stop_and_show("ms (pose net total)");

    otter::draw_coco_detection(image_final, pred_fix, width, height);

    otter::cv::save_image(image_final, "final");
    
    return 0;
}
