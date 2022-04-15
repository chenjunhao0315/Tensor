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

using namespace std;

struct KeyPoint
{
    otter::cv::Point2f p;
    float prob;
};

int main(int argc, const char * argv[]) {
    
    otter::Net net;
    net.load_otter("nanodet-plus-m-1.5x_416.otter", otter::CompileMode::Inference);
    
    net.load_weight("nanodet-plus-m-1.5x_416-opt.bin", otter::Net::WeightType::Ncnn);
    
    otter::Net pose;
    pose.load_otter("simplepose.otter", otter::CompileMode::Inference);
    pose.load_weight("simplepose-opt.bin", otter::Net::WeightType::Ncnn);
    
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
            
            int x1 = pred_fix_a[i][2];
            int y1 = pred_fix_a[i][3];
            int x2 = x1 + pred_fix_a[i][4];
            int y2 = y1 + pred_fix_a[i][5];
            
            int pw = x2 - x1;
            int ph = y2 - y1;
            int cx = x1 + 0.5 * pw;
            int cy = y1 + 0.5 * ph;
            
            x1 = cx - 0.7 * pw;
            y1 = cy - 0.6 * ph;
            x2 = cx + 0.7 * pw;
            y2 = cy + 0.6 * ph;
            
            if (x1 < 0) x1 = 0;
            if (y1 < 0) y1 = 0;
            if (x2 < 0) x2 = 0;
            if (y2 < 0) y2 = 0;
            
            if(x1 > width)  x1 = width;
            if(y1 > height) y1 = height;
            if(x2 > width)  x2 = width;
            if(y2 > height) y2 = height;
            
            auto image = otter::crop(img, {static_cast<long long>(x1), static_cast<long long>(img.size(3) - x2), static_cast<long long>(y1), static_cast<long long>(img.size(2) - y2)});
            
            int w = image.size(3);
            int h = image.size(2);
            
            auto in = otter::Interpolate(image, {256, 192}, {}, otter::InterpolateMode::BILINEAR);
            
            const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
            const float norm_vals[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f};
            
            in[0][0] -= mean_vals[0];
            in[0][1] -= mean_vals[1];
            in[0][2] -= mean_vals[2];
            in[0][0] *= norm_vals[0];
            in[0][1] *= norm_vals[1];
            in[0][2] *= norm_vals[2];
            
            otter::Clock p;
            auto ex = pose.create_extractor();
            ex.input("data_1", in);
            
            otter::Tensor out;
            ex.extract("conv_56", out, 0);
            p.stop_and_show("ms (pose net)");
            
            std::vector<KeyPoint> keypoints;
            
            for (int p = 0; p < out.size(1); p++) {
                const otter::Tensor m = out[0][p];
                
                float max_prob = 0.f;
                int max_x = 0;
                int max_y = 0;
                for (int y = 0; y < out.size(2); y++) {
                    const float* ptr = m[y].data_ptr<float>();
                    for (int x = 0; x < out.size(3); x++)
                    {
                        float prob = ptr[x];
                        if (prob > max_prob)
                        {
                            max_prob = prob;
                            max_x = x;
                            max_y = y;
                        }
                    }
                }
                
                KeyPoint keypoint;
                keypoint.p = otter::cv::Point2f(max_x * w / (float)out.size(3) + x1, max_y * h / (float)out.size(2) + y1);
                keypoint.prob = max_prob;
                keypoints.push_back(keypoint);
            }
            
            static const int joint_pairs[16][2] = {
                {0, 1}, {1, 3}, {0, 2}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}
            };
            for (int i = 0; i < 16; i++)
            {
                const KeyPoint& p1 = keypoints[joint_pairs[i][0]];
                const KeyPoint& p2 = keypoints[joint_pairs[i][1]];
                if (p1.prob < 0.2f || p2.prob < 0.2f)
                    continue;
                otter::cv::line(image_final, p1.p, p2.p, otter::cv::Color(255, 0, 0), 5);
            }
            // draw joint
            for (size_t i = 0; i < keypoints.size(); i++)
            {
                const KeyPoint& keypoint = keypoints[i];
                //fprintf(stderr, "%.2f %.2f = %.5f\n", keypoint.p.x, keypoint.p.y, keypoint.prob);
                if (keypoint.prob < 0.2f)
                    continue;
                otter::cv::circle(image_final, keypoint.p, 7, otter::cv::Color(0, 255, 0), -1);
            }
            
        }
    }
    p.stop_and_show("ms (pose net total)");
    
    otter::cv::save_image(image_final, "final");
    
    return 0;
}
