//
//  PoseEstimation.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/29.
//

#include "PoseEstimation.hpp"
#include "TensorInterpolation.hpp"
#include "TensorTransform.hpp"
#include "Drawing.hpp"

namespace otter {

PoseInput pose_pre_process(const Tensor& pred, const Tensor& img) {
    int width = (int)img.size(3);
    int height = (int)img.size(2);
    
    auto pred_a = pred.accessor<float, 1>();
    
    int x1 = pred_a[2];
    int y1 = pred_a[3];
    int x2 = x1 + pred_a[4];
    int y2 = y1 + pred_a[5];
    
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
    
    auto image = otter::crop(img, {static_cast<long long>(x1), static_cast<long long>(width - x2), static_cast<long long>(y1), static_cast<long long>(height - y2)});
    
    int w = image.size(3);
    int h = image.size(2);
    
    auto norm = otter::Interpolate(image, {256, 192}, {0, 0}, otter::InterpolateMode::BILINEAR, false).clone();
    
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_vals[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f};
    
    norm[0][0] -= mean_vals[0];
    norm[0][1] -= mean_vals[1];
    norm[0][2] -= mean_vals[2];
    norm[0][0] *= norm_vals[0];
    norm[0][1] *= norm_vals[1];
    norm[0][2] *= norm_vals[2];
    
    return {norm, w, h, x1, y1};
}

std::vector<KeyPoint> pose_post_process(const Tensor& pred, const PoseInput& preprocess) {
    auto in = preprocess.image;
    int x1 = preprocess.x1;
    int y1 = preprocess.y1;
    int w = preprocess.w;
    int h = preprocess.h;
    
    std::vector<KeyPoint> keypoints;
    
    for (int p = 0; p < pred.size(1); p++) {
        const otter::Tensor m = pred[0][p];
        
        float max_prob = 0.f;
        int max_x = 0;
        int max_y = 0;
        for (int y = 0; y < pred.size(2); y++) {
            const float* ptr = m.accessor<float, 2>()[y].data();
            for (int x = 0; x < pred.size(3); x++)
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
        keypoint.p = otter::cv::Point2f(max_x * w / (float)pred.size(3) + x1, max_y * h / (float)pred.size(2) + y1);
        keypoint.prob = max_prob;
        keypoints.push_back(keypoint);
    }
    
    return keypoints;
}

void draw_pose_detection(Tensor& img, std::vector<otter::KeyPoint>& keypoints) {
    for (int i = 0; i < 16; i++) {
        const otter::KeyPoint& p1 = keypoints[joint_pairs[i][0]];
        const otter::KeyPoint& p2 = keypoints[joint_pairs[i][1]];
        if (p1.prob < 0.2f || p2.prob < 0.2f)
            continue;
        otter::cv::line(img, p1.p, p2.p, otter::cv::getDefaultColor(otter::cv::SKY_BLUE), 5);
    }
    // draw joint
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        const otter::KeyPoint& keypoint = keypoints[i];
        if (keypoint.prob < 0.2f)
            continue;
        otter::cv::circle(img, keypoint.p, 7, otter::cv::Color(0, 255, 0), -1);
    }
}

}   // end namespace otter
