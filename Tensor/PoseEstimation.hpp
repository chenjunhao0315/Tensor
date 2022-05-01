//
//  PoseEstimation.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/4/29.
//

#ifndef PoseEstimation_hpp
#define PoseEstimation_hpp

#include "Tensor.hpp"
#include "GraphicAPI.hpp"

namespace otter {

struct KeyPoint {
    otter::cv::Point2f p;
    float prob;
};

struct PoseInput {
    otter::Tensor image;
    int w;
    int h;
    int x1;
    int y1;
};

PoseInput pose_pre_process(const Tensor& pred, const Tensor& img);

std::vector<KeyPoint> pose_post_process(const Tensor& pred, const PoseInput& proprocess);

static const int joint_pairs[16][2] = {
    {0, 1}, {1, 3}, {0, 2}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, {12, 14}, {13, 15}, {14, 16}
};

void draw_pose_detection(Tensor& img, std::vector<otter::KeyPoint>& keypoints);

}

#endif /* PoseEstimation_hpp */
