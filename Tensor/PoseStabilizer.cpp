//
//  PoseStabilizer.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/10/7.
//

#include "PoseStabilizer.hpp"

namespace otter {
namespace cv {

std::vector<KeyPoint> PoseStabilizer::track(std::vector<KeyPoint> keypoints) {
    if (kpts.size() > 0) {
        for (size_t i = 0; i < kpts.size(); ++i) {
            kpts[i].update(keypoints[i].p);
        }
    } else {
        for (auto& keypoint : keypoints) {
            kpts.push_back(KalmanPointTracker(keypoint.p));
        }
    }
    
    for (size_t i = 0; i < kpts.size(); ++i) {
        keypoints[i].p = kpts[i].predict();
    }
    
    return keypoints;
}

}   // end namespace cv
}   // end namespace otter
