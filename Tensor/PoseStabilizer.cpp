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
            checklist[i] += keypoints[i].prob > 0.2 ? 1 : -5;
        }
    } else {
        checklist.resize(keypoints.size(), 0);
        for (auto& keypoint : keypoints) {
            kpts.push_back(KalmanPointTracker(keypoint.p));
        }
        return keypoints;
    }
    
    for (size_t i = 0; i < kpts.size(); ++i) {
        keypoints[i].p = kpts[i].predict();
        if (keypoints[i].prob < 0 && checklist[i] > 0) {
            keypoints[i].prob = 0.25;
        }
    }
    
    return keypoints;
}

}   // end namespace cv
}   // end namespace otter
